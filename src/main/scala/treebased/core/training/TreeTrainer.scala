package treebased.core.training

import treebased.config.{ForestConfig, TreeBaseConfig}
import treebased.core.prediction.LeafEstimator
import treebased.core.training.split.SplitRule
import treebased.core.training.split.guard.SplitGuard
import treebased.core.domain.data.DataPoint
import treebased.core.domain.tree.{LeafNode, Node, TreeNode}

import scala.reflect.ClassTag
import scala.util.Random
import scala.util.control.TailCalls.*


object TreeTrainer {

  def buildTree[O <: DataPoint : ClassTag](
    predictionData: Array[O],
    predictionRowIds: Array[Int],
    splitData: Array[O],
    sortedFeatureIndices: Array[Array[Int]],
    config: TreeBaseConfig,
    splitter: SplitRule[O],
    leaf: LeafEstimator[O, ?],
    sizeGuard: SplitGuard[O]
  ): Node = {
    val root = buildTreeRecursive(predictionData, predictionRowIds, splitData, sortedFeatureIndices, config, splitter, leaf, sizeGuard, curDepth = 0, parentNodeId = "0").result
    HonestyPruner.pruneEmptyHonestyLeaves(root)
  }

  /**
   * Shared honest-tree entry point behind [[treebased.api.CausalTree.trainTree]] and
   * [[treebased.api.GeneralizedRandomTree.trainTree]]: splits `trainingData` into the
   * split-determining half and the honest prediction half, remaps the (possibly
   * forest-subsampled) global sorted-feature-index arrays down to the split half's local
   * positions, and builds the tree. The two callers differ only in which [[SplitRule]] /
   * [[SplitGuard]] / [[LeafEstimator]] they inject — exact-causal vs. gradient splitting.
   */
  def trainTree[O <: DataPoint : ClassTag](
    trainingData: Array[O],
    config: TreeBaseConfig,
    splitter: SplitRule[O],
    leaf: LeafEstimator[O, ?],
    sizeGuard: SplitGuard[O],
    sortedSubSampleFeatureIndices: Option[Array[Array[Int]]] = None,
    subsampleRowIds: Array[Int] = Array.emptyIntArray
  ): Node = {
    val allSortedIndices = sortedSubSampleFeatureIndices.getOrElse(sortFeatureIndices(trainingData, config))
    val splitPoint = (trainingData.length * config.predictionDataRatio).toInt
    val (predictionData, splitData) = trainingData.splitAt(splitPoint)
    val rowIds = if (subsampleRowIds.isEmpty) Array.range(0, trainingData.length) else subsampleRowIds
    val predictionRowIds = rowIds.take(splitPoint)

    // Filter sorted indices to the split half and remap to local positions (0..splitData.length-1)
    val splitSortedIndices: Array[Array[Int]] =
      allSortedIndices.map { arr =>
        var count = 0; var i = 0
        while (i < arr.length) { if (arr(i) >= splitPoint) count += 1; i += 1 }
        val out = new Array[Int](count)
        var w = 0; i = 0
        while (i < arr.length) {
          if (arr(i) >= splitPoint) { out(w) = arr(i) - splitPoint; w += 1 }
          i += 1
        }
        out
      }

    buildTree(predictionData, predictionRowIds, splitData, splitSortedIndices, config, splitter, leaf, sizeGuard)
  }

  /**
   * Bootstrap subsample for a single tree, shared by [[treebased.api.CausalForest]] and
   * [[treebased.api.GeneralizedRandomForest]]: draws `config.subsampleRatio * N` rows without
   * replacement and filters/remaps the forest's global per-feature sorted-index arrays down to
   * the subsample's own positions, so the subsampled tree never re-sorts its features from
   * scratch.
   *
   * `rowPool` restricts the draw to a subset of the global row indices (the shared half-sample of
   * a bootstrap-of-little-bags group); `null` means "all N rows". The returned `sampledIndices`
   * are global either way, so honest-leaf membership still indexes into the forest's training
   * array. `count` is a fraction of N (not of the pool): `subsampleRatio` must not
   * exceed `ciGroupHalfFraction`, or the pool is smaller than the requested draw.
   */
  private[treebased] def subSampleTrainingData[O <: DataPoint : ClassTag](
    treeRng: Random,
    trainingData: Array[O],
    sortedFeatureIndices: Array[Array[Int]],
    config: ForestConfig,
    rowPool: Array[Int] = null
  ): (Array[O], Array[Array[Int]], Array[Int]) = {
    val N = trainingData.length
    val pool = if (rowPool == null) (0 until N).toArray else rowPool
    val count = math.min((config.subsampleRatio * N).toInt, pool.length)
    val sampledIndices = treeRng.shuffle(pool.toIndexedSeq).take(count).toArray

    // inSubsample(i) = true if original index i was sampled
    // subsamplePos(i) = position of original index i in the subsample array (0..count-1)
    val inSubsample  = new Array[Boolean](N)
    val subsamplePos = new Array[Int](N)
    var p = 0
    while (p < count) { inSubsample(sampledIndices(p)) = true; subsamplePos(sampledIndices(p)) = p; p += 1 }

    // Filter each feature's sorted indices to subsample members, remap to subsample positions.
    // Iteration order of sortedFeatureIndices(f) is feature-sorted, so output is also feature-sorted.
    val sortedSubSampleFeatureIndices: Array[Array[Int]] =
      sortedFeatureIndices.map { arr =>
        val out = new Array[Int](count)
        var w = 0; var i = 0
        while (i < arr.length) {
          if (inSubsample(arr(i))) { out(w) = subsamplePos(arr(i)); w += 1 }
          i += 1
        }
        out
      }

    (sampledIndices.map(trainingData).toArray, sortedSubSampleFeatureIndices, sampledIndices)
  }

  private def buildTreeRecursive[O <: DataPoint : ClassTag](
    predictionData: Array[O],
    predictionRowIds: Array[Int],
    splitData: Array[O],
    sortedFeatureIndices: Array[Array[Int]],
    config: TreeBaseConfig,
    splitter: SplitRule[O],
    leaf: LeafEstimator[O, ?],
    sizeGuard: SplitGuard[O],
    curDepth: Int,
    parentNodeId: String
  ): TailRec[Node] = {
    val maxDepthReached = config.maxDepth <= curDepth
    val enoughData      = sizeGuard.hasEnoughDataForSplit(splitData, config)

    if (maxDepthReached || !enoughData)
      done(LeafNode.build(predictionData, predictionRowIds, config, parentNodeId, curDepth, leaf))
    else
      val featureIndexesForSplit = splitter.candidateFeatureSplitIndices(config)
      splitter.findBestSplitPoint(splitData, featureIndexesForSplit, sortedFeatureIndices) match {
        case None =>
          done(LeafNode.build(predictionData, predictionRowIds, config, parentNodeId, curDepth, leaf))

        case Some(split) =>
          val (leftSplitData, rightSplitData) = partitionInPlace(splitData, row => split.goLeft(row.features))
          val (leftPredictionData, leftPredictionRowIds, rightPredictionData, rightPredictionRowIds) =
            partitionPaired(predictionData, predictionRowIds, row => split.goLeft(row.features))

          val leftNewIdx  = new Array[Int](splitData.length)
          val rightNewIdx = new Array[Int](splitData.length)
          var lw = 0; var rw = 0; var k = 0
          while (k < splitData.length) {
            if (split.goLeft(splitData(k).features)) { leftNewIdx(k) = lw;  lw += 1; rightNewIdx(k) = -1 }
            else                                     { rightNewIdx(k) = rw; rw += 1; leftNewIdx(k)  = -1 }
            k += 1
          }

          val leftSortedIndices  = sortedFeatureIndices.map(remapSorted(_, leftNewIdx,  lw))
          val rightSortedIndices = sortedFeatureIndices.map(remapSorted(_, rightNewIdx, rw))

          tailcall(buildTreeRecursive(leftPredictionData,  leftPredictionRowIds,  leftSplitData,  leftSortedIndices,  config, splitter, leaf, sizeGuard, curDepth + 1, parentNodeId + "L")).flatMap {
          leftNode =>
          tailcall(buildTreeRecursive(rightPredictionData, rightPredictionRowIds, rightSplitData, rightSortedIndices, config, splitter, leaf, sizeGuard, curDepth + 1, parentNodeId + "R")).map {
          rightNode =>
          TreeNode(id = parentNodeId, depth = curDepth, split = split, leftNode = leftNode, rightNode = rightNode)
          }}
      }
  }

  private def remapSorted(oldSorted: Array[Int], newIdx: Array[Int], newSize: Int): Array[Int] = {
    val out = new Array[Int](newSize)
    var w = 0; var i = 0
    while (i < oldSorted.length) {
      val mappedIdx = newIdx(oldSorted(i))
      if (mappedIdx >= 0) { out(w) = mappedIdx; w += 1 }
      i += 1
    }
    out
  }

  private def partitionInPlace[O : ClassTag](
    data: Array[O],
    predicate: O => Boolean
  ): (Array[O], Array[O]) = {
    var leftCount = 0
    var i = 0
    while (i < data.length) { if (predicate(data(i))) leftCount += 1; i += 1 }
    val left  = new Array[O](leftCount)
    val right = new Array[O](data.length - leftCount)
    var lw = 0; var rw = 0; i = 0
    while (i < data.length) {
      val row = data(i)
      if (predicate(row)) { left(lw)  = row; lw += 1 }
      else                { right(rw) = row; rw += 1 }
      i += 1
    }
    (left, right)
  }

  /** Partitions `data` and its parallel `ids` array by the same predicate in one pass. */
  private def partitionPaired[O : ClassTag](
    data: Array[O],
    ids: Array[Int],
    predicate: O => Boolean
  ): (Array[O], Array[Int], Array[O], Array[Int]) = {
    var leftCount = 0
    var i = 0
    while (i < data.length) { if (predicate(data(i))) leftCount += 1; i += 1 }
    val leftData  = new Array[O](leftCount)
    val rightData = new Array[O](data.length - leftCount)
    val leftIds   = new Array[Int](leftCount)
    val rightIds  = new Array[Int](data.length - leftCount)
    var lw = 0; var rw = 0; i = 0
    while (i < data.length) {
      if (predicate(data(i))) { leftData(lw)  = data(i); leftIds(lw)  = ids(i); lw += 1 }
      else                    { rightData(rw) = data(i); rightIds(rw) = ids(i); rw += 1 }
      i += 1
    }
    (leftData, leftIds, rightData, rightIds)
  }

  def sortFeatureIndices[O <: DataPoint](trainingData: Array[O], config: TreeBaseConfig): Array[Array[Int]] =
    Array.tabulate(config.nFeatures) { f =>
      trainingData.indices.toArray.sortBy(i => trainingData(i).features(f))
    }


}
