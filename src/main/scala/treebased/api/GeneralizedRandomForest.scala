package treebased.api

import treebased.config.{CausalForestConfig, ForestConfig, RegressionForestConfig}
import treebased.estimand.{CausalSpec, ForestSpec, RegressionSpec}
import treebased.core.training.TreeTrainer
import treebased.core.domain.data.{CausalObservation, DataPoint, Observation}
import treebased.core.domain.tree.Node
import treebased.core.domain.model.ForestModel
import treebased.core.prediction.{ForestAggregation, ForestKernel, KernelRow, VarianceEstimator, WeightedEstimator}

import scala.reflect.ClassTag
import scala.util.Random
import scala.collection.parallel.CollectionConverters.*
import scala.collection.parallel.immutable.ParVector

/**
 * User-facing API for any forest from the Generalized Random Forests family (Athey,
 * Tibshirani & Wager, 2019) — e.g. a causal forest built via gradient-based splitting, as
 * opposed to the exact-splitting honest tree used by [[CausalForest]]. The estimand is
 * selected by the [[ForestSpec]] passed to `train`.
 */
case class GeneralizedRandomForest private (
  forestConfig: ForestConfig,
  trees: Vector[Node],
  predOps: ForestAggregation,
  estimator: WeightedEstimator = WeightedEstimator.Mean,
  kernelRows: Array[KernelRow] = Array.empty,
  // Bootstrap-of-little-bags group id per tree, aligned to `trees`. Empty unless the forest was
  // grown with `forestConfig.ciGroupSize >= 2`; required by `predictInterval`.
  treeGroups: Array[Int] = Array.empty
) extends ForestModel {

  /**
   * With `forestConfig.kernelPrediction` (and a forest that kept its training rows) the estimate
   * solves the adaptive-kernel-weighted moment equation: alpha_i(x) from [[ForestKernel]] fed to
   * the estimand's [[WeightedEstimator]]. Otherwise it averages the per-tree scalar predictions,
   * which coincides with the weighted solution only for linear estimands.
   */
  def predictSingle(features: Array[Double]): Double =
    if (forestConfig.kernelPrediction && kernelRows.nonEmpty) {
      val alpha = ForestKernel.alpha(trees, features, kernelRows.length, i => kernelRows(i).weight)
      estimator.solve(kernelRows, alpha)
    } else {
      predOps.aggregate(trees.map(tree => tree.predict(features)), trees.size)
    }

  override def predict(features: Array[Array[Double]]): Array[Double] =
    features.par.map(predictSingle).toArray

  /**
   * Pointwise estimate with a confidence interval for `theta_hat(x)`, via the grouped
   * half-sampling / bootstrap-of-little-bags variance estimator (Athey, Tibshirani & Wager 2019,
   * §4). Requires a forest grown with `CausalForestConfig`/`RegressionForestConfig`
   * `ciGroupSize >= 2` and `kernelPrediction = true`.
   */
  def predictInterval(features: Array[Double], level: Double = 0.95): VarianceEstimator.Interval = {
    require(
      treeGroups.nonEmpty && kernelRows.nonEmpty,
      "predictInterval needs a forest trained with ciGroupSize >= 2 and kernelPrediction = true"
    )
    VarianceEstimator.interval(trees, treeGroups, kernelRows, estimator, features, level)
  }

  /** Batch [[predictInterval]]. */
  def predictIntervals(features: Array[Array[Double]], level: Double = 0.95): Array[VarianceEstimator.Interval] =
    features.par.map(predictInterval(_, level)).toArray

  def size: Int = trees.size
}

object GeneralizedRandomForest {

  def apply(
    forestConfig: ForestConfig,
    trees: Vector[Node],
    predOps: ForestAggregation,
    estimator: WeightedEstimator = WeightedEstimator.Mean,
    kernelRows: Array[KernelRow] = Array.empty,
    treeGroups: Array[Int] = Array.empty
  ): GeneralizedRandomForest = {
    require(trees.nonEmpty, "Forest must have at least one tree")
    require(
      trees.size == forestConfig.nTrees,
      s"Expected ${forestConfig.nTrees} trees but got ${trees.size}"
    )
    require(
      treeGroups.isEmpty || treeGroups.length == trees.size,
      s"treeGroups length ${treeGroups.length} does not match ${trees.size} trees"
    )
    new GeneralizedRandomForest(forestConfig, trees, predOps, estimator, kernelRows, treeGroups)
  }

  /**
   * Causal-effect forest via GRF gradient splitting. [[CausalForestConfig.centering]] selects
   * whether and how the training data is locally centered before growing;
   * [[CausalForestConfig.kernelPrediction]] (on by default) selects the α-weighted moment solve
   * over per-tree leaf-slope averaging.
   */
  def causal(
     trainingData: Array[CausalObservation],
     config: CausalForestConfig
  ): GeneralizedRandomForest = {
    val forest = config.toForestConfig(nFeaturesOf(trainingData))
    train(trainingData, forest, new CausalSpec(config.centering.transform(forest)))
  }

  /** Regression forest: `E[Y | X]` via GRF gradient splitting. Never locally centered. */
  def regression(
     trainingData: Array[Observation],
     config: RegressionForestConfig
  ): GeneralizedRandomForest =
    train(trainingData, config.toForestConfig(nFeaturesOf(trainingData)), RegressionSpec)

  private def nFeaturesOf[O <: DataPoint](trainingData: Array[O]): Int =
    trainingData.headOption.map(_.features.length).getOrElse(0)

  /**
   * The shared training engine. Estimand selection is the caller's job — use the [[causal]] /
   * [[regression]] factories rather than building a [[ForestSpec]] by hand.
   */
  private[treebased] def train[O <: DataPoint : ClassTag](
     trainingData: Array[O],
     config: ForestConfig,
     strategy: ForestSpec[O]
  ): GeneralizedRandomForest = {

    val sortedFeatureIndices = TreeTrainer.sortFeatureIndices(trainingData, config)
    val centeredData = strategy.centering.centered(trainingData, sortedFeatureIndices)
    val spec = strategy.calibrate(centeredData)
    val grown = trainTreesWithMembership(centeredData, config, spec, Some(sortedFeatureIndices))
    val nodes = grown.map(_.node)
    val treeGroups =
      if (config.varianceEstimationEnabled) grown.map(_.group).toArray
      else Array.empty[Int]
    // honestRowIds on the leaves index into `centeredData`, so kernel prediction must keep it too.
    val kernelRows =
      if (config.kernelPrediction) centeredData.map(spec.kernelRow)
      else Array.empty[KernelRow]
    GeneralizedRandomForest(config, nodes, spec.predictOps, spec.estimator, kernelRows, treeGroups)
  }

  /** One grown tree with the training rows it saw and its bootstrap-of-little-bags group id
   *  (`0` for every tree when variance estimation is off). */
  private[treebased] final case class GrownTree(node: Node, rowIds: Array[Int], group: Int)

  /**
   * Grow the forest's trees, pairing each with the original training-row indices it was grown on.
   * The membership arrays are what out-of-bag machinery (local centering) needs to know which
   * rows a given tree never saw; the plain [[train]] path uses only the nodes.
   *
   * With `config.ciGroupSize >= 2` the trees are grown in `ceil(nTrees / ciGroupSize)` groups,
   * each sharing one half-sample draw of size `ciGroupHalfFraction * N`; each tree then subsamples
   * from within its group's half-sample. That grouped structure is what
   * [[treebased.core.prediction.VarianceEstimator]] reads for pointwise confidence intervals.
   */
  private[treebased] def trainTreesWithMembership[O <: DataPoint : ClassTag](
    trainingData: Array[O],
    config: ForestConfig,
    strategy: ForestSpec[O],
    precomputedSortedIndices: Option[Array[Array[Int]]] = None
  ): Vector[GrownTree] = {
    val n = trainingData.length
    val rng = new Random(config.seed)
    val sortedFeatureIndices = precomputedSortedIndices.getOrElse(TreeTrainer.sortFeatureIndices(trainingData, config))

    def growOne(seed: Long, group: Int, rowPool: Array[Int]): GrownTree = {
      val treeRng = new Random(seed)
      val treeConfig = config.toTreeConfig(seed)
      val (subsample, subSortedIdx, sampledIndices) =
        TreeTrainer.subSampleTrainingData(treeRng, trainingData, sortedFeatureIndices, config, rowPool)
      GrownTree(
        GeneralizedRandomTree.trainTree(subsample, treeConfig, strategy, Some(subSortedIdx), sampledIndices),
        sampledIndices,
        group
      )
    }

    if (!config.varianceEstimationEnabled) {
      val seeds = ParVector.fill(config.nTrees)(rng.nextLong())
      seeds.map(seed => growOne(seed, 0, null)).toVector
    } else {
      val groupSize = config.ciGroupSize
      val nGroups   = (config.nTrees + groupSize - 1) / groupSize
      val halfSize  = math.max(1, math.min(n, (n * config.ciGroupHalfFraction).toInt))
      // Sequential plan build keeps the seed / half-sample stream deterministic in `config.seed`.
      val plan: Vector[(Long, Int, Array[Int])] =
        (0 until nGroups).toVector.flatMap { g =>
          val groupRng = new Random(rng.nextLong())
          val half = groupRng.shuffle((0 until n).toIndexedSeq).take(halfSize).toArray
          val treesInGroup = math.min(groupSize, config.nTrees - g * groupSize)
          Vector.tabulate(treesInGroup)(_ => (groupRng.nextLong(), g, half))
        }
      plan.par.map { case (seed, group, half) => growOne(seed, group, half) }.toVector
    }
  }

  /**
   * Grow the forest and, in the same parallel pass, accumulate each tree's prediction for the
   * rows it did **not** sample. Returns `(oobSum, oobCount)` aligned to `trainingData`: row `i`'s
   * out-of-bag mean is `oobSum(i) / oobCount(i)` where `oobCount(i) > 0`. This is what local
   * centering needs from a nuisance forest — it never needs the trees themselves — so folding the
   * out-of-bag scoring into the grow avoids a second `O(nRows * nTrees)` descent pass over a
   * retained tree vector. `sortedFeatureIndices` is supplied by the caller (see [[train]]) so the
   * nuisance and causal forests share one index build.
   */
  private[treebased] def trainTreesOob[O <: DataPoint : ClassTag](
    trainingData: Array[O],
    config: ForestConfig,
    strategy: ForestSpec[O],
    sortedFeatureIndices: Array[Array[Int]]
  ): (Array[Double], Array[Int]) = {
    val n = trainingData.length
    val rng = new Random(config.seed)
    val seeds = ParVector.fill(config.nTrees)(rng.nextLong())

    val contributions = seeds.map { seed =>
      val treeRng    = new Random(seed)
      val treeConfig = config.toTreeConfig(seed)
      val (subsample, subSortedIdx, sampledIndices) =
        TreeTrainer.subSampleTrainingData(treeRng, trainingData, sortedFeatureIndices, config)
      val tree = GeneralizedRandomTree.trainTree(subsample, treeConfig, strategy, Some(subSortedIdx), sampledIndices)

      // sampledIndices is drawn without replacement, so its complement is exactly the OOB set.
      val inBag = new Array[Boolean](n)
      var k = 0
      while (k < sampledIndices.length) { inBag(sampledIndices(k)) = true; k += 1 }

      val oobIds   = new Array[Int](n - sampledIndices.length)
      val oobPreds = new Array[Double](n - sampledIndices.length)
      var w = 0; var i = 0
      while (i < n) {
        if (!inBag(i)) { oobIds(w) = i; oobPreds(w) = tree.predict(trainingData(i).features); w += 1 }
        i += 1
      }
      (oobIds, oobPreds)
    }.seq

    // Sequential scatter-merge — no cross-thread writes to the shared row accumulators.
    val oobSum = new Array[Double](n)
    val oobCnt = new Array[Int](n)
    contributions.foreach { case (ids, preds) =>
      var k = 0
      while (k < ids.length) { oobSum(ids(k)) += preds(k); oobCnt(ids(k)) += 1; k += 1 }
    }
    (oobSum, oobCnt)
  }
}
