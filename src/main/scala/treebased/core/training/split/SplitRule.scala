package treebased.core.training.split

import treebased.config.TreeBaseConfig
import treebased.core.domain.data.DataPoint
import treebased.core.domain.tree.TreeSplit


trait SplitRule[O <: DataPoint] {
  def candidateFeatureSplitIndices(config: TreeBaseConfig): Vector[Int]
  def findBestSplitPoint(
    nodeDataPoints: Array[O],
    featureIndexesForSplit: Vector[Int],
    sortedFeatureIndices: Array[Array[Int]],
  ): Option[TreeSplit]
}
