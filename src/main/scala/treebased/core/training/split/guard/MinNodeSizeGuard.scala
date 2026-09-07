package treebased.core.training.split.guard

import treebased.config.TreeBaseConfig
import treebased.core.domain.data.Observation

object MinNodeSizeGuard extends SplitGuard[Observation] {
  def hasEnoughDataForSplit(data: Array[Observation], config: TreeBaseConfig): Boolean =
    data.nonEmpty && data.length >= config.minNodeSize * 2

  def trimToValidWindow(sortedData: Array[Observation], minSize: Int): Array[Observation] =
    sortedData.drop(minSize).dropRight(minSize)
}
