package treebased.core.training.split.guard

import treebased.config.TreeBaseConfig
import treebased.core.domain.data.CausalObservation

/**
 * Causal-forest node-size guard: a split is only valid if both treatment groups reach
 * `minNodeSizePerTreatmentGroup` on each side, so every candidate child remains estimable.
 * Used by both [[treebased.core.training.split.ExactCausalSplitRule]] and
 * [[treebased.core.training.split.GradientSplitRule]].
 */
object TreatmentBalanceGuard extends SplitGuard[CausalObservation] {

  private def hasEnoughDataForCausalSplit(data: Array[CausalObservation], config: TreeBaseConfig): Boolean =
    data.iterator
      .scanLeft((0, 0)) { case ((treated, control), dp) =>
        if (dp.w > 0) (treated + 1, control) else (treated, control + 1)
      }
      .drop(config.minNodeSizePerTreatmentGroup * 2 - 1)
      .exists { case (treated, control) =>
        treated >= config.minNodeSizePerTreatmentGroup &&
        control >= config.minNodeSizePerTreatmentGroup
      }

  def hasEnoughDataForSplit(data: Array[CausalObservation], config: TreeBaseConfig): Boolean =
    data.nonEmpty &&
    data.size * 2 >= config.minNodeSizePerTreatmentGroup &&
    hasEnoughDataForCausalSplit(data, config)

  def trimToValidWindow(sortedData: Array[CausalObservation], minSize: Int): Array[CausalObservation] = {

    @annotation.tailrec
    def findStartIndex(idx: Int, count0: Int, count1: Int): Int = {
      if (idx >= sortedData.length) return -1
      val dp = sortedData(idx)
      val (newC0, newC1) = if (dp.w == 0) (count0 + 1, count1) else (count0, count1 + 1)
      if (count0 >= minSize && count1 >= minSize) return idx - 1
      findStartIndex(idx + 1, newC0, newC1)
    }

    @annotation.tailrec
    def findEndIndex(idx: Int, count0: Int, count1: Int): Int = {
      if (idx < 0) return -1
      val dp = sortedData(idx)
      val (newC0, newC1) = if (dp.w == 0) (count0 + 1, count1) else (count0, count1 + 1)
      if (count0 >= minSize && count1 >= minSize) return idx + 2
      findEndIndex(idx - 1, newC0, newC1)
    }

    val startIdx = findStartIndex(0, 0, 0)
    if (startIdx == -1) return Array.empty
    val endIdx = findEndIndex(sortedData.length - 1, 0, 0)
    if (endIdx == -1) return Array.empty

    if (startIdx >= endIdx) Array.empty
    else sortedData.slice(startIdx, endIdx)
  }
}
