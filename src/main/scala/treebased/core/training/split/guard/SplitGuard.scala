package treebased.core.training.split.guard

import treebased.config.TreeBaseConfig
import treebased.core.domain.data.DataPoint

/**
 * Split-validity checks, pluggable per estimand: which candidate splits leave enough data on
 * each side to be worth evaluating. A causal forest needs both treatment arms represented; a
 * regression forest just needs a plain minimum count.
 */
trait SplitGuard[O <: DataPoint] {
  /** Whether it's worth attempting to find a split at this node at all. */
  def hasEnoughDataForSplit(data: Array[O], config: TreeBaseConfig): Boolean

  /** Trims feature-sorted data to the largest contiguous window where every candidate split is valid. */
  def trimToValidWindow(sortedData: Array[O], minSize: Int): Array[O]

  /**
   * Per-row non-negative weights for the `alpha` child-balance rule, aligned to `nodeData`
   * positions. A split is rejected unless each child keeps at least `splitBalanceAlpha` of the
   * node's total weight. The default is uniform (`1` per row → a plain count fraction); a causal
   * guard weights by the treatment-residual variance so the rule protects overlap, not just size.
   */
  def splitBalanceWeights(nodeData: Array[O]): Array[Double] = Array.fill(nodeData.length)(1.0)
}
