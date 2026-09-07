package treebased.config

trait TreeBaseConfig {
  def maxDepth: Int
  def nFeatures: Int
  def maxFeaturesForSplit: Int
  def minInfoGain: Double
  def minNodeSize: Int
  def minNodeSizePerTreatmentGroup: Int
  def predictionDataRatio: Double
  /** `alpha`: each child of a split must retain at least this fraction of the parent node's
   *  split-balance weight (uniform for regression, treatment-residual variance for a causal forest).
   *  `0` disables the rule. */
  def splitBalanceAlpha: Double

  /** Split stabilization: when true, the gradient split criterion weights each child's score
   *  by that child's treatment-residual variance (relative to the parent's), so a child that keeps
   *  little treatment overlap cannot win a split on noise alone. A no-op wherever the split-balance
   *  weights are uniform (i.e. regression). */
  def stabilizeSplits: Boolean

  def seed: Long
}

