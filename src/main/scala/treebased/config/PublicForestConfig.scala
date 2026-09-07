package treebased.config

/**
 * Fields shared verbatim by every public forest config ([[CausalForestConfig]],
 * [[RegressionForestConfig]], [[ExactCausalForestConfig]]) on their way to a [[ForestConfig]].
 * Factored out so the mapping isn't independently re-derived — and free to drift — in each of
 * the three; a config-specific `toForestConfig(nFeatures)` supplies only what genuinely differs
 * per estimand (`minNodeSizePerTreatmentGroup`, `kernelPrediction`, split-balance tuning) as
 * arguments to [[baseForestConfig]].
 */
private[treebased] trait PublicForestConfig {
  def nTrees: Int
  def maxDepth: Int
  def minNodeSize: Int
  def subsampleRatio: Double
  def predictionDataRatio: Double
  def maxFeaturesForSplit: Int
  def minInfoGain: Double
  def seed: Long

  /** Bootstrap-of-little-bags grouping for pointwise variance. Off by default; only
   *  [[CausalForestConfig]] / [[RegressionForestConfig]] expose it as a settable field. */
  def ciGroupSize: Int = 1
  def ciGroupHalfFraction: Double = 0.5

  protected def baseForestConfig(
    nFeatures: Int,
    minNodeSizePerTreatmentGroup: Int = 10,
    kernelPrediction: Boolean = false,
    splitBalanceAlpha: Double = 0.0,
    stabilizeSplits: Boolean = false
  ): ForestConfig = ForestConfig(
    maxDepth = maxDepth,
    nTrees = nTrees,
    nFeatures = nFeatures,
    maxFeaturesForSplit = ForestConfig.resolveMtry(maxFeaturesForSplit, nFeatures),
    minInfoGain = minInfoGain,
    minNodeSize = minNodeSize,
    minNodeSizePerTreatmentGroup = minNodeSizePerTreatmentGroup,
    predictionDataRatio = predictionDataRatio,
    subsampleRatio = subsampleRatio,
    kernelPrediction = kernelPrediction,
    splitBalanceAlpha = splitBalanceAlpha,
    stabilizeSplits = stabilizeSplits,
    ciGroupSize = ciGroupSize,
    ciGroupHalfFraction = ciGroupHalfFraction,
    seed = seed
  )
}
