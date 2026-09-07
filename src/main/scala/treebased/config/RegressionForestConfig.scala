package treebased.config

/**
 * User-facing config for [[treebased.api.GeneralizedRandomForest.regression]] — `E[Y | X]` via
 * gradient-based splitting. Every field is defaulted to a standard recommendation, so
 * `RegressionForestConfig()` is a reasonable forest on its own; `nFeatures` isn't here at all —
 * it's read off the training data at `train` time. A regression forest is never locally
 * centered, so there's no `Centering` knob here (contrast [[CausalForestConfig]]).
 */
final case class RegressionForestConfig(
  nTrees: Int = 2000,
  maxDepth: Int = Int.MaxValue,
  minNodeSize: Int = 5,
  subsampleRatio: Double = 0.5,
  /** Fraction of each tree's subsample held out for honest leaf estimation. */
  predictionDataRatio: Double = 0.5,
  /** `mtry` — features considered per split. `<= 0` auto-resolves against the data's feature
   *  count — see [[ForestConfig.resolveMtry]]. */
  maxFeaturesForSplit: Int = -1,
  minInfoGain: Double = 0.0,
  seed: Long = 42L,
  /** Predict by solving the adaptive-kernel-weighted moment equation over all training rows
   *  rather than averaging each tree's leaf mean. See
   *  [[treebased.api.GeneralizedRandomForest.predictSingle]]. */
  kernelPrediction: Boolean = false,
  /** Trees per shared half-sample for pointwise variance / confidence intervals
   *  ([[treebased.core.prediction.VarianceEstimator]]). 1 (default) disables it. */
  override val ciGroupSize: Int = 1,
  override val ciGroupHalfFraction: Double = 0.5,
  splitTuning: GradientSplitParams = GradientSplitParams()
) extends PublicForestConfig {

  private[treebased] def toForestConfig(nFeatures: Int): ForestConfig = baseForestConfig(
    nFeatures = nFeatures,
    kernelPrediction = kernelPrediction,
    splitBalanceAlpha = splitTuning.splitBalanceAlpha,
    stabilizeSplits = splitTuning.stabilizeSplits
  )
}
