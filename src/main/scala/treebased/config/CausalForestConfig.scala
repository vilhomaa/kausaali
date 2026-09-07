package treebased.config

import treebased.estimand.Centering

/**
 * User-facing config for [[treebased.api.GeneralizedRandomForest.causal]] — a causal-effect
 * forest via gradient-based splitting (Athey, Tibshirani & Wager, 2019). Every field is
 * defaulted to a standard recommendation, so `CausalForestConfig()` is a reasonable forest on its
 * own; `nFeatures` isn't here at all — it's read off the training data at `train` time.
 *
 * [[centering]] selects whether and how the training data is locally centered before growing;
 * [[kernelPrediction]] (on by default) selects the α-weighted moment solve over per-tree
 * leaf-slope averaging — see [[treebased.api.GeneralizedRandomForest.predictSingle]].
 */
final case class CausalForestConfig(
  nTrees: Int = 2000,
  maxDepth: Int = Int.MaxValue,
  minNodeSize: Int = 5,
  /** Both treatment arms must reach this count in a node before it's split — see
   *  [[treebased.core.training.split.guard.TreatmentBalanceGuard]]. */
  minNodeSizePerTreatmentGroup: Int = 5,
  subsampleRatio: Double = 0.5,
  /** Fraction of each tree's subsample held out for honest leaf estimation. */
  predictionDataRatio: Double = 0.5,
  /** `mtry` — features considered per split. `<= 0` auto-resolves against the data's feature
   *  count — see [[ForestConfig.resolveMtry]]. */
  maxFeaturesForSplit: Int = -1,
  minInfoGain: Double = 0.0,
  seed: Long = 42L,
  centering: Centering = Centering.CrossFit(),
  kernelPrediction: Boolean = true,
  /** Trees per shared half-sample for pointwise variance / confidence intervals
   *  ([[treebased.core.prediction.VarianceEstimator]]). 1 (default) disables it. */
  override val ciGroupSize: Int = 1,
  override val ciGroupHalfFraction: Double = 0.5,
  splitTuning: GradientSplitParams = GradientSplitParams()
) extends PublicForestConfig {

  private[treebased] def toForestConfig(nFeatures: Int): ForestConfig = baseForestConfig(
    nFeatures = nFeatures,
    minNodeSizePerTreatmentGroup = minNodeSizePerTreatmentGroup,
    kernelPrediction = kernelPrediction,
    splitBalanceAlpha = splitTuning.splitBalanceAlpha,
    stabilizeSplits = splitTuning.stabilizeSplits
  )
}
