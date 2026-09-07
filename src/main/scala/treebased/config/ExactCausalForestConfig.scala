package treebased.config

/**
 * User-facing config for [[treebased.api.CausalForest.train]] — the exact-splitting honest
 * causal tree of Athey & Wager (2017): every candidate threshold is scored directly by the
 * child ATE difference, with no split-balance rule, no split stabilization, and no
 * kernel-weighted prediction — none of those exist in the 2017 paper, so unlike
 * [[CausalForestConfig]] this type has no [[GradientSplitParams]] or `kernelPrediction` field to
 * set. Every field is defaulted, so `ExactCausalForestConfig()` is a reasonable forest on its
 * own; `nFeatures` isn't here at all — it's read off the training data at `train` time.
 */
final case class ExactCausalForestConfig(
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
  seed: Long = 42L
) extends PublicForestConfig {

  // kernelPrediction / splitBalanceAlpha / stabilizeSplits stay at baseForestConfig's defaults
  // (false / 0.0 / false) — none of those exist in Athey-Wager 2017.
  private[treebased] def toForestConfig(nFeatures: Int): ForestConfig = baseForestConfig(
    nFeatures = nFeatures,
    minNodeSizePerTreatmentGroup = minNodeSizePerTreatmentGroup
  )
}
