package treebased.config

case class ForestConfig(
  maxDepth: Int,
  nTrees: Int = 10,
  nFeatures: Int,                          // total number of features in the data
  maxFeaturesForSplit: Int,               // features considered per split (`mtry`)
  minInfoGain: Double = 0.0,
  minNodeSize: Int = 10,                   // unused by causal forests
  minNodeSizePerTreatmentGroup: Int = 10, // used by ExactCausalSplitRule
  predictionDataRatio: Double = 0.5,      // fraction of the tree's rows held out for honest leaf estimation
  subsampleRatio: Double = 0.8,           // per-tree row subsample fraction
  kernelPrediction: Boolean = false,      // solve the alpha_i(x)-weighted moment equation instead of averaging per-tree scalars
  splitBalanceAlpha: Double = 0.05,       // `alpha`: min fraction of the parent's split-balance weight each child must keep; guards overlap, curbs end-cuts
  stabilizeSplits: Boolean = true,        // weight each child's gradient split score by its treatment-residual variance; no-op for uniform balance weights (regression)
  ciGroupSize: Int = 1,                   // trees per shared half-sample for the bootstrap-of-little-bags variance estimator; 1 disables it
  ciGroupHalfFraction: Double = 0.5,      // each ci-group's shared half-sample as a fraction of N; needs subsampleRatio <= ciGroupHalfFraction
  seed: Long = 42L
) extends TreeBaseConfig {

  /** Whether the forest is grown in the grouped structure the [[treebased.core.prediction.VarianceEstimator]] needs. */
  def varianceEstimationEnabled: Boolean = ciGroupSize >= 2

  def toTreeConfig(seed: Long): TreeConfig =
    TreeConfig(
      maxDepth = this.maxDepth,
      nFeatures = this.nFeatures,
      maxFeaturesForSplit = this.maxFeaturesForSplit,
      minInfoGain = this.minInfoGain,
      minNodeSize = this.minNodeSize,
      minNodeSizePerTreatmentGroup = this.minNodeSizePerTreatmentGroup,
      predictionDataRatio = this.predictionDataRatio,
      splitBalanceAlpha = this.splitBalanceAlpha,
      stabilizeSplits = this.stabilizeSplits,
      seed = seed
    )
}

object ForestConfig {

  /**
   * Resolves a public config's `maxFeaturesForSplit` against the data's actual feature count.
   * `requested <= 0` is the "auto" sentinel every user-facing config defaults to: the standard
   * `mtry` default, `min(nFeatures, ceil(sqrt(nFeatures)) + 20)`. A positive `requested` is
   * clamped to `nFeatures` so a stale value from a differently-shaped dataset can't overrun it.
   */
  private[treebased] def resolveMtry(requested: Int, nFeatures: Int): Int =
    if (requested > 0) math.min(requested, nFeatures)
    else math.min(nFeatures, math.ceil(math.sqrt(nFeatures.toDouble)).toInt + 20)
}
