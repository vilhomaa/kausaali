package treebased.config

case class TreeConfig(
  maxDepth: Int,
  nFeatures: Int,
  maxFeaturesForSplit: Int,
  minInfoGain: Double = 0.0,
  minNodeSize: Int = 10,                   // unused by causal forests
  minNodeSizePerTreatmentGroup: Int = 10, // used by ExactCausalSplitRule
  predictionDataRatio: Double = 0.5,      // fraction of the tree's rows held out for honest leaf estimation
  splitBalanceAlpha: Double = 0.0,        // `alpha` child-balance rule; 0 = off (the forest configs set it)
  stabilizeSplits: Boolean = true,        // split stabilization; no-op unless the balance weights are non-uniform (causal)
  seed: Long = 42L
) extends TreeBaseConfig
