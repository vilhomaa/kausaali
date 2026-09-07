package treebased.testutils

import treebased.config.{CausalForestConfig, ExactCausalForestConfig, RegressionForestConfig, TreeConfig}

/** Canonical config shapes so suites stop re-declaring near-identical hyperparameters inline. */
object TestConfigs {

  /** 3-feature shape (suites vary depth / nTrees via `.copy`), one per entry point. */
  val exactCausalForest: ExactCausalForestConfig = ExactCausalForestConfig(
    maxDepth                     = 15,
    nTrees                       = 200,
    maxFeaturesForSplit          = 2,
    minNodeSizePerTreatmentGroup = 5,
    seed                         = 42L
  )

  val causalForest: CausalForestConfig = CausalForestConfig(
    maxDepth                     = 15,
    nTrees                       = 200,
    maxFeaturesForSplit          = 2,
    minNodeSizePerTreatmentGroup = 5,
    seed                         = 42L
  )

  val regressionForest: RegressionForestConfig = RegressionForestConfig(
    maxDepth            = 15,
    nTrees              = 200,
    maxFeaturesForSplit = 2,
    seed                = 42L
  )

  /** Tiny and shallow: serialization / wiring tests that must not spend real time training. */
  val tinyForest: RegressionForestConfig = regressionForest.copy(maxDepth = 4, nTrees = 8)
  val tinyExactCausalForest: ExactCausalForestConfig = exactCausalForest.copy(maxDepth = 4, nTrees = 8)

  val causalTree: TreeConfig = TreeConfig(
    maxDepth            = 3,
    nFeatures           = 3,
    maxFeaturesForSplit = 2,
    seed                = 42L
  )
}
