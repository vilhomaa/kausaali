package treebased.core.training.centering

import treebased.api.GeneralizedRandomForest
import treebased.config.ForestConfig
import treebased.core.domain.data.Observation
import treebased.estimand.RegressionSpec

/**
 * Out-of-bag predictions from a single regression forest, for local centering
 * (Athey, Tibshirani & Wager 2019, §6.1.1; Robinson 1988).
 *
 * A row's prediction aggregates only the trees whose subsample excluded that row, so the residual
 * `target - m̂(x)` is genuinely out-of-sample without an explicit K-fold split — each tree is an
 * out-of-sample predictor for its own OOB rows. This is one forest per nuisance instead of `K`
 * disjoint refits, and the OOB average over many trees gives lower-variance residuals than a
 * single held-out fold model.
 *
 * A row covered by fewer than [[MinOobTrees]] out-of-bag trees falls back to the global target
 * mean rather than a thin, noisy OOB average.
 */
private[treebased] object OobNuisanceForest {

  /** Minimum out-of-bag trees for a row to use its OOB mean; below this, the global target mean. */
  val MinOobTrees: Int = 10

  def predictOob(
    features: Array[Array[Double]],
    target: Array[Double],
    nuisanceForest: ForestConfig,
    sortedFeatureIndices: Array[Array[Int]]
  ): Array[Double] = {
    val n = features.length
    if (n == 0) return Array.empty
    val globalMean = target.sum / n

    val rows = Array.tabulate(n)(i => Observation(features(i), 1.0, target(i)))
    // Grow the nuisance forest and collect each row's out-of-bag prediction sum/count in the
    // same parallel pass — no retained tree vector, no second descent over every (row, tree).
    val (oobSum, oobCnt) =
      GeneralizedRandomForest.trainTreesOob(rows, nuisanceForest, RegressionSpec, sortedFeatureIndices)

    Array.tabulate(n)(i => if (oobCnt(i) >= MinOobTrees) oobSum(i) / oobCnt(i) else globalMean)
  }

  /** Convenience for callers without a precomputed feature index (tests, ad-hoc use). */
  def predictOob(
    features: Array[Array[Double]],
    target: Array[Double],
    nuisanceForest: ForestConfig
  ): Array[Double] = {
    val idx = Array.tabulate(nuisanceForest.nFeatures) { f =>
      features.indices.toArray.sortBy(i => features(i)(f))
    }
    predictOob(features, target, nuisanceForest, idx)
  }
}
