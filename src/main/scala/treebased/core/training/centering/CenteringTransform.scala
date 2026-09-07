package treebased.core.training.centering

import treebased.config.ForestConfig
import treebased.core.domain.data.{CausalObservation, DataPoint}
import treebased.core.domain.model.ForestModel
import treebased.core.training.TreeTrainer

/**
 * Local centering / orthogonalization (Athey, Tibshirani & Wager 2019, §6.1.1; Robinson 1988;
 * the R-learner, Nie & Wager 2021).
 *
 * A single training-pipeline stage: runs once per forest, upstream of subsampling. It replaces
 * each row's outcome (and, for causal estimands, its treatment) with the corresponding residual
 * against a nuisance estimate, so the GRF split criterion then targets treatment-effect
 * heterogeneity rather than variation in baseline `Y`.
 *
 * The user chooses a strategy via [[treebased.estimand.Centering]]; that choice is resolved to a
 * concrete transform here by [[treebased.estimand.Centering.transform]].
 */
trait CenteringTransform[O <: DataPoint] {
  /**
   * Residualize `data` against the nuisance estimate. `sortedFeatureIndices` is the global
   * per-feature row ordering of `data`, built once by the caller ([[treebased.api.GeneralizedRandomForest.train]])
   * and shared with the forest it grows afterwards — centering never reorders rows or touches
   * `features`, so any nuisance forest here can reuse it instead of re-sorting X.
   */
  def centered(data: Array[O], sortedFeatureIndices: Array[Array[Int]]): Array[O]
}

/** Identity transform for estimands that need no orthogonalization (regression, quantile). */
final class NoCentering[O <: DataPoint] extends CenteringTransform[O] {
  def centered(data: Array[O], sortedFeatureIndices: Array[Array[Int]]): Array[O] = data
}

/**
 * Cross-fitted causal local centering via **out-of-bag** predictions: `m̂(x) = E[Y|X=x]` and
 * `ê(x) = E[W|X=x]` are each a single regression forest, and every training row is residualized
 * against only the trees that did not sample it, then `Y <- Y - m̂(X)` and `W <- W - ê(X)`.
 *
 * OOB is a valid cross-fitting scheme — each tree is out-of-sample for its own OOB rows — at
 * roughly `1/K` the cost of `K` disjoint refits, and the OOB average over many trees yields
 * lower-variance residuals than a single held-out fold model. `nuisanceForest` configures those
 * regression forests independently of the causal forest being centered; see
 * [[OobNuisanceForest]] for the coverage fallback.
 */
final class OobCentering(nuisanceForest: ForestConfig)
  extends CenteringTransform[CausalObservation] {

  def centered(data: Array[CausalObservation], sortedFeatureIndices: Array[Array[Int]]): Array[CausalObservation] = {
    val x    = data.map(_.features)
    val mHat = OobNuisanceForest.predictOob(x, data.map(_.y), nuisanceForest, sortedFeatureIndices)
    val eHat = OobNuisanceForest.predictOob(x, data.map(_.w), nuisanceForest, sortedFeatureIndices)
    Array.tabulate(data.length)(i => data(i).copy(y = data(i).y - mHat(i), w = data(i).w - eHat(i)))
  }

  /** Ad-hoc / test entry point: derive the shared feature index from `data` itself. */
  def centered(data: Array[CausalObservation]): Array[CausalObservation] =
    centered(data, TreeTrainer.sortFeatureIndices(data, nuisanceForest))
}

/**
 * Causal local centering against nuisance models fitted elsewhere — typically on a larger,
 * independent sample. Statistically valid when that sample is drawn from the same population and
 * is disjoint from the causal forest's training data (so the residuals are genuinely
 * out-of-sample); a bigger nuisance sample only tightens the DML rate condition.
 */
final class PrefittedCentering(outcomeModel: ForestModel, treatmentModel: ForestModel)
  extends CenteringTransform[CausalObservation] {

  def centered(data: Array[CausalObservation], sortedFeatureIndices: Array[Array[Int]]): Array[CausalObservation] = {
    val x    = data.map(_.features)
    val mHat = outcomeModel.predict(x)
    val eHat = treatmentModel.predict(x)
    Array.tabulate(data.length)(i => data(i).copy(y = data(i).y - mHat(i), w = data(i).w - eHat(i)))
  }
}
