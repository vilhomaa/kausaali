package treebased.estimand

import treebased.config.{ForestConfig, RegressionForestConfig}
import treebased.core.domain.data.CausalObservation
import treebased.core.domain.model.ForestModel
import treebased.core.training.centering.{CenteringTransform, NoCentering, OobCentering, PrefittedCentering}

/**
 * User-facing choice of how [[treebased.api.GeneralizedRandomForest.causal]] orthogonalizes its
 * training data before growing (Athey, Tibshirani & Wager 2019, §6.1.1). Carried by
 * [[CausalForestConfig]] and resolved to a
 * [[treebased.core.training.centering.CenteringTransform]] by [[transform]].
 */
enum Centering {

  /** No centering: split and leaf-estimate on the raw outcome and treatment. */
  case Off

  /**
   * Cross-fitted centering: fit `E[Y|X]` and `E[W|X]` as regression forests and residualize each
   * training row against its own out-of-bag trees (a valid cross-fitting scheme without an
   * explicit fold split — see [[treebased.core.training.centering.OobCentering]]).
   *
   * `nuisanceForest` tunes those forests independently of the causal forest. `None` derives a
   * config from the causal forest's own: `nTrees / 4` (floored at 50), honesty fraction 0.5 and
   * min-node-size 5, with `mtry`, subsample fraction and max depth inherited from the causal
   * forest. An explicit config is used as given, except its subsample ratio is capped at 0.8 so
   * every row still gets enough out-of-bag trees.
   */
  case CrossFit(nuisanceForest: Option[RegressionForestConfig] = None)

  /**
   * Center against nuisance models fitted elsewhere (e.g. on a larger, independent sample).
   * Statistically valid when that sample comes from the same population and is disjoint from this
   * forest's training data.
   */
  case Prefitted(outcomeModel: ForestModel, treatmentModel: ForestModel)

  def transform(causalForest: ForestConfig): CenteringTransform[CausalObservation] = this match {
    case Off                           => new NoCentering[CausalObservation]
    case CrossFit(nuisance)            => new OobCentering(Centering.nuisanceConfig(nuisance, causalForest))
    case Prefitted(outcome, treatment) => new PrefittedCentering(outcome, treatment)
  }
}

object Centering {

  /** Max nuisance subsample ratio for an explicit config — keeps out-of-bag coverage healthy. */
  private val MaxNuisanceSubsampleRatio = 0.8

  /**
   * `None`-nuisance defaults: the orthogonalization forests get `nTrees / 4` trees (never fewer
   * than 50), honesty fraction 0.5 and min-node-size 5; everything else (`mtry`, subsample
   * fraction, max depth, `alpha`) is inherited from the causal-forest config.
   */
  private val NuisanceTreeDivisor       = 4
  private val MinNuisanceTrees          = 50
  private val NuisanceMinNodeSize       = 5
  private val NuisanceHonestyFraction   = 0.5

  private[estimand] def nuisanceConfig(explicit: Option[RegressionForestConfig], causalForest: ForestConfig): ForestConfig =
    explicit match {
      case Some(c) => c.toForestConfig(causalForest.nFeatures)
                        .copy(subsampleRatio = math.min(c.subsampleRatio, MaxNuisanceSubsampleRatio))
      case None    => causalForest.copy(
        nTrees              = math.max(MinNuisanceTrees, causalForest.nTrees / NuisanceTreeDivisor),
        minNodeSize         = NuisanceMinNodeSize,
        predictionDataRatio = NuisanceHonestyFraction,
        kernelPrediction    = false
      )
    }
}
