package treebased.estimand

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/**
 * Placeholder coverage for the quantile regression forest (Meinshausen 2006; Athey, Tibshirani &
 * Wager 2019, §5).
 *
 * [[QuantileForest]] is an unimplemented stub; these pending tests keep the missing estimand
 * visible in the test report. Flesh them out alongside the `ForestSpec[Observation]` wiring
 * described in `QuantileForest.scala` (check-loss relabeling, weighted-CDF leaf estimator).
 */
class QuantileForestSuite extends AnyFlatSpec with Matchers {

  "GeneralizedRandomForest.quantile" should "recover conditional quantiles of Y given X" in pending

  it should "invert the alpha-weighted empirical CDF in kernel-prediction mode" in pending

  it should "return several requested quantiles from one fitted forest" in pending
}
