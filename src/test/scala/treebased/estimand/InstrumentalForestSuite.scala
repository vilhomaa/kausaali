package treebased.estimand

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/**
 * Placeholder coverage for the instrumental forest (Athey, Tibshirani & Wager 2019, §7).
 *
 * [[InstrumentalForest]] is an unimplemented stub; these pending tests keep the missing estimand
 * visible in the test report. Flesh them out alongside the `ForestSpec` wiring described in
 * `InstrumentalForest.scala` (IvObservation row type, Wald-ratio leaf estimator,
 * instrument-balance guard).
 */
class InstrumentalForestSuite extends AnyFlatSpec with Matchers {

  "GeneralizedRandomForest.instrumental" should "recover the LATE through a valid instrument" in pending

  it should "solve the alpha-weighted Wald ratio in kernel-prediction mode" in pending

  it should "collapse a weak-instrument leaf via the honesty pruner" in pending
}
