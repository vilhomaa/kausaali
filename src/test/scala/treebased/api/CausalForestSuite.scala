package treebased.api

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import treebased.testutils.SyntheticData
import treebased.testutils.SyntheticData.generatePoints
import treebased.testutils.TestConfigs
import treebased.testutils.Numeric.{mean, pearson}

/**
 * End-to-end checks for the Athey-Wager (2018) exact-splitting honest forest. It is a weaker
 * estimator than the GRF forest ([[GeneralizedRandomForestSuite]]), so the assertions here are
 * deliberately loose: the point is that training wires together and the forest lands in the right
 * neighbourhood, not that it is sharp.
 */
class CausalForestSuite extends AnyFlatSpec with Matchers {

  private val data = generatePoints(5000)
  private val (trainData, testData) = data.splitAt(4000)
  private val config = TestConfigs.exactCausalForest

  private val forest = CausalForest.train(trainData, config)

  private val predictRows = testData.take(600)
  private val preds       = forest.predict(predictRows.map(_.features)).toIndexedSeq

  "CausalForest.train" should "build the configured number of trees" in {
    forest.size shouldBe config.nTrees
  }

  it should "produce finite, bounded per-point effect estimates" in {
    all (preds.map(_.isFinite)) shouldBe true
    all (preds.map(math.abs)) should be < 10.0
  }

  it should "estimate an average effect near the true ATE" in {
    mean(preds) shouldBe SyntheticData.TrueAte +- 0.3
  }

  // True HTE is -2 * x3, so the per-point estimates must fall as x3 rises.
  it should "recover the sign of the heterogeneity in x3" in {
    val x3 = predictRows.map(_.features(2)).toIndexedSeq
    pearson(preds, x3) should be < 0.0
  }
}
