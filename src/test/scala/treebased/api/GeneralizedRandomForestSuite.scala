package treebased.api

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import treebased.core.domain.data.Observation
import treebased.estimand.Centering
import treebased.testutils.SyntheticData
import treebased.testutils.SyntheticData.generatePoints
import treebased.testutils.{ForestFixtures, TestConfigs}

class GeneralizedRandomForestSuite extends AnyFlatSpec with Matchers {

  private val data = generatePoints(5000)
  private val (trainData, testData) = data.splitAt(4000)
  private val testConfig = TestConfigs.causalForest

  private def estimatedAte(forest: GeneralizedRandomForest): Double = {
    val predictions = forest.predict(testData.take(600).map(_.features))
    predictions.sum / predictions.length
  }

  "GeneralizedRandomForest.causal" should "estimate the ATE via gradient-based (GRF) splitting" in {
    val forest = GeneralizedRandomForest.causal(trainData, testConfig)
    estimatedAte(forest) should be(SyntheticData.TrueAte +- 0.3)
  }

  // Regression guard: OOB local centering must not let the pseudo-outcome / leaf slope blow up
  // when the (near-randomized) treatment residual variance collapses in deep nodes.
  it should "produce finite, bounded per-point predictions with default (cross-fit) centering" in {
    val forest = GeneralizedRandomForest.causal(trainData, testConfig)
    val preds  = forest.predict(testData.take(600).map(_.features))
    all (preds.toIndexedSeq.map(_.isFinite)) shouldBe true
    all (preds.toIndexedSeq.map(math.abs)) should be < 10.0
  }

  it should "estimate the ATE with local centering turned off" in {
    val config = testConfig.copy(centering = Centering.Off)
    val forest = GeneralizedRandomForest.causal(trainData, config)
    estimatedAte(forest) should be(SyntheticData.TrueAte +- 0.3)
  }

  it should "estimate the ATE with caller-supplied nuisance models" in {
    val (outcomeModel, treatmentModel) = ForestFixtures.loadNuisanceForests()

    val smallConfig = testConfig.copy(nTrees = 10, maxDepth = 6)
    val config = smallConfig.copy(centering = Centering.Prefitted(outcomeModel, treatmentModel))
    val forest = GeneralizedRandomForest.causal(trainData, config)

    val predictions = forest.predict(testData.take(50).map(_.features))
    forest.size shouldBe smallConfig.nTrees
    predictions should have length 50
    predictions.foreach(p => assert(p.isFinite))
  }

  "GeneralizedRandomForest.regression" should "estimate the labels" in {
    val regressionTrain = trainData.map(d => Observation(d.features, d.weight, d.y))
    val forest = GeneralizedRandomForest.regression(regressionTrain, TestConfigs.regressionForest)

    val predictions = forest.predict(testData.take(600).map(_.features))
    val estimatedAvgLabel = predictions.sum / predictions.length

    estimatedAvgLabel should be(SyntheticData.TrueMeanOutcome +- 0.2)
  }
}
