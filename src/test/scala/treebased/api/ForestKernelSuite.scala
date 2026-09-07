package treebased.api

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import treebased.core.domain.data.Observation
import treebased.core.prediction.ForestKernel
import treebased.estimand.Centering
import treebased.serialization.ForestSerializer
import treebased.testutils.SyntheticData
import treebased.testutils.SyntheticData.generatePoints
import treebased.testutils.TestConfigs
import treebased.testutils.EitherOps.orThrow
import treebased.testutils.Numeric.mean

import java.nio.file.Files

class ForestKernelSuite extends AnyFlatSpec with Matchers {

  private val data = generatePoints(4000)
  private val (trainData, testData) = data.splitAt(3200)
  private val testFeatures = testData.take(300).map(_.features)

  private val baseConfig   = TestConfigs.causalForest.copy(maxDepth = 12)
  private val kernelConfig  = baseConfig.copy(kernelPrediction = true)
  private val baseRegressionConfig   = TestConfigs.regressionForest.copy(maxDepth = 12)
  private val kernelRegressionConfig = baseRegressionConfig.copy(kernelPrediction = true)

  "GeneralizedRandomForest.causal with kernelPrediction" should "keep the per-row honest sample on every leaf" in {
    val forest = GeneralizedRandomForest.causal(trainData, kernelConfig.copy(centering = Centering.Off))
    val nTrain = forest.kernelRows.length
    nTrain shouldBe trainData.length

    val leaves = testFeatures.map(x => forest.trees.head.locate(x))
    all(leaves.toIndexedSeq.map(_.honestRowIds.nonEmpty)) shouldBe true
    all(leaves.toIndexedSeq.map(l => l.honestRowIds.forall(i => i >= 0 && i < nTrain))) shouldBe true
  }

  "ForestKernel.alpha" should "return non-negative weights that sum to one over the training rows" in {
    val forest = GeneralizedRandomForest.causal(trainData, kernelConfig.copy(centering = Centering.Off))
    val nTrain = forest.kernelRows.length

    testFeatures.take(50).foreach { x =>
      val a = ForestKernel.alpha(forest.trees, x, nTrain, i => forest.kernelRows(i).weight)
      a.length shouldBe nTrain
      all(a.toIndexedSeq.map(_ >= 0.0)) shouldBe true
      a.sum shouldBe 1.0 +- 1e-9
    }
  }

  it should "recover the ATE via the alpha-weighted slope, uncentered" in {
    val forest = GeneralizedRandomForest.causal(trainData, kernelConfig.copy(centering = Centering.Off))
    val preds  = forest.predict(testFeatures)
    all(preds.toIndexedSeq.map(_.isFinite)) shouldBe true
    mean(preds.toSeq) should be(SyntheticData.TrueAte +- 0.3)
  }

  it should "stay finite and bounded with default cross-fit centering" in {
    val forest = GeneralizedRandomForest.causal(trainData, kernelConfig)
    val preds  = forest.predict(testFeatures)
    all(preds.toIndexedSeq.map(_.isFinite)) shouldBe true
    all(preds.toIndexedSeq.map(math.abs)) should be < 10.0
    mean(preds.toSeq) should be(SyntheticData.TrueAte +- 0.3)
  }

  "GeneralizedRandomForest.regression with kernelPrediction" should "estimate the mean label from the alpha-weighted sample" in {
    val regressionTrain = trainData.map(d => Observation(d.features, d.weight, d.y))
    val forest = GeneralizedRandomForest.regression(regressionTrain, kernelRegressionConfig)
    mean(forest.predict(testFeatures).toSeq) should be(SyntheticData.TrueMeanOutcome +- 0.2)
  }

  "A kernel-mode forest" should "round-trip through JSON with identical predictions" in {
    val forest = GeneralizedRandomForest.causal(
      trainData, kernelConfig.copy(nTrees = 20, maxDepth = 6, centering = Centering.Off))

    val file = Files.createTempFile("kernel-forest-", ".json")
    val loaded =
      try {
        orThrow(ForestSerializer.save(forest, file.toString))
        orThrow(ForestSerializer.loadGeneralized(file.toString))
      } finally Files.deleteIfExists(file)

    loaded.forestConfig.kernelPrediction shouldBe true
    loaded.kernelRows.length shouldBe forest.kernelRows.length
    loaded.predict(testFeatures).zip(forest.predict(testFeatures)).foreach { case (a, b) => a shouldBe b +- 1e-9 }
  }
}
