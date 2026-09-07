package treebased.api

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import treebased.core.prediction.LeafEstimator
import treebased.core.training.split.criterion.ExactCausalVariance
import treebased.testutils.SyntheticData
import treebased.testutils.SyntheticData.generatePoints
import treebased.testutils.TestConfigs
import treebased.testutils.Numeric.mean

class CausalTreeSuite extends AnyFlatSpec with Matchers {

  private val data = generatePoints(2000)
  private val (trainData, testData) = data.splitAt(1600)
  private val tree = CausalTree.train(trainData, TestConfigs.causalTree, LeafEstimator.ate, ExactCausalVariance)

  "CausalTree.train" should "respect the configured max depth" in {
    tree.depth should be <= TestConfigs.causalTree.maxDepth
  }

  it should "produce finite, heterogeneous leaf predictions" in {
    val preds = testData.map(dp => tree.predict(dp.features))
    all (preds.toIndexedSeq.map(_.isFinite)) shouldBe true
    preds.distinct.length should be > 1 // more than one leaf reached
  }

  // A single shallow tree is a weak ATE estimator — assert only that it lands in the neighbourhood.
  it should "estimate an average effect near the true ATE" in {
    val preds = testData.map(dp => tree.predict(dp.features))
    mean(preds.toSeq) shouldBe SyntheticData.TrueAte +- 1.0
  }
}
