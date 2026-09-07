package treebased.core.prediction

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import treebased.api.GeneralizedRandomForest
import treebased.testutils.SyntheticData.generatePoints
import treebased.testutils.TestConfigs

/**
 * Bootstrap-of-little-bags pointwise intervals for a causal forest. The DGP's true HTE at a
 * point is `-2 * x3` (see [[treebased.testutils.SyntheticData]]), so nominal-95% intervals
 * should cover it for roughly 95% of test points.
 */
class VarianceEstimatorSuite extends AnyFlatSpec with Matchers {

  private val data = generatePoints(3600)
  private val (trainData, testData) = data.splitAt(2800)
  private val ciConfig = TestConfigs.causalForest.copy(nTrees = 800, maxDepth = 10, ciGroupSize = 2)

  "predictInterval" should "reject a forest grown without ciGroupSize >= 2" in {
    val forest = GeneralizedRandomForest.causal(trainData, TestConfigs.causalForest.copy(nTrees = 20))
    forest.treeGroups shouldBe empty
    val thrown = intercept[IllegalArgumentException](forest.predictInterval(testData.head.features))
    thrown.getMessage should include("ciGroupSize")
  }

  it should "bracket its own point estimate with a positive standard error" in {
    val forest = GeneralizedRandomForest.causal(trainData, ciConfig)
    forest.treeGroups.length shouldBe ciConfig.nTrees

    val ci = forest.predictInterval(testData.head.features)
    ci.stdErr should be > 0.0
    ci.stdErr.isFinite shouldBe true
    ci.lower should be < ci.estimate
    ci.estimate should be < ci.upper
    ci.estimate shouldBe forest.predictSingle(testData.head.features) +- 1e-9
  }

  it should "cover the true heterogeneous effect for most test points" in {
    val forest = GeneralizedRandomForest.causal(trainData, ciConfig)
    val points = testData.take(250)

    val covered = points.count { o =>
      val trueHte = -2.0 * o.features(2)
      val ci = forest.predictInterval(o.features)
      ci.lower <= trueHte && trueHte <= ci.upper
    }
    // Smoke check, not a calibration guarantee: nominal-95% pointwise intervals on a
    // finite forest routinely land at ~0.80-0.90 empirical coverage because theta_hat(x) carries
    // shrinkage bias toward the ATE and the grouped half-samples overlap. The precise structural
    // properties are covered by the bracket test above.
    (covered.toDouble / points.length) should be >= 0.75
  }
}
