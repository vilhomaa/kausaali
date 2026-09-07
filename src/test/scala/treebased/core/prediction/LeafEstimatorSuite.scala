package treebased.core.prediction

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import treebased.testutils.SyntheticData
import treebased.testutils.SyntheticData.generatePoints

class LeafEstimatorSuite extends AnyFlatSpec with Matchers {

  private val rows = generatePoints(2000)

  "LeafEstimator.ate.aggregate" should "partition the sample into the two arms" in {
    val stats = LeafEstimator.ate.aggregate(rows)
    stats.treatmentCount + stats.controlCount shouldBe rows.length
    stats.treatmentCount should be > 0
    stats.controlCount should be > 0
  }

  it should "derive `ate` as the difference of the arm means" in {
    val stats = LeafEstimator.ate.aggregate(rows)
    val expected = stats.treatmentSum / stats.treatmentCount - stats.controlSum / stats.controlCount
    stats.ate shouldBe expected +- 1e-9
  }

  // Treatment is randomized in the DGP, so the raw arm-mean difference is already an unbiased
  // ATE estimate — it should land near the known truth without any adjustment.
  it should "recover the true ATE on randomized synthetic data" in {
    LeafEstimator.ate.prediction(rows) shouldBe SyntheticData.TrueAte +- 0.3
  }
}
