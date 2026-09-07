package treebased.core.training.split.criterion

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import treebased.core.training.aggregate.TreatmentEffectStats

import scala.math.pow

class ExactCausalVarianceSuite extends AnyFlatSpec with Matchers {

  private def stats(tSum: Double, cSum: Double, tCount: Int, cCount: Int) =
    TreatmentEffectStats(tSum, cSum, tCount, cCount)

  "ExactCausalVariance.calculate" should "equal P_L * P_R * (ate_L - ate_R)^2" in {
    val left  = stats(tSum = 30.0, cSum = 10.0, tCount = 10, cCount = 10) // ate_L = 2.0
    val right = stats(tSum = 20.0, cSum = 40.0, tCount = 10, cCount = 20) // ate_R = 0.0
    val nL = left.count.toDouble; val nR = right.count.toDouble
    val expected = pow(left.ate - right.ate, 2) * (nL * nR) / pow(nL + nR, 2)
    ExactCausalVariance.calculate(left, right) shouldBe expected +- 1e-12
  }

  it should "be zero when both children have the same treatment effect" in {
    val left  = stats(20.0, 0.0, 10, 10) // ate 2.0
    val right = stats(40.0, 0.0, 20, 20) // ate 2.0
    ExactCausalVariance.calculate(left, right) shouldBe 0.0 +- 1e-12
  }

  it should "be symmetric in its arguments and non-negative" in {
    val a = stats(30.0, 10.0, 10, 10)
    val b = stats(5.0, 25.0, 10, 20)
    val ab = ExactCausalVariance.calculate(a, b)
    ab shouldBe ExactCausalVariance.calculate(b, a) +- 1e-12
    ab should be >= 0.0
  }
}
