package treebased.testutils

import treebased.core.domain.data.CausalObservation

import scala.util.Random

/**
 * Synthetic causal data used across the test suites.
 *
 * The DGP: three independent Gaussian covariates, a randomized 0/1 treatment, and
 *   y = x1 + 0.5*x2 - 2*x3*w + noise
 *
 * Because the treatment is independent of the covariates:
 *   - true ATE                 = -2 * E[x3] = -2 * 0.8 = [[TrueAte]]
 *   - true average outcome     = 0.2 - 0.5*0.5 - 2*0.8*0.5 = [[TrueMeanOutcome]]
 *   - true HTE at a point x    = -2 * x3
 */
object SyntheticData {

  val DefaultSeed: Int      = 42
  val TrueAte: Double       = -1.6
  val TrueMeanOutcome: Double = -0.85

  def generatePoints(n: Int, seed: Int = DefaultSeed): Array[CausalObservation] = {
    val random = new Random(seed)
    Array.fill(n) {
      val x1    = random.nextGaussian() + 0.2
      val x2    = random.nextGaussian() - 0.5
      val x3    = random.nextGaussian() + 0.8
      val w     = random.nextInt(2).toDouble
      val noise = random.nextGaussian()
      val y     = x1 + x2 * 0.5 + x3 * -2 * w + noise
      CausalObservation(Array(x1, x2, x3), 1.0, y, w)
    }
  }
}
