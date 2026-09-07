package treebased.core.prediction

/**
 * The per-row fields the forest keeps for kernel-weighted prediction: the sample weight plus the
 * (possibly locally centered) outcome and treatment. `w` is unused by single-response estimands
 * and carries `NaN` there.
 */
final case class KernelRow(weight: Double, y: Double, w: Double)

/**
 * Solves the adaptive-kernel-weighted estimating equation
 *
 *     sum_i alpha_i(x) * psi_theta(O_i) = 0
 *
 * for theta(x), given the forest weights alpha_i(x) from [[ForestKernel]] (Athey, Tibshirani &
 * Wager 2019, eq. 2-3). This is the per-estimand axis of GRF prediction: linear estimands
 * (regression mean, partially-linear ATE, Wald IV) have a closed form here; genuinely non-linear
 * ones (quantile) would iterate. Replaces per-tree scalar averaging
 * ([[ForestAggregation]]), which only coincides with the weighted solution for linear psi.
 */
trait WeightedEstimator extends Serializable {
  def solve(rows: Array[KernelRow], alpha: Array[Double]): Double

  /**
   * Per-row influence contributions rho_i at the solution `theta`, i.e. `-A^{-1} psi_theta(O_i)`
   * (Athey, Tibshirani & Wager 2019, §4), computed with the same weights `alpha` used for
   * [[solve]]. They satisfy `sum_i alpha_i * rho_i = 0` at `theta`, and turn the estimator into
   * the locally linear functional `theta(x) ~= theta + sum_i alpha_i(x) * rho_i` that the
   * bootstrap-of-little-bags variance estimator ([[VarianceEstimator]]) needs — so a single
   * tree's aggregate `sum_i alpha_i^b rho_i` stays well-behaved even though re-solving the
   * estimator on one tree's handful of honest rows would not.
   */
  def influence(rows: Array[KernelRow], alpha: Array[Double], theta: Double): Array[Double]
}

object WeightedEstimator {

  /** theta = sum_i alpha_i y_i / sum_i alpha_i — the regression-forest conditional mean. */
  case object Mean extends WeightedEstimator {
    def solve(rows: Array[KernelRow], alpha: Array[Double]): Double = {
      var aSum  = 0.0
      var aySum = 0.0
      var i = 0
      while (i < rows.length) {
        val a = alpha(i)
        if (a != 0.0) { aSum += a; aySum += a * rows(i).y }
        i += 1
      }
      if (aSum == 0.0) Double.NaN else aySum / aSum
    }

    def influence(rows: Array[KernelRow], alpha: Array[Double], theta: Double): Array[Double] = {
      var aSum = 0.0
      var i = 0
      while (i < rows.length) { aSum += alpha(i); i += 1 }
      val rho = new Array[Double](rows.length)
      if (aSum != 0.0) {
        i = 0
        while (i < rows.length) { rho(i) = (rows(i).y - theta) / aSum; i += 1 }
      }
      rho
    }
  }

  /**
   * Weighted least-squares slope of Y on W — the partially-linear-model ATE, the kernel-weighted
   * analogue of [[LeafEstimator.ateSlope]]. `treatmentVarianceFloor` clamps the weighted
   * treatment-residual variance exactly as [[treebased.estimand.CausalSpec]] does for the split
   * moment and the leaf slope: below it the effect is unidentified and the point is `NaN`.
   */
  final case class Slope(treatmentVarianceFloor: Double = 0.0) extends WeightedEstimator {
    def solve(rows: Array[KernelRow], alpha: Array[Double]): Double = {
      var aSum   = 0.0
      var awSum  = 0.0
      var aySum  = 0.0
      var awwSum = 0.0
      var awySum = 0.0
      var i = 0
      while (i < rows.length) {
        val a = alpha(i)
        if (a != 0.0) {
          val row = rows(i)
          aSum   += a
          awSum  += a * row.w
          aySum  += a * row.y
          awwSum += a * row.w * row.w
          awySum += a * row.w * row.y
        }
        i += 1
      }
      if (aSum == 0.0) Double.NaN
      else {
        val denom = awwSum - awSum * awSum / aSum
        if (denom <= treatmentVarianceFloor * aSum) Double.NaN
        else (awySum - awSum * aySum / aSum) / denom
      }
    }

    def influence(rows: Array[KernelRow], alpha: Array[Double], theta: Double): Array[Double] = {
      var aSum = 0.0; var awSum = 0.0; var aySum = 0.0; var awwSum = 0.0
      var i = 0
      while (i < rows.length) {
        val a = alpha(i)
        if (a != 0.0) {
          val row = rows(i)
          aSum   += a
          awSum  += a * row.w
          aySum  += a * row.y
          awwSum += a * row.w * row.w
        }
        i += 1
      }
      val rho = new Array[Double](rows.length)
      if (aSum != 0.0 && theta.isFinite) {
        val wBar  = awSum / aSum
        val yBar  = aySum / aSum
        val denom = awwSum - awSum * awSum / aSum        // sum_i alpha_i (w_i - wBar)^2
        if (denom > treatmentVarianceFloor * aSum) {
          i = 0
          while (i < rows.length) {
            val row       = rows(i)
            val wCentered = row.w - wBar
            rho(i) = wCentered * (row.y - yBar - theta * wCentered) / denom
            i += 1
          }
        }
      }
      rho
    }
  }
}
