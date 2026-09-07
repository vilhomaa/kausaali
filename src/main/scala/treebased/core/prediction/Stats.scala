package treebased.core.prediction

/** Small numeric helpers with no external dependency. */
private[treebased] object Stats {

  private val a = Array(-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
    1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
  private val b = Array(-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
    6.680131188771972e+01, -1.328068155288572e+01)
  private val c = Array(-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
    -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
  private val d = Array(7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
    3.754408661907416e+00)
  private val pLow = 0.02425
  private val pHigh = 1.0 - pLow

  /**
   * Inverse standard-normal CDF (Acklam's rational approximation, |error| < 1.15e-9).
   * `normalQuantile(0.975) ~= 1.959964`.
   */
  def normalQuantile(p: Double): Double = {
    require(p > 0.0 && p < 1.0, s"normalQuantile expects p in (0,1), got $p")
    if (p < pLow) {
      val q = math.sqrt(-2.0 * math.log(p))
      (((((c(0) * q + c(1)) * q + c(2)) * q + c(3)) * q + c(4)) * q + c(5)) /
        ((((d(0) * q + d(1)) * q + d(2)) * q + d(3)) * q + 1.0)
    } else if (p <= pHigh) {
      val q = p - 0.5
      val r = q * q
      (((((a(0) * r + a(1)) * r + a(2)) * r + a(3)) * r + a(4)) * r + a(5)) * q /
        (((((b(0) * r + b(1)) * r + b(2)) * r + b(3)) * r + b(4)) * r + 1.0)
    } else {
      val q = math.sqrt(-2.0 * math.log(1.0 - p))
      -(((((c(0) * q + c(1)) * q + c(2)) * q + c(3)) * q + c(4)) * q + c(5)) /
        ((((d(0) * q + d(1)) * q + d(2)) * q + d(3)) * q + 1.0)
    }
  }
}
