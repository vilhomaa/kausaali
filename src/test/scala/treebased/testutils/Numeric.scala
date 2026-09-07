package treebased.testutils

/** Small numeric helpers for assertions: point estimates and their agreement with a target. */
object Numeric {

  def mean(xs: Seq[Double]): Double = xs.sum / xs.length

  def variance(xs: Seq[Double]): Double = {
    val m = mean(xs)
    xs.map(x => (x - m) * (x - m)).sum / xs.length
  }

  def rmse(pred: Seq[Double], actual: Seq[Double]): Double =
    math.sqrt(mean(pred.zip(actual).map { case (p, a) => (p - a) * (p - a) }))

  /** Fraction of the target's variance explained by `pred` (can go negative for a bad fit). */
  def r2(pred: Seq[Double], actual: Seq[Double]): Double =
    1 - mean(pred.zip(actual).map { case (p, a) => (p - a) * (p - a) }) / variance(actual)

  def pearson(a: Seq[Double], b: Seq[Double]): Double = {
    val ma = mean(a); val mb = mean(b)
    val cov = a.zip(b).map { case (x, y) => (x - ma) * (y - mb) }.sum
    val sa  = math.sqrt(a.map(x => (x - ma) * (x - ma)).sum)
    val sb  = math.sqrt(b.map(y => (y - mb) * (y - mb)).sum)
    cov / (sa * sb)
  }
}
