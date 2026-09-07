package treebased.core.training.aggregate

/**
 * Running sum and count of a scalar quantity. A pure record; the additive group lives in
 * [[treebased.core.training.aggregate.SumCountGroup]] and backs two estimand-agnostic uses:
 *
 *  - the pseudo-outcome scan in
 *    [[treebased.core.training.split.GradientSplitRule]], where `sum` accumulates
 *    per-point gradients and `mean` is not a meaningful quantity;
 *  - a plain regression forest's honest leaf, where `mean` is the conditional-mean prediction.
 *
 * `mean` is deliberately left as `0.0 / 0` = `NaN` on an empty sample rather than substituting
 * `0.0`, so [[treebased.core.training.HonestyPruner]] can collapse the leaf into its sibling.
 */
final case class SumCount(sum: Double, count: Int) {
  def mean: Double = sum / count
}

object SumCount {
  val zero: SumCount = SumCount(0.0, 0)
}
