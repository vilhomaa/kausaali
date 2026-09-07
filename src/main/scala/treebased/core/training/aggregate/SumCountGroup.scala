package treebased.core.training.aggregate


/** The `SumCount` commutative group, shared by the gradient split scan and the regression leaf
 *  (same carrier, different `lift`). */
trait SumCountGroup {
  final val empty: SumCount = SumCount.zero
  final def combine(a: SumCount, b: SumCount): SumCount = SumCount(a.sum + b.sum, a.count + b.count)
  final def remove(whole: SumCount, part: SumCount): SumCount =
    SumCount(whole.sum - part.sum, whole.count - part.count)
}
