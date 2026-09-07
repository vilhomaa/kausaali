package treebased.core.training.aggregate

/**
 * Commutative group over a sufficient statistic `S`, with a lift `I => S`. The single home for
 * "fold a stream into `S` and slide one element back out" - consumed by the split scan
 * (`right = remove(total, left)`) and by leaf estimation ([[foldMap]]). Laws:
 *   combine(empty, s) == s ; combine associative & commutative ; remove(combine(a, b), b) == a
 */
trait Accumulator[I, S] extends Serializable {
  def empty: S
  def lift(i: I): S
  def combine(a: S, b: S): S
  def remove(whole: S, part: S): S

  final def foldMap(is: Array[I]): S = {
    var acc = empty
    var k = 0
    while (k < is.length) { acc = combine(acc, lift(is(k))); k += 1 }
    acc
  }
}
