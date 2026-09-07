package treebased.core.training

import treebased.core.domain.data.CausalObservation

/**
 * First/second moments of the (possibly locally centered) treatment `w` over a node's rows.
 * Several causal-forest components divide by `Var(w | node)` — the split moment, the leaf slope,
 * the variance guard, the split-balance weights — so it is formed in exactly one place here.
 */
private[treebased] object TreatmentMoments {

  /** `w_i - w̄` for each row (empty for an empty node). */
  def centeredW(data: Array[CausalObservation]): Array[Double] = {
    val n = data.length
    if (n == 0) return Array.emptyDoubleArray
    var sum = 0.0
    var i = 0
    while (i < n) { sum += data(i).w; i += 1 }
    val mean = sum / n
    val out = new Array[Double](n)
    i = 0
    while (i < n) { out(i) = data(i).w - mean; i += 1 }
    out
  }

  /** Population variance `Σ(w_i - w̄)² / n` (`0` for an empty node). */
  def variance(data: Array[CausalObservation]): Double = {
    val centered = centeredW(data)
    if (centered.isEmpty) return 0.0
    var sumSq = 0.0
    var i = 0
    while (i < centered.length) { sumSq += centered(i) * centered(i); i += 1 }
    sumSq / centered.length
  }
}
