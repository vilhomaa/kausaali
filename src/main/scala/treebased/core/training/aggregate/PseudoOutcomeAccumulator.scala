package treebased.core.training.aggregate


/** Running sum and count of gradient pseudo-outcomes, scanned during the GRF split search
 *  ([[treebased.core.training.split.GradientSplitRule]]). */
object PseudoOutcomeAccumulator extends Accumulator[Double, SumCount] with SumCountGroup {
  def lift(pseudoOutcome: Double): SumCount = SumCount(pseudoOutcome, 1)
}
