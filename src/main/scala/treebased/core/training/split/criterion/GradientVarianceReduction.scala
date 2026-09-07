package treebased.core.training.split.criterion

import treebased.core.training.aggregate.SumCount

/**
 * Impurity used in GENERALIZED RANDOM FORESTS by Athey, Tibshirani, and Wager (2019).
 */
object GradientVarianceReduction extends SplitCriterion[SumCount] {
  def calculate(left: SumCount, right: SumCount): Double =
    (1.0 / left.count) * math.pow(left.sum, 2) + (1.0 / right.count) * math.pow(right.sum, 2)
}
