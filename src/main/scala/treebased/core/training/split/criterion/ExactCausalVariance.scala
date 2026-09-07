package treebased.core.training.split.criterion

import treebased.core.training.aggregate.TreatmentEffectStats

import scala.math.pow


/**
 * Splitting rule used in the paper "Estimation and Inference of Heterogeneous Treatment
 * Effects using Random Forests" by Athey and Wager (2017)
 */
object ExactCausalVariance extends SplitCriterion[TreatmentEffectStats] {
  def calculate(left: TreatmentEffectStats, right: TreatmentEffectStats): Double =
    // P_L·P_R·(τ_L − τ_R)², with P_L·P_R rewritten as n_L·n_R / (n_L + n_R)².
    pow(left.ate - right.ate, 2) *
    (left.count * right.count) / pow(left.count + right.count, 2)
}
