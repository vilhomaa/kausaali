package treebased.core.prediction

import treebased.core.domain.tree.Node

/**
 * The GRF adaptive weighting kernel alpha_i(x) (Athey, Tibshirani & Wager 2019, §2, eq. 2-3):
 *
 *     alpha_i(x) = (1 / B) * sum_b  s_i * 1{ i in L_b(x) } / sum_{j in L_b(x)} s_j
 *
 * where L_b(x) is the honest sample of the leaf tree b routes x into ([[Node.locate]]) and s_i is
 * row i's sample weight. The weights are non-negative and sum to 1 over the trees that route x
 * into a non-empty leaf; empty-honest leaves (kept only until [[treebased.core.training.HonestyPruner]]
 * would collapse them) contribute nothing.
 *
 * [[WeightedEstimator]] then solves the alpha-weighted estimating equation for theta(x). This is
 * also the object a bootstrap-of-little-bags variance estimator would build on (see
 * [[VarianceEstimator]]).
 */
object ForestKernel {

  def alpha(
    trees: Vector[Node],
    x: Array[Double],
    nTrain: Int,
    sampleWeight: Int => Double = _ => 1.0
  ): Array[Double] = {
    val weights = new Array[Double](nTrain)
    var usedTrees = 0
    trees.foreach { tree =>
      val leafRows = tree.locate(x).honestRowIds
      var leafWeight = 0.0
      var k = 0
      while (k < leafRows.length) { leafWeight += sampleWeight(leafRows(k)); k += 1 }
      if (leafWeight > 0.0) {
        usedTrees += 1
        k = 0
        while (k < leafRows.length) { weights(leafRows(k)) += sampleWeight(leafRows(k)) / leafWeight; k += 1 }
      }
    }
    if (usedTrees > 0) {
      val inv = 1.0 / usedTrees
      var i = 0
      while (i < nTrain) { weights(i) *= inv; i += 1 }
    }
    weights
  }
}
