package treebased.core.prediction

import treebased.core.domain.tree.Node

/**
 * Pointwise variance / confidence intervals for `theta_hat(x)`, from Athey, Tibshirani & Wager
 * (2019), "Generalized Random Forests", §4, via the bootstrap-of-little-bags / grouped
 * half-sample estimator (Sexton & Laake 2009).
 *
 * The forest must be grown in groups that share a half-sample draw (see
 * [[treebased.api.GeneralizedRandomForest.trainTreesWithMembership]] with
 * `ForestConfig.ciGroupSize >= 2`). Each tree `b` contributes a locally linear pseudo-estimate
 *
 *   T_b(x) = theta_hat(x) + sum_i alpha_i^b(x) * rho_i
 *
 * where `alpha_i^b(x)` is tree `b`'s leaf kernel ([[ForestKernel]]) and `rho_i` the estimand's
 * influence contribution at the forest solution ([[WeightedEstimator.influence]]). Writing
 * `T_g(x)` for a group mean over `S_g` trees and `T_bar(x)` for the mean over all `M` usable
 * trees:
 *
 *   between = (1 / (G - 1)) * sum_g (T_g - T_bar)^2
 *   within  = (1 / G) * sum_g [ (1 / (S_g - 1)) * sum_{b in g} (T_b - T_g)^2 ]
 *   Var_hat(theta_hat(x)) = max(between - within / S_bar, 0)
 *
 * `within / S_bar` debiases `between` for the Monte-Carlo noise it absorbs from finitely many
 * trees per group; in the linear case this coincides with the infinitesimal jackknife. The exact
 * finite-forest constants of Athey, Tibshirani & Wager (2019, §4) would refine the debiasing;
 * this is the standard Sexton-Laake form.
 */
object VarianceEstimator {

  /** `estimate` +- `stdErr`, with the `level`-confidence normal interval `[lower, upper]`. */
  final case class Interval(estimate: Double, stdErr: Double, lower: Double, upper: Double)

  def interval(
    trees: Vector[Node],
    treeGroups: Array[Int],
    rows: Array[KernelRow],
    estimator: WeightedEstimator,
    x: Array[Double],
    level: Double = 0.95
  ): Interval = {
    require(trees.length == treeGroups.length, "treeGroups must be aligned to trees")
    require(level > 0.0 && level < 1.0, s"level must be in (0,1), got $level")

    val n = rows.length
    val sampleWeight: Int => Double = i => rows(i).weight

    val alphaFull = ForestKernel.alpha(trees, x, n, sampleWeight)
    val thetaHat  = estimator.solve(rows, alphaFull)
    val rho       = estimator.influence(rows, alphaFull, thetaHat)

    // Per-tree locally linear pseudo-estimate; None for a tree whose leaf at x had no honest rows.
    val perTree: Array[Option[Double]] = Array.tabulate(trees.length) { t =>
      val alphaB = ForestKernel.alpha(Vector(trees(t)), x, n, sampleWeight)
      var used = 0.0
      var agg  = 0.0
      var i = 0
      while (i < n) {
        val a = alphaB(i)
        if (a != 0.0) { used += a; agg += a * rho(i) }
        i += 1
      }
      if (used == 0.0) None else Some(thetaHat + agg)
    }

    val groups: Array[Array[Double]] =
      perTree.indices
        .collect { case i if perTree(i).isDefined => i }
        .groupBy(i => treeGroups(i))
        .values
        .map(_.map(i => perTree(i).get).toArray)
        .filter(_.length >= 2)
        .toArray

    val nGroups = groups.length
    if (nGroups < 2) return Interval(thetaHat, Double.NaN, Double.NaN, Double.NaN)

    val groupMeans = groups.map(group => group.sum / group.length)
    val totalTrees = groups.map(_.length).sum
    val grandMean  = groups.iterator.flatMap(_.iterator).sum / totalTrees

    var betweenRaw = 0.0
    var withinMean = 0.0
    var k = 0
    while (k < nGroups) {
      val group     = groups(k)
      val groupMean = groupMeans(k)
      betweenRaw += (groupMean - grandMean) * (groupMean - grandMean)
      var sumSqWithin = 0.0
      var j = 0
      while (j < group.length) { val d = group(j) - groupMean; sumSqWithin += d * d; j += 1 }
      withinMean += sumSqWithin / (group.length - 1)
      k += 1
    }
    withinMean /= nGroups
    val sBar = totalTrees.toDouble / nGroups

    val varHat = math.max(betweenRaw / (nGroups - 1) - withinMean / sBar, 0.0)
    val se     = math.sqrt(varHat)
    val z      = Stats.normalQuantile(1.0 - (1.0 - level) / 2.0)
    Interval(thetaHat, se, thetaHat - z * se, thetaHat + z * se)
  }
}
