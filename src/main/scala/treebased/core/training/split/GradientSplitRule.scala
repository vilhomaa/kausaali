package treebased.core.training.split

import treebased.config.TreeBaseConfig
import treebased.core.training.split.criterion.SplitCriterion
import treebased.core.training.moment.Moment
import treebased.core.training.split.guard.SplitGuard
import treebased.core.training.aggregate.{PseudoOutcomeAccumulator, SumCount}
import treebased.core.domain.data.DataPoint
import treebased.core.domain.tree.{Split, TreeSplit}

import scala.reflect.ClassTag
import scala.util.Random

/**
 * Gradient-based tree splitting rule from Athey, Tibshirani & Wager
 * (2019), "Generalized Random Forests": relabels a node's points via the injected
 * [[Moment]], then scores candidate splits by a standard weighted-variance-reduction
 * criterion. The moment condition is what makes this splitter reusable across estimands - e.g.
 * [[treebased.core.training.moment.CausalMoment]] for a causal forest,
 * [[treebased.core.training.moment.RegressionMoment]] for a regression forest.
 *
 * Three details that materially affect accuracy:
 *  - `mtry`: `featureIndexesForSplit` arrives as a full random permutation; the first
 *    `maxFeaturesForSplit` are evaluated, and more only if none of those yields a valid split.
 *  - split threshold placed at the midpoint between the two straddling feature values, not on a
 *    sample value (which also removes an off-by-one between the scored and realized partition).
 *  - split stabilization: when [[TreeBaseConfig.stabilizeSplits]] and the split-balance weights
 *    are non-uniform (causal), each child's score is weighted by its treatment-residual variance
 *    relative to the parent's, so a thin-overlap child cannot win on noise.
 */
case class GradientSplitRule[O <: DataPoint : ClassTag](
  treeConfig: TreeBaseConfig,
  rng: Random,
  impurity: SplitCriterion[SumCount],
  moment: Moment[O],
  sizeGuard: SplitGuard[O]
) extends SplitRule[O] with FeatureSubsampling {

  def findBestSplitPoint(
    nodeDataPoints: Array[O],
    featureIndexesForSplit: Vector[Int],
    sortedFeatureIndices: Array[Array[Int]]
  ): Option[TreeSplit] = {

    val pseudoOutcomes = moment.pseudoOutcomes(nodeDataPoints)
    val n              = nodeDataPoints.length

    // `alpha` (child-balance rule): each child must retain at least this fraction of the node's
    // split-balance weight (uniform, or treatment-residual variance for a causal guard). 0 disables it.
    // split stabilization: re-weight each child's score by its share of that weight per point.
    val alpha       = treeConfig.splitBalanceAlpha
    val stabilize   = treeConfig.stabilizeSplits
    val needWeights = alpha > 0.0 || stabilize
    val rowWeights  = if (needWeights) sizeGuard.splitBalanceWeights(nodeDataPoints) else Array.emptyDoubleArray
    val totalWeight = if (needWeights) { var s = 0.0; var i = 0; while (i < rowWeights.length) { s += rowWeights(i); i += 1 }; s } else 0.0
    val weightFloor = alpha * totalWeight
    // a_P = totalWeight / n (mean of the per-point weight); its inverse scales the stabilized score.
    val invParentVar = if (stabilize && totalWeight > 0.0) n.toDouble / totalWeight else 0.0

    val mtry = treeConfig.maxFeaturesForSplit
    var best: Option[(Int, Double, Double)] = None // (featureIdx, threshold, gain)
    var k = 0
    while (k < featureIndexesForSplit.length && (k < mtry || best.isEmpty)) {
      val featureIdx         = featureIndexesForSplit(k)
      val rowsByFeatureValue = sortedFeatureIndices(featureIdx)

      if (sizeGuard.trimToValidWindow(rowsByFeatureValue.map(nodeDataPoints), treeConfig.minNodeSizePerTreatmentGroup).nonEmpty) {
        val pseudoCum = rowsByFeatureValue.scanLeft(PseudoOutcomeAccumulator.empty) { (acc, rowIdx) =>
          PseudoOutcomeAccumulator.combine(acc, PseudoOutcomeAccumulator.lift(pseudoOutcomes(rowIdx)))
        }
        val nodeTotal = pseudoCum.last
        val weightCum = if (needWeights) rowsByFeatureValue.scanLeft(0.0)((acc, rowIdx) => acc + rowWeights(rowIdx)) else Array.emptyDoubleArray

        val lo = math.max(1, treeConfig.minNodeSize - 1)
        val hi = rowsByFeatureValue.length - treeConfig.minNodeSize - 1
        var idx = lo
        while (idx < hi) {
          val weightLeft = if (needWeights) weightCum(idx) else 0.0
          val alphaOk    = alpha <= 0.0 || (weightLeft >= weightFloor && totalWeight - weightLeft >= weightFloor)
          if (alphaOk) {
            val leftStats  = pseudoCum(idx)
            val rightStats = PseudoOutcomeAccumulator.remove(nodeTotal, leftStats)
            val nL = leftStats.count
            val nR = rightStats.count
            val gain =
              if (stabilize && totalWeight > 0.0 && nL > 0 && nR > 0) {
                val avgWeightLeft  = weightLeft / nL
                val avgWeightRight = (totalWeight - weightLeft) / nR
                avgWeightLeft  * invParentVar * leftStats.sum  * leftStats.sum  / nL +
                avgWeightRight * invParentVar * rightStats.sum * rightStats.sum / nR
              } else impurity.calculate(leftStats, rightStats)

            if (gain.isFinite && best.forall(_._3 < gain)) {
              val threshold = (nodeDataPoints(rowsByFeatureValue(idx - 1)).features(featureIdx) +
                               nodeDataPoints(rowsByFeatureValue(idx)).features(featureIdx)) / 2.0
              best = Some((featureIdx, threshold, gain))
            }
          }
          idx += 1
        }
      }
      k += 1
    }

    best.map { case (featureIdx, threshold, gain) => Split(featureIdx, threshold, gain) }
  }
}
