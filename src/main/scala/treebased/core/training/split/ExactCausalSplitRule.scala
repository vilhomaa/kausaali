package treebased.core.training.split

import treebased.config.TreeBaseConfig
import treebased.core.training.split.criterion.SplitCriterion
import treebased.core.training.split.{FeatureSubsampling, SplitRule}
import treebased.core.training.split.guard.SplitGuard
import treebased.core.training.aggregate.{TreatmentEffectStats, TreatmentEffectStatsAccumulator}
import treebased.core.domain.data.CausalObservation
import treebased.core.domain.tree.{Split, TreeSplit}

import scala.util.Random

/**
 * Exact causal-tree splitting rule (Athey & Wager, 2019, "Estimation and Inference of
 * Heterogeneous Treatment Effects using Random Forests"): for every candidate threshold,
 * exactly scores the resulting children by treatment-effect heterogeneity via [[SplitCriterion]].
 */
case class ExactCausalSplitRule(
  treeConfig: TreeBaseConfig,
  rng: Random,
  impurity: SplitCriterion[TreatmentEffectStats],
  sizeGuard: SplitGuard[CausalObservation]
) extends SplitRule[CausalObservation] with FeatureSubsampling {

  private val acc = TreatmentEffectStatsAccumulator

  private def findBestExactSplit(
    data: Array[CausalObservation],
    sortedFeatureIndices: Array[Array[Int]],
    featureIndex: Int
  ): Option[TreeSplit] = {
    val rowsByFeatureValue = sortedFeatureIndices(featureIndex).map(data)
    val candidates         = sizeGuard.trimToValidWindow(rowsByFeatureValue, treeConfig.minNodeSizePerTreatmentGroup)
    if (candidates.isEmpty) return None

    // Seed the right child with every candidate; slide one point at a time into the left child,
    // evaluating gain only at feature-value boundaries.
    var left  = acc.empty
    var right = acc.foldMap(candidates)
    var bestSplit: Option[TreeSplit] = None
    var i = 0
    while (i < candidates.length - 1) {
      val row = candidates(i)
      val moved = acc.lift(row)
      left  = acc.combine(left, moved)
      right = acc.remove(right, moved)
      if (row.features(featureIndex) != candidates(i + 1).features(featureIndex)) {
        val gain      = impurity.calculate(left, right)
        // Threshold at the midpoint between the two straddling values, not on the sample value.
        val threshold = (row.features(featureIndex) + candidates(i + 1).features(featureIndex)) / 2.0
        val split     = Split(featureIndex, threshold, gain)
        if (gain.isFinite && bestSplit.forall(_.infoGain < gain)) bestSplit = Some(split)
      }
      i += 1
    }
    bestSplit
  }

  def findBestSplitPoint(
    nodeDataPoints: Array[CausalObservation],
    featureIndexesForSplit: Vector[Int],
    sortedFeatureIndices: Array[Array[Int]],
  ): Option[TreeSplit] = {
    // `mtry`: evaluate the first `maxFeaturesForSplit` features in the (already shuffled)
    // list; keep going only if none of them yielded a valid split.
    val mtry = treeConfig.maxFeaturesForSplit
    var best: Option[TreeSplit] = None
    var k = 0
    while (k < featureIndexesForSplit.length && (k < mtry || best.isEmpty)) {
      findBestExactSplit(nodeDataPoints, sortedFeatureIndices, featureIndexesForSplit(k)) match {
        case Some(s) if best.forall(_.infoGain < s.infoGain) => best = Some(s)
        case _                                               => ()
      }
      k += 1
    }
    best
  }
}
