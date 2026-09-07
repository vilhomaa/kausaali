package treebased.core.training.split

import treebased.config.TreeBaseConfig

import scala.util.Random

/**
 * Shared feature-subsampling behavior for [[SplitRule]] implementations: returns the node's
 * features in a fresh random order. The splitter evaluates the first `maxFeaturesForSplit` of
 * them (`mtry`) and, if none yields a valid split, keeps going down the list until one does
 * or the features are exhausted — a "draw more variables rather than give up" fallback. Mix
 * this into a splitter so exact and gradient-based rules don't each redefine the logic.
 *
 * The order is a full permutation, not a random-sized subset: a subset of size drawn from
 * `1..mtry` would, at `mtry = nFeatures`, still look at only ~half the features per node.
 */
trait FeatureSubsampling {
  def rng: Random

  def candidateFeatureSplitIndices(config: TreeBaseConfig): Vector[Int] =
    rng.shuffle((0 until config.nFeatures).toVector)
}
