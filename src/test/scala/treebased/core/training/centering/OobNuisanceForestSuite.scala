package treebased.core.training.centering

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import treebased.api.GeneralizedRandomForest
import treebased.config.ForestConfig
import treebased.core.domain.data.{CausalObservation, Observation}
import treebased.estimand.RegressionSpec
import treebased.testutils.Numeric.pearson

import scala.util.Random

/** White-box tests for the out-of-bag nuisance forest behind [[OobCentering]]. Lives in the
 *  `treebased` package tree so it can see the `private[treebased]` OOB helpers. */
class OobNuisanceForestSuite extends AnyFlatSpec with Matchers {

  private val rng      = new Random(1)
  private val n        = 400
  private val features = Array.fill(n)(Array(rng.nextGaussian(), rng.nextGaussian()))
  private val target   = features.map(x => x(0) * 2.0 - x(1) + rng.nextGaussian() * 0.1)

  private val cfg = ForestConfig(
    maxDepth = 8, nTrees = 200, nFeatures = 2, maxFeaturesForSplit = 2,
    minNodeSize = 5, subsampleRatio = 0.5, seed = 7L
  )

  private def rows = Array.tabulate(n)(i => Observation(features(i), 1.0, target(i)))

  "trainTreesWithMembership" should "pair every tree with the rows it was grown on" in {
    val trees = GeneralizedRandomForest.trainTreesWithMembership(rows, cfg, RegressionSpec)
    trees should have length cfg.nTrees
    val expected = (cfg.subsampleRatio * n).toInt
    trees.foreach { grown =>
      grown.rowIds.length shouldBe expected
      grown.rowIds.distinct.length shouldBe grown.rowIds.length   // sampled without replacement
      all (grown.rowIds.toIndexedSeq) should (be >= 0 and be < n)
    }
  }

  "predictOob" should "give finite predictions that track the target" in {
    val preds = OobNuisanceForest.predictOob(features, target, cfg)
    preds should have length n
    all (preds.toIndexedSeq.map(_.isFinite)) shouldBe true
    pearson(preds.toSeq, target.toSeq) should be > 0.85
  }

  it should "average only out-of-bag trees for each row" in {
    val trees = GeneralizedRandomForest.trainTreesWithMembership(rows, cfg, RegressionSpec)
    val i = 0
    val inBagForRow0 = trees.zipWithIndex.collect { case (grown, t) if grown.rowIds.contains(i) => t }.toSet
    val oobMean = trees.zipWithIndex.collect {
      case (grown, t) if !inBagForRow0.contains(t) => grown.node.predict(features(i))
    }
    val expected = oobMean.sum / oobMean.length
    oobMean.length should be >= OobNuisanceForest.MinOobTrees
    OobNuisanceForest.predictOob(features, target, cfg)(i) shouldBe expected +- 1e-9
  }

  it should "fall back to the global mean when a row has no out-of-bag trees" in {
    val noOob = cfg.copy(subsampleRatio = 1.0, nTrees = 20) // every tree sees every row
    val preds = OobNuisanceForest.predictOob(features, target, noOob)
    val mean  = target.sum / n
    all (preds.toIndexedSeq.map(p => math.abs(p - mean))) should be < 1e-9
  }

  "OobCentering" should "residualize without inflating the treatment scale (near-RCT data)" in {
    val cRng = new Random(2)
    val cData = Array.tabulate(1200) { _ =>
      val x = Array(cRng.nextGaussian(), cRng.nextGaussian(), cRng.nextGaussian())
      val w = cRng.nextInt(2).toDouble
      val y = x(0) + 0.5 * x(1) - 2.0 * x(2) * w + cRng.nextGaussian()
      CausalObservation(x, 1.0, y, w)
    }
    val centered = new OobCentering(cfg.copy(nFeatures = 3, maxFeaturesForSplit = 2))
      .centered(cData)

    val wRes = centered.map(_.w)
    val yRes = centered.map(_.y)
    all (wRes.toIndexedSeq.map(_.isFinite)) shouldBe true
    // centred treatment stays on roughly its raw scale, not blown up
    math.sqrt(wRes.map(v => v * v).sum / wRes.length) should be < 1.0
    math.abs(yRes.sum / yRes.length) should be < 0.5
  }
}
