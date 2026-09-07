package treebased.core.training.split

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import treebased.config.TreeConfig
import treebased.core.domain.tree.Split
import treebased.core.training.TreeTrainer
import treebased.core.training.moment.CausalMoment
import treebased.core.training.split.criterion.{ExactCausalVariance, GradientVarianceReduction}
import treebased.core.training.split.guard.TreatmentBalanceGuard
import treebased.testutils.SyntheticData.generatePoints

import scala.util.Random

class SplitRuleSuite extends AnyFlatSpec with Matchers {

  private val trainingRows = generatePoints(1000)
  private val treeConfig = TreeConfig(maxDepth = 5, nFeatures = 3, maxFeaturesForSplit = 3)
  private val sortedFeatureIndices = TreeTrainer.sortFeatureIndices(trainingRows, treeConfig)
  // The DGP's heterogeneity is driven entirely by feature 2 (x3), so both rules should split there.
  private val candidateFeatures = Vector(1, 2)

  "ExactCausalSplitRule.findBestSplitPoint" should "split on the heterogeneity-carrying feature" in {
    val splitter = ExactCausalSplitRule(treeConfig, Random(treeConfig.seed), ExactCausalVariance, TreatmentBalanceGuard)
    val split = splitter.findBestSplitPoint(trainingRows, candidateFeatures, sortedFeatureIndices)
    split.map(_.featureIndex) shouldBe Some(2)
    split.map(_.infoGain).foreach(_ should be > 0.0)
  }

  // Characterization test: pins the exact split this deterministic input produces. Expected to
  // change whenever the exact-splitting criterion or its traversal is deliberately modified.
  it should "produce a stable exact split for the fixed seed" in {
    val splitter = ExactCausalSplitRule(treeConfig, Random(treeConfig.seed), ExactCausalVariance, TreatmentBalanceGuard)
    splitter.findBestSplitPoint(trainingRows, candidateFeatures, sortedFeatureIndices) shouldBe
      Some(Split(2, 0.3986128506132841, 2.1190323024957656))
  }

  "GradientSplitRule.findBestSplitPoint" should "also split on the heterogeneity-carrying feature" in {
    val splitter = GradientSplitRule(
      treeConfig, Random(treeConfig.seed), GradientVarianceReduction, CausalMoment(), TreatmentBalanceGuard)
    val split = splitter.findBestSplitPoint(trainingRows, candidateFeatures, sortedFeatureIndices)
    split.map(_.featureIndex) shouldBe Some(2)
    split.map(_.infoGain).foreach(_ should be > 0.0)
  }
}
