package treebased.api

import treebased.config.TreeBaseConfig
import treebased.core.prediction.LeafEstimator
import treebased.core.training.split.criterion.SplitCriterion
import treebased.core.training.split.ExactCausalSplitRule
import treebased.core.training.split.guard.TreatmentBalanceGuard
import treebased.core.training.aggregate.TreatmentEffectStats
import treebased.core.training.TreeTrainer
import treebased.core.domain.data.CausalObservation
import treebased.core.domain.tree.Node
import treebased.core.domain.model.TreeModel

import scala.util.Random


/** User-facing API for a single exact-split causal tree (Athey & Wager, 2018).  */
case class CausalTree(treeConfig: TreeBaseConfig, rootNode: Node) extends TreeModel {

  override def predict(features: Array[Double]): Double = rootNode.predict(features)

  override val depth: Int = rootNode.depth
}

object CausalTree {

  def train(
             trainingData: Array[CausalObservation],
             treeConfig: TreeBaseConfig,
             leaf: LeafEstimator[CausalObservation, TreatmentEffectStats],
             impurity: SplitCriterion[TreatmentEffectStats]
           ): CausalTree = {
    val rootNode = trainTree(trainingData, treeConfig, leaf, impurity)
    CausalTree(treeConfig, rootNode)
  }

  def trainTree(
                 trainingData: Array[CausalObservation],
                 config: TreeBaseConfig,
                 leaf: LeafEstimator[CausalObservation, TreatmentEffectStats],
                 impurity: SplitCriterion[TreatmentEffectStats],
                 sortedSubSampleFeatureIndices : Option[Array[Array[Int]]] = None,
                 subsampleRowIds: Array[Int] = Array.emptyIntArray
               ): Node = {
    val splitter = ExactCausalSplitRule(config, Random(config.seed), impurity, TreatmentBalanceGuard)
    TreeTrainer.trainTree(trainingData, config, splitter, leaf, TreatmentBalanceGuard, sortedSubSampleFeatureIndices, subsampleRowIds)
  }
}
