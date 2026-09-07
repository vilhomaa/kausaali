package treebased.api

import treebased.config.TreeBaseConfig
import treebased.core.training.split.criterion.GradientVarianceReduction
import treebased.core.training.split.GradientSplitRule
import treebased.estimand.ForestSpec
import treebased.core.training.TreeTrainer
import treebased.core.domain.data.DataPoint
import treebased.core.domain.tree.Node
import treebased.core.domain.model.TreeModel

import scala.reflect.ClassTag
import scala.util.Random

/**
 * User-facing API for a single tree from the Generalized Random Forests family (Athey,
 * Tibshirani & Wager, 2019). Always splits via [[GradientSplitRule]] with
 * [[GradientVarianceReduction]]; the estimand (causal ATE, regression mean, ...) is selected by
 * the injected [[ForestSpec]].
 */
case class GeneralizedRandomTree(treeConfig: TreeBaseConfig, rootNode: Node) extends TreeModel {

  override def predict(features: Array[Double]): Double = rootNode.predict(features)

  override val depth: Int = rootNode.depth
}

object GeneralizedRandomTree {

  def train[O <: DataPoint : ClassTag](
             trainingData: Array[O],
             treeConfig: TreeBaseConfig,
             strategy: ForestSpec[O]
           ): GeneralizedRandomTree = {
    val rootNode = trainTree(trainingData, treeConfig, strategy)
    GeneralizedRandomTree(treeConfig, rootNode)
  }

  def trainTree[O <: DataPoint : ClassTag](
                 trainingData: Array[O],
                 config: TreeBaseConfig,
                 strategy: ForestSpec[O],
                 sortedSubSampleFeatureIndices: Option[Array[Array[Int]]] = None,
                 subsampleRowIds: Array[Int] = Array.emptyIntArray
               ): Node = {
    val splitter = GradientSplitRule(config, Random(config.seed), GradientVarianceReduction, strategy.moment, strategy.sizeGuard)
    TreeTrainer.trainTree(trainingData, config, splitter, strategy.leaf, strategy.sizeGuard, sortedSubSampleFeatureIndices, subsampleRowIds)
  }
}
