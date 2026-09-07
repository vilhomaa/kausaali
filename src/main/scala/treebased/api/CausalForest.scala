package treebased.api

import treebased.config.{ExactCausalForestConfig, ForestConfig}
import treebased.core.prediction.LeafEstimator
import treebased.core.training.split.criterion.{ExactCausalVariance, SplitCriterion}
import treebased.core.training.aggregate.TreatmentEffectStats
import treebased.core.training.TreeTrainer
import treebased.core.domain.data.CausalObservation
import treebased.core.domain.tree.Node
import treebased.core.domain.model.ForestModel
import treebased.core.prediction.ForestAggregation

import scala.util.Random
import scala.collection.parallel.CollectionConverters.*
import scala.collection.parallel.immutable.ParVector


/**
 * User-facing API for a causal forest built with the exact-splitting honest tree of Athey &
 * Wager (2018): every candidate threshold is scored directly by the child ATE difference
 * `P_L·P_R·(τ_L − τ_R)²`, with no stabilization. That criterion tracks sampling noise in `τ̂` and
 * is markedly less accurate than gradient splitting on small samples — prefer
 * [[GeneralizedRandomForest.causal]] (Athey, Tibshirani & Wager, 2019) for estimation; this class
 * is kept for the exact-2018 comparison.
 */
case class CausalForest private (
  forestConfig: ForestConfig,
  trees: Vector[Node],
  predOps: ForestAggregation
) extends ForestModel {

  def predictSingle(features: Array[Double]): Double =
    predOps.aggregate(trees.map(tree => tree.predict(features)), trees.size)

  override def predict(features: Array[Array[Double]]): Array[Double] =
    features.par.map(predictSingle).toArray

  def size: Int = trees.size
}

object CausalForest {

  def apply(
    forestConfig: ForestConfig,
    trees: Vector[Node],
    predOps: ForestAggregation
  ): CausalForest = {
    require(trees.nonEmpty, "Forest must have at least one tree")
    require(
      trees.size == forestConfig.nTrees,
      s"Expected ${forestConfig.nTrees} trees but got ${trees.size}"
    )
    new CausalForest(forestConfig, trees, predOps)
  }

  def train(
     trainingData: Array[CausalObservation],
     config: ExactCausalForestConfig
  ): CausalForest = {
    val nFeatures = trainingData.headOption.map(_.features.length).getOrElse(0)
    train(trainingData, config.toForestConfig(nFeatures), LeafEstimator.ate, ExactCausalVariance, ForestAggregation.mean)
  }
  
  def train(
     trainingData: Array[CausalObservation],
     config: ForestConfig,
     leaf: LeafEstimator[CausalObservation, TreatmentEffectStats],
     impurity: SplitCriterion[TreatmentEffectStats],
     predOps: ForestAggregation
  ): CausalForest = {

    val rng = new Random(config.seed)
    val seeds = ParVector.fill(config.nTrees)(rng.nextLong())
    val sortedFeatureIndices = TreeTrainer.sortFeatureIndices(trainingData, config)

    val nodes = seeds.map { seed =>
      val treeRng = new Random(seed)
      val treeConfig = config.toTreeConfig(seed)
      val (trainingDataSubsample, sortedSubSampleFeatureIndices, sampledIndices) = TreeTrainer.subSampleTrainingData(treeRng, trainingData, sortedFeatureIndices, config)
      CausalTree.trainTree(trainingDataSubsample, treeConfig, leaf, impurity, Some(sortedSubSampleFeatureIndices), sampledIndices)
    }
    CausalForest(config, nodes.toVector, predOps)
  }
}
