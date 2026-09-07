package treebased.core.domain.tree

import treebased.config.TreeBaseConfig
import treebased.core.prediction.LeafEstimator
import treebased.core.domain.data.DataPoint

/**
 * `honestRowIds` are the indices, into the forest's training array, of the honest-sample rows
 * that landed in this leaf. They are what [[treebased.core.prediction.ForestKernel]] needs to
 * build the adaptive weights alpha_i(x); the scalar `prediction` is retained as the fast path
 * for linear estimands (and as the NaN honesty-prune signal).
 */
case class LeafNode(
                     id: String,
                     depth: Int,
                     prediction: Double,
                     nDataPoints: Int,
                     honestRowIds: Array[Int]
                   ) extends Node {
  override def predict(features: Array[Double]): Double = prediction
  override def locate(features: Array[Double]): LeafNode = this
}

object LeafNode {
  def build[O <: DataPoint](
             predictionData: Array[O],
             predictionRowIds: Array[Int],
             config: TreeBaseConfig,
             parentNodeId: String,
             curDepth: Int,
             leaf: LeafEstimator[O, ?]
           ): LeafNode = {
    LeafNode(
      id = parentNodeId,
      depth = curDepth,
      prediction = leaf.prediction(predictionData),
      nDataPoints = predictionData.length,
      honestRowIds = predictionRowIds
    )
  }
}
