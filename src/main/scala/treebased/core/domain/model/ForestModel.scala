package treebased.core.domain.model

import treebased.config.ForestConfig
import treebased.core.domain.tree.Node

trait ForestModel extends Serializable {
  def forestConfig: ForestConfig
  def trees: Vector[Node]
  def predict(features: Array[Array[Double]]): Array[Double]
  def predictSingle(features: Array[Double]): Double
}
