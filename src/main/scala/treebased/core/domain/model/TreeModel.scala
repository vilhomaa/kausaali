package treebased.core.domain.model

import treebased.config.TreeBaseConfig
import treebased.core.domain.tree.Node

trait TreeModel extends Serializable {
  val treeConfig: TreeBaseConfig
  val rootNode: Node

  def predict(features: Array[Double]): Double

  val depth: Int = rootNode.depth
}
