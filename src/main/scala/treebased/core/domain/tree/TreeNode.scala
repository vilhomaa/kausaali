package treebased.core.domain.tree

case class TreeNode(
                     id: String,
                     depth: Int,
                     split: TreeSplit,
                     leftNode: Node,
                     rightNode: Node,
 ) extends Node {
  override def predict(features: Array[Double]): Double =
   if (split.goLeft(features)) leftNode.predict(features)
   else rightNode.predict(features)

  override def locate(features: Array[Double]): LeafNode =
   if (split.goLeft(features)) leftNode.locate(features)
   else rightNode.locate(features)
}
