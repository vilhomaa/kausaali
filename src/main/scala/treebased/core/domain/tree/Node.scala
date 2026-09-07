package treebased.core.domain.tree

trait Node {
  val id: String
  val depth: Int
  def predict(features: Array[Double]): Double

  /** The leaf this node routes `features` into — the honest sample behind that leaf is what
   *  [[treebased.core.prediction.ForestKernel]] turns into the adaptive weights alpha_i(x). */
  def locate(features: Array[Double]): LeafNode
}
