package treebased.core.domain.tree


sealed trait TreeSplit {
  def featureIndex: Int
  def infoGain: Double
  def goLeft(features: Array[Double]): Boolean
}

case class Split(
  featureIndex: Int,
  threshold: Double,
  infoGain: Double
) extends TreeSplit {
  def goLeft(features: Array[Double]): Boolean =
    features(featureIndex) <= threshold
}
