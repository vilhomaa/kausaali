package treebased.core.prediction

trait ForestAggregation {
  def aggregate(values: Seq[Double], nTrees: Int): Double
}

object ForestAggregation {
  val mean: ForestAggregation = new ForestAggregation {
    def aggregate(predictions: Seq[Double], nTrees: Int): Double =
      predictions.sum / nTrees
  }
}
