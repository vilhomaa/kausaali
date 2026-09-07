package treebased.core.training.split.criterion

trait SplitCriterion[Stats] extends Serializable {
  def calculate(left: Stats, right: Stats): Double
}
