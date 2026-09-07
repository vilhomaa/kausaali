package treebased.core.domain.data

/** Everything the estimand-agnostic tree machinery needs from a training row: the covariates it
 *  splits on and the sample weight. Estimand-specific fields live on the concrete subtypes. */
trait DataPoint {
  def features: Array[Double]
  def weight: Double
}

/** A single-response row: regression and quantile forests both consume this shape. */
final case class Observation(
  features: Array[Double],
  weight: Double,
  y: Double
) extends DataPoint

/** A response row plus a treatment assignment, consumed by the causal forests. `w` is `Double`
 *  so it can carry a residualized (locally centered) treatment as well as a raw assignment. */
final case class CausalObservation(
  features: Array[Double],
  weight: Double,
  y: Double,
  w: Double
) extends DataPoint
