package treebased.preprocessing.Encoders

import treebased.preprocessing.{Encoder, EncodingPipeline}

case class PassThroughEncoder() extends Encoder[Any] {
  def encode(value: Any): Either[Throwable, Array[Double]] = EncodingPipeline.anyToDouble(value).map(Array(_))
  def decode(encodedValue: Array[Double]): Any = encodedValue.head
  def outputSize: Int = 1
}
