package treebased.preprocessing.Encoders

import treebased.preprocessing.{Encoder, EncoderFitter}

case class LabelEncoder[T](encoding: Map[T, Double]) extends Encoder[T] {
  private lazy val decoding: Map[Double, T] = encoding.map(_.swap)
  def decode(c: Array[Double]): T = decoding(c.head)
  def encode(input: T): Either[Throwable, Array[Double]] =
    encoding.get(input)
      .toRight(new NoSuchElementException(s"Value $input not found in encoding map"))
      .map(Array(_))
  def outputSize: Int = 1
}

object LabelEncoder {
  def apply[T](values: Seq[T]) : LabelEncoder[T] = {
    new LabelEncoder(values.distinct.zipWithIndex.map(p => (p._1, (p._2 + 1).toDouble)).toMap)
  }
}
