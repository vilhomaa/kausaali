package treebased.preprocessing.Encoders

import treebased.preprocessing.Encoder

case class OneHotEncoder[T](categories: IndexedSeq[T]) extends Encoder[T] {
  private val indexMap: Map[T, Int] = categories.zipWithIndex.toMap
  lazy val decodingMap: Map[Int, T] = indexMap.map(_.swap)

  def encode(value: T): Either[Throwable, Array[Double]] =
    indexMap.get(value)
      .toRight(new NoSuchElementException(s"Unknown category: $value"))
      .map { idx =>
        val arr = new Array[Double](categories.size)
        arr(idx) = 1.0
        arr
      }

  def decode(encodedValue: Array[Double]): T = decodingMap(encodedValue.indexOf(1.0))
  def outputSize: Int = categories.size
}

object OneHotEncoder {
  def apply[T](values: Seq[T]): OneHotEncoder[T] =
    new OneHotEncoder(values.distinct.toIndexedSeq)
}
