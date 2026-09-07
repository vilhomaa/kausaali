package treebased.preprocessing

import scala.reflect.ClassTag
import scala.util.Try

/**
 * A fitted encoding pipeline that transforms raw mixed-type feature rows into Array[Double].
 * Fitted on training data, then reused for prediction data to ensure consistent encoding.
 */
case class EncodingPipeline(columnEncoders: IndexedSeq[Encoder[Any]]) {

  /** Transform a single row of raw features into a Double array. `Left` on the first column
   *  whose encoder cannot represent its value (e.g. a category unseen when the pipeline was
   *  fitted) — see [[Encoder.encode]]. */
  def transform(features: IndexedSeq[Any]): Either[Throwable, Array[Double]] =
    EncodingPipeline.traverse(features.zip(columnEncoders)) { case (v, enc) => enc.encode(v) }.map(_.flatten)

  /**
   * Encodes the exact rows this pipeline was [[fit]] on — see [[Encoder.encodeTrainingRows]].
   * For a label-derived column (`MeanEncoder`) this avoids leaking a row's own label into its own
   * feature value; for every other column it's equivalent to calling [[transform]] per row.
   * `rows` must be those same fit-time rows, in the same order.
   */
  def transformTrainingRows(rows: Seq[IndexedSeq[Any]]): Either[Throwable, Array[Array[Double]]] =
    EncodingPipeline.traverse(columnEncoders.indices) { colIdx =>
      columnEncoders(colIdx).encodeTrainingRows(rows.map(_(colIdx)))
    }.map(perColumn => Array.tabulate(rows.size)(row => perColumn.flatMap(_(row))))

  /** Total number of output features (may differ from input due to one-hot expansion). */
  def outputSize: Int = columnEncoders.map(_.outputSize).sum
}

object EncodingPipeline {

  /**
   * Runs `f` over every element of `xs`, short-circuiting on the first `Left`. The one place this
   * "sequence a batch of fallible per-row/per-column encodes" pattern lives — [[transform]] uses
   * it across a row's columns, [[treebased.preprocessing.DataLoader]] across a dataset's rows.
   */
  private[preprocessing] def traverse[A, B : ClassTag](xs: Seq[A])(f: A => Either[Throwable, B]): Either[Throwable, Array[B]] =
    xs.foldLeft[Either[Throwable, Vector[B]]](Right(Vector.empty)) { (acc, a) =>
      for { bs <- acc; b <- f(a) } yield bs :+ b
    }.map(_.toArray)

  /** Best-effort widening of a raw cell to `Double`. `Left` (never a throw) when the value is
   *  non-numeric — the shared home for [[PassThroughEncoder]] and
   *  [[treebased.preprocessing.DataLoader]]. */
  private[preprocessing] def anyToDouble(value: Any): Either[Throwable, Double] = value match {
    case d: Double  => Right(d)
    case b: Boolean => Right(if (b) 1.0 else 0.0)
    case n: Number  => Right(n.doubleValue())
    case s: String  => Try(s.toDouble).toEither
    case other      => Left(new IllegalArgumentException(s"Cannot convert to Double: $other (${other.getClass})"))
  }

  /**
   * Fit an encoding pipeline on training data.
   *
   * Columns in `encodings` use the specified strategy. Columns not in `encodings` are
   * auto-detected: numeric values get PassThrough, non-numeric values get `defaultEncoding`.
   *
   * @param featureColumns raw feature values per row (each row is an IndexedSeq of Any)
   * @param labels         the label values (needed for mean encoding)
   * @param encodings      user-specified encoding per feature column index
   * @param defaultEncoding encoding to use for auto-detected non-numeric columns
   */
  def fit(
    featureColumns: Seq[IndexedSeq[Any]],
    labels: Seq[Double],
    encodings: Map[Int, EncodingType] = Map.empty,
    defaultEncoding: EncodingType = EncodingType.Mean
  ): EncodingPipeline = {
    val numCols = if (featureColumns.isEmpty) 0 else featureColumns.head.size
    val fitted = (0 until numCols).map { idx =>
      val colValues = featureColumns.map(_(idx))
      val strategy = encodings.getOrElse(idx, detectEncodingType(colValues, defaultEncoding))
      EncoderFitter.fit(strategy, colValues, labels)
    }
    EncodingPipeline(fitted)
  }

  private def detectEncodingType(values: Seq[Any], defaultEncoding: EncodingType): EncodingType =
    values.find(_ != null) match {
      case Some(_: Number) => EncodingType.PassThrough
      case _               => defaultEncoding
    }
}
