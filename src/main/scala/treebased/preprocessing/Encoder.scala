package treebased.preprocessing

import treebased.preprocessing.Encoders.{LabelEncoder, MeanEncoder, OneHotEncoder, PassThroughEncoder}

enum EncodingType:
  case Label
  case OneHot
  case Mean
  case PassThrough


trait Encoder[T] {
  /** `Left` when `value` cannot be represented (e.g. a category unseen at fit time) rather than
   *  throwing, so a held-out row with novel data at prediction time is a typed failure the caller
   *  can handle instead of a crash. */
  def encode(value: T): Either[Throwable, Array[Double]]
  def decode(encodedValue: Array[Double]): T
  def outputSize: Int

  /**
   * Encodes the exact rows this encoder was fit on, aligned by index to those rows. Encoders
   * whose mapping doesn't depend on the label (`Label`/`OneHot`/`PassThrough`) have nothing to
   * leak, so the default just calls [[encode]] row by row. [[MeanEncoder]] — whose mapping *is* a
   * function of the label — overrides this to use only the row's held-out cross-validation fold,
   * so a row's own label never leaks back into its own training-time feature value.
   */
  def encodeTrainingRows(values: Seq[T]): Either[Throwable, Array[Array[Double]]] =
    EncodingPipeline.traverse(values)(encode)
}

object EncoderFitter:
  def fit(
           strategy: EncodingType,
           values: Seq[Any],
           labels: Seq[Double]
         ): Encoder[Any] = strategy match {
    case EncodingType.Label => LabelEncoder(values)
    case EncodingType.OneHot => OneHotEncoder(values)
    case EncodingType.Mean => MeanEncoder(values, labels)
    case EncodingType.PassThrough => PassThroughEncoder()
  }
