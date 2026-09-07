package treebased.preprocessing.Encoders

import treebased.preprocessing.Encoder

import scala.util.Random

/**
 * Target encoder: maps each category to the mean label of observations with that category.
 *
 * [[mapping]]/[[globalMean]] are fit on the *entire* training set and are what [[encode]] uses —
 * safe for genuinely new rows at prediction time, since those rows never contributed to the
 * mapping. Applying that same full-sample mapping back to the rows it was fit *on* would leak
 * each row's own label into its own feature value — with a single row in a category the
 * "feature" is essentially that row's own `Y` — and the GRF split search would then partly be
 * "predicting" `Y` from a covariate that already contains it (target leakage), contaminating
 * every downstream estimate this codebase otherwise goes to great lengths to keep unbiased.
 * [[encodeTrainingRows]] exists to avoid exactly that: each fit-time row is encoded from a
 * category mean computed with that row's own cross-validation fold held out, mirroring the
 * out-of-bag cross-fitting [[treebased.core.training.centering.OobNuisanceForest]] already uses
 * for local centering.
 */
case class MeanEncoder[T](
  mapping: Map[String, Double],
  globalMean: Double,
  private val foldOf: Array[Int],
  private val foldMapping: Array[Map[String, Double]],
  private val foldGlobalMean: Array[Double]
) extends Encoder[T] {
  lazy val decoding: Map[Double, String] =
    mapping
      .toArray
      .map((k, v) => (v, k))
      .groupBy(_._1)
      .view
      .mapValues(_.map(_._2).mkString(" | "))
      .toMap

  def encode(value: T): Either[Throwable, Array[Double]] =
    Right(Array(mapping.getOrElse(value.toString, globalMean)))

  def decode(encodedValue: Array[Double]): T =
    // idempotency is not guaranteed because multiple categories can have the same mean
    decoding(encodedValue.head).asInstanceOf[T]

  def outputSize: Int = 1

  /** Out-of-fold target encoding for the exact rows this encoder was fit on — see the class doc.
   *  `values` must be those same fit-time rows, in the same order (as `DataLoader` guarantees by
   *  construction: it fits and then out-of-fold-transforms from the same `featureRows`). */
  override def encodeTrainingRows(values: Seq[T]): Either[Throwable, Array[Array[Double]]] = {
    require(
      values.length == foldOf.length,
      s"MeanEncoder.encodeTrainingRows: got ${values.length} rows but this encoder was fit on ${foldOf.length}"
    )
    Right(Array.tabulate(values.length) { i =>
      val fold = foldOf(i)
      Array(foldMapping(fold).getOrElse(values(i).toString, foldGlobalMean(fold)))
    })
  }
}

object MeanEncoder {

  /** Cross-validation folds behind [[MeanEncoder.encodeTrainingRows]]'s out-of-fold means: each
   *  fit-time row is encoded using only the other `Folds - 1` folds, so its own label never
   *  contributes to its own feature value. Clamped down to the sample size for small datasets. */
  val Folds: Int = 5

  def apply[T](values: Seq[T], labels: Seq[Double], seed: Long = 42L): MeanEncoder[T] = {
    val n = labels.size
    val globalMean = if (n == 0) 0.0 else labels.sum / n
    val categories = values.map(_.toString)
    val mapping = meanByCategory(categories, labels)

    val folds = math.max(1, math.min(Folds, n))
    val foldOf = new Array[Int](n)
    new Random(seed).shuffle((0 until n).toVector).zipWithIndex.foreach {
      case (rowIdx, pos) => foldOf(rowIdx) = pos % folds
    }

    val foldMapping = Array.tabulate(folds) { f =>
      val complement = (0 until n).filter(foldOf(_) != f)
      meanByCategory(complement.map(categories), complement.map(labels))
    }
    val foldGlobalMean = Array.tabulate(folds) { f =>
      val complementLabels = (0 until n).filter(foldOf(_) != f).map(labels)
      if (complementLabels.isEmpty) globalMean else complementLabels.sum / complementLabels.size
    }

    MeanEncoder(mapping, globalMean, foldOf, foldMapping, foldGlobalMean)
  }

  private def meanByCategory(categories: Seq[String], labels: Seq[Double]): Map[String, Double] =
    categories.zip(labels).groupMap(_._1)(_._2).map((k, vs) => k -> vs.sum / vs.size)
}
