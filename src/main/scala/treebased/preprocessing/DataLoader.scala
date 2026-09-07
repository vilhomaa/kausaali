package treebased.preprocessing

import treebased.core.domain.data.CausalObservation

/**
 * Schema describing which columns in a tabular dataset map to features, treatment, label, and weight.
 *
 * @param featureColumns ordered indices of columns to use as features
 * @param treatmentColumn index of the treatment assignment column
 * @param labelColumn index of the outcome/label column
 * @param weightColumn optional index of the sample weight column (defaults to weight = 1.0)
 * @param encodings per-feature-column encoding overrides (keyed by position in featureColumns, not raw column index).
 *                  Columns not in this map are auto-detected: numeric pass through, non-numeric use defaultEncoding.
 * @param defaultEncoding encoding applied to non-numeric columns that have no explicit encoding specified
 */
case class DataSchema(
  featureColumns: Vector[Int],
  treatmentColumn: Int,
  labelColumn: Int,
  weightColumn: Option[Int] = None,
  encodings: Map[Int, EncodingType] = Map.empty,
  defaultEncoding: EncodingType = EncodingType.Mean
)

/**
 * The same mapping as [[DataSchema]] but addressing columns by *name* rather than by position.
 * Resolve it against a data source's header ([[RowSource.columns]]) to get a positional
 * [[DataSchema]]. Names are safer than indices for real files — reordering columns, or reading a
 * projection, no longer silently shifts the mapping — and `encodings` keyed by feature name
 * removes the "is this a raw index or a position within featureColumns?" ambiguity that
 * [[DataSchema.encodings]] carries.
 *
 * @param featureColumns ordered names of the columns to use as features
 * @param treatmentColumn name of the treatment assignment column
 * @param labelColumn name of the outcome/label column
 * @param weightColumn optional name of the sample-weight column (defaults to weight = 1.0)
 * @param encodings encoding override per feature column, keyed by that column's name. Names not
 *                  in `featureColumns` are rejected by [[resolve]]. Feature columns absent from
 *                  this map are auto-detected exactly as in [[DataSchema]].
 * @param defaultEncoding encoding applied to auto-detected non-numeric feature columns
 */
case class NamedSchema(
  featureColumns: Vector[String],
  treatmentColumn: String,
  labelColumn: String,
  weightColumn: Option[String] = None,
  encodings: Map[String, EncodingType] = Map.empty,
  defaultEncoding: EncodingType = EncodingType.Mean
) {

  /**
   * Bind this schema to a concrete column ordering. `Left` (never a throw) when a referenced
   * column is missing from `header`, when a referenced name is duplicated in `header` (so the
   * mapping would be ambiguous), or when `encodings` names a column that isn't a feature.
   */
  def resolve(header: Seq[String]): Either[Throwable, DataSchema] = {
    val position: Map[String, Int] = header.zipWithIndex.toMap
    val referenced = featureColumns ++ Vector(treatmentColumn, labelColumn) ++ weightColumn.toVector

    val missing    = referenced.filterNot(position.contains).distinct
    val ambiguous  = referenced.filter(name => header.count(_ == name) > 1).distinct
    val notFeature = encodings.keys.filterNot(featureColumns.contains).toVector.sorted

    if (missing.nonEmpty)
      Left(new IllegalArgumentException(
        s"Column(s) not found in data: ${missing.mkString(", ")}. Available: ${header.mkString(", ")}"))
    else if (ambiguous.nonEmpty)
      Left(new IllegalArgumentException(
        s"Column name(s) appear more than once in the header, mapping is ambiguous: ${ambiguous.mkString(", ")}"))
    else if (notFeature.nonEmpty)
      Left(new IllegalArgumentException(
        s"encodings names column(s) that are not in featureColumns: ${notFeature.mkString(", ")}"))
    else
      Right(DataSchema(
        featureColumns  = featureColumns.map(position),
        treatmentColumn = position(treatmentColumn),
        labelColumn     = position(labelColumn),
        weightColumn    = weightColumn.map(position),
        // DataSchema.encodings is keyed by position within featureColumns, not raw column index.
        encodings       = encodings.map { case (name, enc) => featureColumns.indexOf(name) -> enc },
        defaultEncoding = defaultEncoding
      ))
  }
}

/**
 * Utilities for converting raw tabular data (e.g. parsed CSV rows) into typed data points.
 * The user is responsible for parsing the file; this handles the mapping to domain types.
 */
object DataLoader {

  /**
   * Load training data: fits an encoding pipeline on the feature columns and transforms them.
   * Returns both the data and the fitted pipeline (reuse the pipeline for prediction data).
   *
   * @param rows each row is an IndexedSeq of Any values (numeric or categorical)
   * @param schema describes column mapping and encoding configuration
   * @return `Left` on the first row/column an encoder cannot represent (see [[Encoder.encode]]),
   *         otherwise `Right` of (training data as CausalObservations with all-Double features,
   *         fitted encoding pipeline)
   */
  def loadTrainingData(rows: Seq[IndexedSeq[Any]], schema: DataSchema): Either[Throwable, (Array[CausalObservation], EncodingPipeline)] = {
    val featureRows = rows.map(row => schema.featureColumns.map(i => row(i)))
    for {
      labels <- EncodingPipeline.traverse(rows)(row => toDouble(row(schema.labelColumn)))
      pipeline = EncodingPipeline.fit(featureRows, labels.toIndexedSeq, schema.encodings, schema.defaultEncoding)
      // `transformTrainingRows`, not `transform`: these are the very rows the pipeline was fit on,
      // so a label-derived column (mean encoding) must encode each one out-of-fold — see
      // `Encoder.encodeTrainingRows`. `loadPredictionData` below uses `transform` because its rows
      // never contributed to the fit.
      encodedRows <- pipeline.transformTrainingRows(featureRows)
      data <- EncodingPipeline.traverse(rows.zip(encodedRows.toIndexedSeq)) { case (row, features) =>
        for {
          y <- toDouble(row(schema.labelColumn))
          w <- toDouble(row(schema.treatmentColumn))
          weight <- schema.weightColumn.fold[Either[Throwable, Double]](Right(1.0))(i => toDouble(row(i)))
        } yield CausalObservation(features, weight, y, w)
      }
    } yield (data, pipeline)
  }

  /**
   * Load prediction data using an already-fitted encoding pipeline.
   *
   * @param rows each row is an IndexedSeq of Any values
   * @param schema describes column mapping
   * @param pipeline a previously fitted EncodingPipeline (from loadTrainingData)
   * @return `Left` on the first row an encoder cannot represent (e.g. a category unseen at fit
   *         time), otherwise `Right` of one all-Double feature vector per row
   */
  def loadPredictionData(rows: Seq[IndexedSeq[Any]], schema: DataSchema, pipeline: EncodingPipeline): Either[Throwable, Array[Array[Double]]] =
    EncodingPipeline.traverse(rows)(row => pipeline.transform(schema.featureColumns.map(i => row(i))))

  /**
   * [[loadTrainingData]] driven by a named schema and a [[RowSource]]: resolves the schema against
   * `source.columns`, then loads `source.rows`. Short-circuits with the resolution error if a
   * column is missing/ambiguous.
   */
  def loadTrainingData(source: RowSource, schema: NamedSchema): Either[Throwable, (Array[CausalObservation], EncodingPipeline)] =
    schema.resolve(source.columns).flatMap(resolved => loadTrainingData(source.rows, resolved))

  /** [[loadPredictionData]] driven by a named schema and a [[RowSource]]. The `pipeline` and the
   *  `schema` must be the ones used at training time; `source` only needs the feature columns. */
  def loadPredictionData(source: RowSource, schema: NamedSchema, pipeline: EncodingPipeline): Either[Throwable, Array[Array[Double]]] =
    schema.resolve(source.columns).flatMap(resolved => loadPredictionData(source.rows, resolved, pipeline))

  private def toDouble(v: Any): Either[Throwable, Double] = EncodingPipeline.anyToDouble(v)
}
