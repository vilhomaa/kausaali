package treebased.preprocessing

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/**
 * [[DataLoader]] behaviour: mapping raw CSV-style rows to training/prediction rows through a
 * [[DataSchema]], and reusing the fitted pipeline at prediction time. Encoder internals are
 * covered by [[EncoderSuite]].
 */
class DataLoaderSuite extends AnyFlatSpec with Matchers {

  /** Unwraps a successful `loadTrainingData`/`loadPredictionData`, failing the test otherwise. */
  private def load(rows: Seq[IndexedSeq[Any]], schema: DataSchema) =
    DataLoader.loadTrainingData(rows, schema).getOrElse(fail("loadTrainingData should succeed"))

  private def predict(rows: Seq[IndexedSeq[Any]], schema: DataSchema, pipeline: EncodingPipeline) =
    DataLoader.loadPredictionData(rows, schema, pipeline).getOrElse(fail("loadPredictionData should succeed"))

  private val numericSchema = DataSchema(
    featureColumns = Vector(0, 1, 2),
    treatmentColumn = 3,
    labelColumn = 4
  )

  private val numericCsvRows: Seq[IndexedSeq[Any]] = Seq(
    IndexedSeq(1.0, 2.0, 3.0, 1, 10.5),
    IndexedSeq(4.0, 5.0, 6.0, 0, 20.3),
    IndexedSeq(7.0, 8.0, 9.0, 1, 30.1)
  )

  "DataLoader.loadTrainingData" should "map rows to feature vectors" in {
    val (data, _) = load(numericCsvRows, numericSchema)
    data.size shouldBe 3
    data(0).features shouldBe Array(1.0, 2.0, 3.0)
    data(1).features shouldBe Array(4.0, 5.0, 6.0)
    data(2).features shouldBe Array(7.0, 8.0, 9.0)
  }

  it should "map treatment and label correctly" in {
    val (data, _) = load(numericCsvRows, numericSchema)
    data(0).w shouldBe 1.0
    data(0).y shouldBe 10.5
    data(1).w shouldBe 0.0
    data(1).y shouldBe 20.3
  }

  it should "default weight to 1.0 when weightColumn is not specified" in {
    val (data, _) = load(numericCsvRows, numericSchema)
    data.foreach(_.weight shouldBe 1.0)
  }

  it should "use the weight column when specified" in {
    val schemaWithWeight = DataSchema(
      featureColumns = Vector(0, 1),
      treatmentColumn = 2,
      labelColumn = 3,
      weightColumn = Some(4)
    )
    val rows: Seq[IndexedSeq[Any]] = Seq(IndexedSeq(1.0, 2.0, 1, 5.0, 0.8))
    val (data, _) = load(rows, schemaWithWeight)
    data.head.weight shouldBe 0.8
  }

  it should "support PassThrough encoding for numeric-as-Any columns" in {
    val schema = DataSchema(
      featureColumns = Vector(0, 1),
      treatmentColumn = 2,
      labelColumn = 3,
      encodings = Map(0 -> EncodingType.PassThrough)
    )
    val rows: Seq[IndexedSeq[Any]] = Seq(
      IndexedSeq(1.0, 2.0, 1, 10.0),
      IndexedSeq(3.0, 4.0, 0, 20.0)
    )
    val (data, _) = load(rows, schema)
    data(0).features shouldBe Array(1.0, 2.0)
    data(1).features shouldBe Array(3.0, 4.0)
  }

  "DataLoader.loadPredictionData" should "map rows to feature vectors using the fitted pipeline" in {
    val (_, pipeline) = load(numericCsvRows, numericSchema)
    val predRows: Seq[IndexedSeq[Any]] = Seq(IndexedSeq(2.0, 3.0, 4.0, 0, 0.0))
    val data = predict(predRows, numericSchema, pipeline)
    data.size shouldBe 1
    data(0) shouldBe Array(2.0, 3.0, 4.0)
  }

  // ── schema-driven categorical encoding ─────────────────────────────────────

  "DataLoader" should "label-encode categorical features and reuse the mapping at prediction time" in {
    val catSchema = DataSchema(
      featureColumns = Vector(0, 1),
      treatmentColumn = 2,
      labelColumn = 3,
      encodings = Map(0 -> EncodingType.Label)
    )
    val rows: Seq[IndexedSeq[Any]] = Seq(
      IndexedSeq("red",  1.0, 1, 10.0),
      IndexedSeq("blue", 2.0, 0, 20.0),
      IndexedSeq("red",  3.0, 1, 30.0)
    )
    val (data, pipeline) = load(rows, catSchema)
    data(0).features(1) shouldBe 1.0                         // numeric column preserved
    data(1).features(1) shouldBe 2.0
    data(0).features(0) shouldBe data(2).features(0)         // same category -> same code
    data(0).features(0) should not be data(1).features(0)    // different category -> different code

    val predData = predict(
      Seq(IndexedSeq[Any]("red", 5.0, 0, 0.0)), catSchema, pipeline
    )
    predData.head(0) shouldBe data(0).features(0)
  }

  it should "one-hot encode categorical features and expand columns" in {
    val ohSchema = DataSchema(
      featureColumns = Vector(0, 1),
      treatmentColumn = 2,
      labelColumn = 3,
      encodings = Map(0 -> EncodingType.OneHot)
    )
    val rows: Seq[IndexedSeq[Any]] = Seq(
      IndexedSeq("a", 1.0, 1, 10.0),
      IndexedSeq("b", 2.0, 0, 20.0),
      IndexedSeq("c", 3.0, 1, 30.0)
    )
    val (data, _) = load(rows, ohSchema)
    data(0).features.size shouldBe 4          // 3 one-hot cols + 1 numeric
    data(0).features.take(3).sum shouldBe 1.0
    data(0).features(3) shouldBe 1.0
  }

  it should "mean-encode categorical features out-of-fold, not by the full-sample mean" in {
    val meanSchema = DataSchema(
      featureColumns = Vector(0, 1),
      treatmentColumn = 2,
      labelColumn = 3
    )
    val rows: Seq[IndexedSeq[Any]] = Seq(
      IndexedSeq("x", 1.0, 1, 10.0),
      IndexedSeq("x", 2.0, 0, 30.0),
      IndexedSeq("y", 3.0, 1, 50.0)
    )
    val (data, _) = load(rows, meanSchema)
    // With only 3 rows every row gets its own cross-validation fold (leave-one-out), so each
    // row's "x" is encoded from the *other* "x" row only, and "y" -- unique to row 2 -- has no
    // "x"-excluded "y" rows to average, so it falls back to that fold's complement mean.
    // The naive full-sample mean ((10+30)/2 = 20.0 for "x", 50.0 for "y" -- row 2's own label)
    // would leak each row's own label into its own feature; see EncoderSuite for that comparison.
    data(0).features(0) shouldBe 30.0 +- 1e-9   // row 0's "x", excluding itself -> row 1's label
    data(1).features(0) shouldBe 10.0 +- 1e-9   // row 1's "x", excluding itself -> row 0's label
    data(2).features(0) shouldBe 20.0 +- 1e-9   // row 2's "y" unseen in its fold -> fold's mean(10,30)
    data(0).features(1) shouldBe 1.0            // numeric column preserved
  }

  it should "not let a training row's own label leak through mean-encoding" in {
    val schema = DataSchema(
      featureColumns = Vector(0, 1),
      treatmentColumn = 2,
      labelColumn = 3,
      encodings = Map(0 -> EncodingType.Mean)
    )
    // Every category is unique to its own row, so naive full-sample mean-encoding would make each
    // row's encoded feature equal to its own label exactly -- the leak this fix closes. Labels
    // have no arithmetic symmetry, so no row's out-of-fold fallback mean can coincidentally equal
    // its own excluded label (an arithmetic sequence's middle element equals the mean of the rest).
    val labels = Seq(1.0, 2.0, 4.0, 8.0, 16.0)
    val rows: Seq[IndexedSeq[Any]] = labels.zipWithIndex.map { case (y, i) => IndexedSeq[Any](s"cat$i", i.toDouble, i % 2, y) }
    val (data, _) = load(rows, schema)
    data.zip(labels).foreach { case (obs, ownLabel) => obs.features(0) should not be ownLabel }
  }

  it should "handle a schema with multiple categorical columns using different encoders" in {
    val mixedSchema = DataSchema(
      featureColumns = Vector(0, 1, 2),
      treatmentColumn = 3,
      labelColumn = 4,
      encodings = Map(
        0 -> EncodingType.Label,
        1 -> EncodingType.OneHot
      )
    )
    val rows: Seq[IndexedSeq[Any]] = Seq(
      IndexedSeq("red",   "small",  1.0, 1, 10.0),
      IndexedSeq("blue",  "large",  2.0, 0, 20.0),
      IndexedSeq("red",   "medium", 3.0, 1, 30.0)
    )
    val (data, pipeline) = load(rows, mixedSchema)
    data(0).features(0) shouldBe data(2).features(0)          // label-encoded: same color, same code
    data(0).features(0) should not equal data(1).features(0)
    data(0).features.length shouldBe 5                        // 1 label + 3 one-hot + 1 numeric
    data(0).features.slice(1, 4).sum shouldBe 1.0
    data(0).features.last shouldBe 1.0                        // numeric col preserved at the end
    data(1).features.last shouldBe 2.0

    val predRows: Seq[IndexedSeq[Any]] = Seq(IndexedSeq("red", "small", 5.0, 0, 0.0))
    val predData = predict(predRows, mixedSchema, pipeline)
    predData.head(0) shouldBe data(0).features(0)
    predData.head.slice(1, 4) shouldBe data(0).features.slice(1, 4)
  }

  it should "use the global mean for an unseen category when mean-encoding at prediction time" in {
    val schema = DataSchema(
      featureColumns = Vector(0, 1),
      treatmentColumn = 2,
      labelColumn = 3,
      encodings = Map(0 -> EncodingType.Mean)
    )
    val trainRows: Seq[IndexedSeq[Any]] = Seq(
      IndexedSeq("p", 1.0, 1, 10.0),
      IndexedSeq("q", 2.0, 0, 30.0)
    )
    val (_, pipeline) = load(trainRows, schema)
    val predRows: Seq[IndexedSeq[Any]] = Seq(IndexedSeq("unseen", 3.0, 0, 0.0))
    val predData = predict(predRows, schema, pipeline)
    predData.head(0) shouldBe 20.0 +- 1e-9   // global mean = (10+30)/2
  }

  it should "return Left instead of crashing on an unseen category when label-encoding at prediction time" in {
    val schema = DataSchema(
      featureColumns = Vector(0, 1),
      treatmentColumn = 2,
      labelColumn = 3,
      encodings = Map(0 -> EncodingType.Label)
    )
    val trainRows: Seq[IndexedSeq[Any]] = Seq(
      IndexedSeq("p", 1.0, 1, 10.0),
      IndexedSeq("q", 2.0, 0, 30.0)
    )
    val (_, pipeline) = load(trainRows, schema)
    val predRows: Seq[IndexedSeq[Any]] = Seq(IndexedSeq("unseen", 3.0, 0, 0.0))
    DataLoader.loadPredictionData(predRows, schema, pipeline) shouldBe a[Left[?, ?]]
  }

  // ── RowSource + NamedSchema entry points ───────────────────────────────────

  "DataLoader.loadTrainingData(RowSource, NamedSchema)" should "resolve names and load through the pipeline" in {
    val source = CsvRowSource.fromString(
      "color,x1,treatment,weight,label\nred,1.0,1,1.0,10.0\nblue,2.0,0,1.0,20.0\nred,3.0,1,1.0,30.0"
    ).getOrElse(fail("fromString should succeed"))
    val named = NamedSchema(
      featureColumns  = Vector("color", "x1"),
      treatmentColumn = "treatment",
      weightColumn    = Some("weight"),
      labelColumn     = "label",
      encodings       = Map("color" -> EncodingType.Label)
    )
    val (data, pipeline) = DataLoader.loadTrainingData(source, named).getOrElse(fail("should succeed"))
    data.size shouldBe 3
    data(0).features(1) shouldBe 1.0
    data(0).features(0) shouldBe data(2).features(0)          // same colour, same code
    data(0).w shouldBe 1.0
    data(1).y shouldBe 20.0

    val predSrc = RowSource(source.columns, Vector(IndexedSeq[Any]("blue", 9.0, 0, 1.0, 0.0)))
    val pred = DataLoader.loadPredictionData(predSrc, named, pipeline).getOrElse(fail("should succeed"))
    pred.head(0) shouldBe data(1).features(0)                 // blue encoded consistently
  }

  it should "propagate the resolution error when a named column is missing" in {
    val source = CsvRowSource.fromString("x1,treatment,label\n1.0,1,10.0")
      .getOrElse(fail("fromString should succeed"))
    val named = NamedSchema(Vector("x1"), treatmentColumn = "treatment", labelColumn = "outcome")
    DataLoader.loadTrainingData(source, named) shouldBe a[Left[?, ?]]
  }
}
