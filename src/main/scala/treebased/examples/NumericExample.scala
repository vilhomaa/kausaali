package treebased.examples

import treebased.api.CausalForest
import treebased.config.ExactCausalForestConfig
import treebased.preprocessing.{CsvRowSource, DataLoader, NamedSchema}

/**
 * End-to-end example: train a CausalForest on purely numeric CSV data.
 *
 * CSV layout (data/numeric_example.csv):
 *   x1,x2,x3,treatment,weight,label
 *
 * No encoding configuration is needed — all feature columns are numeric and CsvRowSource infers
 * them as such, so they pass through the pipeline unchanged.
 *
 * Run with:
 *   sbt "runMain treebased.examples.NumericExample"
 */
object NumericExample {

  def main(args: Array[String]): Unit = {

    // ── 1. Read CSV ────────────────────────────────────────────────────────────
    // CsvRowSource parses the file and infers per-column types from the header row down.
    val source = CsvRowSource.fromFile("data/numeric_example.csv")
      .fold(err => throw err, identity)

    // ── 2. Define schema by column name ───────────────────────────────────────
    // No `encodings` map needed: all feature columns are numeric.
    val schema = NamedSchema(
      featureColumns  = Vector("x1", "x2", "x3"),
      treatmentColumn = "treatment",
      weightColumn    = Some("weight"),
      labelColumn     = "label"
    )

    // ── 3. Load training data and fit the encoding pipeline ────────────────────
    val (allData, pipeline) = DataLoader.loadTrainingData(source, schema)
      .fold(err => throw err, identity)

    val splitIdx                  = (allData.size * 0.8).toInt
    val (trainData, testData)     = allData.splitAt(splitIdx)

    println(s"Training rows : ${trainData.size}")
    println(s"Test rows     : ${testData.size}")
    println(s"Feature count : ${pipeline.outputSize}")

    // ── 4. Configure and train the forest ─────────────────────────────────────
    val config = ExactCausalForestConfig(
      maxDepth                   = 10,
      nTrees                     = 50,
      maxFeaturesForSplit        = 2,
      minNodeSizePerTreatmentGroup = 10,
      subsampleRatio             = 0.8,
      seed                       = 42L
    )

    val forest = CausalForest.train(trainData, config)

    println(s"Trees built   : ${forest.size}")

    // ── 5. Predict treatment effects on held-out data ─────────────────────────
    val predictions = forest.predict(testData.map(_.features))
    val ate         = predictions.sum / predictions.size

    println(f"Estimated ATE : $ate%.4f  (true ≈ -1.65)")
    println(s"First 5 ITEs  : ${predictions.take(5).map(v => f"$v%.4f").mkString(", ")}")
  }
}
