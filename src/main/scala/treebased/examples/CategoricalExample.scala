package treebased.examples

import treebased.api.CausalForest
import treebased.config.ExactCausalForestConfig
import treebased.preprocessing.{CsvRowSource, DataLoader, EncodingType, NamedSchema, RowSource}

/**
 * End-to-end example: train a CausalForest on mixed categorical/numeric CSV data.
 *
 * CSV layout (data/categorical_example.csv):
 *   color,size,region,x4,treatment,weight,label
 *
 * The three categorical columns take a different encoder each via the NamedSchema encodings
 * map (color → Label, size → OneHot, region → Mean); the continuous column (x4) is inferred
 * numeric by CsvRowSource and passes through unchanged. The true HTE is colorEffect +
 * regionEffect, which varies per observation.
 *
 * Run with:
 *   sbt "runMain treebased.examples.CategoricalExample"
 */
object CategoricalExample {

  def main(args: Array[String]): Unit = {

    // ── 1. Read CSV ────────────────────────────────────────────────────────────
    val source = CsvRowSource.fromFile("data/categorical_example.csv")
      .fold(err => throw err, identity)

    // ── 2. Define schema with explicit encodings, keyed by column name ─────────
    // color/size/region are categorical strings and need an encoder before the forest can use
    // them. This example deliberately uses a different one per column to show all three; any of
    // them can encode any categorical column. x4 is numeric → PassThrough is detected automatically.
    val schema = NamedSchema(
      featureColumns  = Vector("color", "size", "region", "x4"),
      treatmentColumn = "treatment",
      weightColumn    = Some("weight"),
      labelColumn     = "label",
      encodings = Map(
        "color"  -> EncodingType.Label,     // one integer code per category
        "size"   -> EncodingType.OneHot,    // one indicator column per category
        "region" -> EncodingType.Mean       // category replaced by its (out-of-fold) mean outcome
      )
    )

    // ── 3. Load training data and fit the encoding pipeline ────────────────────
    // `pipeline` is returned so the same category-to-code mapping can be reused consistently
    // when encoding prediction data later.
    val (allData, pipeline) = DataLoader.loadTrainingData(source, schema)
      .fold(err => throw err, identity)

    val splitIdx              = (allData.size * 0.8).toInt
    val (trainData, testData) = allData.splitAt(splitIdx)

    println(s"Training rows : ${trainData.size}")
    println(s"Test rows     : ${testData.size}")
    println(s"Feature count : ${pipeline.outputSize}  (1 label + 3 one-hot + 1 mean + 1 numeric)")

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

    println(f"Estimated ATE : $ate%.4f")
    println(s"First 5 ITEs  : ${predictions.take(5).map(v => f"$v%.4f").mkString(", ")}")

    // ── 6. Predict on new rows using the fitted pipeline ──────────────────────
    // Wrap the new observations in a RowSource with the SAME column names, then pass the same
    // `pipeline` and `schema` — this guarantees the encoding matches what the forest trained on.
    // loadPredictionData only touches the feature columns, so treatment/weight/label can be dummy.
    val newRows = RowSource(
      source.columns,
      Vector(
        IndexedSeq[Any]("red", "large", "north", 0.5, 1, 1.0, 0.0),
        IndexedSeq[Any]("green", "small", "south", -1.2, 0, 1.0, 0.0)
      )
    )
    val newPredData = DataLoader.loadPredictionData(newRows, schema, pipeline)
      .fold(err => throw err, identity)
    val newPreds    = forest.predict(newPredData)

    println("\nNew observation predictions:")
    newPreds.zip(newRows.rows).foreach { case (pred, row) =>
      println(f"  color=${row(0)}, size=${row(1)}, region=${row(2)}, x4=${row(3).asInstanceOf[Double]}%.2f  →  ITE=$pred%.4f")
    }
  }
}
