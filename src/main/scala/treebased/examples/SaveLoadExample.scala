package treebased.examples

import treebased.api.CausalForest
import treebased.config.ExactCausalForestConfig
import treebased.preprocessing.{CsvRowSource, DataLoader, NamedSchema}
import treebased.serialization.ForestSerializer

/**
 * Demonstrates saving and loading a trained CausalForest to/from JSON.
 *
 * Run with:
 *   sbt "runMain treebased.examples.SaveLoadExample"
 */
object SaveLoadExample:

  def main(args: Array[String]): Unit =

    // ── 1. Load CSV (same layout as NumericExample) ───────────────────────────
    val source = CsvRowSource.fromFile("data/numeric_example.csv")
      .fold(err => throw err, identity)

    val schema = NamedSchema(
      featureColumns  = Vector("x1", "x2", "x3"),
      treatmentColumn = "treatment",
      weightColumn    = Some("weight"),
      labelColumn     = "label"
    )

    val (allData, pipeline)        = DataLoader.loadTrainingData(source, schema)
      .fold(err => throw err, identity)
    val splitIdx                   = (allData.size * 0.8).toInt
    val (trainData, testData)      = allData.splitAt(splitIdx)

    println(s"Training rows : ${trainData.size}")
    println(s"Test rows     : ${testData.size}")

    // ── 2. Train a small forest ────────────────────────────────────────────────
    val config = ExactCausalForestConfig(
      maxDepth                   = 5,
      nTrees                     = 10,
      maxFeaturesForSplit        = 2,
      minNodeSizePerTreatmentGroup = 10,
      seed                       = 42L
    )

    val forest = CausalForest.train(trainData, config)

    println(s"Trees built   : ${forest.size}")

    // ── 3. Predict before saving ──────────────────────────────────────────────
    val predsBefore = forest.predict(testData.map(_.features))
    println(s"\nPredictions before save (first 5): ${predsBefore.take(5).map(v => f"$v%.4f").mkString(", ")}")

    // ── 4. Save to disk ───────────────────────────────────────────────────────
    val savePath = "data/causal_forest.json"
    ForestSerializer.save(forest, savePath)
      .fold(err => throw err, _ => println(s"Forest saved  : $savePath"))

    // ── 5. Load from disk ─────────────────────────────────────────────────────
    val loadedForest = ForestSerializer.load(savePath)
      .fold(err => throw err, identity)

    println(s"Forest loaded : ${loadedForest.size} trees")

    // ── 6. Predict with loaded forest and verify round-trip ───────────────────
    val predsAfter = loadedForest.predict(testData.map(_.features))
    println(s"Predictions after load (first 5): ${predsAfter.take(5).map(v => f"$v%.4f").mkString(", ")}")

    val maxDiff = predsBefore.zip(predsAfter).map((a, b) => math.abs(a - b)).max
    println(f"\nMax prediction difference: $maxDiff%.2e")

    if maxDiff < 1e-9 then
      println("Save/load round-trip successful — predictions match exactly.")
    else
      throw AssertionError(s"Predictions differ after save/load! Max diff: $maxDiff")
