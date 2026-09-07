# Kausaali

## Overview

This Scala 3 library collects machine learning methods for estimating the treatment effect CATE (Conditional Average Treatment Effect).

Included are the models:

- **`GeneralizedRandomForest`** — the Generalized Random Forests family of Athey, Tibshirani & Wager (2019): gradient-based ("pseudo-outcome") splitting, honest leaves, local centering, adaptive-kernel prediction, and pointwise confidence intervals. Entry point: [`treebased.api.GeneralizedRandomForest`](src/main/scala/treebased/api/GeneralizedRandomForest.scala).
- **`CausalForest`** — the earlier exact-splitting honest causal forest of Athey & Wager (2018): every threshold scored directly by the child ATE difference `P_L·P_R·(τ_L − τ_R)²`. Kept for comparison; prefer the GRF causal forest for estimation. Entry point: [`treebased.api.CausalForest`](src/main/scala/treebased/api/CausalForest.scala).

Build: `sbt compile` / `sbt test`. Scala 3.3, JSON serialization via circe.

Use it from another sbt project via `sbt publishLocal`:

```scala
libraryDependencies += "io.github.vilhomaa" %% "kausaali" % "0.1.0-SNAPSHOT"
```

Cross-framework speed/accuracy benchmarks (vs R `grf` and Python EconML) live in the
separate [`kausaali-benchmarks`](https://github.com/vilhomaa/kausaali-benchmarks) repository.

Runnable end-to-end examples live in [`src/main/scala/treebased/examples`](src/main/scala/treebased/examples):

```bash
sbt "runMain treebased.examples.CsvExampleData"       # generate data/*.csv
sbt "runMain treebased.examples.NumericExample"       # numeric CSV -> CausalForest
sbt "runMain treebased.examples.CategoricalExample"   # mixed CSV -> CausalForest
sbt "runMain treebased.examples.SaveLoadExample"      # train, save to JSON, reload
```

---

## 1. Reading data

[`RowSource`](src/main/scala/treebased/preprocessing/RowSource.scala) is a materialised table of untyped cells with named columns — the format-agnostic input. [`CsvRowSource`](src/main/scala/treebased/preprocessing/RowSource.scala) is the built-in reader (RFC-4180 quoting, per-column numeric/String type inference). A [`NamedSchema`](src/main/scala/treebased/preprocessing/DataLoader.scala) says which columns are features, treatment, label, and (optionally) sample weight; [`DataLoader`](src/main/scala/treebased/preprocessing/DataLoader.scala) resolves it against the source's header, maps rows to typed observations, and fits an encoding pipeline for any categorical columns. (A `DataSchema` with integer column indices + a `Seq[IndexedSeq[Any]]` you parsed yourself is still accepted for headerless data or a non-CSV source.)

### All-numeric CSV

`data/numeric_example.csv` — `x1,x2,x3,treatment,weight,label`

```scala
import treebased.preprocessing.{CsvRowSource, DataLoader, NamedSchema}

val source = CsvRowSource.fromFile("data/numeric_example.csv").fold(throw _, identity)

val schema = NamedSchema(
  featureColumns  = Vector("x1", "x2", "x3"),
  treatmentColumn = "treatment",
  weightColumn    = Some("weight"),
  labelColumn     = "label"
)

// Right((data, pipeline)) or Left(throwable)
val (data, pipeline) = DataLoader.loadTrainingData(source, schema).fold(throw _, identity)

val cut = (data.size * 0.8).toInt
val (trainData, testData) = data.splitAt(cut)   // Array[CausalObservation]
```

### Mixed categorical / numeric CSV

`data/categorical_example.csv` — `color,size,region,x4,treatment,weight,label`. Give categorical feature columns an [`EncodingType`](src/main/scala/treebased/preprocessing/Encoder.scala) (`Label`, `OneHot`, `Mean`), keyed by column name; numeric columns are inferred and pass through automatically.

```scala
import treebased.preprocessing.{CsvRowSource, DataLoader, EncodingType, NamedSchema, RowSource}

val source = CsvRowSource.fromFile("data/categorical_example.csv").fold(throw _, identity)

val schema = NamedSchema(
  featureColumns  = Vector("color", "size", "region", "x4"),
  treatmentColumn = "treatment",
  weightColumn    = Some("weight"),
  labelColumn     = "label",
  encodings = Map(
    "color"  -> EncodingType.Label,
    "size"   -> EncodingType.OneHot,
    "region" -> EncodingType.Mean
  )
)

val (data, pipeline) = DataLoader.loadTrainingData(source, schema).fold(throw _, identity)

// Encode new rows for prediction with the SAME fitted pipeline — wrap them in a RowSource
// carrying the same column names (only the feature columns are read):
val newRows = RowSource(source.columns, Vector(
  IndexedSeq[Any]("red", "large", "north", 0.5, 1, 1.0, 0.0)
))
val newX: Array[Array[Double]] =
  DataLoader.loadPredictionData(newRows, schema, pipeline).fold(throw _, identity)
```

`DataLoader` produces `Array[CausalObservation]`. For a regression forest, map to `Observation`:

```scala
import treebased.core.domain.data.Observation
val regressionTrain = trainData.map(d => Observation(d.features, d.weight, d.y))
```

---

## 2. Training the forests

### GRF causal forest — Athey, Tibshirani & Wager (2019)

```scala
import treebased.api.GeneralizedRandomForest
import treebased.config.CausalForestConfig
import treebased.estimand.Centering

val config = CausalForestConfig(
  nTrees               = 2000,
  minNodeSize          = 5,
  subsampleRatio       = 0.5,
  centering            = Centering.CrossFit(),  // .Off / .Prefitted(...) also available
  kernelPrediction     = true,                  // adaptive-kernel moment solve
  seed                 = 42L
)

val forest = GeneralizedRandomForest.causal(trainData, config)

val tauHat: Array[Double] = forest.predict(testData.map(_.features))
val ate = tauHat.sum / tauHat.length
```

Pointwise confidence intervals (bootstrap-of-little-bags) need `ciGroupSize >= 2` and `kernelPrediction = true`:

```scala
val ciConfig = CausalForestConfig(nTrees = 2000, kernelPrediction = true, ciGroupSize = 4)
val ciForest = GeneralizedRandomForest.causal(trainData, ciConfig)

val interval = ciForest.predictInterval(testData.head.features, level = 0.95)
// interval.estimate, interval.lower, interval.upper
```

### GRF regression forest — `E[Y | X]`

```scala
import treebased.api.GeneralizedRandomForest
import treebased.config.RegressionForestConfig

val rf = GeneralizedRandomForest.regression(
  regressionTrain,                          // Array[Observation]
  RegressionForestConfig(nTrees = 2000, seed = 42L)
)
val yHat = rf.predict(testData.map(_.features))
```

### Exact causal forest — Athey & Wager (2018)

```scala
import treebased.api.CausalForest
import treebased.config.ExactCausalForestConfig

val config = ExactCausalForestConfig(
  maxDepth                     = 10,
  nTrees                       = 50,
  maxFeaturesForSplit          = 2,
  minNodeSizePerTreatmentGroup = 10,
  subsampleRatio               = 0.8,
  seed                         = 42L
)

val forest = CausalForest.train(trainData, config)   // Array[CausalObservation]
val tauHat = forest.predict(testData.map(_.features))
```

All configs are fully defaulted — `CausalForestConfig()`, `RegressionForestConfig()`, `ExactCausalForestConfig()` are each a reasonable forest on their own. `nFeatures` is read off the training data at `train` time.

---

## 3. Saving & loading trees

[`ForestSerializer`](src/main/scala/treebased/serialization/ForestSerializer.scala) writes the whole forest (config + every tree) to JSON. A `.gz` path is transparently gzip-compressed (~10× smaller for large forests).

```scala
import treebased.serialization.ForestSerializer

// Exact CausalForest
ForestSerializer.save(forest, "data/causal_forest.json").fold(throw _, identity)
val reloaded = ForestSerializer.load("data/causal_forest.json").fold(throw _, identity)

// GeneralizedRandomForest — note the distinct loader name (can't overload on return type)
ForestSerializer.save(grfForest, "data/grf.json.gz").fold(throw _, identity)
val grfReloaded = ForestSerializer.loadGeneralized("data/grf.json.gz").fold(throw _, identity)

// Round-trips exactly: reloaded.predict(x) == forest.predict(x)
```

Every method returns `Either[Throwable, _]`; `.fold(throw _, identity)` unwraps or rethrows.

---

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
Contributions are welcome; see [`CONTRIBUTING.md`](CONTRIBUTING.md).
