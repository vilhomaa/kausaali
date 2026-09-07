package treebased.testutils

import treebased.api.GeneralizedRandomForest
import treebased.config.RegressionForestConfig
import treebased.core.domain.data.{CausalObservation, Observation}
import treebased.serialization.ForestSerializer
import treebased.testutils.EitherOps.orThrow
import treebased.testutils.SyntheticData.generatePoints

import java.nio.file.{Files, Path, StandardCopyOption}

/**
 * Small forests shared by more than one suite. Deliberately tiny: they exist to exercise
 * serialization and the `Centering.Prefitted` wiring, not to produce accurate nuisance estimates.
 *
 * [[treebased.serialization.ForestSerializationSuite]] trains the two nuisance forests and writes
 * them to the fixture paths; [[treebased.api.GeneralizedRandomForestSuite]] loads them back as
 * caller-supplied nuisance models. Either suite order works — [[loadNuisanceForests]] regenerates
 * the files if they are missing — and the fit is fully deterministic, so a regenerated file is
 * byte-identical to the original.
 */
object ForestFixtures {

  /** Shallow, few trees — keep every forest built from this cheap. */
  val config: RegressionForestConfig = TestConfigs.tinyForest

  /** Stand-in for a nuisance sample fit "elsewhere", independent of any suite's own data. */
  val trainRows: Array[CausalObservation] = generatePoints(1500)

  private val dir                 = "target/test-fixtures"
  val outcomeForestPath: String   = s"$dir/nuisance_outcome_forest.json"
  val treatmentForestPath: String = s"$dir/nuisance_treatment_forest.json"

  private def regressionForest(label: CausalObservation => Double): GeneralizedRandomForest =
    GeneralizedRandomForest.regression(
      trainRows.map(d => Observation(d.features, d.weight, label(d))),
      config
    )

  /** Train both nuisance forests without touching disk. */
  def nuisanceForests(): (GeneralizedRandomForest, GeneralizedRandomForest) =
    (regressionForest(_.y), regressionForest(_.w))

  /** Train both nuisance forests, write them to their fixture paths, and return them. */
  def writeNuisanceForests(): (GeneralizedRandomForest, GeneralizedRandomForest) = {
    Files.createDirectories(Path.of(dir))
    val (outcome, treatment) = nuisanceForests()
    saveAtomic(outcome, outcomeForestPath)
    saveAtomic(treatment, treatmentForestPath)
    (outcome, treatment)
  }

  /** Load both nuisance forests from disk, regenerating the files first if either is absent. */
  def loadNuisanceForests(): (GeneralizedRandomForest, GeneralizedRandomForest) =
    (ForestSerializer.loadGeneralized(outcomeForestPath),
     ForestSerializer.loadGeneralized(treatmentForestPath)) match {
      case (Right(outcome), Right(treatment)) => (outcome, treatment)
      case _                                  => writeNuisanceForests()
    }

  // Write to a sibling temp file then atomically rename, so a suite reading the fixture while
  // another suite regenerates it never sees a half-written file (sbt runs suites in parallel).
  private def saveAtomic(forest: GeneralizedRandomForest, path: String): Unit = {
    val target = Path.of(path)
    val tmp    = Files.createTempFile(target.getParent, ".fixture-", ".tmp")
    orThrow(ForestSerializer.save(forest, tmp.toString))
    Files.move(tmp, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
  }
}
