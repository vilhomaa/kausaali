package treebased.serialization

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import treebased.api.CausalForest
import treebased.core.domain.model.ForestModel
import treebased.testutils.{ForestFixtures, TestConfigs}
import treebased.testutils.EitherOps.orThrow

import java.nio.file.Files

/**
 * Round-trip guarantees for [[ForestSerializer]]: a trained forest saved and reloaded must
 * reproduce its predictions. All forests here are tiny, shallow shapes — see
 * [[treebased.testutils.TestConfigs.tinyExactCausalForest]] / [[ForestFixtures.config]].
 *
 * The GeneralizedRandomForest case also writes the shared nuisance fixtures that
 * [[treebased.api.GeneralizedRandomForestSuite]] loads as caller-supplied nuisance models.
 */
class ForestSerializationSuite extends AnyFlatSpec with Matchers {

  private val trainRows   = ForestFixtures.trainRows.take(1200)
  private val featureRows  = ForestFixtures.trainRows.drop(1200).map(_.features)

  private def predictionsMatch(a: ForestModel, b: ForestModel): Unit =
    a.predict(featureRows).zip(b.predict(featureRows)).foreach { case (x, y) => x shouldBe y +- 1e-9 }

  private def inTempFile[A](suffix: String)(body: String => A): A = {
    val file = Files.createTempFile("forest-", suffix)
    try body(file.toString)
    finally Files.deleteIfExists(file)
  }

  "ForestSerializer" should "round-trip a trained CausalForest through JSON" in {
    val forest = CausalForest.train(trainRows, TestConfigs.tinyExactCausalForest)

    val loaded = inTempFile(".json") { path =>
      orThrow(ForestSerializer.save(forest, path))
      orThrow(ForestSerializer.load(path))
    }

    loaded.forestConfig shouldBe forest.forestConfig
    predictionsMatch(loaded, forest)
  }

  it should "write the shared nuisance forests and reload them with identical predictions" in {
    val (outcome, treatment) = ForestFixtures.writeNuisanceForests()

    val reloadedOutcome   = orThrow(ForestSerializer.loadGeneralized(ForestFixtures.outcomeForestPath))
    val reloadedTreatment = orThrow(ForestSerializer.loadGeneralized(ForestFixtures.treatmentForestPath))

    reloadedOutcome.forestConfig shouldBe outcome.forestConfig
    predictionsMatch(reloadedOutcome, outcome)
    predictionsMatch(reloadedTreatment, treatment)
  }

  it should "round-trip a GeneralizedRandomForest through gzip-compressed JSON" in {
    val (outcome, _) = ForestFixtures.nuisanceForests()

    val loaded = inTempFile(".json.gz") { path =>
      orThrow(ForestSerializer.save(outcome, path))
      orThrow(ForestSerializer.loadGeneralized(path))
    }

    predictionsMatch(loaded, outcome)
  }
}
