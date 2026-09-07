package treebased.testutils

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import treebased.testutils.SyntheticData.generatePoints

/** Guards the shared generator itself: shape and determinism the other suites rely on. */
class SyntheticDataSuite extends AnyFlatSpec with Matchers {

  "generatePoints" should "produce n rows with three features, a 0/1 treatment and unit weight" in {
    val rows = generatePoints(200)
    rows.length shouldBe 200
    rows.foreach { r =>
      r.features.length shouldBe 3
      r.w should (be(0.0) or be(1.0))
      r.weight shouldBe 1.0
    }
  }

  it should "be deterministic for a fixed seed" in {
    val a = generatePoints(50, seed = 7)
    val b = generatePoints(50, seed = 7)
    a.map(_.y).toSeq shouldBe b.map(_.y).toSeq
  }
}
