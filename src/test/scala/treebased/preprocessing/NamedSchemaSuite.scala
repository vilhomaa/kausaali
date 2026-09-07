package treebased.preprocessing

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** [[NamedSchema.resolve]]: binding a name-addressed schema to a concrete column ordering. */
class NamedSchemaSuite extends AnyFlatSpec with Matchers {

  private val header = Vector("color", "x1", "treatment", "weight", "label")

  private val schema = NamedSchema(
    featureColumns  = Vector("color", "x1"),
    treatmentColumn = "treatment",
    weightColumn    = Some("weight"),
    labelColumn     = "label",
    encodings       = Map("color" -> EncodingType.Label)
  )

  "resolve" should "map names to positions in the given header" in {
    val ds = schema.resolve(header).getOrElse(fail("resolve should succeed"))
    ds.featureColumns  shouldBe Vector(0, 1)
    ds.treatmentColumn shouldBe 2
    ds.weightColumn    shouldBe Some(3)
    ds.labelColumn     shouldBe 4
  }

  it should "key encodings by position within featureColumns, not by raw column index" in {
    val reordered = Vector("label", "weight", "treatment", "x1", "color")  // color is raw index 4
    val ds = schema.resolve(reordered).getOrElse(fail("resolve should succeed"))
    ds.featureColumns shouldBe Vector(4, 3)                        // color, x1
    ds.encodings      shouldBe Map(0 -> EncodingType.Label)        // position 0 of featureColumns
  }

  it should "leave weightColumn as None when not requested" in {
    val noWeight = schema.copy(weightColumn = None)
    noWeight.resolve(header).getOrElse(fail("resolve should succeed")).weightColumn shouldBe None
  }

  it should "fail when a referenced column is missing" in {
    val res = schema.resolve(Vector("color", "x1", "treatment", "label"))  // no "weight"
    res shouldBe a[Left[?, ?]]
    res.left.getOrElse(fail()).getMessage should include("weight")
  }

  it should "fail when a referenced column name is duplicated in the header" in {
    schema.resolve(header :+ "x1") shouldBe a[Left[?, ?]]
  }

  it should "fail when encodings names a column that is not a feature" in {
    val bad = schema.copy(encodings = Map("treatment" -> EncodingType.OneHot))
    val res = bad.resolve(header)
    res shouldBe a[Left[?, ?]]
    res.left.getOrElse(fail()).getMessage should include("treatment")
  }
}
