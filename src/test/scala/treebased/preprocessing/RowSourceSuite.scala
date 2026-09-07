package treebased.preprocessing

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/**
 * [[CsvRowSource]] parsing and per-column type inference. Schema resolution is covered by
 * [[NamedSchemaSuite]] and the end-to-end mapping by [[DataLoaderSuite]].
 */
class RowSourceSuite extends AnyFlatSpec with Matchers {

  private def parse(text: String, header: Boolean = true) =
    CsvRowSource.fromString(text, header = header).getOrElse(fail("fromString should succeed"))

  "CsvRowSource" should "read the header row into column names" in {
    val src = parse("a,b,c\n1,2,3")
    src.columns shouldBe Vector("a", "b", "c")
  }

  it should "infer an all-numeric column as Double and a mixed column as String" in {
    val src = parse("x,label\n1.5,foo\n2,bar")
    src.rows.head shouldBe IndexedSeq(1.5, "foo")
    src.rows(1)   shouldBe IndexedSeq(2.0, "bar")
    src.rows.head.head shouldBe a[java.lang.Double]
    src.rows.head(1)   shouldBe a[String]
  }

  it should "leave a numeric-looking column as String if any cell is non-numeric" in {
    val src = parse("v\n1\n2\nNA")
    src.rows.map(_.head) shouldBe Vector("1", "2", "NA")
  }

  it should "treat an empty field as null without disqualifying a numeric column" in {
    val src = parse("a,b\n1,\n3,4")
    src.rows shouldBe Vector(IndexedSeq(1.0, null), IndexedSeq(3.0, 4.0))
  }

  it should "honour quoted fields containing the separator" in {
    val src = parse("name,note\n\"Doe, Jane\",\"a, b, c\"")
    src.rows.head shouldBe IndexedSeq("Doe, Jane", "a, b, c")
  }

  it should "unescape doubled quotes inside a quoted field" in {
    val src = parse("q\n\"she said \"\"hi\"\"\"")
    src.rows.head.head shouldBe """she said "hi""""
  }

  it should "handle quoted fields that span newlines" in {
    val src = parse("a,b\n\"line1\nline2\",x")
    src.rows.head shouldBe IndexedSeq("line1\nline2", "x")
  }

  it should "skip fully blank lines and tolerate a trailing newline" in {
    val src = parse("a\n1\n\n2\n")
    src.rows.map(_.head) shouldBe Vector(1.0, 2.0)
  }

  it should "accept CRLF line endings" in {
    val src = parse("a,b\r\n1,2\r\n3,4\r\n")
    src.rows shouldBe Vector(IndexedSeq(1.0, 2.0), IndexedSeq(3.0, 4.0))
  }

  it should "trim unquoted fields but keep quoted fields verbatim" in {
    val src = parse("a,b\n  1  ,\"  spaced  \"")
    src.rows.head shouldBe IndexedSeq(1.0, "  spaced  ")
  }

  it should "synthesise c0,c1,... names when header = false" in {
    val src = parse("1,2,3\n4,5,6", header = false)
    src.columns shouldBe Vector("c0", "c1", "c2")
    src.rows shouldBe Vector(IndexedSeq(1.0, 2.0, 3.0), IndexedSeq(4.0, 5.0, 6.0))
  }

  it should "fail on a ragged row" in {
    CsvRowSource.fromString("a,b,c\n1,2") shouldBe a[Left[?, ?]]
  }

  it should "return an empty source for empty input" in {
    val src = parse("")
    src.columns shouldBe empty
    src.rows shouldBe empty
  }
}
