package treebased.preprocessing

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import treebased.preprocessing.*
import treebased.preprocessing.Encoders.*

/**
 * Unit tests for the individual encoders and the [[EncodingPipeline]]. Their behaviour *through*
 * [[DataLoader]] (schema-driven column selection, prediction-time reuse) lives in
 * [[DataLoaderSuite]].
 */
class EncoderSuite extends AnyFlatSpec with Matchers {

  /** Unwraps a successful `encode`, failing the test with a clear message otherwise. */
  private def code[T](enc: Encoder[T], v: T): Array[Double] =
    enc.encode(v).getOrElse(fail(s"encode($v) should succeed"))

  // ── LabelEncoder ───────────────────────────────────────────────────────────

  "LabelEncoder" should "assign distinct integer codes starting at 1" in {
    val enc = LabelEncoder(Seq("cat", "dog", "bird"))
    enc.outputSize shouldBe 1
    val codes = Seq("cat", "dog", "bird").map(v => code(enc, v).head)
    codes.distinct.size shouldBe 3
    codes.forall(_ >= 1.0) shouldBe true
  }

  it should "return the same code for repeated values" in {
    val enc = LabelEncoder(Seq("yes", "no", "yes", "yes"))
    code(enc, "yes") shouldBe code(enc, "yes")
    code(enc, "yes") should not equal code(enc, "no")
  }

  it should "round-trip via decode" in {
    val enc = LabelEncoder(Seq("alpha", "beta", "gamma"))
    Seq("alpha", "beta", "gamma").foreach(v => enc.decode(code(enc, v)) shouldBe v)
  }

  it should "return Left for an unknown value instead of throwing" in {
    val enc = LabelEncoder(Seq("a", "b"))
    enc.encode("z") shouldBe a[Left[?, ?]]
    enc.encode("z").swap.getOrElse(fail("expected Left")) shouldBe a[NoSuchElementException]
  }

  // ── OneHotEncoder ──────────────────────────────────────────────────────────

  "OneHotEncoder" should "produce a vector of length equal to the number of categories" in {
    val enc = OneHotEncoder(Seq("red", "green", "blue"))
    enc.outputSize shouldBe 3
    code(enc, "red").length shouldBe 3
  }

  it should "have exactly one 1.0 per encoded row" in {
    val enc = OneHotEncoder(Seq("red", "green", "blue"))
    Seq("red", "green", "blue").foreach(v => code(enc, v).sum shouldBe 1.0)
  }

  it should "place the 1.0 at different positions for different categories" in {
    val enc = OneHotEncoder(Seq("red", "green", "blue"))
    code(enc, "red").indexOf(1.0) should not equal code(enc, "green").indexOf(1.0)
    code(enc, "green").indexOf(1.0) should not equal code(enc, "blue").indexOf(1.0)
  }

  it should "round-trip via decode" in {
    val enc = OneHotEncoder(Seq("x", "y", "z"))
    Seq("x", "y", "z").foreach(v => enc.decode(code(enc, v)) shouldBe v)
  }

  it should "return Left for an unknown category instead of throwing" in {
    val enc = OneHotEncoder(Seq("a", "b"))
    enc.encode("c") shouldBe a[Left[?, ?]]
    enc.encode("c").swap.getOrElse(fail("expected Left")) shouldBe a[NoSuchElementException]
  }

  // ── MeanEncoder ────────────────────────────────────────────────────────────

  "MeanEncoder" should "encode each category as the mean label for that category" in {
    val enc = MeanEncoder(Seq("a", "a", "b"), Seq(10.0, 20.0, 50.0))
    enc.outputSize shouldBe 1
    code(enc, "a").head shouldBe 15.0 +- 1e-9 // (10+20)/2
    code(enc, "b").head shouldBe 50.0 +- 1e-9
  }

  it should "fall back to global mean for unseen categories" in {
    val enc = MeanEncoder(Seq("a", "b"), Seq(10.0, 30.0))
    code(enc, "unknown").head shouldBe 20.0 +- 1e-9 // (10+30)/2
  }

  it should "decode to all contributing values" in {
    val enc = MeanEncoder(Seq("a", "b"), Seq(5.0, 5.0))
    enc.decode(Array(5.0)) shouldBe "a | b"
  }

  it should "not let a singleton category's own label leak into its out-of-fold training encoding" in {
    // Every category is unique to its row, so the naive full-sample `encode` reproduces each
    // row's own label exactly -- the leak this whole fix exists to close. Labels have no
    // arithmetic symmetry, so no row's out-of-fold fallback mean can coincidentally equal its own
    // excluded label (an arithmetic sequence's middle element equals the mean of the rest).
    val values = Seq("a", "b", "c", "d", "e")
    val labels = Seq(1.0, 2.0, 4.0, 8.0, 16.0)
    val enc = MeanEncoder(values, labels)

    val naive = values.map(v => code(enc, v).head)
    naive shouldBe labels // sanity check: this is the leak, if we asked `encode` for these rows

    val oof = enc.encodeTrainingRows(values).getOrElse(fail("encodeTrainingRows should succeed")).map(_.head)
    oof.zip(labels).foreach { case (encoded, ownLabel) => encoded should not be ownLabel }
  }

  it should "reject encodeTrainingRows when called with a different row count than it was fit on" in {
    val enc = MeanEncoder(Seq("a", "b"), Seq(1.0, 2.0))
    an[IllegalArgumentException] should be thrownBy enc.encodeTrainingRows(Seq("a"))
  }

  // ── PassThroughEncoder ─────────────────────────────────────────────────────

  "PassThroughEncoder" should "pass Double values unchanged" in {
    val enc = PassThroughEncoder()
    code(enc, 3.14).head shouldBe 3.14
    enc.outputSize shouldBe 1
  }

  it should "convert Int, Long, Float, Boolean, and numeric String" in {
    val enc = PassThroughEncoder()
    code(enc, 42).head shouldBe 42.0
    code(enc, 7L).head shouldBe 7.0
    code(enc, 2.5f).head shouldBe 2.5 +- 1e-6
    code(enc, true).head shouldBe 1.0
    code(enc, false).head shouldBe 0.0
    code(enc, "9.9").head shouldBe 9.9 +- 1e-9
  }

  it should "decode back to the double value" in {
    PassThroughEncoder().decode(Array(1.5)) shouldBe 1.5
  }

  it should "return Left for non-numeric strings instead of throwing" in {
    val result = PassThroughEncoder().encode("hello")
    result shouldBe a[Left[?, ?]]
    result.swap.getOrElse(fail("expected Left")) shouldBe a[NumberFormatException]
  }

  // ── EncodingPipeline ───────────────────────────────────────────────────────

  "EncodingPipeline" should "pass through numeric columns and encode categorical ones" in {
    val rows: Seq[IndexedSeq[Any]] = Seq(IndexedSeq("red", 1.0), IndexedSeq("blue", 2.0))
    val pipeline = EncodingPipeline.fit(rows, Seq(10.0, 20.0), Map(0 -> EncodingType.Label))
    val result = pipeline.transform(IndexedSeq[Any]("red", 3.0)).getOrElse(fail("transform should succeed"))
    result.size shouldBe 2
    result(1) shouldBe 3.0
  }

  it should "auto-detect numeric columns as PassThrough and mean-encode categoricals by default" in {
    val rows: Seq[IndexedSeq[Any]] = Seq(IndexedSeq(1.0, "cat"), IndexedSeq(2.0, "dog"))
    val pipeline = EncodingPipeline.fit(rows, Seq(10.0, 20.0))
    val result = pipeline.transform(IndexedSeq[Any](3.0, "cat")).getOrElse(fail("transform should succeed"))
    result(0) shouldBe 3.0
    result.size shouldBe 2
  }

  it should "return Left when a column's encoder can't represent its value" in {
    val rows: Seq[IndexedSeq[Any]] = Seq(IndexedSeq("red", 1.0), IndexedSeq("blue", 2.0))
    val pipeline = EncodingPipeline.fit(rows, Seq(10.0, 20.0), Map(0 -> EncodingType.Label))
    pipeline.transform(IndexedSeq[Any]("unseen", 3.0)) shouldBe a[Left[?, ?]]
  }
}
