package treebased.preprocessing

import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scala.util.{Try, Using}

/**
 * A materialised, in-memory table of untyped cells with named columns — the format-agnostic
 * input to [[DataLoader.loadTrainingData]]/[[DataLoader.loadPredictionData]] when paired with a
 * [[NamedSchema]].
 *
 * Row-oriented on purpose: every cell is `Any` (a `Double`, a `String`, a `Boolean`, or `null`
 * for a missing value), each row positionally aligned to [[columns]]. The tree machinery needs
 * the whole dataset in memory anyway, so there's no streaming contract to honour here — an
 * implementation reads its source fully up front.
 *
 * [[CsvRowSource]] is the built-in implementation. A Parquet/Arrow reader would be another one,
 * ideally in a separate module so the core carries no such dependency; it only has to produce
 * `columns` and `rows` in this shape.
 */
trait RowSource {
  /** Column names, positionally aligned to every row's cells. */
  def columns: Vector[String]

  /** Every data row; cell `j` of each row belongs to `columns(j)`. */
  def rows: Vector[IndexedSeq[Any]]
}

object RowSource {

  /** Wrap an already-built grid (useful for tests and for hand-constructed prediction rows). */
  def apply(cols: Vector[String], data: Vector[IndexedSeq[Any]]): RowSource =
    new RowSource {
      val columns: Vector[String]          = cols
      val rows: Vector[IndexedSeq[Any]]    = data
    }
}

/**
 * Reads delimited text (CSV/TSV) into a [[RowSource]].
 *
 * Parsing follows RFC 4180: fields may be double-quoted, a quote inside a quoted field is
 * written `""`, and quoted fields may contain the separator, newlines, and quotes. Unquoted
 * fields are trimmed; quoted fields are taken verbatim. Fully blank lines are skipped.
 *
 * Cell types are inferred per column: a column whose every non-blank cell parses as a `Double`
 * becomes a `Double` column (blanks → `null`); otherwise every cell is the raw `String` (blanks
 * → `null`). This matters because [[EncodingPipeline]] auto-detects numeric vs. categorical
 * columns by runtime type — a numeric column left as strings would be treated as categorical.
 */
object CsvRowSource {

  def fromString(text: String, separator: Char = ',', header: Boolean = true): Either[Throwable, RowSource] =
    Try(build(parseGrid(text, separator), header)).toEither

  def fromFile(path: String, separator: Char = ',', header: Boolean = true): Either[Throwable, RowSource] =
    Using(Source.fromFile(path))(_.mkString).toEither.flatMap(fromString(_, separator, header))

  private def build(grid: Vector[Vector[String]], header: Boolean): RowSource = {
    val nonBlank = grid.filterNot(_.forall(_.isEmpty))
    if (nonBlank.isEmpty) return RowSource(Vector.empty, Vector.empty)

    val (colNames, body) =
      if (header) (nonBlank.head, nonBlank.tail)
      else        (nonBlank.head.indices.map(i => s"c$i").toVector, nonBlank)

    val width = colNames.length
    body.zipWithIndex.foreach { case (row, i) =>
      if (row.length != width)
        throw new IllegalArgumentException(
          s"Row ${i + (if (header) 2 else 1)} has ${row.length} fields, expected $width")
    }

    val columnsTyped: Vector[Array[Any]] =
      (0 until width).toVector.map(c => inferColumn(body.map(_(c))))
    val typedRows: Vector[IndexedSeq[Any]] =
      body.indices.map(r => columnsTyped.map(_(r)): IndexedSeq[Any]).toVector

    RowSource(colNames, typedRows)
  }

  /** All non-blank cells parse as Double → a Double column; otherwise a String column. */
  private def inferColumn(raw: Seq[String]): Array[Any] = {
    def cell(s: String): Option[String] = if (s == null || s.isEmpty) None else Some(s)
    val numeric = raw.forall(s => cell(s).forall(_.toDoubleOption.isDefined))
    raw.iterator.map { s =>
      cell(s) match {
        case None                 => null
        case Some(v) if numeric   => v.toDouble: Any
        case Some(v)              => v: Any
      }
    }.toArray
  }

  /** RFC-4180 field scanner. Returns the raw string grid; blank-line filtering and typing happen
   *  in [[build]]. */
  private def parseGrid(text: String, sep: Char): Vector[Vector[String]] = {
    val records = ArrayBuffer.empty[Vector[String]]
    val record  = ArrayBuffer.empty[String]
    val field   = new StringBuilder
    var inQuotes    = false
    var fieldQuoted = false
    var dirty       = false // any char (incl. a separator) seen since the last record boundary
    var i = 0
    val n = text.length

    def endField(): Unit = {
      record += (if (fieldQuoted) field.toString else field.toString.trim)
      field.setLength(0)
      fieldQuoted = false
    }
    def endRecord(): Unit = {
      endField()
      records += record.toVector
      record.clear()
      dirty = false
    }

    while (i < n) {
      val c = text.charAt(i)
      if (inQuotes) {
        if (c == '"') {
          if (i + 1 < n && text.charAt(i + 1) == '"') { field.append('"'); i += 1 }
          else inQuotes = false
        } else field.append(c)
      } else c match {
        case '"'  => inQuotes = true; fieldQuoted = true; dirty = true
        case `sep` => dirty = true; endField()
        case '\n' => endRecord()
        case '\r' =>
          endRecord()
          if (i + 1 < n && text.charAt(i + 1) == '\n') i += 1
        case _ => dirty = true; field.append(c)
      }
      i += 1
    }
    if (dirty || field.nonEmpty || record.nonEmpty) endRecord()
    records.toVector
  }
}
