package treebased.serialization

import io.circe.parser.decode
import io.circe.syntax.*
import treebased.api.{CausalForest, GeneralizedRandomForest}
import treebased.serialization.ForestCodecs.given

import java.io.{ByteArrayOutputStream, InputStream, OutputStream}
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files, Path}
import java.util.zip.{GZIPInputStream, GZIPOutputStream}
import scala.util.{Try, Using}

object ForestSerializer:

  def save(forest: CausalForest, path: String): Either[Throwable, Unit] =
    writeJson(path, forest.asJson.spaces2)

  def load(path: String): Either[Throwable, CausalForest] =
    readJson(path).flatMap(decode[CausalForest](_).left.map(err => err: Throwable))

  def save(forest: GeneralizedRandomForest, path: String): Either[Throwable, Unit] =
    writeJson(path, forest.asJson.spaces2)

  // `load` can't overload on return type alone, hence the distinct name.
  def loadGeneralized(path: String): Either[Throwable, GeneralizedRandomForest] =
    readJson(path).flatMap(decode[GeneralizedRandomForest](_).left.map(err => err: Throwable))

  // A `.gz` path is transparently gzip-compressed on write and decompressed on read; the tree
  // JSON is highly repetitive, so this is roughly a 10x saving for large forests.

  private def writeJson(path: String, json: String): Either[Throwable, Unit] =
    Try {
      val p = Path.of(path)
      if isGzip(path) then
        Using.resource(gzipOut(Files.newOutputStream(p)))(_.write(json.getBytes(UTF_8)))
      else Files.writeString(p, json)
    }.toEither.map(_ => ())

  private def readJson(path: String): Either[Throwable, String] =
    Try {
      if isGzip(path) then
        Using.resource(GZIPInputStream(Files.newInputStream(Path.of(path))))(readAll)
      else Files.readString(Path.of(path))
    }.toEither

  private def isGzip(path: String): Boolean = path.endsWith(".gz")

  private def gzipOut(out: OutputStream): GZIPOutputStream = GZIPOutputStream(out)

  private def readAll(in: InputStream): String =
    val buf = ByteArrayOutputStream()
    in.transferTo(buf)
    buf.toString(UTF_8)
