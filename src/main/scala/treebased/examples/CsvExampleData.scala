package treebased.examples

import java.io.{BufferedWriter, FileWriter}
import java.nio.file.{Files, Paths}
import scala.util.Random

/**
 * Generates the example CSV files consumed by [[NumericExample]] and [[CategoricalExample]].
 *
 * Run with:
 *   sbt "runMain treebased.examples.CsvExampleData"
 */
object CsvExampleData {

  def main(args: Array[String]): Unit = {
    writeNumericCsv("data/numeric_example.csv", n = 2000)
    writeCategoricalCsv("data/categorical_example.csv", n = 2000)
    println("Wrote data/numeric_example.csv and data/categorical_example.csv")
  }

  /**
   * Numeric example data. Columns: x1,x2,x3,treatment,weight,label
   *
   * label = x1 + x2*0.5 + x3*(-2)*treatment + noise, so the true HTE is x3 * -2.
   */
  def writeNumericCsv(filePath: String, n: Int = 1000, seed: Int = 42): Unit =
    withCsv(filePath) { bw =>
      bw.write("x1,x2,x3,treatment,weight,label")
      bw.newLine()
      val random = new Random(seed)
      (1 to n).foreach { _ =>
        val x1        = random.nextGaussian() + 0.2
        val x2        = random.nextGaussian() - 0.5
        val x3        = random.nextGaussian() + 0.8
        val treatment = random.nextInt(2)
        val noise     = random.nextGaussian()
        val label     = x1 + x2 * 0.5 + x3 * -2 * treatment + noise
        bw.write(s"$x1,$x2,$x3,$treatment,1.0,$label")
        bw.newLine()
      }
    }

  /**
   * Categorical example data. Columns: color,size,region,x4,treatment,weight,label
   *
   * color, size and region are categorical treatment-effect modifiers; x4 is a continuous
   * covariate. The HTE is colorEffect + regionEffect.
   */
  def writeCategoricalCsv(filePath: String, n: Int = 1000, seed: Int = 42): Unit =
    withCsv(filePath) { bw =>
      bw.write("color,size,region,x4,treatment,weight,label")
      bw.newLine()
      val random  = new Random(seed)
      val colors  = Seq("red", "green", "blue")
      val sizes   = Seq("small", "medium", "large")
      val regions = Seq("north", "south", "east", "west")
      (1 to n).foreach { _ =>
        val color     = colors(random.nextInt(colors.length))
        val size      = sizes(random.nextInt(sizes.length))
        val region    = regions(random.nextInt(regions.length))
        val x4        = random.nextGaussian()
        val treatment = random.nextInt(2)
        val noise     = random.nextGaussian() * 0.5
        val colorEffect = color match {
          case "red" => 2.0; case "green" => -1.0; case _ => 0.5
        }
        val regionEffect = region match {
          case "north" => 1.5; case "south" => -1.5; case "east" => 0.0; case _ => 0.5
        }
        val sizeEffect = size match {
          case "small" => -1.0; case "large" => 1.0; case _ => 0.0
        }
        val label = sizeEffect + x4 * 0.3 + treatment * (colorEffect + regionEffect) + regionEffect * 0.5 + noise
        bw.write(s"$color,$size,$region,$x4,$treatment,1.0,$label")
        bw.newLine()
      }
    }

  private def withCsv(filePath: String)(body: BufferedWriter => Unit): Unit = {
    Files.createDirectories(Paths.get(filePath).getParent)
    val bw = new BufferedWriter(new FileWriter(filePath))
    try body(bw)
    finally bw.close()
  }
}
