package treebased.serialization

import io.circe.*
import io.circe.syntax.*
import io.circe.generic.semiauto.{deriveDecoder, deriveEncoder}
import scala.util.Try
import treebased.api.{CausalForest, GeneralizedRandomForest}
import treebased.config.ForestConfig
import treebased.core.domain.tree.{LeafNode, Node, Split, TreeNode, TreeSplit}
import treebased.core.prediction.{ForestAggregation, KernelRow, WeightedEstimator}

object ForestCodecs:

  // ── Split ──────────────────────────────────────────────────────────────────

  given Encoder[Split] = Encoder.forProduct4("type", "featureIndex", "threshold", "infoGain")(
    s => ("numeric", s.featureIndex, s.threshold, s.infoGain)
  )

  given Decoder[Split] = Decoder.instance { c =>
    for
      fi <- c.downField("featureIndex").as[Int]
      th <- c.downField("threshold").as[Double]
      ig <- c.downField("infoGain").as[Double]
    yield Split(fi, th, ig)
  }

  // ── TreeSplit (dispatch on "type" for extensibility) ───────────────────────

  given Encoder[TreeSplit] = Encoder.instance {
    case s: Split => s.asJson
  }

  given Decoder[TreeSplit] = Decoder.instance { c =>
    c.downField("type").as[String].flatMap {
      case "numeric" => c.as[Split]
      case other     => Left(DecodingFailure(s"Unknown split type: $other", c.history))
    }
  }

  // ── LeafNode ───────────────────────────────────────────────────────────────

  given Encoder[LeafNode] = Encoder.forProduct6("type", "id", "depth", "prediction", "nDataPoints", "honestRowIds")(
    n => ("leaf", n.id, n.depth, n.prediction, n.nDataPoints, n.honestRowIds)
  )

  given Decoder[LeafNode] = Decoder.instance { c =>
    for
      id    <- c.downField("id").as[String]
      depth <- c.downField("depth").as[Int]
      pred  <- c.downField("prediction").as[Double]
      n     <- c.downField("nDataPoints").as[Int]
      ids   <- c.downField("honestRowIds").as[Option[Array[Int]]]
    yield LeafNode(id, depth, pred, n, ids.getOrElse(Array.emptyIntArray))
  }

  // ── Node (recursive — references itself for left/right children) ───────────

  given nodeEncoder: Encoder[Node] = Encoder.instance {
    case n: LeafNode => n.asJson
    case n: TreeNode =>
      Json.obj(
        "type"      -> "internal".asJson,
        "id"        -> n.id.asJson,
        "depth"     -> n.depth.asJson,
        "split"     -> n.split.asJson,
        "leftNode"  -> nodeEncoder(n.leftNode),
        "rightNode" -> nodeEncoder(n.rightNode)
      )
  }

  given nodeDecoder: Decoder[Node] = Decoder.instance { c =>
    c.downField("type").as[String].flatMap {
      case "leaf" => c.as[LeafNode]
      case "internal" =>
        for
          id        <- c.downField("id").as[String]
          depth     <- c.downField("depth").as[Int]
          split     <- c.downField("split").as[TreeSplit]
          leftNode  <- c.downField("leftNode").as[Node](using nodeDecoder)
          rightNode <- c.downField("rightNode").as[Node](using nodeDecoder)
        yield TreeNode(id, depth, split, leftNode, rightNode)
      case other => Left(DecodingFailure(s"Unknown node type: $other", c.history))
    }
  }

  // ── ForestConfig ──────────────────────────────────────────────────────────

  given Encoder[ForestConfig] = deriveEncoder[ForestConfig]
  given Decoder[ForestConfig] = deriveDecoder[ForestConfig]

  // ── Kernel prediction (GeneralizedRandomForest only) ──────────────────────

  given Encoder[KernelRow] = Encoder.forProduct3("weight", "y", "w")(r => (r.weight, r.y, r.w))
  given Decoder[KernelRow] = Decoder.forProduct3("weight", "y", "w")(KernelRow.apply)

  given Encoder[WeightedEstimator] = Encoder.instance {
    case WeightedEstimator.Mean     => Json.obj("type" -> "mean".asJson)
    case WeightedEstimator.Slope(f) => Json.obj("type" -> "slope".asJson, "treatmentVarianceFloor" -> f.asJson)
  }

  given Decoder[WeightedEstimator] = Decoder.instance { c =>
    c.downField("type").as[String].flatMap {
      case "mean"  => Right(WeightedEstimator.Mean)
      case "slope" => c.downField("treatmentVarianceFloor").as[Option[Double]].map(f => WeightedEstimator.Slope(f.getOrElse(0.0)))
      case other   => Left(DecodingFailure(s"Unknown estimator type: $other", c.history))
    }
  }

  // ── CausalForest (predOps is not serialized; always restored as mean) ──────

  given Encoder[CausalForest] = Encoder.instance { f =>
    Json.obj(
      "forestConfig" -> f.forestConfig.asJson,
      "trees"        -> f.trees.asJson
    )
  }

  given Decoder[CausalForest] = Decoder.instance { c =>
    for
      config <- c.downField("forestConfig").as[ForestConfig]
      trees  <- c.downField("trees").as[Vector[Node]]
      forest <- Try(CausalForest(config, trees, ForestAggregation.mean)).toEither
                  .left.map(err => DecodingFailure(err.getMessage, c.history))
    yield forest
  }

  // ── GeneralizedRandomForest (predOps is not serialized; restored as mean) ──
  // `estimator` and `kernelRows` are only written (and only needed) when the forest was trained
  // with `kernelPrediction`; both fields decode tolerantly for older / averaging-only forests.

  given Encoder[GeneralizedRandomForest] = Encoder.instance { f =>
    Json.obj(
      "forestConfig" -> f.forestConfig.asJson,
      "trees"        -> f.trees.asJson,
      "estimator"    -> f.estimator.asJson,
      "kernelRows"   -> f.kernelRows.asJson,
      "treeGroups"   -> f.treeGroups.asJson
    )
  }

  given Decoder[GeneralizedRandomForest] = Decoder.instance { c =>
    for
      config     <- c.downField("forestConfig").as[ForestConfig]
      trees      <- c.downField("trees").as[Vector[Node]]
      estimator  <- c.downField("estimator").as[Option[WeightedEstimator]]
      kernelRows <- c.downField("kernelRows").as[Option[Array[KernelRow]]]
      treeGroups <- c.downField("treeGroups").as[Option[Array[Int]]]
      forest <- Try(GeneralizedRandomForest(config, trees, ForestAggregation.mean,
                  estimator.getOrElse(WeightedEstimator.Mean), kernelRows.getOrElse(Array.empty),
                  treeGroups.getOrElse(Array.empty))).toEither
                  .left.map(err => DecodingFailure(err.getMessage, c.history))
    yield forest
  }
