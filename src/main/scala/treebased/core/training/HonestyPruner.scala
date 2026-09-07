package treebased.core.training

import treebased.core.domain.tree.{LeafNode, Node, TreeNode}

/**
 * Post-hoc repair for honest splitting: [[TreeTrainer]] chooses each split using only
 * `splitData`, so the disjoint `predictionData` can still end up empty overall, or nonempty but
 * empty in one treatment arm of a causal leaf, on one side of a split that was otherwise valid.
 * Either way the leaf carries an undefined (0/0) prediction (see
 * [[treebased.core.training.aggregate.SumCount.mean]] and
 * [[treebased.core.training.aggregate.TreatmentEffectStats.ate]]), surfaced as `NaN`, so this
 * collapses any such leaf back into its honesty-sample sibling. Implemented as a pure bottom-up
 * rewrite of the immutable [[Node]] tree rather than in-place index splicing.
 */
object HonestyPruner {
  def pruneEmptyHonestyLeaves(node: Node): Node = node match {
    case leaf: LeafNode => leaf
    case TreeNode(id, depth, split, left, right) =>
      (pruneEmptyHonestyLeaves(left), pruneEmptyHonestyLeaves(right)) match {
        case (l: LeafNode, r) if l.prediction.isNaN => r
        case (l, r: LeafNode) if r.prediction.isNaN => l
        case (l, r)                                 => TreeNode(id, depth, split, l, r)
      }
  }
}
