package treebased.core.training

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import treebased.core.domain.tree.{LeafNode, Split, TreeNode}

class HonestyPrunerSuite extends AnyFlatSpec with Matchers {

  val split: Split = Split(featureIndex = 0, threshold = 0.5, infoGain = 1.0)

  def leaf(id: String, prediction: Double, nDataPoints: Int): LeafNode =
    LeafNode(id = id, depth = 1, prediction = prediction, nDataPoints = nDataPoints, honestRowIds = Array.emptyIntArray)

  "HonestyPruner.pruneEmptyHonestyLeaves" should "leave a bare leaf unchanged" in {
    val singleLeaf = leaf("0", prediction = 3.0, nDataPoints = 10)
    HonestyPruner.pruneEmptyHonestyLeaves(singleLeaf) shouldBe singleLeaf
  }

  it should "leave a tree with two non-empty leaves unchanged" in {
    val tree = TreeNode("0", depth = 0, split, leaf("0L", 1.0, 5), leaf("0R", 2.0, 5))
    HonestyPruner.pruneEmptyHonestyLeaves(tree) shouldBe tree
  }

  it should "collapse a split whose left leaf has zero honesty points into the right leaf" in {
    val right = leaf("0R", 2.0, 5)
    val tree = TreeNode("0", depth = 0, split, leaf("0L", prediction = Double.NaN, nDataPoints = 0), right)
    HonestyPruner.pruneEmptyHonestyLeaves(tree) shouldBe right
  }

  it should "collapse a split whose right leaf has zero honesty points into the left leaf" in {
    val left = leaf("0L", 1.0, 5)
    val tree = TreeNode("0", depth = 0, split, left, leaf("0R", prediction = Double.NaN, nDataPoints = 0))
    HonestyPruner.pruneEmptyHonestyLeaves(tree) shouldBe left
  }

  it should "cascade the collapse up through an ancestor whose subtree is entirely empty" in {
    // root
    //  ├─ left:  TreeNode(0L)          <- both of its own children are empty leaves
    //  │    ├─ 0LL: empty leaf (nDataPoints = 0)
    //  │    └─ 0LR: empty leaf (nDataPoints = 0)
    //  └─ right: 0R, non-empty leaf
    val emptySubtree = TreeNode(
      "0L", depth = 1, split,
      leaf("0LL", prediction = Double.NaN, nDataPoints = 0),
      leaf("0LR", prediction = Double.NaN, nDataPoints = 0)
    )
    val nonEmptyRight = leaf("0R", prediction = 4.0, nDataPoints = 8)
    val tree = TreeNode("0", depth = 0, split, emptySubtree, nonEmptyRight)

    HonestyPruner.pruneEmptyHonestyLeaves(tree) shouldBe nonEmptyRight
  }

  it should "prune nested empty leaves several levels below an otherwise healthy split" in {
    // root
    //  ├─ left: 0L, non-empty leaf
    //  └─ right: TreeNode(0R)
    //       ├─ 0RL: TreeNode whose two children are both empty leaves
    //       └─ 0RR: non-empty leaf
    val deeplyEmptySubtree = TreeNode(
      "0RL", depth = 2, split,
      leaf("0RLL", prediction = Double.NaN, nDataPoints = 0),
      leaf("0RLR", prediction = Double.NaN, nDataPoints = 0)
    )
    val nonEmptyRightRight = leaf("0RR", prediction = 6.0, nDataPoints = 3)
    val rightSubtree = TreeNode("0R", depth = 1, split, deeplyEmptySubtree, nonEmptyRightRight)
    val nonEmptyLeft = leaf("0L", prediction = 1.0, nDataPoints = 9)
    val tree = TreeNode("0", depth = 0, split, nonEmptyLeft, rightSubtree)

    val pruned = HonestyPruner.pruneEmptyHonestyLeaves(tree)
    pruned shouldBe TreeNode("0", depth = 0, split, nonEmptyLeft, nonEmptyRightRight)
  }
}
