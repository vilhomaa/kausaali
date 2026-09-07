package treebased.config

/**
 * Tuning for gradient-based splitting (Athey, Tibshirani & Wager, 2019) — meaningless
 * for the exact-splitting honest tree of Athey & Wager (2017), so it lives apart from
 * [[ExactCausalForestConfig]] rather than on a shared base config.
 */
final case class GradientSplitParams(
  /** `alpha`: each child of a split must retain at least this fraction of the parent
   *  node's split-balance weight. `0` disables the rule. */
  splitBalanceAlpha: Double = 0.05,
  /** Split stabilization: weight each child's gradient split score by its treatment-residual
   *  variance so a child with thin treatment overlap can't win a split on noise alone. */
  stabilizeSplits: Boolean = true
)
