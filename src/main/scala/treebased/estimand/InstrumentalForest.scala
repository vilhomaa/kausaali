package treebased.estimand

/*
 * PLACEHOLDER — not implemented yet. Intentionally no code.
 *
 * Instrumental forest: Athey, Tibshirani & Wager (2019), "Generalized Random Forests",
 * section 7 — the paper's flagship non-trivial estimand.
 *
 * Estimand: theta(x) = local average treatment effect of W on Y given X = x, identified
 * through an instrument Z (Z affects W, and affects Y only through W).
 *
 * Pieces needed to add it as a `ForestSpec`:
 *   - New DataPoint subtype, e.g. `IvObservation(features, weight, y, w, z)` in
 *     core/domain/data (mirrors `CausalObservation` plus the instrument z).
 *   - Moment / relabeling: eq. (34)-(35). Pseudo-outcome orthogonalises (Y, W) against
 *     the instrument, using node-level averages of Z, W, Y; reduces to `CausalMoment`
 *     when Z == W (the unconfounded case).
 *   - SplitCriterion: `GradientVarianceReduction` on the relabeled outcomes.
 *   - SplitGuard: instrument-balance variant of `TreatmentBalanceGuard` (both Z groups
 *     must stay estimable on each side), plus a minimum on the node's Z-W covariance so
 *     the first stage does not degenerate.
 *   - LeafEstimator: 2SLS / Wald ratio on the honest sample
 *     ( cov(Z, Y) / cov(Z, W) ), with the same NaN-on-degenerate contract that
 *     `TreatmentEffectStats.ate` uses so `HonestyPruner` can collapse a weak-instrument leaf.
 *   - WeightedEstimator: a closed-form weighted Wald ratio over the alpha_i(x)-weighted honest
 *     sample (see `ForestKernel`), not an average of per-tree Wald ratios. Needs a `KernelRow`
 *     carrying z as well — either widen `KernelRow` or add an IV-specific row/estimator pair.
 */
