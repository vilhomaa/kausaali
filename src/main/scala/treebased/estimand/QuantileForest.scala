package treebased.estimand

/*
 * PLACEHOLDER — not implemented yet. Intentionally no code.
 *
 * Quantile regression forest: Meinshausen (2006), "Quantile Regression Forests",
 * recovered inside the GRF framework in Athey, Tibshirani & Wager (2019), section 5.
 *
 * Estimand: theta(x) = q-th conditional quantile of Y given X = x (or several q at once).
 *
 * Pieces needed to add it as another `ForestSpec[Observation]`:
 *   - Moment / relabeling: pseudo-outcome from the check-loss gradient, i.e.
 *     rho_i = q - 1{ Y_i <= theta_hat_parent } (sign-based relabeling); one per target
 *     quantile, or a multi-quantile score.
 *   - SplitCriterion: reuse `GradientVarianceReduction` on the relabeled outcomes.
 *   - SplitGuard: `MinNodeSizeGuard` (no treatment arms) — same as regression.
 *   - LeafEstimator: empty weighted CDF of the honest Y sample, read off at level q;
 *     needs the leaf to keep Y values (or the forest weights from `ForestKernel`),
 *     not just a mean.
 *   - WeightedEstimator: a new impl that inverts the alpha_i(x)-weighted empirical CDF of Y
 *     at level q (with `kernelPrediction` on), NOT an average of per-tree quantiles. Unlike
 *     the linear estimands it is a sort, not a closed-form solve.
 *
 * No new DataPoint subtype required — `Observation` already carries (features, weight, y);
 * `RegressionSpec.kernelRow` already projects it to a `KernelRow`.
 */
