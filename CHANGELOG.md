# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(`early-semver`: while on `0.x`, the minor version may carry breaking changes).


## [0.1.0] - 7.9.2026

Initial public release.

### Added
- `GeneralizedRandomForest` — GRF family (Athey, Tibshirani & Wager 2019):
  gradient-based splitting, honest leaves, local centering, adaptive-kernel
  prediction, pointwise confidence intervals.
- `CausalForest` — exact-splitting honest causal forest (Athey & Wager 2018).
- CSV / row-source data loading with a categorical encoding pipeline.
- JSON forest serialization via circe, with transparent gzip.
- `LICENSE` (Apache-2.0), `NOTICE`, and project metadata for publishing.


[Unreleased]: https://github.com/vilhomaa/kausaali/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/vilhomaa/kausaali/releases/tag/v0.1.0
