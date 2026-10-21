# Changelog

## 0.2.0 - 2026-10-21

### Added

- Particle filter with `f32`, `f64`, and `no_std` support, stable log weights,
  effective sample size, posterior moments, and systematic resampling.
- Reusable buffers for prediction, likelihood updates, and resampling without
  internal allocations after construction.
- Accuracy and allocation tests, usage documentation, and benchmarks against
  `kalman_filters` 1.0.1.
