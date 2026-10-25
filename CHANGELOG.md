# Changelog

## 0.2.0 - 2026-10-26

### Added

- `ParticleFilter` with stable log weights, effective sample size, posterior moments, and systematic resampling. Reusable buffers for prediction, likelihood updates, and resampling without internal allocations after construction.
- Fixed-size `SKalmanFilter`, `SExtendedKalmanFilter`, `SUnscentedKalmanFilter`, and `SParticleFilter` with inline storage and compile-time dimensions.

### Changed

- Move `kalman_filters` to dev-dependencies and remove the `comparison` feature. Comparison tests and benchmarks now run without feature flags.
