# Smelly: Kalman Filters for Rust

Your smelly Unscented Kalman Filter implementation and more.

<p>
    <a href="https://opensource.org/license/BSD-3-clause">
        <img src="https://img.shields.io/badge/License-BSD--3--Clause-brightgreen.svg" alt="License">
    </a>
    <a href="https://crates.io/crates/smelly">
        <img src="https://img.shields.io/crates/v/smelly" alt="Crate">
    </a>
    <a href="https://crates.io/crates/smelly">
        <img src="https://img.shields.io/crates/d/smelly" alt="Total Downloads">
    </a>
    <a href="https://docs.rs/smelly">
        <img src="https://img.shields.io/badge/Docs-docs.rs-blue" alt="Documentation">
    </a>
</p>

```rust
use nalgebra::{DMatrix, DVector};
use smelly::KalmanFilter;

let mut filter = KalmanFilter::<f64>::new(2, 1);
filter.set_transition(DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 0.0, 1.0])).unwrap();
filter.set_measurement(DMatrix::from_row_slice(1, 2, &[1.0, 0.0])).unwrap();
filter.predict()?;
filter.update(&DVector::from_row_slice(&[1.2]))?;
# Ok::<(), smelly::FilterError>(())
```

## Features

- Kalman filter (KF), Extended Kalman filter (EKF), Unscented Kalman filter (UKF), and particle filter
- Works with `f32` and `f64`
- `no_std` support

## Particle filter

Particles are columns of a `DMatrix`. Supply your own initial samples and process
noise; resampling takes a uniform random value from `[0, 1)`, so you can use your
preferred RNG, including seeded generators in `no_std` applications.

```rust
use nalgebra::dmatrix;
use smelly::ParticleFilter;

let mut filter = ParticleFilter::<f64>::new(dmatrix![0.0, 0.5, 1.0; 1.0, 1.0, 1.0])?;
filter.predict(|state, output| {
    output[0] = state[0] + 0.1 * state[1]; // Add sampled process noise here.
})?;
filter.update_log_likelihood(|state| -0.5 * (state[0] - 0.8).powi(2) / 0.04)?;
let estimate = filter.state();
let covariance = filter.covariance();
if filter.effective_sample_size() < filter.weights().len() as f64 / 2.0 {
    filter.resample(0.37)?; // Replace with a fresh uniform random draw in production.
}
# Ok::<(), smelly::FilterError>(())
```

`update` accepts ordinary likelihoods; `update_log_likelihood` avoids underflow
for very small likelihoods. Both retain log weights between observations.
Prediction, weight updates, and systematic resampling reuse buffers and allocate
nothing internally after construction. Model callbacks may allocate; computing
`state()` and `covariance()` allocates their results. Invalid inputs return
`FilterError` and preserve the posterior for retry. Resampling is explicit so you
can estimate the weighted posterior before introducing resampling variance.

## Performance

Time per predict/update cycle for a 4-state, 2-measurement `f64` constant-velocity
model, measured with Criterion on an AMD Ryzen 5 4600H (rustc 1.98.1):

| Filter | smelly | [kalman-filter-rs](https://github.com/destenson/kalman-filter-rs/tree/v1.0.1) | Speedup |
| --- | ---: | ---: | ---: |
| Linear | 512 ns | 687 ns | 1.34× |
| Extended | 535 ns | 660 ns | 1.23× |
| Unscented | 1.28 µs | 1.78 µs | 1.40× |

Speedup is reference time divided by smelly time, calculated before rounding.
Results depend on the model and hardware. The UKFs differ: smelly regenerates
sigma points after adding process noise; kalman-filter-rs reuses propagated
points, so their numerical results can differ.

Run the comparison on your hardware:

```sh
cargo bench --bench comparison --features comparison
```

### Particle filter comparison

Mean time per cycle for a 4-state `f64` constant-velocity model on the same
Ryzen 5 4600H and rustc 1.98.1, comparing `kalman_filters` 1.0.1:

| Cycle | Particles | smelly | kalman-filter-rs | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Predict/update | 128 | 3.50 µs | 6.63 µs | 1.90× |
| Predict/update | 1,024 | 27.27 µs | 53.56 µs | 1.96× |
| Predict/update | 8,192 | 220.73 µs | 406.29 µs | 1.84× |
| Predict/update/resample | 128 | 3.94 µs | 11.40 µs | 2.89× |
| Predict/update/resample | 1,024 | 30.44 µs | 86.23 µs | 2.83× |
| Predict/update/resample | 8,192 | 258.22 µs | 688.74 µs | 2.67× |

Each cycle starts with identical samples and uniform weights, uses the same
motion and likelihood functions, and excludes setup and destruction. Process
noise and automatic resampling are disabled. The resampling group explicitly
performs systematic resampling once per cycle; smelly uses a supplied offset,
while the reference draws its offset internally. Random process noise generation
and posterior moment calculations are outside this comparison. Results depend
on hardware and workload; the 95% confidence intervals for mean cycle times did
not overlap in this run.

Reproduce with Criterion (30 samples, 1-second warmup, 3-second measurement):

```sh
cargo bench --bench particle --features comparison -- \
    --warm-up-time 1 --measurement-time 3 --sample-size 30 --noplot
```

Particle tests check exact discrete Bayes weights, an analytic Gaussian
posterior, repeated nonlinear updates against the reference library, underflow
recovery, `f32` accuracy, systematic resampling, and zero allocations in the core
cycle. Run the reference comparison tests with `cargo test --all-features`.

## Testing

The test data is generated with [FilterPy](https://github.com/rlabbe/filterpy):

```sh
uv run tests/generate_filterpy_reference.py
cargo test
```

## Acknowledgments

Smelly is built upon [FilterPy](https://github.com/rlabbe/filterpy) by Roger Labbe. While FilterPy's philosophy is pedagogy over performance, Smelly adapts the core algorithms and optimizes for performance and memory efficiency.