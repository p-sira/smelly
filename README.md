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
    <a href="https://app.codspeed.io/p-sira/smelly?utm_source=badge"><img src="https://img.shields.io/endpoint?url=https://codspeed.io/badge.json" alt="CodSpeed"/></a>
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
- Dynamic and fixed-size variants of every filter
- Works with `f32` and `f64`
- `no_std` support

## Static filters

`SKalmanFilter<T, N, M>`, `SExtendedKalmanFilter<T, N, M>`, and
`SUnscentedKalmanFilter<T, N, M>` use `SVector` and `SMatrix`, with state dimension
`N` and measurement dimension `M` checked at compile time. Their constructors
omit dimension arguments; the UKF still takes `MerweScaledSigmaPoints`.
Configuration and callbacks otherwise follow the dynamic APIs.

```rust
use nalgebra::{SMatrix, SVector};
use smelly::SKalmanFilter;

let mut filter = SKalmanFilter::<f64, 2, 1>::new();
filter.set_transition(SMatrix::from_row_slice(&[1.0, 1.0, 0.0, 1.0]))?;
filter.set_measurement(SMatrix::from_row_slice(&[1.0, 0.0]))?;
filter.predict()?;
filter.update(&SVector::from_row_slice(&[1.2]))?;
# Ok::<(), smelly::FilterError>(())
```

`SParticleFilter<T, N, P>` fixes both the state dimension and particle count.
Construct it with an `SMatrix<T, N, P>` whose columns are the initial samples;
it shares the dynamic particle filter's algorithms and slice callbacks.
All static variants store their matrices and workspaces inline and perform no
internal heap allocations, including particle posterior moments. Callbacks may
allocate. Large dimensions increase stack usage, so use the dynamic variants
when inline storage is impractical.

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
Ordinary likelihood updates normalize linear products directly when they are
well represented, falling back to log-sum-exp for extreme weights.
Prediction, weight updates, and systematic resampling reuse buffers and allocate
nothing internally after construction. Model callbacks may allocate; computing
`state()` and `covariance()` allocates their results. Invalid inputs return
`FilterError` and preserve the posterior for retry. Resampling is explicit so you
can estimate the weighted posterior before introducing resampling variance.

## Performance

Mean time per predict/update cycle for a 4-state, 2-measurement `f64` constant-velocity
model, measured with Criterion on an AMD Ryzen 5 4600H (rustc 1.99.1):

| Filter | smelly (dynamic) | smelly (static) | [kalman-filter-rs](https://github.com/destenson/kalman-filter-rs/tree/v1.0.1) | Dynamic speedup | Static speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear | 484 ns | 114 ns | 682 ns | 1.41× | 6.00× |
| Extended | 516 ns | 114 ns | 722 ns | 1.40× | 6.33× |
| Unscented | 1.27 µs | 354 ns | 1.75 µs | 1.38× | 4.96× |

Speedups are reference time divided by the corresponding smelly time, calculated
before rounding. All columns were measured in the same run.
Results depend on the model and hardware. The UKFs differ: smelly regenerates
sigma points after adding process noise; kalman-filter-rs reuses propagated
points, so their numerical results can differ.

Run the comparison on your hardware:

```sh
cargo bench --bench comparison -- \
    --warm-up-time 1 --measurement-time 3 --sample-size 30 --noplot
```

### Particle filter comparison

Mean time per cycle for a 4-state `f64` constant-velocity model on the same
Ryzen 5 4600H and rustc 1.99.1, comparing `kalman_filters` 1.0.1:

| Cycle | Particles | smelly (dynamic) | smelly (static) | kalman-filter-rs | Dynamic speedup | Static speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Predict/update | 128 | 3.02 µs | 2.62 µs | 6.41 µs | 2.12× | 2.45× |
| Predict/update | 1,024 | 23.39 µs | 20.96 µs | 49.38 µs | 2.11× | 2.36× |
| Predict/update/resample | 128 | 3.41 µs | 3.00 µs | 10.94 µs | 3.21× | 3.65× |
| Predict/update/resample | 1,024 | 27.55 µs | 23.96 µs | 84.03 µs | 3.05× | 3.51× |

Each cycle starts with identical samples and uniform weights, uses the same
motion and likelihood functions, and excludes setup and destruction. Process
noise and automatic resampling are disabled. Static and dynamic smelly filters
use the same callbacks and resampling offset. The resampling group explicitly
performs systematic resampling once per cycle; smelly uses a supplied offset,
while the reference draws its offset internally. Random process noise generation
and posterior moment calculations are outside this comparison. Results depend
on hardware and workload.

Benchmarks cover 128, 512, and 1,024 particles. Timings for 512 particles are
not included above.

Reproduce with Criterion (30 samples, 1-second warmup, 3-second measurement):

```sh
cargo bench --bench particle -- \
    --warm-up-time 1 --measurement-time 3 --sample-size 30 --noplot
```

Particle tests check exact discrete Bayes weights, an analytic Gaussian
posterior, repeated nonlinear updates against the reference library, underflow
recovery, `f32` accuracy, systematic resampling, and zero allocations in the core
cycle. Run the reference comparison tests with `cargo test`.
Smelly's benchmarks are also tracked continuously in CI with [CodSpeed](https://app.codspeed.io/p-sira/smelly).

## Testing

The test data is generated with [FilterPy](https://github.com/rlabbe/filterpy):

```sh
uv run tests/generate_filterpy_reference.py
cargo test
```

## Acknowledgments

Smelly is built upon [FilterPy](https://github.com/rlabbe/filterpy) by Roger Labbe. While FilterPy's philosophy is pedagogy over performance, Smelly adapts the core algorithms and optimizes for performance and memory efficiency.
