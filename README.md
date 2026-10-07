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

- Kalman filter (KF), Extended Kalman filter (EKF), and Unscented Kalman filter (UKF)
- Works with `f32` and `f64`
- `no_std` support

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

## Testing

The test data is generated with [FilterPy](https://github.com/rlabbe/filterpy):

```sh
uv run tests/generate_filterpy_reference.py
cargo test
```

## Acknowledgments

Smelly is built upon [FilterPy](https://github.com/rlabbe/filterpy) by Roger Labbe. While FilterPy's philosophy is pedagogy over performance, Smelly adapts the core algorithms and optimizes for performance and memory efficiency.