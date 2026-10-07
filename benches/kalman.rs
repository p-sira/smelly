use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nalgebra::{DMatrix, DVector};
use smelly::{ExtendedKalmanFilter, KalmanFilter, MerweScaledSigmaPoints, UnscentedKalmanFilter};

fn transition_matrix(state_dim: usize) -> DMatrix<f64> {
    let mut transition = DMatrix::identity(state_dim, state_dim);
    for index in 0..state_dim / 2 {
        transition[(index, state_dim / 2 + index)] = 0.1;
    }
    transition
}

fn measurement_matrix(state_dim: usize, measurement_dim: usize) -> DMatrix<f64> {
    let mut measurement = DMatrix::zeros(measurement_dim, state_dim);
    for index in 0..measurement_dim.min(state_dim) {
        measurement[(index, index)] = 1.0;
    }
    measurement
}

fn linear_filter(state_dim: usize, measurement_dim: usize) -> KalmanFilter<f64> {
    let mut filter = KalmanFilter::new(state_dim, measurement_dim);
    filter.set_transition(transition_matrix(state_dim)).unwrap();
    filter
        .set_process_noise(DMatrix::identity(state_dim, state_dim) * 0.001)
        .unwrap();
    filter
        .set_measurement(measurement_matrix(state_dim, measurement_dim))
        .unwrap();
    filter
        .set_measurement_noise(DMatrix::identity(measurement_dim, measurement_dim) * 0.1)
        .unwrap();
    filter
}

fn extended_filter(state_dim: usize, measurement_dim: usize) -> ExtendedKalmanFilter<f64> {
    let mut filter = ExtendedKalmanFilter::new(state_dim, measurement_dim);
    filter.set_transition(transition_matrix(state_dim)).unwrap();
    filter
        .set_process_noise(DMatrix::identity(state_dim, state_dim) * 0.001)
        .unwrap();
    filter
        .set_measurement_noise(DMatrix::identity(measurement_dim, measurement_dim) * 0.1)
        .unwrap();
    filter
}

fn unscented_filter(state_dim: usize, measurement_dim: usize) -> UnscentedKalmanFilter<f64> {
    let mut filter = UnscentedKalmanFilter::new(
        state_dim,
        measurement_dim,
        MerweScaledSigmaPoints::new(0.1, 2.0, 0.0),
    );
    filter
        .set_process_noise(DMatrix::identity(state_dim, state_dim) * 0.001)
        .unwrap();
    filter
        .set_measurement_noise(DMatrix::identity(measurement_dim, measurement_dim) * 0.1)
        .unwrap();
    filter
}

fn kalman_cycle(c: &mut Criterion) {
    let mut group = c.benchmark_group("kalman_cycle");
    for (state_dim, measurement_dim) in [(4, 2), (16, 8), (64, 16)] {
        let measurement = DVector::from_element(measurement_dim, 1.0);
        let mut filter = linear_filter(state_dim, measurement_dim);
        group.bench_with_input(
            BenchmarkId::new("predict_update", format!("{state_dim}x{measurement_dim}")),
            &(state_dim, measurement_dim),
            |b, _| {
                b.iter(|| {
                    filter.predict().unwrap();
                    black_box(filter.update(black_box(&measurement)).unwrap());
                });
            },
        );
    }
    group.finish();
}

fn extended_cycle(c: &mut Criterion) {
    let mut group = c.benchmark_group("extended_cycle");
    for (state_dim, measurement_dim) in [(4, 2), (16, 8), (64, 16)] {
        let measurement = DVector::from_element(measurement_dim, 1.0);
        let mut filter = extended_filter(state_dim, measurement_dim);
        let h = measurement_matrix(state_dim, measurement_dim);
        group.bench_with_input(
            BenchmarkId::new("predict_update", format!("{state_dim}x{measurement_dim}")),
            &(state_dim, measurement_dim),
            |b, _| {
                b.iter(|| {
                    filter.predict().unwrap();
                    black_box(
                        filter
                            .update(black_box(&measurement), |state| &h * state, |_| h.clone())
                            .unwrap(),
                    );
                });
            },
        );
    }
    group.finish();
}

fn unscented_cycle(c: &mut Criterion) {
    let mut group = c.benchmark_group("unscented_cycle");
    for (state_dim, measurement_dim) in [(4, 2), (16, 8), (64, 16)] {
        let measurement = DVector::from_element(measurement_dim, 1.0);
        let mut filter = unscented_filter(state_dim, measurement_dim);
        let f = transition_matrix(state_dim);
        let h = measurement_matrix(state_dim, measurement_dim);
        group.bench_with_input(
            BenchmarkId::new("predict_update", format!("{state_dim}x{measurement_dim}")),
            &(state_dim, measurement_dim),
            |b, _| {
                b.iter(|| {
                    filter.predict(|state| &f * state).unwrap();
                    black_box(
                        filter
                            .update(black_box(&measurement), |state| &h * state)
                            .unwrap(),
                    );
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, kalman_cycle, extended_cycle, unscented_cycle);
criterion_main!(benches);
