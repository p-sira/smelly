//! Compare dynamically sized and fixed-size (static) variants of each filter
//! on the same 4-state constant-velocity model.

use core::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use nalgebra::{DMatrix, DVector, SMatrix, SVector};
use smelly::{
    ExtendedKalmanFilter, KalmanFilter, MerweScaledSigmaPoints, SExtendedKalmanFilter,
    SKalmanFilter, SUnscentedKalmanFilter, UnscentedKalmanFilter,
};

const STATE_DIM: usize = 4;
const MEASUREMENT_DIM: usize = 2;
const DT: f64 = 0.1;

fn transition() -> DMatrix<f64> {
    DMatrix::from_row_slice(STATE_DIM, STATE_DIM, &transition_vec(DT))
}

fn observation() -> DMatrix<f64> {
    DMatrix::from_row_slice(MEASUREMENT_DIM, STATE_DIM, &observation_vec())
}

fn transition_vec(dt: f64) -> Vec<f64> {
    vec![
        1.0, 0.0, dt, 0.0, //
        0.0, 1.0, 0.0, dt, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    ]
}

fn observation_vec() -> Vec<f64> {
    vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
}

fn scaled_identity(size: usize, scale: f64) -> DMatrix<f64> {
    DMatrix::identity(size, size) * scale
}

fn smelly_kf() -> KalmanFilter<f64> {
    let mut filter = KalmanFilter::new(STATE_DIM, MEASUREMENT_DIM);
    filter.set_transition(transition()).unwrap();
    filter
        .set_process_noise(scaled_identity(STATE_DIM, 0.001))
        .unwrap();
    filter.set_measurement(observation()).unwrap();
    filter
        .set_measurement_noise(scaled_identity(MEASUREMENT_DIM, 0.1))
        .unwrap();
    filter
}

fn smelly_ekf() -> ExtendedKalmanFilter<f64> {
    let mut filter = ExtendedKalmanFilter::new(STATE_DIM, MEASUREMENT_DIM);
    filter.set_transition(transition()).unwrap();
    filter
        .set_process_noise(scaled_identity(STATE_DIM, 0.001))
        .unwrap();
    filter
        .set_measurement_noise(scaled_identity(MEASUREMENT_DIM, 0.1))
        .unwrap();
    filter
}

fn smelly_ukf() -> UnscentedKalmanFilter<f64> {
    let mut filter = UnscentedKalmanFilter::new(
        STATE_DIM,
        MEASUREMENT_DIM,
        MerweScaledSigmaPoints::new(0.1, 2.0, 0.0),
    );
    filter
        .set_process_noise(scaled_identity(STATE_DIM, 0.001))
        .unwrap();
    filter
        .set_measurement_noise(scaled_identity(MEASUREMENT_DIM, 0.1))
        .unwrap();
    filter
}

fn compare(c: &mut Criterion) {
    let measurement = DVector::from_vec(vec![1.0, 1.0]);
    let static_measurement = SVector::<f64, 2>::repeat(1.0);
    let f = SMatrix::<f64, 4, 4>::from_row_slice(&transition_vec(DT));
    let h = SMatrix::<f64, 2, 4>::from_row_slice(&observation_vec());

    let mut group = c.benchmark_group("filter_type/linear");
    let mut filter = smelly_kf();
    group.bench_function("smelly", |b| {
        b.iter(|| {
            filter.predict().unwrap();
            black_box(filter.update(black_box(&measurement)).unwrap());
        });
    });
    let mut filter = SKalmanFilter::<f64, 4, 2>::new();
    filter.set_transition(f).unwrap();
    filter.set_measurement(h).unwrap();
    filter
        .set_process_noise(SMatrix::identity() * 0.001)
        .unwrap();
    filter
        .set_measurement_noise(SMatrix::identity() * 0.1)
        .unwrap();
    group.bench_function("smelly_static", |b| {
        b.iter(|| {
            filter.predict().unwrap();
            black_box(filter.update(black_box(&static_measurement)).unwrap());
        });
    });
    group.finish();

    let mut group = c.benchmark_group("filter_type/extended");
    let mut filter = smelly_ekf();
    group.bench_function("smelly", |b| {
        b.iter(|| {
            filter.predict().unwrap();
            black_box(
                filter
                    .update(
                        black_box(&measurement),
                        |state| DVector::from_vec(vec![state[0], state[1]]),
                        |_| observation(),
                    )
                    .unwrap(),
            );
        });
    });
    let mut filter = SExtendedKalmanFilter::<f64, 4, 2>::new();
    filter.set_transition(f).unwrap();
    filter
        .set_process_noise(SMatrix::identity() * 0.001)
        .unwrap();
    filter
        .set_measurement_noise(SMatrix::identity() * 0.1)
        .unwrap();
    group.bench_function("smelly_static", |b| {
        b.iter(|| {
            filter.predict().unwrap();
            black_box(
                filter
                    .update(black_box(&static_measurement), |x| h * x, |_| h)
                    .unwrap(),
            );
        });
    });
    group.finish();

    let mut group = c.benchmark_group("filter_type/unscented");
    let mut filter = smelly_ukf();
    group.bench_function("smelly", |b| {
        b.iter(|| {
            filter
                .predict(|state| {
                    DVector::from_vec(vec![
                        state[0] + DT * state[2],
                        state[1] + DT * state[3],
                        state[2],
                        state[3],
                    ])
                })
                .unwrap();
            black_box(
                filter
                    .update(black_box(&measurement), |state| {
                        DVector::from_vec(vec![state[0], state[1]])
                    })
                    .unwrap(),
            );
        });
    });
    let mut filter =
        SUnscentedKalmanFilter::<f64, 4, 2>::new(MerweScaledSigmaPoints::new(0.1, 2.0, 0.0));
    filter
        .set_process_noise(SMatrix::identity() * 0.001)
        .unwrap();
    filter
        .set_measurement_noise(SMatrix::identity() * 0.1)
        .unwrap();
    group.bench_function("smelly_static", |b| {
        b.iter(|| {
            filter.predict(|x| f * x).unwrap();
            black_box(
                filter
                    .update(black_box(&static_measurement), |x| h * x)
                    .unwrap(),
            );
        });
    });
    group.finish();
}

criterion_group!(benches, compare);
criterion_main!(benches);
