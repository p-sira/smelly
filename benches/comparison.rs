use core::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use kalman_filters::{
    ExtendedKalmanFilterBuilder as ReferenceExtendedBuilder, KalmanFilter as ReferenceKalmanFilter,
    KalmanFilterBuilder as ReferenceKalmanBuilder, NonlinearSystem,
    UnscentedKalmanFilterBuilder as ReferenceUnscentedBuilder,
};
use nalgebra::{DMatrix, DVector};
use smelly::{ExtendedKalmanFilter, KalmanFilter, MerweScaledSigmaPoints, UnscentedKalmanFilter};

const STATE_DIM: usize = 4;
const MEASUREMENT_DIM: usize = 2;
const DT: f64 = 0.1;

#[derive(Clone, Copy)]
struct ConstantVelocity;

impl NonlinearSystem<f64> for ConstantVelocity {
    fn state_transition(&self, state: &[f64], _control: Option<&[f64]>, dt: f64) -> Vec<f64> {
        vec![
            state[0] + dt * state[2],
            state[1] + dt * state[3],
            state[2],
            state[3],
        ]
    }

    fn measurement(&self, state: &[f64]) -> Vec<f64> {
        vec![state[0], state[1]]
    }

    fn state_jacobian(&self, _state: &[f64], _control: Option<&[f64]>, dt: f64) -> Vec<f64> {
        transition_vec(dt)
    }

    fn measurement_jacobian(&self, _state: &[f64]) -> Vec<f64> {
        observation_vec()
    }

    fn state_dim(&self) -> usize {
        STATE_DIM
    }

    fn measurement_dim(&self) -> usize {
        MEASUREMENT_DIM
    }
}

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

fn reference_kf() -> ReferenceKalmanFilter<f64> {
    ReferenceKalmanBuilder::new(STATE_DIM, MEASUREMENT_DIM)
        .initial_state(vec![0.0; STATE_DIM])
        .initial_covariance(identity_vec(STATE_DIM, 1.0))
        .transition_matrix(transition_vec(DT))
        .process_noise(identity_vec(STATE_DIM, 0.001))
        .observation_matrix(observation_vec())
        .measurement_noise(identity_vec(MEASUREMENT_DIM, 0.1))
        .build()
        .unwrap()
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

fn reference_ekf() -> kalman_filters::ExtendedKalmanFilter<f64, ConstantVelocity> {
    ReferenceExtendedBuilder::new(ConstantVelocity)
        .initial_state(vec![0.0; STATE_DIM])
        .initial_covariance(identity_vec(STATE_DIM, 1.0))
        .process_noise(identity_vec(STATE_DIM, 0.001))
        .measurement_noise(identity_vec(MEASUREMENT_DIM, 0.1))
        .dt(DT)
        .build()
        .unwrap()
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

fn reference_ukf() -> kalman_filters::UnscentedKalmanFilter<f64, ConstantVelocity> {
    ReferenceUnscentedBuilder::new(ConstantVelocity)
        .initial_state(vec![0.0; STATE_DIM])
        .initial_covariance(identity_vec(STATE_DIM, 1.0))
        .process_noise(identity_vec(STATE_DIM, 0.001))
        .measurement_noise(identity_vec(MEASUREMENT_DIM, 0.1))
        .dt(DT)
        .alpha(0.1)
        .beta(2.0)
        .kappa(0.0)
        .build()
        .unwrap()
}

fn identity_vec(size: usize, scale: f64) -> Vec<f64> {
    let mut matrix = vec![0.0; size * size];
    for index in 0..size {
        matrix[index * size + index] = scale;
    }
    matrix
}

fn compare(c: &mut Criterion) {
    let measurement = DVector::from_vec(vec![1.0, 1.0]);
    let reference_measurement = [1.0, 1.0];

    let mut group = c.benchmark_group("filter_type/linear");
    let mut filter = smelly_kf();
    group.bench_function("smelly", |b| {
        b.iter(|| {
            filter.predict().unwrap();
            black_box(filter.update(black_box(&measurement)).unwrap());
        });
    });
    let mut filter = reference_kf();
    group.bench_function("kalman-filter-rs", |b| {
        b.iter(|| {
            filter.predict();
            filter.update(black_box(&reference_measurement)).unwrap();
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
    let mut filter = reference_ekf();
    group.bench_function("kalman-filter-rs", |b| {
        b.iter(|| {
            filter.predict();
            filter.update(black_box(&reference_measurement)).unwrap();
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
    let mut filter = reference_ukf();
    group.bench_function("kalman-filter-rs", |b| {
        b.iter(|| {
            filter.predict().unwrap();
            filter.update(black_box(&reference_measurement)).unwrap();
        });
    });
    group.finish();
}

criterion_group!(benches, compare);
criterion_main!(benches);
