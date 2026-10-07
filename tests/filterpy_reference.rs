#[path = "data/reference_data.rs"]
mod reference_data;

use nalgebra::{DMatrix, DVector, RealField, dmatrix, dvector};
use smelly::{ExtendedKalmanFilter, KalmanFilter, MerweScaledSigmaPoints, UnscentedKalmanFilter};

use reference_data::*;

const MEASUREMENTS: [f64; 3] = [1.2, 1.85, 2.45];

fn assert_vector_close<T: RealField + Copy + Into<f64>>(
    actual: &DVector<T>,
    expected: &[f64],
    tolerance: f64,
) {
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            ((*actual).into() - expected).abs() <= tolerance,
            "{actual:?} != {expected:?}"
        );
    }
}

fn assert_matrix_close<T: RealField + Copy + Into<f64>>(
    actual: &DMatrix<T>,
    expected: &[f64],
    tolerance: f64,
) {
    assert_eq!(actual.len(), expected.len());
    let columns = actual.ncols();
    for row in 0..actual.nrows() {
        for column in 0..columns {
            let actual = actual[(row, column)];
            let expected = expected[row * columns + column];
            assert!(
                (actual.into() - expected).abs() <= tolerance,
                "{actual:?} != {expected:?} at ({row}, {column})"
            );
        }
    }
}

fn run_linear<T>()
where
    T: RealField + Copy + Into<f64>,
{
    let mut filter = KalmanFilter::<T>::new(2, 1);
    filter
        .set_state(dvector![
            T::from_f64(0.2).unwrap(),
            T::from_f64(1.1).unwrap()
        ])
        .unwrap();
    filter.set_covariance(dmatrix![T::from_f64(2.0).unwrap(), T::from_f64(0.3).unwrap(); T::from_f64(0.3).unwrap(), T::from_f64(1.0).unwrap()]).unwrap();
    filter
        .set_transition(
            dmatrix![T::from_f64(1.0).unwrap(), T::from_f64(0.5).unwrap(); T::zero(), T::one()],
        )
        .unwrap();
    filter.set_process_noise(dmatrix![T::from_f64(0.02).unwrap(), T::from_f64(0.01).unwrap(); T::from_f64(0.01).unwrap(), T::from_f64(0.03).unwrap()]).unwrap();
    filter
        .set_measurement(dmatrix![T::one(), T::zero()])
        .unwrap();
    filter
        .set_measurement_noise(dmatrix![T::from_f64(0.4).unwrap()])
        .unwrap();
    for (step, z) in MEASUREMENTS.into_iter().enumerate() {
        filter.predict().unwrap();
        let innovation = filter.update(&dvector![T::from_f64(z).unwrap()]).unwrap();
        let tolerance = if std::mem::size_of::<T>() == 4 {
            2e-5
        } else {
            1e-12
        };
        assert_vector_close(filter.state(), &LINEAR_X[step * 2..step * 2 + 2], tolerance);
        assert_matrix_close(
            filter.covariance(),
            &LINEAR_P[step * 4..step * 4 + 4],
            tolerance,
        );
        assert_vector_close(&innovation, &LINEAR_Y[step..step + 1], tolerance);
    }
}

fn run_extended<T>()
where
    T: RealField + Copy + Into<f64>,
{
    let mut filter = ExtendedKalmanFilter::<T>::new(2, 1);
    filter
        .set_state(dvector![
            T::from_f64(0.2).unwrap(),
            T::from_f64(1.1).unwrap()
        ])
        .unwrap();
    filter.set_covariance(dmatrix![T::from_f64(2.0).unwrap(), T::from_f64(0.3).unwrap(); T::from_f64(0.3).unwrap(), T::from_f64(1.0).unwrap()]).unwrap();
    filter
        .set_transition(dmatrix![T::one(), T::from_f64(0.5).unwrap(); T::zero(), T::one()])
        .unwrap();
    filter.set_process_noise(dmatrix![T::from_f64(0.02).unwrap(), T::from_f64(0.01).unwrap(); T::from_f64(0.01).unwrap(), T::from_f64(0.03).unwrap()]).unwrap();
    filter
        .set_measurement_noise(dmatrix![T::from_f64(0.4).unwrap()])
        .unwrap();
    for (step, z) in MEASUREMENTS.into_iter().enumerate() {
        filter.predict().unwrap();
        let innovation = filter
            .update(
                &dvector![T::from_f64(z).unwrap()],
                |x| dvector![x[0] * x[0] + T::from_f64(0.25).unwrap() * x[1]],
                |x| dmatrix![T::from_f64(2.0).unwrap() * x[0], T::from_f64(0.25).unwrap()],
            )
            .unwrap();
        let tolerance = if std::mem::size_of::<T>() == 4 {
            3e-5
        } else {
            1e-12
        };
        assert_vector_close(
            filter.state(),
            &EXTENDED_X[step * 2..step * 2 + 2],
            tolerance,
        );
        assert_matrix_close(
            filter.covariance(),
            &EXTENDED_P[step * 4..step * 4 + 4],
            tolerance,
        );
        assert_vector_close(&innovation, &EXTENDED_Y[step..step + 1], tolerance);
    }
}

fn run_unscented<T>()
where
    T: RealField + Copy + Into<f64>,
{
    let points = MerweScaledSigmaPoints::new(
        T::from_f64(0.3).unwrap(),
        T::from_f64(2.0).unwrap(),
        T::zero(),
    );
    let mut filter = UnscentedKalmanFilter::<T>::new(2, 1, points);
    filter
        .set_state(dvector![
            T::from_f64(0.2).unwrap(),
            T::from_f64(1.1).unwrap()
        ])
        .unwrap();
    filter.set_covariance(dmatrix![T::from_f64(2.0).unwrap(), T::from_f64(0.3).unwrap(); T::from_f64(0.3).unwrap(), T::from_f64(1.0).unwrap()]).unwrap();
    filter.set_process_noise(dmatrix![T::from_f64(0.02).unwrap(), T::from_f64(0.01).unwrap(); T::from_f64(0.01).unwrap(), T::from_f64(0.03).unwrap()]).unwrap();
    filter
        .set_measurement_noise(dmatrix![T::from_f64(0.4).unwrap()])
        .unwrap();
    for (step, z) in MEASUREMENTS.into_iter().enumerate() {
        filter
            .predict(|x| {
                dvector![
                    x[0] + T::from_f64(0.5).unwrap() * x[1]
                        + T::from_f64(0.05).unwrap() * x[0] * x[0],
                    x[1] + T::from_f64(0.02).unwrap() * x[0].sin(),
                ]
            })
            .unwrap();
        let innovation = filter
            .update(&dvector![T::from_f64(z).unwrap()], |x| {
                dvector![x[0] * x[0] + T::from_f64(0.25).unwrap() * x[1]]
            })
            .unwrap();
        let tolerance = if std::mem::size_of::<T>() == 4 {
            2e-4
        } else {
            1e-11
        };
        assert_vector_close(
            filter.state(),
            &UNSCENTED_X[step * 2..step * 2 + 2],
            tolerance,
        );
        assert_matrix_close(
            filter.covariance(),
            &UNSCENTED_P[step * 4..step * 4 + 4],
            tolerance,
        );
        assert_vector_close(&innovation, &UNSCENTED_Y[step..step + 1], tolerance);
    }
}

#[test]
fn multivariate_linear_matches_filterpy() {
    let mut filter = KalmanFilter::<f64>::new(3, 2);
    filter.set_state(dvector![0.5, -0.2, 1.0]).unwrap();
    filter
        .set_covariance(dmatrix![1.2, 0.1, 0.0; 0.1, 0.8, 0.2; 0.0, 0.2, 1.5])
        .unwrap();
    filter
        .set_transition(dmatrix![1.0, 0.2, 0.0; 0.0, 1.0, 0.3; 0.0, 0.0, 1.0])
        .unwrap();
    filter
        .set_process_noise(dmatrix![0.01, 0.0, 0.0; 0.0, 0.02, 0.0; 0.0, 0.0, 0.03])
        .unwrap();
    filter
        .set_measurement(dmatrix![1.0, 0.0, 0.5; 0.0, 1.0, -0.25])
        .unwrap();
    filter
        .set_measurement_noise(dmatrix![0.3, 0.05; 0.05, 0.4])
        .unwrap();

    for (step, measurement) in [[1.1, -0.1], [1.7, 0.05]].into_iter().enumerate() {
        filter.predict().unwrap();
        let innovation = filter
            .update(&DVector::from_row_slice(&measurement))
            .unwrap();
        assert_vector_close(
            filter.state(),
            &MULTIVARIATE_LINEAR_X[step * 3..step * 3 + 3],
            1e-12,
        );
        assert_matrix_close(
            filter.covariance(),
            &MULTIVARIATE_LINEAR_P[step * 9..step * 9 + 9],
            1e-12,
        );
        assert_vector_close(
            &innovation,
            &MULTIVARIATE_LINEAR_Y[step * 2..step * 2 + 2],
            1e-12,
        );
    }
}

#[test]
fn matches_filterpy_for_f64() {
    run_linear::<f64>();
    run_extended::<f64>();
    run_unscented::<f64>();
}

#[test]
fn matches_filterpy_for_f32() {
    run_linear::<f32>();
    run_extended::<f32>();
    run_unscented::<f32>();
}
