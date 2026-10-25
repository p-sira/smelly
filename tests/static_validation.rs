use nalgebra::{SMatrix, SVector, matrix, vector};
use smelly::{
    FilterError, MerweScaledSigmaPoints, SExtendedKalmanFilter, SKalmanFilter, SParticleFilter,
    SUnscentedKalmanFilter,
};

#[test]
fn singular_updates_preserve_static_filters() {
    let mut linear = SKalmanFilter::<f64, 2, 2>::new();
    linear.set_measurement_noise(SMatrix::zeros()).unwrap();
    let state = *linear.state();
    let covariance = *linear.covariance();
    assert_eq!(
        linear.update(&vector![1.0, 2.0]),
        Err(FilterError::SingularInnovation)
    );
    assert_eq!(*linear.state(), state);
    assert_eq!(*linear.covariance(), covariance);

    let mut extended = SExtendedKalmanFilter::<f64, 2, 2>::new();
    extended.set_measurement_noise(SMatrix::zeros()).unwrap();
    assert_eq!(
        extended.update(
            &vector![1.0, 2.0],
            |_| SVector::zeros(),
            |_| SMatrix::zeros()
        ),
        Err(FilterError::SingularInnovation)
    );
    assert_eq!(*extended.state(), state);
    assert_eq!(*extended.covariance(), covariance);
}

#[test]
fn unscented_errors_preserve_prediction_for_retry() {
    let mut filter =
        SUnscentedKalmanFilter::<f64, 2, 1>::new(MerweScaledSigmaPoints::new(0.3, 2.0, 0.0));
    assert_eq!(
        filter.update(&vector![1.0], |x| vector![x[0]]),
        Err(FilterError::PredictRequired)
    );
    filter.points.alpha = 0.0;
    assert_eq!(
        filter.predict(|x| *x),
        Err(FilterError::InvalidSigmaPointScaling)
    );
    filter.points.alpha = 0.3;
    filter.set_covariance(-SMatrix::identity()).unwrap();
    assert_eq!(
        filter.predict(|x| *x),
        Err(FilterError::NonPositiveDefiniteCovariance)
    );
    filter.set_covariance(SMatrix::identity()).unwrap();
    filter.predict(|x| *x).unwrap();
    let state = *filter.state();
    let covariance = *filter.covariance();
    filter.set_measurement_noise(SMatrix::zeros()).unwrap();
    assert_eq!(
        filter.update(&vector![1.0], |_| vector![0.0]),
        Err(FilterError::SingularInnovation)
    );
    assert_eq!(*filter.state(), state);
    assert_eq!(*filter.covariance(), covariance);
    filter.update(&vector![1.0], |x| vector![x[0]]).unwrap();
    assert_eq!(
        filter.update(&vector![1.0], |x| vector![x[0]]),
        Err(FilterError::PredictRequired)
    );
}

#[test]
fn static_particles_match_bayes_and_recover_from_underflow() {
    let mut filter =
        SParticleFilter::<f64, 2, 3>::new(matrix![0.0, 1.0, 2.0; 1.0, 1.0, 1.0]).unwrap();
    filter.update(|x| x[0] + 1.0).unwrap();
    for (actual, expected) in filter
        .weights()
        .iter()
        .zip([1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0])
    {
        assert!((actual - expected).abs() < 1e-14);
    }
    assert!((filter.state()[0] - 4.0 / 3.0).abs() < 1e-14);
    assert!((filter.covariance()[(0, 0)] - 5.0 / 9.0).abs() < 1e-14);
    let posterior = filter.clone();
    assert_eq!(
        filter.predict(|_, out| out[0] = f64::NAN),
        Err(FilterError::InvalidParticles)
    );
    assert_eq!(
        filter.update(|_| -1.0),
        Err(FilterError::InvalidParticleWeights)
    );
    assert_eq!(
        filter.resample(1.0),
        Err(FilterError::InvalidResamplingOffset)
    );
    assert_eq!(filter.particles(), posterior.particles());
    assert_eq!(filter.weights(), posterior.weights());
    filter.update_log_likelihood(|x| -1000.0 * x[0]).unwrap();
    filter.update_log_likelihood(|x| 1000.0 * x[0]).unwrap();
    for (actual, expected) in filter.weights().iter().zip(posterior.weights()) {
        assert!((actual - expected).abs() < 1e-12);
    }
    filter.resample(0.37).unwrap();
    assert_eq!(filter.weights(), &[1.0 / 3.0; 3]);
    assert!(SParticleFilter::<f32, 0, 3>::new(SMatrix::zeros()).is_err());
    assert!(SParticleFilter::<f32, 2, 0>::new(SMatrix::zeros()).is_err());
}
