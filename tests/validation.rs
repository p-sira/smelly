use nalgebra::{DMatrix, DVector, dmatrix, dvector};
use smelly::{
    ExtendedKalmanFilter, FilterError, KalmanFilter, MerweScaledSigmaPoints, UnscentedKalmanFilter,
};

#[test]
fn linear_filter_rejects_invalid_shapes() {
    let mut filter = KalmanFilter::<f64>::new(2, 1);
    assert!(matches!(
        filter.set_covariance(DMatrix::identity(1, 1)),
        Err(FilterError::Dimension { name: "P", .. })
    ));

    let mut filter = KalmanFilter::<f64>::new(2, 1);
    assert!(matches!(
        filter.update(&dvector![1.0, 2.0]),
        Err(FilterError::Dimension { name: "z", .. })
    ));

    assert!(matches!(
        filter.set_measurement(DMatrix::zeros(1, 3)),
        Err(FilterError::Dimension { name: "H", .. })
    ));
}

#[test]
fn linear_filter_reports_singular_innovation_without_changing_state() {
    let mut filter = KalmanFilter::<f64>::new(1, 1);
    filter.set_state(dvector![3.0]).unwrap();
    filter
        .set_covariance(DMatrix::zeros(filter.state().len(), filter.state().len()))
        .unwrap();
    filter
        .set_process_noise(DMatrix::zeros(
            filter.process_noise().nrows(),
            filter.process_noise().ncols(),
        ))
        .unwrap();
    filter
        .set_measurement(DMatrix::from_element(1, 1, 1.0))
        .unwrap();
    filter
        .set_measurement_noise(DMatrix::zeros(
            filter.measurement_noise().nrows(),
            filter.measurement_noise().ncols(),
        ))
        .unwrap();
    let original_x = filter.state().clone();
    let original_p = filter.covariance().clone();

    assert_eq!(
        filter.update(&dvector![4.0]),
        Err(FilterError::SingularInnovation)
    );
    assert_eq!(filter.state(), &original_x);
    assert_eq!(filter.covariance(), &original_p);
}

#[test]
fn extended_filter_validates_nonlinear_function_outputs() {
    let mut filter = ExtendedKalmanFilter::<f64>::new(2, 1);
    assert!(matches!(
        filter.update(&dvector![1.0], |_| dvector![1.0], |_| DMatrix::zeros(2, 2),),
        Err(FilterError::Dimension {
            name: "measurement Jacobian",
            ..
        })
    ));

    assert!(matches!(
        filter.update(
            &dvector![1.0],
            |_| dvector![1.0, 2.0],
            |_| dmatrix![1.0, 0.0],
        ),
        Err(FilterError::Dimension {
            name: "measurement function result",
            ..
        })
    ));
}

#[test]
fn unscented_filter_requires_prediction_before_update() {
    let points = MerweScaledSigmaPoints::new(0.3, 2.0, 0.0);
    let mut filter = UnscentedKalmanFilter::<f64>::new(2, 1, points);
    assert_eq!(
        filter.update(&dvector![1.0], |x| dvector![x[0]]),
        Err(FilterError::PredictRequired)
    );
}

#[test]
fn unscented_filter_rejects_invalid_scaling_and_covariance() {
    let points = MerweScaledSigmaPoints::new(0.0, 2.0, 0.0);
    let mut filter = UnscentedKalmanFilter::<f64>::new(2, 1, points);
    assert_eq!(
        filter.predict(|x| x.clone()),
        Err(FilterError::InvalidSigmaPointScaling)
    );

    let points = MerweScaledSigmaPoints::new(0.3, 2.0, 0.0);
    let mut filter = UnscentedKalmanFilter::<f64>::new(2, 1, points);
    filter.set_covariance(dmatrix![1.0, 2.0; 2.0, 1.0]).unwrap();
    assert_eq!(
        filter.predict(|x| x.clone()),
        Err(FilterError::NonPositiveDefiniteCovariance)
    );
}

#[test]
fn unscented_filter_validates_model_outputs() {
    let points = MerweScaledSigmaPoints::new(0.3, 2.0, 0.0);
    let mut filter = UnscentedKalmanFilter::<f64>::new(2, 1, points);
    assert!(matches!(
        filter.predict(|_| dvector![1.0]),
        Err(FilterError::Dimension {
            name: "process function result",
            ..
        })
    ));

    filter.predict(|x| x.clone()).unwrap();
    assert!(matches!(
        filter.update(&dvector![1.0], |x| x.clone()),
        Err(FilterError::Dimension {
            name: "measurement function result",
            ..
        })
    ));
}

#[test]
fn unscented_filter_matches_linear_filter_with_process_noise() {
    for (n, m) in [(2, 1), (4, 2), (16, 8)] {
        let mut linear = KalmanFilter::<f64>::new(n, m);
        let mut transition = linear.transition().clone();
        let mut measurement = linear.measurement().clone();
        for i in 0..m {
            transition[(i, n - m + i)] = 0.1;
            measurement[(i, i)] = 1.0;
        }
        linear.set_transition(transition).unwrap();
        linear.set_measurement(measurement).unwrap();
        linear
            .set_process_noise(linear.process_noise() * 0.05)
            .unwrap();
        linear
            .set_measurement_noise(linear.measurement_noise() * 0.2)
            .unwrap();
        let mut unscented =
            UnscentedKalmanFilter::new(n, m, MerweScaledSigmaPoints::new(0.3, 2.0, 0.0));
        unscented
            .set_process_noise(linear.process_noise().clone())
            .unwrap();
        unscented
            .set_measurement_noise(linear.measurement_noise().clone())
            .unwrap();
        for step in 0..100 {
            let z = DVector::from_fn(m, |i, _| (step as f64 * 0.1 + i as f64).sin());
            linear.predict().unwrap();
            unscented.predict(|x| linear.transition() * x).unwrap();
            let expected = linear.update(&z).unwrap();
            let actual = unscented.update(&z, |x| linear.measurement() * x).unwrap();
            assert!((&actual - expected).norm() < 1e-10);
            assert!((unscented.state() - linear.state()).norm() < 1e-10);
            assert!((unscented.covariance() - linear.covariance()).norm() < 1e-10);
            assert!(unscented.covariance().clone().cholesky().is_some());
        }
    }
}

#[test]
fn unscented_failures_preserve_prediction_for_retry() {
    let mut filter =
        UnscentedKalmanFilter::<f64>::new(2, 1, MerweScaledSigmaPoints::new(0.3, 2.0, 0.0));
    filter.predict(|x| x.clone()).unwrap();
    let mut expected = filter.clone();
    let x = filter.state().clone();
    let p = filter.covariance().clone();

    assert!(filter.predict(|_| dvector![1.0]).is_err());
    // Fail the second Cholesky decomposition, after transforming the points.
    filter
        .set_process_noise(-DMatrix::identity(2, 2) * 100.0)
        .unwrap();
    assert_eq!(
        filter.predict(|x| x.clone()),
        Err(FilterError::NonPositiveDefiniteCovariance)
    );
    assert!(filter.update(&dvector![1.0], |x| x.clone()).is_err());
    filter
        .set_measurement_noise(DMatrix::zeros(
            filter.measurement_noise().nrows(),
            filter.measurement_noise().ncols(),
        ))
        .unwrap();
    assert_eq!(
        filter.update(&dvector![1.0], |_| dvector![0.0]),
        Err(FilterError::SingularInnovation)
    );
    assert_eq!(filter.state(), &x);
    assert_eq!(filter.covariance(), &p);

    filter
        .set_measurement_noise(expected.measurement_noise().clone())
        .unwrap();
    let actual = filter.update(&dvector![1.0], |x| dvector![x[0]]).unwrap();
    let innovation = expected.update(&dvector![1.0], |x| dvector![x[0]]).unwrap();
    assert_eq!(actual, innovation);
    assert_eq!(filter.state(), expected.state());
    assert_eq!(filter.covariance(), expected.covariance());
    assert_eq!(
        filter.update(&dvector![1.0], |x| dvector![x[0]]),
        Err(FilterError::PredictRequired)
    );
}

#[test]
fn successful_updates_keep_covariance_symmetric_positive_definite() {
    let mut filter = KalmanFilter::<f64>::new(2, 1);
    filter.set_transition(dmatrix![1.0, 0.5; 0.0, 1.0]).unwrap();
    filter.set_measurement(dmatrix![1.0, 0.0]).unwrap();
    filter
        .set_process_noise(DMatrix::identity(2, 2) * 0.01)
        .unwrap();
    filter.set_measurement_noise(dmatrix![0.2]).unwrap();

    for measurement in [0.2, 0.9, 1.4, 2.1] {
        filter.predict().unwrap();
        filter.update(&dvector![measurement]).unwrap();
        assert!((filter.covariance() - filter.covariance().transpose()).norm() < 1e-12);
        assert!(filter.covariance().clone().cholesky().is_some());
    }
}

#[test]
fn covariance_stays_positive_definite_over_many_updates() {
    let mut filter = KalmanFilter::<f64>::new(4, 2);
    filter
        .set_transition(dmatrix![
            1.0, 0.0, 0.1, 0.0;
            0.0, 1.0, 0.0, 0.1;
            0.0, 0.0, 1.0, 0.0;
            0.0, 0.0, 0.0, 1.0
        ])
        .unwrap();
    filter
        .set_measurement(dmatrix![1.0, 0.0, 0.0, 0.0; 0.0, 1.0, 0.0, 0.0])
        .unwrap();
    filter
        .set_process_noise(DMatrix::identity(4, 4) * 1e-12)
        .unwrap();
    filter
        .set_measurement_noise(DMatrix::identity(2, 2) * 1e-9)
        .unwrap();

    for step in 0..10_000 {
        filter.predict().unwrap();
        let position = step as f64 * 0.01;
        filter.update(&dvector![position, -position]).unwrap();
        assert!((filter.covariance() - filter.covariance().transpose()).norm() < 1e-14);
        assert!(filter.covariance().clone().cholesky().is_some());
    }
}

#[test]
fn filter_dimensions_cannot_change_after_construction() {
    let mut linear = KalmanFilter::<f64>::new(2, 1);
    assert!(matches!(
        linear.set_state(DVector::zeros(3)),
        Err(FilterError::Dimension { name: "x", .. })
    ));
    assert!(matches!(
        linear.set_covariance(DMatrix::identity(3, 3)),
        Err(FilterError::Dimension { name: "P", .. })
    ));
}
