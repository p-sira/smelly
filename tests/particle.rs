use nalgebra::{DMatrix, dmatrix};
use smelly::{FilterError, ParticleFilter};

#[test]
fn posterior_matches_exact_discrete_bayes_and_moments() {
    let mut filter = ParticleFilter::<f64>::new(dmatrix![0.0, 1.0, 2.0; 2.0, 1.0, 0.0]).unwrap();
    for _ in 0..2 {
        filter.update(|x| x[0] + 1.0).unwrap();
    }
    let expected = [1.0 / 14.0, 4.0 / 14.0, 9.0 / 14.0];
    for (actual, expected) in filter.weights().iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-14);
    }
    let mean = filter.state();
    assert!((mean[0] - 11.0 / 7.0).abs() < 1e-14);
    assert!((mean[1] - 3.0 / 7.0).abs() < 1e-14);
    let covariance = filter.covariance();
    assert!((covariance[(0, 0)] - 19.0 / 49.0).abs() < 1e-14);
    assert!((covariance[(0, 1)] + 19.0 / 49.0).abs() < 1e-14);
    assert!((filter.effective_sample_size() - 2.0).abs() < 1e-14);
}

#[test]
fn log_weights_retain_information_after_linear_underflow() {
    let mut filter = ParticleFilter::<f64>::new(dmatrix![0.0, 1.0]).unwrap();
    filter
        .update_log_likelihood(|x| -10000.0 - x[0] * 1000.0)
        .unwrap();
    assert_eq!(filter.weights()[1], 0.0);
    filter.update_log_likelihood(|x| x[0] * 1000.0).unwrap();
    assert!((filter.weights()[0] - 0.5).abs() < 1e-12);
    assert!((filter.weights()[1] - 0.5).abs() < 1e-12);
}

#[test]
fn invalid_inputs_preserve_posterior_and_allow_retry() {
    for matrix in [
        DMatrix::<f64>::zeros(0, 2),
        DMatrix::zeros(2, 0),
        dmatrix![f64::NAN],
    ] {
        assert!(matches!(
            ParticleFilter::<f64>::new(matrix),
            Err(FilterError::InvalidParticles)
        ));
    }
    let mut filter = ParticleFilter::<f64>::new(dmatrix![0.0, 1.0, 2.0]).unwrap();
    filter.update(|x| x[0]).unwrap();
    let original = filter.clone();
    for bad in [-1.0, f64::NAN, f64::INFINITY, 0.0] {
        assert_eq!(
            filter.update(|_| bad),
            Err(FilterError::InvalidParticleWeights)
        );
        assert_eq!(filter.weights(), original.weights());
    }
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(filter.update_log_likelihood(|_| bad).is_err());
        assert_eq!(filter.weights(), original.weights());
    }
    assert!(
        filter
            .predict(|x, out| out[0] = if x[0] == 2.0 { f64::NAN } else { 99.0 })
            .is_err()
    );
    assert_eq!(filter.particles(), original.particles());
    for offset in [-0.1, 1.0, f64::NAN, f64::INFINITY] {
        assert!(filter.resample(offset).is_err());
        assert_eq!(filter.particles(), original.particles());
        assert_eq!(filter.weights(), original.weights());
    }
    filter.predict(|x, out| out[0] = x[0] + 1.0).unwrap();
    filter.update(|_| 1.0).unwrap();
    assert!((filter.state()[0] - 8.0 / 3.0).abs() < 1e-14);
}

#[test]
fn systematic_resampling_matches_inverse_cdf_and_skips_zero_mass() {
    let mut filter = ParticleFilter::<f64>::new(dmatrix![0.0, 1.0, 2.0, 3.0]).unwrap();
    filter
        .update(|x| match x[0] as usize {
            1 => 1.0,
            3 => 3.0,
            _ => 0.0,
        })
        .unwrap();
    filter.resample(0.0).unwrap();
    assert_eq!(filter.particles(), &dmatrix![1.0, 3.0, 3.0, 3.0]);
    assert_eq!(filter.weights(), &[0.25; 4]);
    filter.update(|_| 1.0).unwrap();
    assert_eq!(filter.weights(), &[0.25; 4]);
}

#[test]
fn supports_f32_and_single_particle() {
    let mut filter = ParticleFilter::<f32>::new(dmatrix![2.0; 3.0]).unwrap();
    filter.predict(|x, out| out[0] += x[1]).unwrap();
    filter.update_log_likelihood(|_| -1000.0).unwrap();
    filter.resample(0.999).unwrap();
    assert_eq!(filter.state().as_slice(), &[5.0, 3.0]);
    assert_eq!(filter.covariance(), DMatrix::zeros(2, 2));
    assert_eq!(filter.effective_sample_size(), 1.0);
}

#[test]
fn gaussian_posterior_matches_analytic_solution() {
    // Midpoint quadrature of N(0,1), observed at z=1 with variance 0.25.
    let count = 20_000;
    let mut filter = ParticleFilter::<f64>::new(DMatrix::from_fn(1, count, |_, i| {
        -8.0 + (i as f64 + 0.5) * 16.0 / count as f64
    }))
    .unwrap();
    filter
        .update_log_likelihood(|x| -0.5 * x[0] * x[0])
        .unwrap();
    filter
        .update_log_likelihood(|x| -0.5 * (x[0] - 1.0).powi(2) / 0.25)
        .unwrap();
    assert!((filter.state()[0] - 0.8).abs() < 1e-12);
    assert!((filter.covariance()[(0, 0)] - 0.2).abs() < 1e-12);
}

#[cfg(feature = "comparison")]
#[test]
fn matches_reference_over_repeated_nonlinear_updates() {
    use kalman_filters::ParticleFilterBuilder;
    let samples = DMatrix::from_fn(2, 128, |row, i| (i as f64 * 0.1 + row as f64).sin());
    let mut filter = ParticleFilter::<f64>::new(samples.clone()).unwrap();
    let mut reference = ParticleFilterBuilder::new(2, 128)
        .initial_mean(vec![0.0; 2])
        .initial_std(vec![0.0; 2])
        .process_noise_std(vec![0.0; 2])
        .measurement_noise_std(vec![1.0])
        .dt(0.1)
        .ess_threshold(0.0)
        .build()
        .unwrap();
    for (particle, column) in reference.particles.iter_mut().zip(samples.column_iter()) {
        particle.state.copy_from_slice(column.as_slice());
    }
    for step in 0..20 {
        filter.predict(|x, out| out[0] = x[0] + 0.1 * x[1]).unwrap();
        reference.predict(|x, dt| vec![x[0] + dt * x[1], x[1]]);
        let z = (step as f64 * 0.1).sin();
        let likelihood = |x: &[f64]| (-0.5 * (x[0] * x[0] - z).powi(2)).exp();
        filter.update(likelihood).unwrap();
        reference.update(&[z], |x, _| likelihood(x)).unwrap();
        for (weight, particle) in filter.weights().iter().zip(&reference.particles) {
            assert!((weight - particle.weight).abs() < 1e-13);
        }
        for (actual, expected) in filter.state().iter().zip(reference.mean()) {
            assert!((actual - expected).abs() < 1e-13);
        }
        let cov = reference.covariance();
        for row in 0..2 {
            for col in 0..2 {
                assert!((filter.covariance()[(row, col)] - cov[row * 2 + col]).abs() < 1e-13);
            }
        }
    }
}

#[test]
fn f32_posterior_matches_exact_bayes() {
    let mut filter = ParticleFilter::<f32>::new(dmatrix![0.0, 1.0, 2.0]).unwrap();
    for _ in 0..2 {
        filter.update(|x| x[0] + 1.0).unwrap();
    }
    assert!((filter.state()[0] - 11.0 / 7.0).abs() < 1e-6);
    assert!((filter.covariance()[(0, 0)] - 19.0 / 49.0).abs() < 1e-6);
    assert!((filter.effective_sample_size() - 2.0).abs() < 1e-6);
}

#[test]
fn systematic_resampling_preserves_expected_mass_across_offsets() {
    let mut filter = ParticleFilter::<f64>::new(dmatrix![0.0, 1.0, 2.0]).unwrap();
    filter.update(|x| [0.1, 0.3, 0.6][x[0] as usize]).unwrap();
    let mut counts = [0; 3];
    for draw in 0..100 {
        let mut sampled = filter.clone();
        sampled.resample((draw as f64 + 0.5) / 100.0).unwrap();
        for value in sampled.particles().iter() {
            counts[*value as usize] += 1;
        }
    }
    assert_eq!(counts, [30, 90, 180]);
}
