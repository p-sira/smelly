use alloc::{string::ToString, vec::Vec};
use getset::Getters;
use nalgebra::{DMatrix, DVector, RealField};

use crate::{FilterError, replace_square, right_solve};

struct SigmaWeights<T> {
    mean_first: T,
    covariance_first: T,
    common: T,
    scale: T,
}

impl<T: Copy> SigmaWeights<T> {
    #[inline]
    fn mean(&self, index: usize) -> T {
        if index == 0 {
            self.mean_first
        } else {
            self.common
        }
    }

    #[inline]
    fn covariance(&self, index: usize) -> T {
        if index == 0 {
            self.covariance_first
        } else {
            self.common
        }
    }
}

/// Van der Merwe scaled sigma-point parameters.
#[derive(Debug, Clone, Copy)]
pub struct MerweScaledSigmaPoints<T: RealField = f64> {
    /// Spread of sigma points around the mean, usually a small positive value.
    pub alpha: T,
    /// Prior knowledge of the distribution; `2` is optimal for Gaussian states.
    pub beta: T,
    /// Secondary scaling parameter, commonly `0`.
    pub kappa: T,
}

impl<T: RealField + Copy> MerweScaledSigmaPoints<T> {
    /// Creates Van der Merwe sigma-point scaling parameters.
    ///
    /// The effective scaling must be positive for the state dimension. Invalid
    /// parameters are reported by the filter when sigma points are generated.
    ///
    /// # Example
    ///
    /// ```
    /// use smelly::MerweScaledSigmaPoints;
    ///
    /// let points = MerweScaledSigmaPoints::new(0.1_f64, 2.0, 0.0);
    /// assert_eq!(points.beta, 2.0);
    /// ```
    #[inline]
    pub fn new(alpha: T, beta: T, kappa: T) -> Self {
        Self { alpha, beta, kappa }
    }

    fn weights(&self, n: usize) -> Result<SigmaWeights<T>, FilterError> {
        let dimension = T::from_usize(n).ok_or(FilterError::InvalidSigmaPointScaling)?;
        let alpha2 = self.alpha * self.alpha;
        let scale = alpha2 * (dimension + self.kappa);
        let lambda = scale - dimension;
        if !scale.is_finite() || scale <= -T::zero() {
            return Err(FilterError::InvalidSigmaPointScaling);
        }
        let inv_scale = T::one() / scale;
        let mean_first = lambda * inv_scale;
        Ok(SigmaWeights {
            mean_first,
            covariance_first: mean_first + T::one() - alpha2 + self.beta,
            common: T::from_f64(0.5).ok_or(FilterError::InvalidSigmaPointScaling)? * inv_scale,
            scale,
        })
    }

    fn sigma_points(
        &self,
        state: &DVector<T>,
        covariance: &DMatrix<T>,
        weights: &SigmaWeights<T>,
        points: &mut Vec<DVector<T>>,
    ) -> Result<(), FilterError> {
        let n = state.len();
        let mut scaled = covariance.clone();
        scaled.scale_mut(weights.scale);
        let lower = scaled
            .cholesky()
            .ok_or(FilterError::NonPositiveDefiniteCovariance)?
            .unpack();
        points.resize_with(2 * n + 1, || DVector::zeros(n));
        for point in points.iter_mut() {
            if point.len() != n {
                point.resize_vertically_mut(n, T::zero());
            }
        }
        points[0].copy_from(state);
        for (index, column) in lower.column_iter().enumerate() {
            state.add_to(&column, &mut points[index + 1]);
            state.sub_to(&column, &mut points[n + index + 1]);
        }
        Ok(())
    }
}

/// An unscented Kalman filter using Van der Merwe scaled sigma points.
#[derive(Debug, Clone, Getters)]
pub struct UnscentedKalmanFilter<T: RealField = f64> {
    /// Current state estimate.
    #[getset(get = "pub")]
    state: DVector<T>,
    /// State covariance.
    #[getset(get = "pub")]
    covariance: DMatrix<T>,
    /// Process-noise covariance.
    #[getset(get = "pub")]
    process_noise: DMatrix<T>,
    /// Measurement-noise covariance.
    #[getset(get = "pub")]
    measurement_noise: DMatrix<T>,
    /// Sigma-point scaling parameters.
    pub points: MerweScaledSigmaPoints<T>,
    measurement_dim: usize,
    predicted_sigmas: Option<Vec<DVector<T>>>,
    sigma_workspace: Vec<DVector<T>>,
    measurement_sigmas: Vec<DVector<T>>,
    transform_mean: DVector<T>,
    transform_covariance: DMatrix<T>,
    transform_delta: DVector<T>,
    measurement_mean: DVector<T>,
    measurement_covariance: DMatrix<T>,
    measurement_delta: DVector<T>,
    cross_covariance: DMatrix<T>,
}

impl<T: RealField + Copy> UnscentedKalmanFilter<T> {
    /// Creates a filter with zero state and identity covariance and noise
    /// matrices.
    ///
    /// # Example
    ///
    /// ```
    /// use smelly::{MerweScaledSigmaPoints, UnscentedKalmanFilter};
    ///
    /// let points = MerweScaledSigmaPoints::new(0.1_f64, 2.0, 0.0);
    /// let filter = UnscentedKalmanFilter::new(3, 1, points);
    /// assert_eq!(filter.state().len(), 3);
    /// assert_eq!(filter.measurement_noise().shape(), (1, 1));
    /// ```
    #[inline]
    pub fn new(
        state_dim: usize,
        measurement_dim: usize,
        points: MerweScaledSigmaPoints<T>,
    ) -> Self {
        Self {
            state: DVector::zeros(state_dim),
            covariance: DMatrix::identity(state_dim, state_dim),
            process_noise: DMatrix::identity(state_dim, state_dim),
            measurement_noise: DMatrix::identity(measurement_dim, measurement_dim),
            points,
            measurement_dim,
            predicted_sigmas: None,
            sigma_workspace: Vec::new(),
            measurement_sigmas: Vec::new(),
            transform_mean: DVector::zeros(0),
            transform_covariance: DMatrix::zeros(0, 0),
            transform_delta: DVector::zeros(0),
            measurement_mean: DVector::zeros(measurement_dim),
            measurement_covariance: DMatrix::zeros(measurement_dim, measurement_dim),
            measurement_delta: DVector::zeros(measurement_dim),
            cross_covariance: DMatrix::zeros(state_dim, measurement_dim),
        }
    }

    pub fn set_state(&mut self, value: DVector<T>) -> Result<(), FilterError> {
        if value.len() != self.state.len() {
            return Err(FilterError::Dimension {
                name: "x",
                expected: self.state.len().to_string(),
                actual: value.len().to_string(),
            });
        }
        self.state = value;
        Ok(())
    }
    pub fn set_covariance(&mut self, value: DMatrix<T>) -> Result<(), FilterError> {
        replace_square("P", &mut self.covariance, value, self.state.len())
    }
    pub fn set_process_noise(&mut self, value: DMatrix<T>) -> Result<(), FilterError> {
        replace_square("Q", &mut self.process_noise, value, self.state.len())
    }
    pub fn set_measurement_noise(&mut self, value: DMatrix<T>) -> Result<(), FilterError> {
        replace_square(
            "R",
            &mut self.measurement_noise,
            value,
            self.measurement_dim,
        )
    }

    /// Propagates sigma points through `process` and adds process noise.
    ///
    /// `process` must return a vector with the same dimension as the state.
    /// Call this before each [`update`](Self::update). The prediction is left
    /// unchanged if validation or sigma-point generation fails.
    pub fn predict<F>(&mut self, process: F) -> Result<(), FilterError>
    where
        F: Fn(&DVector<T>) -> DVector<T>,
    {
        let n = self.state.len();
        let weights = self.points.weights(n)?;
        self.points.sigma_points(
            &self.state,
            &self.covariance,
            &weights,
            &mut self.sigma_workspace,
        )?;
        for point in &mut self.sigma_workspace {
            *point = process(point);
        }
        if self.sigma_workspace.iter().any(|point| point.len() != n) {
            let actual = self
                .sigma_workspace
                .iter()
                .map(DVector::len)
                .find(|&len| len != n)
                .unwrap();
            return Err(FilterError::Dimension {
                name: "process function result",
                expected: n.to_string(),
                actual: actual.to_string(),
            });
        }
        transform(
            &self.sigma_workspace,
            &weights,
            &self.process_noise,
            &mut self.transform_mean,
            &mut self.transform_covariance,
            &mut self.transform_delta,
        );
        // Resample after adding Q, reusing the propagated vectors.
        self.points.sigma_points(
            &self.transform_mean,
            &self.transform_covariance,
            &weights,
            &mut self.sigma_workspace,
        )?;
        core::mem::swap(&mut self.state, &mut self.transform_mean);
        core::mem::swap(&mut self.covariance, &mut self.transform_covariance);
        let predicted = core::mem::take(&mut self.sigma_workspace);
        let previous = self.predicted_sigmas.replace(predicted);
        self.sigma_workspace = previous.unwrap_or_default();
        Ok(())
    }

    /// Incorporates a measurement transformed by `measurement`.
    ///
    /// `measurement` must return a vector of the configured measurement
    /// dimension. The returned vector is the innovation. A successful update
    /// consumes the prediction, so another update requires a new
    /// [`predict`](Self::predict) call.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::{DMatrix, DVector};
    /// use smelly::{MerweScaledSigmaPoints, UnscentedKalmanFilter};
    ///
    /// let points = MerweScaledSigmaPoints::new(0.1_f64, 2.0, 0.0);
    /// let mut filter = UnscentedKalmanFilter::new(2, 1, points);
    /// filter.set_state(DVector::from_vec(vec![0.0, 1.0]))?;
    /// filter.set_process_noise(DMatrix::identity(2, 2) * 0.01)?;
    /// filter.set_measurement_noise(DMatrix::identity(1, 1) * 0.1)?;
    ///
    /// filter.predict(|x| DVector::from_vec(vec![x[0] + x[1], x[1]]))?;
    /// let innovation = filter.update(
    ///     &DVector::from_vec(vec![1.2]),
    ///     |x| DVector::from_vec(vec![x[0]]),
    /// )?;
    ///
    /// assert_eq!(innovation.len(), 1);
    /// # Ok::<(), smelly::FilterError>(())
    /// ```
    pub fn update<H>(&mut self, z: &DVector<T>, measurement: H) -> Result<DVector<T>, FilterError>
    where
        H: Fn(&DVector<T>) -> DVector<T>,
    {
        if z.len() != self.measurement_dim {
            return Err(FilterError::Dimension {
                name: "z",
                expected: self.measurement_dim.to_string(),
                actual: z.len().to_string(),
            });
        }
        let sigmas = self
            .predicted_sigmas
            .as_ref()
            .ok_or(FilterError::PredictRequired)?;
        self.measurement_sigmas.clear();
        self.measurement_sigmas
            .extend(sigmas.iter().map(measurement));
        if self
            .measurement_sigmas
            .iter()
            .any(|point| point.len() != self.measurement_dim)
        {
            let actual = self
                .measurement_sigmas
                .iter()
                .map(DVector::len)
                .find(|&len| len != self.measurement_dim)
                .unwrap();
            return Err(FilterError::Dimension {
                name: "measurement function result",
                expected: self.measurement_dim.to_string(),
                actual: actual.to_string(),
            });
        }

        let weights = self.points.weights(self.state.len())?;
        transform(
            &self.measurement_sigmas,
            &weights,
            &self.measurement_noise,
            &mut self.measurement_mean,
            &mut self.measurement_covariance,
            &mut self.measurement_delta,
        );
        let cross = &mut self.cross_covariance;
        cross.fill(T::zero());
        let state_delta = &mut self.transform_delta;
        let measurement_delta = &mut self.measurement_delta;
        for (index, (state, measured)) in sigmas.iter().zip(&self.measurement_sigmas).enumerate() {
            state.sub_to(&self.state, state_delta);
            measured.sub_to(&self.measurement_mean, measurement_delta);
            cross.ger(
                weights.covariance(index),
                state_delta,
                measurement_delta,
                T::one(),
            );
        }
        let k = right_solve(&self.measurement_covariance, cross)?;
        let innovation = z - &self.measurement_mean;
        self.state.gemv(T::one(), &k, &innovation, T::one());
        cross.gemm(T::one(), &k, &self.measurement_covariance, T::zero());
        self.covariance
            .gemm(-T::one(), cross, &k.transpose(), T::one());
        // Keep the sigma vectors for the next prediction instead of reallocating them.
        self.sigma_workspace = self.predicted_sigmas.take().unwrap();
        Ok(innovation)
    }
}

fn transform<T: RealField + Copy>(
    points: &[DVector<T>],
    weights: &SigmaWeights<T>,
    noise: &DMatrix<T>,
    mean: &mut DVector<T>,
    covariance: &mut DMatrix<T>,
    delta: &mut DVector<T>,
) {
    let dimension = points[0].len();
    if mean.len() != dimension {
        mean.resize_vertically_mut(dimension, T::zero());
    }
    mean.fill(T::zero());
    for (index, point) in points.iter().enumerate() {
        mean.axpy(weights.mean(index), point, T::one());
    }
    if covariance.shape() != noise.shape() {
        covariance.resize_mut(noise.nrows(), noise.ncols(), T::zero());
    }
    covariance.copy_from(noise);
    if delta.len() != dimension {
        delta.resize_vertically_mut(dimension, T::zero());
    }
    for (index, point) in points.iter().enumerate() {
        point.sub_to(mean, delta);
        covariance.ger(weights.covariance(index), delta, delta, T::one());
    }
}
