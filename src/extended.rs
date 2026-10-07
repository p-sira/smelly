use alloc::{format, string::ToString};
use getset::Getters;
use nalgebra::{DMatrix, DVector, RealField};

use crate::{FilterError, FilterWorkspace, linear_predict, linear_update, replace_square};

/// An extended Kalman filter with a linear process model and nonlinear measurements.
#[derive(Debug, Clone, Getters)]
#[getset(get = "pub")]
pub struct ExtendedKalmanFilter<T: RealField = f64> {
    /// Current state estimate.
    state: DVector<T>,
    /// State covariance.
    covariance: DMatrix<T>,
    /// State-transition matrix.
    transition: DMatrix<T>,
    /// Process-noise covariance.
    process_noise: DMatrix<T>,
    /// Measurement-noise covariance.
    measurement_noise: DMatrix<T>,
    #[getset(skip)]
    measurement_dim: usize,
    #[getset(skip)]
    workspace: FilterWorkspace<T>,
}

impl<T: RealField + Copy> ExtendedKalmanFilter<T> {
    /// Creates a filter with zero state and identity covariance, transition, and
    /// noise matrices.
    ///
    /// # Example
    ///
    /// ```
    /// use smelly::ExtendedKalmanFilter;
    ///
    /// let filter = ExtendedKalmanFilter::<f64>::new(2, 1);
    /// assert_eq!(filter.state().len(), 2);
    /// assert_eq!(filter.measurement_noise().shape(), (1, 1));
    /// ```
    #[inline]
    pub fn new(state_dim: usize, measurement_dim: usize) -> Self {
        Self {
            state: DVector::zeros(state_dim),
            covariance: DMatrix::identity(state_dim, state_dim),
            transition: DMatrix::identity(state_dim, state_dim),
            process_noise: DMatrix::identity(state_dim, state_dim),
            measurement_noise: DMatrix::identity(measurement_dim, measurement_dim),
            measurement_dim,
            workspace: FilterWorkspace::new(),
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
    pub fn set_transition(&mut self, value: DMatrix<T>) -> Result<(), FilterError> {
        replace_square("F", &mut self.transition, value, self.state.len())
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

    /// Propagates the state and covariance through the linear process model.
    ///
    /// This applies `x = F x` and `P = F P Fᵀ + Q`. Matrix dimensions are
    /// fixed by construction and validated by the configuration methods.
    pub fn predict(&mut self) -> Result<(), FilterError> {
        linear_predict(
            &mut self.state,
            &mut self.covariance,
            &self.transition,
            &self.process_noise,
            &mut self.workspace,
        );
        Ok(())
    }

    /// Updates from a nonlinear measurement function and its Jacobian.
    ///
    /// Both callbacks receive the predicted state. `measurement` must return a
    /// vector of the configured measurement dimension, and `jacobian` must
    /// return a `measurement_dim × state_dim` matrix.
    ///
    /// The returned vector is the innovation: the observed measurement minus
    /// the predicted measurement.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::{DMatrix, DVector};
    /// use smelly::ExtendedKalmanFilter;
    ///
    /// let mut filter = ExtendedKalmanFilter::<f64>::new(2, 1);
    /// filter.set_state(DVector::from_vec(vec![3.0, 4.0]))?;
    /// filter.set_process_noise(DMatrix::identity(2, 2) * 0.01)?;
    /// filter.set_measurement_noise(DMatrix::identity(1, 1) * 0.1)?;
    /// filter.predict()?;
    ///
    /// let innovation = filter.update(
    ///     &DVector::from_vec(vec![5.1]),
    ///     |x| DVector::from_vec(vec![(x[0] * x[0] + x[1] * x[1]).sqrt()]),
    ///     |x| {
    ///         let range = (x[0] * x[0] + x[1] * x[1]).sqrt();
    ///         DMatrix::from_row_slice(1, 2, &[x[0] / range, x[1] / range])
    ///     },
    /// )?;
    ///
    /// assert!((innovation[0] - 0.1).abs() < 1e-12);
    /// # Ok::<(), smelly::FilterError>(())
    /// ```
    pub fn update<H, J>(
        &mut self,
        z: &DVector<T>,
        measurement: H,
        jacobian: J,
    ) -> Result<DVector<T>, FilterError>
    where
        H: FnOnce(&DVector<T>) -> DVector<T>,
        J: FnOnce(&DVector<T>) -> DMatrix<T>,
    {
        let n = self.state.len();
        if z.len() != self.measurement_dim {
            return Err(FilterError::Dimension {
                name: "z",
                expected: self.measurement_dim.to_string(),
                actual: z.len().to_string(),
            });
        }

        let h = jacobian(&self.state);
        if h.nrows() != self.measurement_dim || h.ncols() != n {
            return Err(FilterError::Dimension {
                name: "measurement Jacobian",
                expected: format!("{}x{n}", self.measurement_dim),
                actual: format!("{}x{}", h.nrows(), h.ncols()),
            });
        }
        let predicted = measurement(&self.state);
        if predicted.len() != self.measurement_dim {
            return Err(FilterError::Dimension {
                name: "measurement function result",
                expected: self.measurement_dim.to_string(),
                actual: predicted.len().to_string(),
            });
        }

        let innovation = z - predicted;
        linear_update(
            &mut self.state,
            &mut self.covariance,
            &h,
            &self.measurement_noise,
            innovation,
            &mut self.workspace,
        )
    }
}
