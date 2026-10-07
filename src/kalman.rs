use alloc::{format, string::ToString};
use getset::Getters;
use nalgebra::{DMatrix, DVector, RealField};

use crate::{FilterError, FilterWorkspace, linear_predict, linear_update, replace_square};

/// A discrete linear Kalman filter.
#[derive(Debug, Clone, Getters)]
#[getset(get = "pub")]
pub struct KalmanFilter<T: RealField = f64> {
    /// Current state estimate.
    state: DVector<T>,
    /// State covariance.
    covariance: DMatrix<T>,
    /// State-transition matrix.
    transition: DMatrix<T>,
    /// Process-noise covariance.
    process_noise: DMatrix<T>,
    /// Measurement matrix.
    measurement: DMatrix<T>,
    /// Measurement-noise covariance.
    measurement_noise: DMatrix<T>,
    #[getset(skip)]
    workspace: FilterWorkspace<T>,
}

impl<T: RealField + Copy> KalmanFilter<T> {
    /// Creates a filter with zero state, identity covariance and noise matrices,
    /// and an identity state-transition matrix.
    ///
    /// The measurement matrix is initialized to zero. Configure [`Self::h`]
    /// before processing measurements.
    ///
    /// # Example
    ///
    /// ```
    /// use smelly::KalmanFilter;
    ///
    /// let filter = KalmanFilter::<f32>::new(4, 2);
    /// assert_eq!(filter.state().len(), 4);
    /// assert_eq!(filter.measurement().shape(), (2, 4));
    /// ```
    #[inline]
    pub fn new(state_dim: usize, measurement_dim: usize) -> Self {
        Self {
            state: DVector::zeros(state_dim),
            covariance: DMatrix::identity(state_dim, state_dim),
            transition: DMatrix::identity(state_dim, state_dim),
            process_noise: DMatrix::identity(state_dim, state_dim),
            measurement: DMatrix::zeros(measurement_dim, state_dim),
            measurement_noise: DMatrix::identity(measurement_dim, measurement_dim),
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
    pub fn set_measurement(&mut self, value: DMatrix<T>) -> Result<(), FilterError> {
        if value.nrows() != self.measurement.nrows() || value.ncols() != self.state.len() {
            return Err(FilterError::Dimension {
                name: "H",
                expected: format!("{}x{}", self.measurement.nrows(), self.state.len()),
                actual: format!("{}x{}", value.nrows(), value.ncols()),
            });
        }
        self.measurement = value;
        Ok(())
    }
    pub fn set_measurement_noise(&mut self, value: DMatrix<T>) -> Result<(), FilterError> {
        replace_square(
            "R",
            &mut self.measurement_noise,
            value,
            self.measurement.nrows(),
        )
    }

    /// Propagates the state and covariance through the linear process model.
    ///
    /// This applies `x = F x` and `P = F P Fᵀ + Q` using [`Self::f`] and
    /// [`Self::q`].
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

    /// Incorporates a measurement and returns its innovation `z - H x`.
    ///
    /// Returns [`FilterError::Dimension`] for incompatible matrix or measurement
    /// dimensions, and [`FilterError::SingularInnovation`] when the innovation
    /// covariance cannot be solved.
    pub fn update(&mut self, z: &DVector<T>) -> Result<DVector<T>, FilterError> {
        let measurement_dim = self.measurement.nrows();
        if z.len() != measurement_dim {
            return Err(FilterError::Dimension {
                name: "z",
                expected: measurement_dim.to_string(),
                actual: z.len().to_string(),
            });
        }

        let innovation = z - &self.measurement * &self.state;
        linear_update(
            &mut self.state,
            &mut self.covariance,
            &self.measurement,
            &self.measurement_noise,
            innovation,
            &mut self.workspace,
        )
    }
}
