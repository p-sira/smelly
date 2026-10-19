#![no_std]
#![doc = include_str!("../README.md")]

extern crate alloc;

use alloc::format;

mod error;
mod extended;
mod kalman;
mod unscented;

pub use error::FilterError;
pub use extended::ExtendedKalmanFilter;
pub use kalman::KalmanFilter;
pub use unscented::{MerweScaledSigmaPoints, UnscentedKalmanFilter};

use nalgebra::{DMatrix, DVector, RealField};

#[derive(Debug, Clone)]
pub(crate) struct FilterWorkspace<T: RealField> {
    predicted_x: DVector<T>,
    covariance: DMatrix<T>,
    product: DMatrix<T>,
    transpose: DMatrix<T>,
    hp: DMatrix<T>,
    pht: DMatrix<T>,
    innovation_covariance: DMatrix<T>,
}

impl<T: RealField + Copy> FilterWorkspace<T> {
    #[inline]
    pub(crate) fn new() -> Self {
        Self {
            predicted_x: DVector::zeros(0),
            covariance: DMatrix::zeros(0, 0),
            product: DMatrix::zeros(0, 0),
            transpose: DMatrix::zeros(0, 0),
            hp: DMatrix::zeros(0, 0),
            pht: DMatrix::zeros(0, 0),
            innovation_covariance: DMatrix::zeros(0, 0),
        }
    }

    #[inline]
    fn prepare_state(&mut self, state_dim: usize) {
        resize_vector(&mut self.predicted_x, state_dim);
        resize_matrix(&mut self.covariance, state_dim, state_dim);
        resize_matrix(&mut self.product, state_dim, state_dim);
        resize_matrix(&mut self.transpose, state_dim, state_dim);
    }

    #[inline]
    fn prepare_measurement(&mut self, state_dim: usize, measurement_dim: usize) {
        self.prepare_state(state_dim);
        resize_matrix(&mut self.hp, measurement_dim, state_dim);
        resize_matrix(&mut self.pht, state_dim, measurement_dim);
        resize_matrix(
            &mut self.innovation_covariance,
            measurement_dim,
            measurement_dim,
        );
    }
}

#[inline]
fn resize_vector<T: RealField + Copy>(vector: &mut DVector<T>, len: usize) {
    if vector.len() != len {
        vector.resize_vertically_mut(len, T::zero());
    }
}

#[inline]
fn resize_matrix<T: RealField + Copy>(matrix: &mut DMatrix<T>, rows: usize, columns: usize) {
    if matrix.shape() != (rows, columns) {
        matrix.resize_mut(rows, columns, T::zero());
    }
}

#[inline]
pub(crate) fn check_square<T: RealField>(
    name: &'static str,
    matrix: &DMatrix<T>,
    size: usize,
) -> Result<(), FilterError> {
    if matrix.nrows() != size || matrix.ncols() != size {
        return Err(FilterError::Dimension {
            name,
            expected: format!("{size}x{size}"),
            actual: format!("{}x{}", matrix.nrows(), matrix.ncols()),
        });
    }
    Ok(())
}

pub(crate) fn replace_square<T: RealField>(
    name: &'static str,
    target: &mut DMatrix<T>,
    value: DMatrix<T>,
    size: usize,
) -> Result<(), FilterError> {
    check_square(name, &value, size)?;
    *target = value;
    Ok(())
}

pub(crate) fn right_solve<T>(a: &DMatrix<T>, b: &DMatrix<T>) -> Result<DMatrix<T>, FilterError>
where
    T: RealField + Copy,
{
    a.transpose()
        .lu()
        .solve(&b.transpose())
        .map(|solution| solution.transpose())
        .ok_or(FilterError::SingularInnovation)
}

pub(crate) fn linear_predict<T>(
    x: &mut DVector<T>,
    p: &mut DMatrix<T>,
    f: &DMatrix<T>,
    q: &DMatrix<T>,
    workspace: &mut FilterWorkspace<T>,
) where
    T: RealField + Copy,
{
    let n = x.len();
    workspace.prepare_state(n);
    workspace.predicted_x.gemv(T::one(), f, x, T::zero());
    core::mem::swap(x, &mut workspace.predicted_x);

    f.transpose_to(&mut workspace.transpose);
    workspace.product.gemm(T::one(), f, &*p, T::zero());
    workspace.covariance.gemm(
        T::one(),
        &workspace.product,
        &workspace.transpose,
        T::zero(),
    );
    core::mem::swap(p, &mut workspace.covariance);
    *p += q;
}

pub(crate) fn linear_update<T>(
    x: &mut DVector<T>,
    p: &mut DMatrix<T>,
    h: &DMatrix<T>,
    r: &DMatrix<T>,
    innovation: DVector<T>,
    workspace: &mut FilterWorkspace<T>,
) -> Result<DVector<T>, FilterError>
where
    T: RealField + Copy,
{
    let n = x.len();
    let m = h.nrows();
    workspace.prepare_measurement(n, m);

    workspace.hp.gemm(T::one(), h, &*p, T::zero());
    workspace.hp.transpose_to(&mut workspace.pht);
    workspace.innovation_covariance.copy_from(r);
    workspace
        .innovation_covariance
        .gemm(T::one(), h, &workspace.pht, T::one());
    let k = right_solve(&workspace.innovation_covariance, &workspace.pht)?;
    x.gemv(T::one(), &k, &innovation, T::one());

    // P = P - KHP avoids the cubic matrix products of the Joseph form.
    workspace
        .covariance
        .gemm(T::one(), &k, &workspace.hp, T::zero());
    *p -= &workspace.covariance;

    // Roundoff can make the algebraically symmetric covariance slightly asymmetric.
    let half = T::from_f64(0.5).unwrap_or_else(|| T::one() / (T::one() + T::one()));
    for row in 0..n {
        for column in 0..row {
            let average = (p[(row, column)] + p[(column, row)]) * half;
            p[(row, column)] = average;
            p[(column, row)] = average;
        }
    }
    Ok(innovation)
}
