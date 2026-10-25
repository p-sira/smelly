use crate::unscented::SigmaWeights;
use crate::{FilterError, MerweScaledSigmaPoints};
use getset::Getters;
use nalgebra::{Const, DimMin, RealField, SMatrix, SVector};

fn predict<T: RealField + Copy, const N: usize>(
    x: &mut SVector<T, N>,
    p: &mut SMatrix<T, N, N>,
    f: &SMatrix<T, N, N>,
    q: &SMatrix<T, N, N>,
) {
    *x = f * *x;
    *p = f * *p * f.transpose() + q;
}

fn right_solve<T: RealField + Copy, const N: usize, const M: usize>(
    a: &SMatrix<T, M, M>,
    b: &SMatrix<T, N, M>,
) -> Result<SMatrix<T, N, M>, FilterError>
where
    Const<M>: DimMin<Const<M>, Output = Const<M>>,
{
    a.transpose()
        .lu()
        .solve(&b.transpose())
        .map(|x| x.transpose())
        .ok_or(FilterError::SingularInnovation)
}

fn update<T: RealField + Copy, const N: usize, const M: usize>(
    x: &mut SVector<T, N>,
    p: &mut SMatrix<T, N, N>,
    h: &SMatrix<T, M, N>,
    r: &SMatrix<T, M, M>,
    innovation: SVector<T, M>,
) -> Result<SVector<T, M>, FilterError>
where
    Const<M>: DimMin<Const<M>, Output = Const<M>>,
{
    let hp = h * *p;
    let pht = hp.transpose();
    let k = right_solve(&(h * pht + r), &pht)?;
    *x += k * innovation;
    *p -= k * hp;
    let half = T::one() / (T::one() + T::one());
    for row in 0..N {
        for column in 0..row {
            let average = (p[(row, column)] + p[(column, row)]) * half;
            p[(row, column)] = average;
            p[(column, row)] = average;
        }
    }
    Ok(innovation)
}

/// Fixed-size counterpart of [`crate::KalmanFilter`]. All internal storage is inline.
/// `N` is the state dimension and `M` is the measurement dimension.
/// The transition starts as identity and the measurement matrix as zero.
#[derive(Debug, Clone, Getters)]
#[getset(get = "pub")]
pub struct SKalmanFilter<T: RealField, const N: usize, const M: usize> {
    state: SVector<T, N>,
    covariance: SMatrix<T, N, N>,
    process_noise: SMatrix<T, N, N>,
    measurement_noise: SMatrix<T, M, M>,
    transition: SMatrix<T, N, N>,
    measurement: SMatrix<T, M, N>,
}
impl<T: RealField + Copy, const N: usize, const M: usize> SKalmanFilter<T, N, M> {
    /// Creates zero state and identity covariance and noise matrices.
    pub fn new() -> Self {
        Self {
            state: SVector::zeros(),
            covariance: SMatrix::identity(),
            process_noise: SMatrix::identity(),
            measurement_noise: SMatrix::identity(),
            transition: SMatrix::identity(),
            measurement: SMatrix::zeros(),
        }
    }
    /// Replaces the configured state. Dimensions are checked at compile time.
    pub fn set_state(&mut self, value: SVector<T, N>) -> Result<(), FilterError> {
        self.state = value;
        Ok(())
    }
    /// Replaces the configured covariance. Dimensions are checked at compile time.
    pub fn set_covariance(&mut self, value: SMatrix<T, N, N>) -> Result<(), FilterError> {
        self.covariance = value;
        Ok(())
    }
    /// Replaces the configured process noise. Dimensions are checked at compile time.
    pub fn set_process_noise(&mut self, value: SMatrix<T, N, N>) -> Result<(), FilterError> {
        self.process_noise = value;
        Ok(())
    }
    /// Replaces the configured measurement noise. Dimensions are checked at compile time.
    pub fn set_measurement_noise(&mut self, value: SMatrix<T, M, M>) -> Result<(), FilterError> {
        self.measurement_noise = value;
        Ok(())
    }
    /// Replaces the configured transition. Dimensions are checked at compile time.
    pub fn set_transition(&mut self, value: SMatrix<T, N, N>) -> Result<(), FilterError> {
        self.transition = value;
        Ok(())
    }
    /// Replaces the configured measurement. Dimensions are checked at compile time.
    pub fn set_measurement(&mut self, value: SMatrix<T, M, N>) -> Result<(), FilterError> {
        self.measurement = value;
        Ok(())
    }
    /// Propagates the state and covariance through the linear process model.
    pub fn predict(&mut self) -> Result<(), FilterError> {
        predict(
            &mut self.state,
            &mut self.covariance,
            &self.transition,
            &self.process_noise,
        );
        Ok(())
    }
    /// Incorporates a measurement and returns its innovation.
    pub fn update(&mut self, z: &SVector<T, M>) -> Result<SVector<T, M>, FilterError>
    where
        Const<M>: DimMin<Const<M>, Output = Const<M>>,
    {
        let innovation = z - self.measurement * self.state;
        update(
            &mut self.state,
            &mut self.covariance,
            &self.measurement,
            &self.measurement_noise,
            innovation,
        )
    }
}
impl<T: RealField + Copy, const N: usize, const M: usize> Default for SKalmanFilter<T, N, M> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fixed-size counterpart of [`crate::ExtendedKalmanFilter`]. All internal storage is inline.
/// `N` is the state dimension and `M` is the measurement dimension.
/// The transition starts as identity.
#[derive(Debug, Clone, Getters)]
#[getset(get = "pub")]
pub struct SExtendedKalmanFilter<T: RealField, const N: usize, const M: usize> {
    state: SVector<T, N>,
    covariance: SMatrix<T, N, N>,
    process_noise: SMatrix<T, N, N>,
    measurement_noise: SMatrix<T, M, M>,
    transition: SMatrix<T, N, N>,
}
impl<T: RealField + Copy, const N: usize, const M: usize> SExtendedKalmanFilter<T, N, M> {
    /// Creates zero state and identity covariance and noise matrices.
    pub fn new() -> Self {
        Self {
            state: SVector::zeros(),
            covariance: SMatrix::identity(),
            process_noise: SMatrix::identity(),
            measurement_noise: SMatrix::identity(),
            transition: SMatrix::identity(),
        }
    }
    /// Replaces the configured state. Dimensions are checked at compile time.
    pub fn set_state(&mut self, value: SVector<T, N>) -> Result<(), FilterError> {
        self.state = value;
        Ok(())
    }
    /// Replaces the configured covariance. Dimensions are checked at compile time.
    pub fn set_covariance(&mut self, value: SMatrix<T, N, N>) -> Result<(), FilterError> {
        self.covariance = value;
        Ok(())
    }
    /// Replaces the configured process noise. Dimensions are checked at compile time.
    pub fn set_process_noise(&mut self, value: SMatrix<T, N, N>) -> Result<(), FilterError> {
        self.process_noise = value;
        Ok(())
    }
    /// Replaces the configured measurement noise. Dimensions are checked at compile time.
    pub fn set_measurement_noise(&mut self, value: SMatrix<T, M, M>) -> Result<(), FilterError> {
        self.measurement_noise = value;
        Ok(())
    }
    /// Replaces the configured transition. Dimensions are checked at compile time.
    pub fn set_transition(&mut self, value: SMatrix<T, N, N>) -> Result<(), FilterError> {
        self.transition = value;
        Ok(())
    }
    /// Propagates the state and covariance through the linear process model.
    pub fn predict(&mut self) -> Result<(), FilterError> {
        predict(
            &mut self.state,
            &mut self.covariance,
            &self.transition,
            &self.process_noise,
        );
        Ok(())
    }
    /// Incorporates a nonlinear measurement and its Jacobian.
    pub fn update<H, J>(
        &mut self,
        z: &SVector<T, M>,
        measurement: H,
        jacobian: J,
    ) -> Result<SVector<T, M>, FilterError>
    where
        Const<M>: DimMin<Const<M>, Output = Const<M>>,
        H: FnOnce(&SVector<T, N>) -> SVector<T, M>,
        J: FnOnce(&SVector<T, N>) -> SMatrix<T, M, N>,
    {
        let h = jacobian(&self.state);
        let innovation = z - measurement(&self.state);
        update(
            &mut self.state,
            &mut self.covariance,
            &h,
            &self.measurement_noise,
            innovation,
        )
    }
}
impl<T: RealField + Copy, const N: usize, const M: usize> Default
    for SExtendedKalmanFilter<T, N, M>
{
    fn default() -> Self {
        Self::new()
    }
}

/// Fixed-size counterpart of [`crate::UnscentedKalmanFilter`]. All internal storage is inline.
/// `N` is the state dimension and `M` is the measurement dimension.
#[derive(Debug, Clone, Getters)]
#[getset(get = "pub")]
pub struct SUnscentedKalmanFilter<T: RealField, const N: usize, const M: usize> {
    state: SVector<T, N>,
    covariance: SMatrix<T, N, N>,
    process_noise: SMatrix<T, N, N>,
    measurement_noise: SMatrix<T, M, M>,
    #[getset(skip)]
    pub points: MerweScaledSigmaPoints<T>,
    #[getset(skip)]
    predicted_sigmas: Option<SigmaPoints<T, N, N>>,
}
impl<T: RealField + Copy, const N: usize, const M: usize> SUnscentedKalmanFilter<T, N, M> {
    /// Creates zero state and identity covariance and noise matrices.
    pub fn new(points: MerweScaledSigmaPoints<T>) -> Self {
        Self {
            state: SVector::zeros(),
            covariance: SMatrix::identity(),
            process_noise: SMatrix::identity(),
            measurement_noise: SMatrix::identity(),
            points,
            predicted_sigmas: None,
        }
    }
    /// Replaces the configured state. Dimensions are checked at compile time.
    pub fn set_state(&mut self, value: SVector<T, N>) -> Result<(), FilterError> {
        self.state = value;
        Ok(())
    }
    /// Replaces the configured covariance. Dimensions are checked at compile time.
    pub fn set_covariance(&mut self, value: SMatrix<T, N, N>) -> Result<(), FilterError> {
        self.covariance = value;
        Ok(())
    }
    /// Replaces the configured process noise. Dimensions are checked at compile time.
    pub fn set_process_noise(&mut self, value: SMatrix<T, N, N>) -> Result<(), FilterError> {
        self.process_noise = value;
        Ok(())
    }
    /// Replaces the configured measurement noise. Dimensions are checked at compile time.
    pub fn set_measurement_noise(&mut self, value: SMatrix<T, M, M>) -> Result<(), FilterError> {
        self.measurement_noise = value;
        Ok(())
    }
    /// Propagates sigma points, adds process noise, and regenerates sigma points.
    pub fn predict<F>(&mut self, process: F) -> Result<(), FilterError>
    where
        F: Fn(&SVector<T, N>) -> SVector<T, N>,
    {
        let weights = self.points.weights(N)?;
        let sigmas = sigma_points(&self.state, &self.covariance, &weights)?.map(process);
        let (state, covariance) = transform(&sigmas, &weights, &self.process_noise);
        let predicted = sigma_points(&state, &covariance, &weights)?;
        self.state = state;
        self.covariance = covariance;
        self.predicted_sigmas = Some(predicted);
        Ok(())
    }
    /// Incorporates a measurement. A successful update consumes the prediction.
    pub fn update<H>(
        &mut self,
        z: &SVector<T, M>,
        measurement: H,
    ) -> Result<SVector<T, M>, FilterError>
    where
        Const<M>: DimMin<Const<M>, Output = Const<M>>,
        H: Fn(&SVector<T, N>) -> SVector<T, M>,
    {
        let sigmas = self
            .predicted_sigmas
            .as_ref()
            .ok_or(FilterError::PredictRequired)?;
        let measured = sigmas.map(measurement);
        let weights = self.points.weights(N)?;
        let (mean, covariance) = transform(&measured, &weights, &self.measurement_noise);
        let mut cross = SMatrix::<T, N, M>::zeros();
        for (i, (x, z)) in sigmas.iter().zip(measured.iter()).enumerate() {
            cross += (x - self.state) * (z - mean).transpose() * weights.covariance(i);
        }
        let k = right_solve(&covariance, &cross)?;
        let innovation = z - mean;
        self.state += k * innovation;
        self.covariance -= k * covariance * k.transpose();
        self.predicted_sigmas = None;
        Ok(innovation)
    }
}

// Split the 2*N+1 points to avoid unstable generic const expressions.
#[derive(Debug, Clone)]
struct SigmaPoints<T: RealField, const D: usize, const N: usize> {
    center: SVector<T, D>,
    positive: [SVector<T, D>; N],
    negative: [SVector<T, D>; N],
}
impl<T: RealField + Copy, const D: usize, const N: usize> SigmaPoints<T, D, N> {
    fn iter(&self) -> impl Iterator<Item = &SVector<T, D>> {
        core::iter::once(&self.center)
            .chain(&self.positive)
            .chain(&self.negative)
    }
    fn map<const M: usize>(
        &self,
        f: impl Fn(&SVector<T, D>) -> SVector<T, M>,
    ) -> SigmaPoints<T, M, N> {
        SigmaPoints {
            center: f(&self.center),
            positive: core::array::from_fn(|i| f(&self.positive[i])),
            negative: core::array::from_fn(|i| f(&self.negative[i])),
        }
    }
}
fn sigma_points<T: RealField + Copy, const N: usize>(
    state: &SVector<T, N>,
    covariance: &SMatrix<T, N, N>,
    weights: &SigmaWeights<T>,
) -> Result<SigmaPoints<T, N, N>, FilterError> {
    let lower = (*covariance * weights.scale)
        .cholesky()
        .ok_or(FilterError::NonPositiveDefiniteCovariance)?
        .unpack();
    Ok(SigmaPoints {
        center: *state,
        positive: core::array::from_fn(|i| state + lower.column(i)),
        negative: core::array::from_fn(|i| state - lower.column(i)),
    })
}
fn transform<T: RealField + Copy, const D: usize, const N: usize>(
    points: &SigmaPoints<T, D, N>,
    weights: &SigmaWeights<T>,
    noise: &SMatrix<T, D, D>,
) -> (SVector<T, D>, SMatrix<T, D, D>) {
    let mut mean = SVector::<T, D>::zeros();
    for (i, point) in points.iter().enumerate() {
        mean += point * weights.mean(i);
    }
    let mut covariance = *noise;
    for (i, point) in points.iter().enumerate() {
        let delta = point - mean;
        covariance += delta * delta.transpose() * weights.covariance(i);
    }
    (mean, covariance)
}
