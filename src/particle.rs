use getset::Getters;
use nalgebra::{
    Const, DefaultAllocator, Dim, Dyn, OMatrix, OVector, RealField, allocator::Allocator,
};

use crate::FilterError;

/// Sequential Monte Carlo filter with contiguous, column-oriented particles.
///
/// Callers supply initial samples, process noise in [`Self::predict`], and a
/// uniform random value for systematic resampling. No RNG or distribution is
/// imposed, so seeded simulations and `no_std` applications use the same API.
#[derive(Debug, Clone, Getters)]
pub struct ParticleFilter<T: RealField = f64, R: Dim = Dyn, C: Dim = Dyn>
where
    DefaultAllocator: Allocator<R, C> + Allocator<C> + Allocator<R> + Allocator<R, R>,
{
    /// State samples, one per column.
    #[getset(get = "pub")]
    particles: OMatrix<T, R, C>,
    weights: OVector<T, C>,
    log_weights: OVector<T, C>,
    particle_workspace: OMatrix<T, R, C>,
    weight_workspace: OVector<T, C>,
}

/// Fixed-size particle filter with inline state samples and weight buffers.
/// `N` is the state dimension and `P` is the particle count.
pub type SParticleFilter<T, const N: usize, const P: usize> = ParticleFilter<T, Const<N>, Const<P>>;

impl<T: RealField + Copy, R: Dim, C: Dim> ParticleFilter<T, R, C>
where
    DefaultAllocator: Allocator<R, C> + Allocator<C> + Allocator<R> + Allocator<R, R>,
{
    /// Creates uniformly weighted particles. Each column is one state sample.
    /// Both dimensions must be nonzero and every component must be finite.
    pub fn new(particles: OMatrix<T, R, C>) -> Result<Self, FilterError> {
        if particles.nrows() == 0 || particles.ncols() == 0 {
            return Err(FilterError::InvalidParticles);
        }
        if !particles.iter().all(|x| x.is_finite()) {
            return Err(FilterError::InvalidParticles);
        }
        let (rows, columns) = particles.shape_generic();
        let count = particles.ncols();
        let weight = T::one() / T::from_usize(count).ok_or(FilterError::InvalidParticles)?;
        Ok(Self {
            particle_workspace: OMatrix::zeros_generic(rows, columns),
            particles,
            weights: OVector::from_element_generic(columns, Const::<1>, weight),
            log_weights: OVector::from_element_generic(columns, Const::<1>, weight.ln()),
            weight_workspace: OVector::zeros_generic(columns, Const::<1>),
        })
    }

    /// Normalized posterior weights.
    pub fn weights(&self) -> &[T] {
        self.weights.as_slice()
    }

    /// Propagates each sample into a reusable output slice of the same length.
    /// The output starts as a copy of the input; add process noise here as needed.
    /// Nonfinite output leaves the entire particle set unchanged.
    pub fn predict<F>(&mut self, mut process: F) -> Result<(), FilterError>
    where
        F: FnMut(&[T], &mut [T]),
    {
        let dim = self.particles.nrows();
        for (input, output) in self
            .particles
            .as_slice()
            .chunks_exact(dim)
            .zip(self.particle_workspace.as_mut_slice().chunks_exact_mut(dim))
        {
            output.copy_from_slice(input);
            process(input, output);
            if !output.iter().all(|x| x.is_finite()) {
                return Err(FilterError::InvalidParticles);
            }
        }
        core::mem::swap(&mut self.particles, &mut self.particle_workspace);
        Ok(())
    }

    /// Multiplies weights by finite, nonnegative likelihoods and normalizes.
    /// All-zero likelihoods return an error without changing the posterior.
    /// Use [`Self::update_log_likelihood`] for very small likelihoods.
    pub fn update<F>(&mut self, mut likelihood: F) -> Result<(), FilterError>
    where
        F: FnMut(&[T]) -> T,
    {
        let negative_infinity = T::zero().ln();
        // A conservative normal-range threshold, available for any RealField.
        let minimum = T::max_value()
            .map(|maximum| T::one() / maximum.sqrt())
            .unwrap_or_else(T::one);
        let mut maximum = negative_infinity;
        let mut sum = T::zero();
        let mut reliable = true;
        for ((((state, prior), log_prior), log), weight) in self
            .particles
            .as_slice()
            .chunks_exact(self.particles.nrows())
            .zip(self.weights.as_slice().iter())
            .zip(self.log_weights.as_slice().iter())
            .zip(self.weight_workspace.as_mut_slice().iter_mut())
            .zip(self.particle_workspace.as_mut_slice())
        {
            let value = likelihood(state);
            if value < T::zero() || !value.is_finite() {
                return Err(FilterError::InvalidParticleWeights);
            }
            *log = *log_prior + value.ln();
            maximum = maximum.max(*log);
            *weight = *prior * value;
            sum += *weight;
            // Subnormal products or priors can lose relative precision. Retained
            // log weights also allow a previously underflowed sample to recover.
            reliable &= (*prior >= minimum && *weight >= minimum)
                || value == T::zero()
                || *log_prior == negative_infinity;
        }
        self.normalize_weights(
            maximum,
            (reliable && sum.is_finite() && sum > T::zero()).then_some(sum),
        )
    }

    /// Adds log likelihoods using log-sum-exp normalization.
    /// Negative infinity excludes a sample; NaN, positive infinity, and an
    /// impossible observation return an error without changing the posterior.
    /// Resampling is explicit, allowing estimates before resampling adds noise.
    pub fn update_log_likelihood<F>(&mut self, mut log_likelihood: F) -> Result<(), FilterError>
    where
        F: FnMut(&[T]) -> T,
    {
        let negative_infinity = T::zero().ln();
        let mut maximum = negative_infinity;
        let dim = self.particles.nrows();
        for ((state, prior), output) in self
            .particles
            .as_slice()
            .chunks_exact(dim)
            .zip(self.log_weights.as_slice().iter())
            .zip(self.weight_workspace.as_mut_slice().iter_mut())
        {
            let likelihood = log_likelihood(state);
            if !likelihood.is_finite() && likelihood != negative_infinity {
                return Err(FilterError::InvalidParticleWeights);
            }
            *output = *prior + likelihood;
            if *output > maximum {
                maximum = *output;
            }
        }
        self.normalize_weights(maximum, None)
    }

    fn normalize_weights(
        &mut self,
        mut maximum: T,
        linear_sum: Option<T>,
    ) -> Result<(), FilterError> {
        if !maximum.is_finite() {
            return Err(FilterError::InvalidParticleWeights);
        }
        let sum = if let Some(sum) = linear_sum {
            maximum = T::zero();
            sum
        } else {
            let mut sum = T::zero();
            // Reuse the inactive particle buffer for candidate linear weights.
            for (log, weight) in self
                .weight_workspace
                .iter()
                .zip(self.particle_workspace.as_mut_slice())
            {
                *weight = (*log - maximum).exp();
                sum += *weight;
            }
            sum
        };
        let log_sum = sum.ln();
        let inverse_sum = T::one() / sum;
        for ((log, candidate), (weight, unnormalized)) in self
            .log_weights
            .as_mut_slice()
            .iter_mut()
            .zip(self.weight_workspace.as_slice().iter())
            .zip(
                self.weights
                    .as_mut_slice()
                    .iter_mut()
                    .zip(self.particle_workspace.as_slice()),
            )
        {
            *log = (*candidate - maximum) - log_sum;
            *weight = *unnormalized * inverse_sum;
        }
        Ok(())
    }

    /// Effective sample size, in the range `1..=particle_count` up to roundoff.
    pub fn effective_sample_size(&self) -> T {
        T::one()
            / self
                .weights
                .as_slice()
                .iter()
                .fold(T::zero(), |sum, w| sum + *w * *w)
    }

    /// Systematic resampling in linear time with no allocations.
    /// `offset` must be sampled uniformly from `[0, 1)` for unbiased resampling.
    /// Weights are reset to uniform. Invalid offsets leave the filter unchanged.
    pub fn resample(&mut self, offset: T) -> Result<(), FilterError> {
        if !offset.is_finite() || offset < T::zero() || offset >= T::one() {
            return Err(FilterError::InvalidResamplingOffset);
        }
        let count = self.weights.len();
        let step = T::one() / T::from_usize(count).unwrap();
        let mut source = 0;
        let mut cumulative = self.weights[0];
        for destination in 0..count {
            let position = (T::from_usize(destination).unwrap() + offset) * step;
            while source < count - 1 && position >= cumulative {
                source += 1;
                cumulative += self.weights[source];
            }
            self.particle_workspace
                .column_mut(destination)
                .copy_from(&self.particles.column(source));
        }
        core::mem::swap(&mut self.particles, &mut self.particle_workspace);
        self.weights.as_mut_slice().fill(step);
        self.log_weights.as_mut_slice().fill(step.ln());
        Ok(())
    }

    /// Weighted posterior mean. Dynamic dimensions allocate one state vector.
    pub fn state(&self) -> OVector<T, R> {
        let mut mean = OVector::zeros_generic(self.particles.shape_generic().0, Const::<1>);
        for (state, weight) in self
            .particles
            .column_iter()
            .zip(self.weights.as_slice().iter())
        {
            mean.axpy(*weight, &state, T::one());
        }
        mean
    }

    /// Weighted population covariance (no sample-size correction).
    pub fn covariance(&self) -> OMatrix<T, R, R> {
        let mean = self.state();
        let mut covariance = OMatrix::zeros_generic(
            self.particles.shape_generic().0,
            self.particles.shape_generic().0,
        );
        let mut delta = OVector::zeros_generic(self.particles.shape_generic().0, Const::<1>);
        for (state, weight) in self
            .particles
            .column_iter()
            .zip(self.weights.as_slice().iter())
        {
            state.sub_to(&mean, &mut delta);
            covariance.ger(*weight, &delta, &delta, T::one());
        }
        covariance
    }
}
