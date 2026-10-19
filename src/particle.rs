use alloc::vec::Vec;
use nalgebra::{DMatrix, DVector, RealField};

use crate::FilterError;

/// Sequential Monte Carlo filter with contiguous, column-oriented particles.
///
/// Callers supply initial samples, process noise in [`Self::predict`], and a
/// uniform random value for systematic resampling. No RNG or distribution is
/// imposed, so seeded simulations and `no_std` applications use the same API.
#[derive(Debug, Clone)]
pub struct ParticleFilter<T: RealField = f64> {
    particles: DMatrix<T>,
    weights: Vec<T>,
    log_weights: Vec<T>,
    particle_workspace: DMatrix<T>,
    weight_workspace: Vec<T>,
}

impl<T: RealField + Copy> ParticleFilter<T> {
    /// Creates uniformly weighted particles. Each column is one state sample.
    /// Both dimensions must be nonzero and every component must be finite.
    pub fn new(particles: DMatrix<T>) -> Result<Self, FilterError> {
        if particles.nrows() == 0 || particles.ncols() == 0 {
            return Err(FilterError::InvalidParticles);
        }
        if !particles.iter().all(|x| x.is_finite()) {
            return Err(FilterError::InvalidParticles);
        }
        let count = particles.ncols();
        let weight = T::one() / T::from_usize(count).ok_or(FilterError::InvalidParticles)?;
        Ok(Self {
            particle_workspace: DMatrix::zeros(particles.nrows(), count),
            particles,
            weights: alloc::vec![weight; count],
            log_weights: alloc::vec![weight.ln(); count],
            weight_workspace: alloc::vec![T::zero(); count],
        })
    }

    /// State samples, one per column.
    pub fn particles(&self) -> &DMatrix<T> {
        &self.particles
    }

    /// Normalized posterior weights.
    pub fn weights(&self) -> &[T] {
        &self.weights
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
        self.update_weights(|state| {
            let value = likelihood(state);
            if value < T::zero() || !value.is_finite() {
                Err(FilterError::InvalidParticleWeights)
            } else {
                Ok(value.ln())
            }
        })
    }

    /// Adds log likelihoods using log-sum-exp normalization.
    /// Negative infinity excludes a sample; NaN, positive infinity, and an
    /// impossible observation return an error without changing the posterior.
    /// Resampling is explicit, allowing estimates before resampling adds noise.
    pub fn update_log_likelihood<F>(&mut self, mut log_likelihood: F) -> Result<(), FilterError>
    where
        F: FnMut(&[T]) -> T,
    {
        self.update_weights(|state| Ok(log_likelihood(state)))
    }

    fn update_weights<F>(&mut self, mut log_likelihood: F) -> Result<(), FilterError>
    where
        F: FnMut(&[T]) -> Result<T, FilterError>,
    {
        let negative_infinity = T::zero().ln();
        let mut maximum = negative_infinity;
        let dim = self.particles.nrows();
        for ((state, prior), output) in self
            .particles
            .as_slice()
            .chunks_exact(dim)
            .zip(&self.log_weights)
            .zip(&mut self.weight_workspace)
        {
            let likelihood = log_likelihood(state)?;
            if !likelihood.is_finite() && likelihood != negative_infinity {
                return Err(FilterError::InvalidParticleWeights);
            }
            *output = *prior + likelihood;
            if *output > maximum {
                maximum = *output;
            }
        }
        if !maximum.is_finite() {
            return Err(FilterError::InvalidParticleWeights);
        }
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
        let log_sum = sum.ln();
        for ((log, candidate), (weight, unnormalized)) in
            self.log_weights.iter_mut().zip(&self.weight_workspace).zip(
                self.weights
                    .iter_mut()
                    .zip(self.particle_workspace.as_slice()),
            )
        {
            *log = (*candidate - maximum) - log_sum;
            *weight = *unnormalized / sum;
        }
        Ok(())
    }

    /// Effective sample size, in the range `1..=particle_count` up to roundoff.
    pub fn effective_sample_size(&self) -> T {
        T::one() / self.weights.iter().fold(T::zero(), |sum, w| sum + *w * *w)
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
        self.weights.fill(step);
        self.log_weights.fill(step.ln());
        Ok(())
    }

    /// Weighted posterior mean. Allocates one state vector.
    pub fn state(&self) -> DVector<T> {
        let mut mean = DVector::zeros(self.particles.nrows());
        for (state, weight) in self.particles.column_iter().zip(&self.weights) {
            mean.axpy(*weight, &state, T::one());
        }
        mean
    }

    /// Weighted population covariance (no sample-size correction).
    pub fn covariance(&self) -> DMatrix<T> {
        let mean = self.state();
        let mut covariance = DMatrix::zeros(mean.len(), mean.len());
        let mut delta = DVector::zeros(mean.len());
        for (state, weight) in self.particles.column_iter().zip(&self.weights) {
            state.sub_to(&mean, &mut delta);
            covariance.ger(*weight, &delta, &delta, T::one());
        }
        covariance
    }
}
