use alloc::string::String;
use core::{error::Error, fmt};

/// Errors returned when filter inputs cannot produce a valid update.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterError {
    /// A vector or matrix has an incompatible shape.
    Dimension {
        /// Name of the invalid input.
        name: &'static str,
        /// Required vector length or matrix shape.
        expected: String,
        /// Supplied vector length or matrix shape.
        actual: String,
    },
    /// Particle samples must have nonzero dimensions and finite components.
    InvalidParticles,
    /// Likelihoods must define a finite, nonzero posterior probability.
    InvalidParticleWeights,
    /// A systematic resampling offset must lie in `[0, 1)`.
    InvalidResamplingOffset,
    /// The innovation covariance cannot be solved.
    SingularInnovation,
    /// Sigma points require a positive-definite covariance matrix.
    NonPositiveDefiniteCovariance,
    /// A sigma-point scaling parameter is invalid.
    InvalidSigmaPointScaling,
    /// An unscented update was attempted without a preceding prediction.
    PredictRequired,
}

impl fmt::Display for FilterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dimension {
                name,
                expected,
                actual,
            } => {
                write!(f, "{name} has shape {actual}; expected {expected}")
            }
            Self::InvalidParticles => write!(f, "particles must be nonempty and finite"),
            Self::InvalidParticleWeights => write!(f, "invalid or all-zero particle likelihoods"),
            Self::InvalidResamplingOffset => write!(f, "resampling offset must be in [0, 1)"),
            Self::SingularInnovation => write!(f, "innovation covariance is singular"),
            Self::NonPositiveDefiniteCovariance => {
                write!(f, "covariance matrix is not positive definite")
            }
            Self::InvalidSigmaPointScaling => write!(f, "sigma-point scaling must be positive"),
            Self::PredictRequired => write!(f, "predict must be called before update"),
        }
    }
}

impl Error for FilterError {}
