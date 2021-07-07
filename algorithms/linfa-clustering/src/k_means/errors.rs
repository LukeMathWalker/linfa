use thiserror::Error;

/// An error when fitting with an invalid hyperparameter
#[derive(Error, Debug)]
pub enum KMeansParamsError {
    #[error("n_clusters cannot be 0")]
    NClusters,
    #[error("n_runs cannot be 0")]
    NRuns,
    #[error("tolerance must be greater than 0")]
    Tolerance,
    #[error("max_n_iterations cannot be 0")]
    MaxIterations,
}

/// An error when modeling a KMeans algorithm
#[derive(Error, Debug)]
pub enum KMeansError {
    /// When any of the hyperparameters are set the wrong value
    #[error("Invalid hyperparameter: {0}")]
    InvalidParams(#[from] KMeansParamsError),
    /// When inertia computation fails
    #[error("Fitting failed: No inertia improvement (-inf)")]
    InertiaError,
    /// When fitting algorithm does not converge
    #[error("Fitting failed: Did not converge. Try different init parameters or check for degenerate data.")]
    NotConverged,
    #[error(transparent)]
    LinfaError(#[from] linfa::error::Error),
}

#[derive(Error, Debug)]
pub enum IncrKMeansError<M: std::fmt::Debug> {
    /// When any of the hyperparameters are set the wrong value
    #[error("Invalid hyperparameter: {0}")]
    InvalidParams(#[from] KMeansParamsError),
    /// When the distance between the old and new centroids exceeds the tolerance parameter. Not an
    /// actual error, just there to signal that the algorithm should keep running.
    #[error("Algorithm has not yet converged, Keep on running the algorithm.")]
    NotConverged(M),
    #[error(transparent)]
    LinfaError(#[from] linfa::error::Error),
}
