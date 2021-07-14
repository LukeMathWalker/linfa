use thiserror::Error;
pub type Result<T> = std::result::Result<T, OpticsError>;

/// An error when modeling a GMM algorithm
#[derive(Error, Debug)]
pub enum OpticsError {
    /// When any of the hyperparameters are set the wrong value
    #[error("Invalid value encountered: {0}")]
    InvalidValue(String),
}
