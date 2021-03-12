use thiserror::Error;
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug, Clone)]
pub enum Error {
    #[error("wrong measure ({0}) for scaler: {1}")]
    WrongMeasureForScaler(String, String),
    #[error("subsamples greater than total samples: {0} > {1}")]
    TooManySubsamples(usize, usize),
    #[error("not enough samples")]
    NotEnoughSamples,
    #[error("not a valid float")]
    InvalidFloat,
    #[error("n_gram boundaries cannot be zero (min = {0}, max = {1})")]
    InvalidNGramBoundaries(usize, usize),
    #[error("n_gram min boundary cannot be greater than max boundary (min = {0}, max = {1})")]
    FlippedNGramBoundaries(usize, usize),
}
