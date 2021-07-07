//! Provide traits for different classes of algorithms
//!

use crate::dataset::{DatasetBase, Records};
use std::convert::From;

/// Transformation algorithms
///
/// A transformer takes a dataset and transforms it into a different one. It has no concept of
/// state and provides therefore no method to predict new data. A typical example are kernel
/// methods.
///
/// It should be implemented for all algorithms, also for those which can be fitted.
///
pub trait Transformer<R: Records, T> {
    fn transform(&self, x: R) -> T;
}

/// Fittable algorithms
///
/// A fittable algorithm takes a dataset and creates a concept of some kind about it. For example
/// in *KMeans* this would be the mean values for each class, or in *SVM* the separating
/// hyperplane. It returns a model, which can be used to predict targets for new data.
pub trait Fit<R: Records, T, E: std::error::Error + From<crate::error::Error>> {
    type Object;

    fn fit(&self, dataset: &DatasetBase<R, T>) -> Result<Self::Object, E>;
}

/// Incremental algorithms
///
/// An incremental algorithm takes a former model and dataset and returns a new model with updated
/// parameters. If the former model is `None`, then the function acts like `Fit::fit` and
/// initializes the model first.
pub trait IncrementalFit<'a, R: Records, T, E: std::error::Error + From<crate::error::Error>> {
    type ObjectIn: 'a;
    type ObjectOut: 'a;

    fn fit_with(
        &self,
        model: Self::ObjectIn,
        dataset: &'a DatasetBase<R, T>,
    ) -> Result<Self::ObjectOut, E>;
}

/// Predict with model
///
/// This trait assumes the `PredictRef` implementation and provides additional input/output
/// combinations.
///
/// # Provided implementation
///
/// * Array2 -> Dataset
/// * Dataset -> Dataset
/// * &Dataset -> Array2
pub trait Predict<R: Records, T> {
    fn predict(&self, x: R) -> T;
}

/// Predict with model for reference of dataset
pub trait PredictRef<R: Records, T> {
    fn predict_ref<'a>(&'a self, x: &'a R) -> T;
}
