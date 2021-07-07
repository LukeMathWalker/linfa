//! Linfa prelude.
//!
//! This module contains the most used types, type aliases, traits and
//! functions that you can import easily as a group.
//!

#[doc(no_inline)]
pub use crate::error::Error;

#[doc(no_inline)]
pub use crate::traits::*;

#[doc(no_inline)]
pub use crate::dataset::{AsTargets, Dataset, DatasetBase, DatasetView, Float, Pr, Records};

#[doc(no_inline)]
pub use crate::metrics_classification::{BinaryClassification, ConfusionMatrix, ToConfusionMatrix};

#[doc(no_inline)]
pub use crate::metrics_regression::{MultiTargetRegression, SingleTargetRegression};

#[doc(no_inline)]
pub use crate::metrics_clustering::SilhouetteScore;

#[doc(no_inline)]
pub use crate::correlation::PearsonCorrelation;

#[doc(no_inline)]
pub use crate::hyperparams::UncheckedHyperParams;
