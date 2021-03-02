//! `linfa` aims to provide a comprehensive toolkit to build Machine Learning applications
//! with Rust.
//!
//! Kin in spirit to Python's `scikit-learn`, it focuses on common preprocessing tasks
//! and classical ML algorithms for your everyday ML tasks.
//!
//! ## Current state
//!
//! Such bold ambitions! Where are we now? [Are we learning yet?](http://www.arewelearningyet.com/)
//!
//! linfa aims to provide a comprehensive toolkit to build Machine Learning applications with Rust.
//!
//! Kin in spirit to Python's scikit-learn, it focuses on common preprocessing tasks and classical ML algorithms for your everyday ML tasks.
//!
//! ## Current state
//!
//! Where does `linfa` stand right now? [Are we learning yet?](http://www.arewelearningyet.com/)
//!
//! `linfa` currently provides sub-packages with the following algorithms:
//!
//!
//! | Name | Purpose | Status | Category |  Notes |
//! | :--- | :--- | :---| :--- | :---|
//! | [clustering](linfa-clustering/) | Data clustering | Tested / Benchmarked  | Unsupervised learning | Clustering of unlabeled data; contains K-Means, Gaussian-Mixture-Model and DBSCAN  |
//! | [kernel](linfa-kernel/) | Kernel methods for data transformation  | Tested  | Pre-processing | Maps feature vector into higher-dimensional space|
//! | [linear](linfa-linear/) | Linear regression | Tested  | Partial fit | Contains Ordinary Least Squares (OLS), Generalized Linear Models (GLM) |
//! | [elasticnet](linfa-elasticnet/) | Elastic Net | Tested | Supervised learning | Linear regression with elastic net constraints |
//! | [logistic](linfa-logistic/) | Logistic regression | Tested  | Partial fit | Builds two-class logistic regression models
//! | [reduction](linfa-reduction/) | Dimensionality reduction | Tested  | Pre-processing | Diffusion mapping and Principal Component Analysis (PCA) |
//! | [trees](linfa-trees/) | Decision trees | Experimental  | Supervised learning | Linear decision trees
//! | [svm](linfa-svm/) | Support Vector Machines | Tested  | Supervised learning | Classification or regression analysis of labeled datasets |
//! | [hierarchical](linfa-hierarchical/) | Agglomerative hierarchical clustering | Tested | Unsupervised learning | Cluster and build hierarchy of clusters |
//! | [bayes](linfa-bayes/) | Naive Bayes | Tested | Supervised learning | Contains Gaussian Naive Bayes |
//! | [ica](linfa-ica/) | Independent component analysis | Tested | Unsupervised learning | Contains FastICA implementation |
//!
//! We believe that only a significant community effort can nurture, build, and sustain a machine learning ecosystem in Rust - there is no other way forward.
//!
//! If this strikes a chord with you, please take a look at the [roadmap](https://github.com/rust-ml/linfa/issues/7) and get involved!
//!

pub mod correlation;
pub mod dataset;
pub mod error;
mod metrics_classification;
mod metrics_regression;
pub mod prelude;
pub mod traits;

pub use dataset::{
    multi_target_model::MultiTargetModel, Dataset, DatasetBase, DatasetPr, DatasetView, Float,
    Label,
};
pub use error::Error;

#[cfg(feature = "ndarray-linalg")]
pub use ndarray_linalg as linalg;

#[cfg(any(feature = "intel-mkl-system", feature = "intel-mkl-static"))]
extern crate intel_mkl_src;

#[cfg(any(feature = "openblas-system", feature = "openblas-static"))]
extern crate openblas_src;

#[cfg(any(feature = "netblas-system", feature = "netblas-static"))]
extern crate netblas_src;

/// Common metrics functions for classification and regression
pub mod metrics {
    pub use crate::metrics_classification::{
        BinaryClassification, ConfusionMatrix, ReceiverOperatingCharacteristic, ToConfusionMatrix,
    };
    pub use crate::metrics_regression::Regression;
}
