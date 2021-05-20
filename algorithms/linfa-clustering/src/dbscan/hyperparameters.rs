use linfa::Float;
use linfa_nn::{distance::Distance, NearestNeighbour};
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug)]
/// The set of hyperparameters that can be specified for the execution of
/// the [DBSCAN algorithm](struct.Dbscan.html).
pub struct DbscanHyperParams<F: Float, D: Distance<F>, N: NearestNeighbour> {
    /// Distance between points for them to be considered neighbours.
    tolerance: F,
    /// Minimum number of neighboring points a point needs to have to be a core
    /// point and not a noise point.
    min_points: usize,
    /// Distance metric used in the DBSCAN calculation
    dist_fn: D,
    /// Nearest neighbour algorithm used for range queries
    nn_impl: N,
}

/// Helper struct used to construct a set of hyperparameters for [DBSCAN algorithm](struct.Dbscan.html).
pub struct DbscanHyperParamsBuilder<F: Float, D: Distance<F>, N: NearestNeighbour> {
    pub(crate) tolerance: F,
    pub(crate) min_points: usize,
    pub(crate) dist_fn: D,
    pub(crate) nn_impl: N,
}

impl<F: Float, D: Distance<F>, N: NearestNeighbour> DbscanHyperParamsBuilder<F, D, N> {
    /// Distance between points for them to be considered neighbours.
    pub fn tolerance(mut self, tolerance: F) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Nearest neighbour algorithm used for range queries
    pub fn nn_impl(mut self, nn_impl: N) -> Self {
        self.nn_impl = nn_impl;
        self
    }

    /// Distance metric used in the DBSCAN calculation
    pub fn dist_fn(mut self, dist_fn: D) -> Self {
        self.dist_fn = dist_fn;
        self
    }

    /// Return an instance of `DbscanHyperParams` after having performed
    /// validation checks on all hyperparameters.
    ///
    /// **Panics** if any of the validation checks fail.
    pub fn build(self) -> DbscanHyperParams<F, D, N> {
        if self.tolerance <= F::zero() {
            panic!("`tolerance` must be greater than 0!");
        }
        // There is always at least one neighbor to a point (itself)
        if self.min_points <= 1 {
            panic!("`min_points` must be greater than 1!");
        }
        DbscanHyperParams {
            tolerance: self.tolerance,
            min_points: self.min_points,
            nn_impl: self.nn_impl,
            dist_fn: self.dist_fn,
        }
    }
}

impl<F: Float, D: Distance<F>, N: NearestNeighbour> DbscanHyperParams<F, D, N> {
    /// Two points are considered neighbors if the euclidean distance between
    /// them is below the tolerance
    pub fn tolerance(&self) -> F {
        self.tolerance
    }

    /// Minimum number of a points in a neighborhood around a point for it to
    /// not be considered noise
    pub fn minimum_points(&self) -> usize {
        self.min_points
    }

    /// Distance metric used in the DBSCAN calculation
    pub fn dist_fn(&self) -> &D {
        &self.dist_fn
    }

    /// Nearest neighbour algorithm used for range queries
    pub fn nn_impl(&self) -> &N {
        &self.nn_impl
    }
}
