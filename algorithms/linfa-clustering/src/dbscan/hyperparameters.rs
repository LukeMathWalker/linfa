use linfa::{prelude::*, Float};
use linfa_nn::{distance::Distance, NearestNeighbour};
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};
use thiserror::Error;

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug)]
/// The set of hyperparameters that can be specified for the execution of
/// the [DBSCAN algorithm](struct.Dbscan.html).
pub struct DbscanHyperParams<F: Float, D: Distance<F>, N: NearestNeighbour> {
    pub(crate) tolerance: F,
    pub(crate) min_points: usize,
    pub(crate) dist_fn: D,
    pub(crate) nn_algo: N,
}

#[derive(Debug)]
/// Helper struct for building a set of [DBSCAN hyperparameters](struct.DbscanHyperParams.html)
pub struct UncheckedDbscanHyperParams<F: Float, D: Distance<F>, N: NearestNeighbour>(
    DbscanHyperParams<F, D, N>,
);

#[derive(Error, Debug)]
pub enum DbscanParamsError {
    #[error("min_points must be greater than 1")]
    MinPoints,
    #[error("tolerance must be greater than 0")]
    Tolerance,
}

impl<F: Float, D: Distance<F>, N: NearestNeighbour> UncheckedDbscanHyperParams<F, D, N> {
    pub(crate) fn new(min_points: usize, dist_fn: D, nn_algo: N) -> Self {
        Self(DbscanHyperParams {
            min_points,
            tolerance: F::cast(1e-4),
            dist_fn,
            nn_algo,
        })
    }

    /// Set the tolerance
    pub fn tolerance(mut self, tolerance: F) -> Self {
        self.0.tolerance = tolerance;
        self
    }

    /// Set the nearest neighbour algorithm to be used
    pub fn nn_algo(mut self, nn_algo: N) -> Self {
        self.0.nn_algo = nn_algo;
        self
    }

    /// Set the distance metric
    pub fn dist_fn(mut self, dist_fn: D) -> Self {
        self.0.dist_fn = dist_fn;
        self
    }
}

impl<F: Float, D: Distance<F>, N: NearestNeighbour> UncheckedHyperParams
    for UncheckedDbscanHyperParams<F, D, N>
{
    type Checked = DbscanHyperParams<F, D, N>;
    type Error = DbscanParamsError;

    fn check_ref(&self) -> Result<&Self::Checked, Self::Error> {
        if self.0.min_points <= 1 {
            Err(DbscanParamsError::MinPoints)
        } else if self.0.tolerance <= F::zero() {
            Err(DbscanParamsError::Tolerance)
        } else {
            Ok(&self.0)
        }
    }

    fn check(self) -> Result<Self::Checked, Self::Error> {
        self.check_ref()?;
        Ok(self.0)
    }
}

impl<F: Float, D: Distance<F>, N: NearestNeighbour> DbscanHyperParams<F, D, N> {
    /// Nearest neighbour algorithm used for range queries
    pub fn tolerance(&self) -> F {
        self.tolerance
    }

    /// Minimum number of neighboring points a point needs to have to be a core                                                                                                
    /// point and not a noise point.
    pub fn minimum_points(&self) -> usize {
        self.min_points
    }

    /// Distance metric used in the DBSCAN calculation
    pub fn dist_fn(&self) -> &D {
        &self.dist_fn
    }

    /// Nearest neighbour algorithm used for range queries
    pub fn nn_algo(&self) -> &N {
        &self.nn_algo
    }
}

#[cfg(test)]
mod tests {
    use linfa_nn::{distance::L2Dist, CommonNearestNeighbour};

    use super::*;

    #[test]
    fn tolerance_cannot_be_zero() {
        let res = UncheckedDbscanHyperParams::new(2, L2Dist, CommonNearestNeighbour::KdTree)
            .tolerance(0.0)
            .check();
        assert!(matches!(res, Err(DbscanParamsError::Tolerance)));
    }

    #[test]
    fn min_points_at_least_2() {
        let res = UncheckedDbscanHyperParams::new(1, L2Dist, CommonNearestNeighbour::KdTree)
            .tolerance(3.3)
            .check();
        assert!(matches!(res, Err(DbscanParamsError::MinPoints)));
    }
}
