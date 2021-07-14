use crate::optics::errors::{OpticsError, Result};
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
/// The set of hyperparameters that can be specified for the execution of
/// the [OPTICS algorithm](struct.Optics.html).
pub struct OpticsHyperParams {
    /// Distance between points for them to be considered neighbours.
    tolerance: f64,
    /// Minimum number of neighboring points a point needs to have to be a core
    /// point and not a noise point.
    min_points: usize,
}

impl OpticsHyperParams {
    /// Minimum number of neighboring points a point needs to have to be a core
    /// point and not a noise point.
    ///
    /// Defaults are provided if the optional parameters are not specified:
    /// * `tolerance = f64::MAX`
    pub fn new(min_points: usize) -> Self {
        OpticsHyperParams {
            min_points,
            tolerance: std::f64::MAX,
        }
    }

    /// Distance between points for them to be considered neighbors. Compared to DBSCAN this
    /// parameter isn't strictly necessary but improves execution time by not considering every
    /// point. If the tolerance is too low the distances calculated are undefined and no clusters
    /// will be returned.
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Two points are considered neighbors if the euclidean distance between
    /// them is below the tolerance
    pub fn get_tolerance(&self) -> f64 {
        self.tolerance
    }

    /// Minimum number of a points in a neighborhood around a point for it to
    /// not be considered noise
    pub fn get_minimum_points(&self) -> usize {
        self.min_points
    }

    pub fn validate(&self) -> Result<()> {
        if self.tolerance <= 0. {
            Err(OpticsError::InvalidValue(
                "`tolerance` must be greater than 0!".to_string(),
            ))
        } else if self.min_points <= 1 {
            // There is always at least one neighbor to a point (itself)
            Err(OpticsError::InvalidValue(
                "`min_points` must be greater than 1!".to_string(),
            ))
        } else {
            Ok(())
        }
    }
}
