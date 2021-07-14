use std::ops::Index;
use std::slice::{Iter, SliceIndex};

/// The analysis from running OPTICS on a dataset, this allows you iterate over the data points and
/// access their core and reachability distances. The ordering of the points also doesn't match
/// that of the dataset instead ordering based on the clustering structure worked out during
/// analysis.
#[derive(Clone, Debug)]
pub struct OpticsAnalysis {
    /// A list of the samples in the dataset sorted and with their reachability and core distances
    /// computed.
    pub(crate) orderings: Vec<Sample>,
}

/// This struct represents a data point in the dataset with it's associated distances obtained from
/// the OPTICS analysis
#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub(crate) index: usize,
    pub(crate) reachability_distance: Option<f64>,
    pub(crate) core_distance: Option<f64>,
}

impl Sample {
    /// Index of the sample in the dataset.
    pub fn index(&self) -> usize {
        self.index
    }

    /// The reachability distance of a sample is the distance between the point and it's cluster
    /// core or another point whichever is larger.
    pub fn reachability_distance(&self) -> &Option<f64> {
        &self.reachability_distance
    }

    /// The distance to the nth closest point where n is the minimum points to form a cluster.
    pub fn core_distance(&self) -> &Option<f64> {
        &self.core_distance
    }
}

impl OpticsAnalysis {
    /// Extracts a slice containing all samples in the dataset
    pub fn as_slice(&self) -> &[Sample] {
        self.orderings.as_slice()
    }

    /// Returns an iterator over the samples in the dataset
    pub fn iter(&self) -> Iter<'_, Sample> {
        self.orderings.iter()
    }
}

impl<I> Index<I> for OpticsAnalysis
where
    I: SliceIndex<[Sample]>,
{
    type Output = I::Output;

    fn index(&self, index: I) -> &Self::Output {
        self.orderings.index(index)
    }
}
