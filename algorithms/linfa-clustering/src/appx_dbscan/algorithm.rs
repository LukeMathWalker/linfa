use crate::appx_dbscan::clustering::AppxDbscanLabeler;
use crate::appx_dbscan::hyperparameters::AppxDbscanHyperParams;
use crate::UncheckedAppxDbscanHyperParams;
use linfa::traits::PredictRef;
use linfa::Float;
use ndarray::{Array1, ArrayBase, Data, Ix2};
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Debug, PartialEq)]
/// DBSCAN (Density-based Spatial Clustering of Applications with Noise)
/// clusters together neighbouring points, while points in sparse regions are labelled
/// as noise. Since points may be part of a cluster or noise the transform method returns
/// `Array1<Option<usize>>`. It should be noted that some "border" points may technically
/// belong to more than one cluster but, since the transform function returns only one
/// label per point (if any), then only one cluster is chosen arbitrarily for those points.
///
/// As DBSCAN groups together points in dense regions, the number of clusters is
/// determined by the dataset and distance tolerance not the user.
///
/// This is an implementation of the approximated version of DBSCAN with
/// complexity O(N). Additional information can be found in
/// [this paper](https://www.cse.cuhk.edu.hk/~taoyf/paper/tods17-dbscan.pdf).
/// Beware of the hidden constant `O((1/slack)^dimensionality)` of the complexity
/// for very small values of `slack` and high dimensionalities.
/// The approximated version of the DBSCAN converges to the exact DBSCAN result
/// for a `slack` that goes to zero. Since only the `tranform` method is provided and
/// border points are not assigned deterministically, it may happen that the two
/// results still differ (in terms of border points) for very small values
/// of `slack`.
///
/// ## The algorithm
///
/// Let `d` be the the number of features of each point:
///
/// * The d-dimensional space is divided in a grid of cells, each containing the points of the dataset that fall
///   inside of them.
/// * All points that have at least `min_points` points in their neighbourhood of size `tolerance` are labeled as "core points". Every
///   cell containing at least one "core point" is labeled as a "core-cell".
/// * All "core cells" are set as vertexes of a graph and an arc is built between any two "core cells" according to the following rules:
///     * If there are two points in the two cells that have distance between them at most `tolerance` then the arc is added to the graph;
///     * If there are two points in the two cells that have distance between them that is between `tolerance` and `tolerance * (1 + slack)` then the arc is added arbitrarily to the graph;
///     * Otherwise the arc is not added;
/// * Every connected component of the graph is a cluster and every core point in the CC is given the label of that cluster.
/// * For all the non-core points in the cells we search for any neighbouring core point (with the same arbirtary rule used to create the arcs)
/// belonging to a cluster and, if at least one is found, then the non-core point is given the label of the cluster of the core-point. If no such core
/// point is found then the point is a "noise" point and it is given a label of `None`.  
///
/// ## Tutorial
///
/// Let's do a walkthrough of an example running the approximated DBSCAN on some data.
///
/// ```rust
/// use linfa_clustering::{AppxDbscan, generate_blobs};
/// use linfa::traits::Predict;
/// use ndarray::{Axis, array, s};
/// use ndarray_rand::rand::SeedableRng;
/// use rand_isaac::Isaac64Rng;
/// use approx::assert_abs_diff_eq;
///
/// // Our random number generator, seeded for reproducibility
/// let seed = 42;
/// let mut rng = Isaac64Rng::seed_from_u64(seed);
///
/// // `expected_centroids` has shape `(n_centroids, n_features)`
/// // i.e. three points in the 2-dimensional plane
/// let expected_centroids = array![[0., 1.], [-10., 20.], [-1., 10.]];
/// // Let's generate a synthetic dataset: three blobs of observations
/// // (100 points each) centered around our `expected_centroids`
/// let observations = generate_blobs(100, &expected_centroids, &mut rng);
///
/// // Let's configure and run our AppxDbscan algorithm
/// // We use the builder pattern to specify the hyperparameters
/// // `min_points` is the only mandatory parameter.
/// // If you don't specify the others (e.g. `tolerance`, `slack`)
/// // default values will be used.
/// let min_points = 3;
/// // Let's run the algorithm!
/// let params =
/// AppxDbscan::params(min_points).tolerance(1e-2).slack(1e-3).predict(&observations).unwrap();
/// // Points are `None` if noise `Some(id)` if belonging to a cluster.
/// ```
///
pub struct AppxDbscan;

impl AppxDbscan {
    /// Configures the hyperparameters with the minimum number of points required to form a cluster
    ///
    /// Defaults are provided if the optional parameters are not specified:
    /// * `tolerance = 1e-4`
    /// * `slack = 1e-2`
    pub fn params<F: Float>(min_points: usize) -> UncheckedAppxDbscanHyperParams<F> {
        UncheckedAppxDbscanHyperParams::new(min_points)
    }
}

impl<F: Float, D: Data<Elem = F>> PredictRef<ArrayBase<D, Ix2>, Array1<Option<usize>>>
    for AppxDbscanHyperParams<F>
{
    fn predict_ref<'a>(&'a self, observations: &'a ArrayBase<D, Ix2>) -> Array1<Option<usize>> {
        if observations.dim().0 == 0 {
            return Array1::from_elem(0, None);
        }

        let labeler = AppxDbscanLabeler::new(&observations.view(), self);
        labeler.into_labels()
    }
}
