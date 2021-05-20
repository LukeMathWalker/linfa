//! `linfa-nn` provides Rust implementations of common spatial indexing algorithms, as well as a
//! trait-based interface for performing nearest-neighbour and range queries using these
//! algorithms.
//!
//! ## The big picture
//!
//! `linfa-nn` is a crate in the `linfa` ecosystem, a wider effort to
//! bootstrap a toolkit for classical Machine Learning implemented in pure Rust,
//! kin in spirit to Python's `scikit-learn`.
//!
//! You can find a roadmap (and a selection of good first issues)
//! [here](https://github.com/LukeMathWalker/linfa/issues) - contributors are more than welcome!
//!
//! ## Current state
//!
//! Right now `linfa-nn` provides the following algorithms:
//! * [Linear Scan](struct.LinearSearch.html)
//! * [KD Tree](struct.KdTree.html)
//! * [Ball Tree](struct.BallTree.html)
//!
//! The [`CommonNearestNeighbour`](struct.CommonNearestNeighbour) enum should be used to dispatch
//! between all of the above algorithms flexibly.

use distance::Distance;
use linfa::Float;
use ndarray::{ArrayBase, ArrayView1, Data, Ix2};
use thiserror::Error;

mod balltree;
mod heap_elem;
mod kdtree;
mod linear;

pub mod distance;

pub use crate::{balltree::*, kdtree::*, linear::*};

pub(crate) type Point<'a, F> = ArrayView1<'a, F>;

/// Error returned when building nearest neighbour indices
#[derive(Error, Debug)]
pub enum BuildError {
    #[error("points have dimension of 0")]
    ZeroDimension,
    #[error("leaf size is 0")]
    EmptyLeaf,
}

/// Error returned when performing spatial queries on nearest neighbour indices
#[derive(Error, Debug)]
pub enum NnError {
    #[error("dimensions of query point and stored points are different")]
    WrongDimension,
}

/// Nearest neighbour algorithm builds a spatial index structure out of a batch of points. The
/// distance between points is calculated using a provided distance function. The index implements
/// the [`NearestNeighbourIndex`](trait.NearestNeighbourIndex.html) trait and allows for efficient
/// computing of nearest neighbour and range queries.
pub trait NearestNeighbour: std::fmt::Debug {
    /// Builds a spatial index using a MxN two-dimensional array representing M points with N
    /// dimensions. Also takes `leaf_size`, which specifies the number of elements in the leaf
    /// nodes of tree-like index structures.
    ///
    /// Returns an error if the points have dimensionality of 0 or if the leaf size is 0. If any
    /// value in the batch is NaN or infinite, the behaviour is unspecified.
    fn from_batch_with_leaf_size<'a, F: Float, DT: Data<Elem = F>, D: 'a + Distance<F>>(
        &self,
        batch: &'a ArrayBase<DT, Ix2>,
        leaf_size: usize,
        dist_fn: D,
    ) -> Result<Box<dyn 'a + NearestNeighbourIndex<F>>, BuildError>;

    /// Builds a spatial index using a default leaf size. See `from_batch_with_leaf_size` for more
    /// information.
    fn from_batch<'a, F: Float, DT: Data<Elem = F>, D: 'a + Distance<F>>(
        &self,
        batch: &'a ArrayBase<DT, Ix2>,
        dist_fn: D,
    ) -> Result<Box<dyn 'a + NearestNeighbourIndex<F>>, BuildError> {
        self.from_batch_with_leaf_size(batch, 2usize.pow(4), dist_fn)
    }
}

/// A spatial index structure over a set of points, created by `NearestNeighbour`. Allows efficient
/// computation of nearest neighbour and range queries over the set of points. Individual points
/// are represented as one-dimensional array views.
pub trait NearestNeighbourIndex<F: Float> {
    /// Returns the `k` points in the index that are the closest to the provided point, along with
    /// their positions in the original dataset. Points are returned in ascending order of the
    /// distance away from the provided points, and less than `k` points will be returned if the
    /// index contains fewer than `k`.
    ///
    /// Returns an error if the provided point has different dimensionality than the index's
    /// points.
    fn k_nearest<'b>(
        &self,
        point: Point<'b, F>,
        k: usize,
    ) -> Result<Vec<(Point<F>, usize)>, NnError>;

    /// Returns all the points in the index that are within the specified distance to the provided
    /// point, along with their positions in the original dataset. The points are not guaranteed to
    /// be in any order, though many algorithms return the points in order of distance.
    ///
    /// Returns an error if the provided point has different dimensionality than the index's
    /// points.
    fn within_range<'b>(
        &self,
        point: Point<'b, F>,
        range: F,
    ) -> Result<Vec<(Point<F>, usize)>, NnError>;
}

/// Enum that dispatches to one of the crate's [`NearestNeighbour`](trait.NearestNeighbour.html)
/// implementations based on value. This enum should be used instead of using types like
/// `LinearSearch` and `KdTree` directly.
///
/// ## Example
///
/// ```rust
/// use rand_isaac::Isaac64Rng;
/// use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
/// use ndarray::{Array1, Array2};
/// use linfa_nn::{distance::*, CommonNearestNeighbour, NearestNeighbour};
///
/// // Use seedable RNG for generating points
/// let mut rng = Isaac64Rng::seed_from_u64(40);
/// let n_features = 3;
/// let distr = Uniform::new(-500., 500.);
/// // Randomly generate points for building the index
/// let points = Array2::random_using((5000, n_features), distr, &mut rng);
///
/// // Build a K-D tree with Euclidean distance as the distance function
/// let nn = CommonNearestNeighbour::KdTree.from_batch(&points, L2Dist).unwrap();
///
/// let pt = Array1::random_using(n_features, distr, &mut rng);
/// // Compute the 10 nearest points to `pt` in the index
/// let nearest = nn.k_nearest(pt.view(), 10).unwrap();
/// // Compute all points within 100 units of `pt`
/// let range = nn.within_range(pt.view(), 100.0).unwrap();
/// ```
#[derive(Debug)]
pub enum CommonNearestNeighbour {
    /// Linear search
    LinearSearch,
    /// KD Tree
    KdTree,
    /// Ball Tree
    BallTree,
}

impl NearestNeighbour for CommonNearestNeighbour {
    fn from_batch_with_leaf_size<'a, F: Float, DT: Data<Elem = F>, D: 'a + Distance<F>>(
        &self,
        batch: &'a ArrayBase<DT, Ix2>,
        leaf_size: usize,
        dist_fn: D,
    ) -> Result<Box<dyn 'a + NearestNeighbourIndex<F>>, BuildError> {
        match self {
            Self::LinearSearch => LinearSearch.from_batch_with_leaf_size(batch, leaf_size, dist_fn),
            Self::KdTree => KdTree.from_batch_with_leaf_size(batch, leaf_size, dist_fn),
            Self::BallTree => BallTree.from_batch_with_leaf_size(batch, leaf_size, dist_fn),
        }
    }
}
