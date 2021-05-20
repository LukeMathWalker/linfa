use linfa::Float;
use ndarray::{aview1, ArrayBase, Data, Ix2};

use crate::{
    distance::Distance, BuildError, NearestNeighbour, NearestNeighbourIndex, NnError, Point,
};

/// Spatial indexing structure created by [`KdTree`](struct.KdTree.html)
#[derive(Debug)]
pub struct KdTreeIndex<'a, F: Float, D: Distance<F>>(
    kdtree::KdTree<F, (Point<'a, F>, usize), &'a [F]>,
    D,
);

impl<'a, F: Float, D: Distance<F>> KdTreeIndex<'a, F, D> {
    /// Creates a new `KdTreeIndex`
    pub fn new<DT: Data<Elem = F>>(
        batch: &'a ArrayBase<DT, Ix2>,
        leaf_size: usize,
        dist_fn: D,
    ) -> Result<Self, BuildError> {
        if leaf_size == 0 {
            Err(BuildError::EmptyLeaf)
        } else if batch.ncols() == 0 {
            Err(BuildError::ZeroDimension)
        } else {
            let mut tree = kdtree::KdTree::with_capacity(batch.ncols().max(1), leaf_size);
            for (i, point) in batch.genrows().into_iter().enumerate() {
                tree.add(
                    point.to_slice().expect("views should be contiguous"),
                    (point, i),
                )
                .unwrap();
            }
            Ok(Self(tree, dist_fn))
        }
    }
}

impl From<kdtree::ErrorKind> for NnError {
    fn from(err: kdtree::ErrorKind) -> Self {
        match err {
            kdtree::ErrorKind::WrongDimension => NnError::WrongDimension,
            kdtree::ErrorKind::NonFiniteCoordinate => panic!("infinite value found"),
            _ => unreachable!(),
        }
    }
}

impl<'a, F: Float, D: Distance<F>> NearestNeighbourIndex<F> for KdTreeIndex<'a, F, D> {
    fn k_nearest<'b>(
        &self,
        point: Point<'b, F>,
        k: usize,
    ) -> Result<Vec<(Point<F>, usize)>, NnError> {
        Ok(self
            .0
            .nearest(
                point.to_slice().expect("views should be contiguous"),
                k,
                &|a, b| self.1.rdistance(aview1(a), aview1(b)),
            )?
            .into_iter()
            .map(|(_, (pt, pos))| (pt.reborrow(), *pos))
            .collect())
    }

    fn within_range<'b>(
        &self,
        point: Point<'b, F>,
        range: F,
    ) -> Result<Vec<(Point<F>, usize)>, NnError> {
        let range = self.1.dist_to_rdist(range);
        Ok(self
            .0
            .within(
                point.to_slice().expect("views should be contiguous"),
                range,
                &|a, b| self.1.rdistance(aview1(a), aview1(b)),
            )?
            .into_iter()
            .map(|(_, (pt, pos))| (pt.reborrow(), *pos))
            .collect())
    }
}

/// Implementation of K-D tree, a fast space-partitioning data structure.  For each parent node,
/// the indexed points are split with a hyperplane into two child nodes. Due to its tree-like
/// structure, the K-D tree performs spatial queries in `O(k * logN)` time, where `k` is the number
/// of points returned by the query. Calling `from_batch` returns a [`KdTree`](struct.KdTree.html).
///
/// More details can be found [here](https://en.wikipedia.org/wiki/K-d_tree).
///
/// Unlike other `NearestNeighbour` implementations, `KdTree` requires that points be laid out
/// contiguously in memory and will panic otherwise.
#[derive(Default, Clone, Debug)]
pub struct KdTree;

impl KdTree {
    /// Creates an instance of `KdTree`
    pub fn new() -> Self {
        Self
    }
}

impl NearestNeighbour for KdTree {
    fn from_batch_with_leaf_size<'a, F: Float, DT: Data<Elem = F>, D: 'a + Distance<F>>(
        &self,
        batch: &'a ArrayBase<DT, Ix2>,
        leaf_size: usize,
        dist_fn: D,
    ) -> Result<Box<dyn 'a + NearestNeighbourIndex<F>>, BuildError> {
        KdTreeIndex::new(batch, leaf_size, dist_fn)
            .map(|v| Box::new(v) as Box<dyn NearestNeighbourIndex<F>>)
    }
}
