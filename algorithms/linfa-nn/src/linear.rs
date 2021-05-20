use std::{cmp::Reverse, collections::BinaryHeap};

use linfa::Float;
use ndarray::{ArrayBase, ArrayView2, Data, Ix2};
use noisy_float::NoisyFloat;

use crate::{
    distance::Distance, heap_elem::MinHeapElem, BuildError, NearestNeighbour,
    NearestNeighbourIndex, NnError, Point,
};

/// Spatial indexing structure created by [`LinearSearch`](struct.LinearSearch.html)
#[derive(Debug)]
pub struct LinearSearchIndex<'a, F: Float, D: Distance<F>>(ArrayView2<'a, F>, D);

impl<'a, F: Float, D: Distance<F>> LinearSearchIndex<'a, F, D> {
    /// Creates a new `LinearSearchIndex`
    pub fn new<DT: Data<Elem = F>>(
        batch: &'a ArrayBase<DT, Ix2>,
        dist_fn: D,
    ) -> Result<Self, BuildError> {
        if batch.ncols() == 0 {
            Err(BuildError::ZeroDimension)
        } else {
            Ok(Self(batch.view(), dist_fn))
        }
    }
}

impl<'a, F: Float, D: Distance<F>> NearestNeighbourIndex<F> for LinearSearchIndex<'a, F, D> {
    fn k_nearest<'b>(
        &self,
        point: Point<'b, F>,
        k: usize,
    ) -> Result<Vec<(Point<F>, usize)>, NnError> {
        if self.0.ncols() != point.len() {
            Err(NnError::WrongDimension)
        } else {
            let mut heap = BinaryHeap::with_capacity(self.0.nrows());
            for (i, pt) in self.0.genrows().into_iter().enumerate() {
                let dist = self.1.rdistance(point.reborrow(), pt.reborrow());
                heap.push(MinHeapElem {
                    elem: (pt.reborrow(), i),
                    dist: Reverse(NoisyFloat::new(dist)),
                });
            }

            Ok((0..k.min(heap.len()))
                .map(|_| heap.pop().unwrap().elem)
                .collect())
        }
    }

    fn within_range<'b>(
        &self,
        point: Point<'b, F>,
        range: F,
    ) -> Result<Vec<(Point<F>, usize)>, NnError> {
        if self.0.ncols() != point.len() {
            Err(NnError::WrongDimension)
        } else {
            let range = self.1.dist_to_rdist(range);
            Ok(self
                .0
                .genrows()
                .into_iter()
                .enumerate()
                .filter(|(_, pt)| self.1.rdistance(point.reborrow(), pt.reborrow()) < range)
                .map(|(i, pt)| (pt, i))
                .collect())
        }
    }
}

/// Implementation of linear search, which is the simplest nearest neighbour algorithm. All queries
/// are implemented by scanning through every point, so all of them are `O(N)`. Calling
/// `from_batch` returns a [`LinearSearchIndex`](struct.LinearSearchIndex.html).
#[derive(Default, Clone, Debug)]
pub struct LinearSearch;

impl LinearSearch {
    /// Creates an instance of `LinearSearch`
    pub fn new() -> Self {
        Self
    }
}

impl NearestNeighbour for LinearSearch {
    fn from_batch_with_leaf_size<'a, F: Float, DT: Data<Elem = F>, D: 'a + Distance<F>>(
        &self,
        batch: &'a ArrayBase<DT, Ix2>,
        leaf_size: usize,
        dist_fn: D,
    ) -> Result<Box<dyn 'a + NearestNeighbourIndex<F>>, BuildError> {
        if leaf_size == 0 {
            return Err(BuildError::EmptyLeaf);
        }
        LinearSearchIndex::new(batch, dist_fn)
            .map(|v| Box::new(v) as Box<dyn NearestNeighbourIndex<F>>)
    }
}
