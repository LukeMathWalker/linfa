use std::{cmp::Reverse, collections::BinaryHeap, marker::PhantomData};

use linfa::Float;
use ndarray::{Array1, Array2};
use noisy_float::{checkers::FiniteChecker, NoisyFloat};

use crate::{
    distance::{CommonDistance, Distance},
    heap_elem::{MaxHeapElem, MinHeapElem},
    BuildError, NearestNeighbour, NearestNeighbourBuilder, NnError, Point,
};

// Partition the points using median value
fn partition<F: Float>(mut points: Vec<Point<F>>) -> (Vec<Point<F>>, Point<F>, Vec<Point<F>>) {
    debug_assert!(points.len() >= 2);

    // Spread of a dimension is measured using range, which is suceptible to skew. It may be better
    // to use STD or variance.
    let max_spread_dim = (0..points[0].len())
        .map(|dim| {
            // Find the range of each dimension
            let it = points
                .iter()
                .map(|p| NoisyFloat::<_, FiniteChecker>::new(p[dim]));
            // May be faster if we can compute min and max with the same iterator, but compiler might
            // have optimized for that
            let max = it.clone().max().expect("partitioned empty vec");
            let min = it.min().expect("partitioned empty vec");
            (dim, max - min)
        })
        .max_by_key(|&(_, range)| range)
        .expect("vec has no dimensions")
        .0;

    let mid = points.len() / 2;
    // Compute median on the chosen dimension in linear time
    let median = order_stat::kth_by(&mut points, mid, |p1, p2| {
        p1[max_spread_dim]
            .partial_cmp(&p2[max_spread_dim])
            .expect("NaN in data")
    })
    .clone();

    let (mut left, mut right): (Vec<_>, Vec<_>) = points
        .into_iter()
        .partition(|pt| pt[max_spread_dim] < median[max_spread_dim]);
    // We can get an empty left partition with degenerate data where all points are equal and
    // gathered in the right partition.  This ensures that the larger partition will always shrink,
    // guaranteeing algorithm termination.
    if left.is_empty() {
        left.push(right.pop().unwrap());
    }
    (left, median, right)
}

// Calculates radius of a bounding sphere
fn calc_radius<'a, F: Float, D: Distance<F>>(
    points: impl Iterator<Item = Point<'a, F>>,
    center: Point<F>,
    dist_fn: &D,
) -> F {
    let r_rad = points
        .map(|pt| NoisyFloat::<_, FiniteChecker>::new(dist_fn.rdistance(pt.clone(), center)))
        .max()
        .unwrap()
        .raw();
    dist_fn.rdist_to_dist(r_rad)
}

#[derive(Debug, PartialEq)]
enum BallTreeInner<'a, F: Float> {
    Leaf {
        center: Array1<F>,
        radius: F,
        points: Vec<Point<'a, F>>,
    },
    // The sphere is a bounding sphere that encompasses this node (both children)
    Branch {
        center: Point<'a, F>,
        radius: F,
        left: Box<BallTreeInner<'a, F>>,
        right: Box<BallTreeInner<'a, F>>,
    },
}

impl<'a, F: Float> BallTreeInner<'a, F> {
    fn new<D: Distance<F>>(points: Vec<Point<'a, F>>, leaf_size: usize, dist_fn: &D) -> Self {
        if points.len() <= leaf_size {
            if let Some(dim) = points.first().map(|p| p.len()) {
                let center = {
                    let mut c = Array1::zeros(dim);
                    points.iter().for_each(|p| c += p);
                    c / F::from(points.len()).unwrap()
                };
                let radius = calc_radius(points.iter().cloned(), center.view(), dist_fn);
                BallTreeInner::Leaf {
                    center,
                    radius,
                    points,
                }
            } else {
                BallTreeInner::Leaf {
                    center: Array1::zeros(0),
                    points,
                    radius: F::zero(),
                }
            }
        } else {
            let (aps, center, bps) = partition(points);
            debug_assert!(!aps.is_empty() && !bps.is_empty());
            let radius = calc_radius(aps.iter().chain(bps.iter()).cloned(), center, dist_fn);
            let a_tree = BallTreeInner::new(aps, leaf_size, dist_fn);
            let b_tree = BallTreeInner::new(bps, leaf_size, dist_fn);
            BallTreeInner::Branch {
                center,
                radius,
                left: Box::new(a_tree),
                right: Box::new(b_tree),
            }
        }
    }

    fn rdistance<D: Distance<F>>(&self, p: Point<F>, dist_fn: &D) -> F {
        let (center, radius) = match self {
            BallTreeInner::Leaf { center, radius, .. } => (center.view(), radius),
            BallTreeInner::Branch { center, radius, .. } => (center.reborrow(), radius),
        };

        // The distance to a branch is the distance to the edge of the bounding sphere
        let border_dist = dist_fn.distance(p, center.clone()) - *radius;
        dist_fn.dist_to_rdist(border_dist.max(F::zero()))
    }
}

/// A `BallTree` is a space-partitioning data-structure that allows for finding
/// nearest neighbors in logarithmic time.
///
/// It does this by partitioning data into a series of nested bounding spheres
/// ("balls" in the literature). Spheres are used because it is trivial to
/// compute the distance between a point and a sphere (distance to the sphere's
/// center minus thte radius). The key observation is that a potential neighbor
/// is necessarily closer than all neighbors that are located inside of a
/// bounding sphere that is farther than the aforementioned neighbor.
pub struct BallTree<'a, F: Float, D: Distance<F> = CommonDistance<F>> {
    tree: BallTreeInner<'a, F>,
    dist_fn: D,
    dim: usize,
    len: usize,
}

impl<'a, F: Float, D: Distance<F>> BallTree<'a, F, D> {
    pub fn new(batch: &'a Array2<F>, leaf_size: usize, dist_fn: D) -> Result<Self, BuildError> {
        let dim = batch.ncols();
        let len = batch.nrows();
        if dim == 0 {
            Err(BuildError::ZeroDimension)
        } else {
            let points: Vec<_> = batch.genrows().into_iter().collect();
            Ok(BallTree {
                tree: BallTreeInner::new(points, leaf_size, &dist_fn),
                dist_fn,
                dim,
                len,
            })
        }
    }

    fn nn_helper<'b>(
        &self,
        point: Point<'b, F>,
        k: usize,
        max_radius: F,
    ) -> Result<Vec<Point<F>>, NnError> {
        if self.dim != point.len() {
            Err(NnError::WrongDimension)
        } else if self.len == 0 {
            Ok(Vec::new())
        } else {
            let mut out: BinaryHeap<MaxHeapElem<_, _>> = BinaryHeap::new();
            let mut queue = BinaryHeap::new();
            queue.push(MinHeapElem::new(
                self.tree.rdistance(point, &self.dist_fn),
                &self.tree,
            ));

            while let Some(MinHeapElem {
                dist: Reverse(dist),
                elem,
            }) = queue.pop()
            {
                if dist >= max_radius || (out.len() == k && dist >= out.peek().unwrap().dist) {
                    break;
                }

                match elem {
                    BallTreeInner::Leaf { points, .. } => {
                        for p in points {
                            let dist = self.dist_fn.rdistance(point, p.reborrow());
                            if dist < max_radius
                                && (out.len() < k || out.peek().unwrap().dist > dist)
                            {
                                out.push(MaxHeapElem::new(dist, p));
                                if out.len() > k {
                                    out.pop();
                                }
                            }
                        }
                    }
                    BallTreeInner::Branch { left, right, .. } => {
                        let dl = left.rdistance(point, &self.dist_fn);
                        let dr = right.rdistance(point, &self.dist_fn);

                        if dl <= max_radius {
                            queue.push(MinHeapElem::new(dl, left));
                        }
                        if dr <= max_radius {
                            queue.push(MinHeapElem::new(dr, right));
                        }
                    }
                }
            }
            Ok(out
                .into_sorted_vec()
                .into_iter()
                .map(|e| e.elem.reborrow())
                .collect())
        }
    }
}

impl<'a, F: Float, D: Distance<F>> NearestNeighbour<F> for BallTree<'a, F, D> {
    fn k_nearest<'b>(&self, point: Point<'b, F>, k: usize) -> Result<Vec<Point<F>>, NnError> {
        self.nn_helper(point, k, F::infinity())
    }

    fn within_range<'b>(&self, point: Point<'b, F>, range: F) -> Result<Vec<Point<F>>, NnError> {
        let range = self.dist_fn.dist_to_rdist(range);
        self.nn_helper(point, self.len, range)
    }
}

#[derive(Default)]
pub struct BallTreeBuilder<F: Float>(PhantomData<F>);

impl<F: Float> BallTreeBuilder<F> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<F: Float, D: 'static + Distance<F>> NearestNeighbourBuilder<F, D> for BallTreeBuilder<F> {
    fn from_batch<'a>(
        &self,
        batch: &'a Array2<F>,
        dist_fn: D,
    ) -> Result<Box<dyn 'a + NearestNeighbour<F>>, BuildError> {
        BallTree::new(batch, 2usize.pow(4), dist_fn)
            .map(|v| Box::new(v) as Box<dyn NearestNeighbour<F>>)
    }
}

#[cfg(test)]
mod test {
    use approx::assert_abs_diff_eq;
    use ndarray::{arr1, arr2, stack, Array1, Array2, Axis};

    use crate::distance::L2Dist;

    use super::*;

    fn assert_partition(
        input: Array2<f64>,
        exp_left: Array2<f64>,
        exp_med: Array1<f64>,
        exp_right: Array2<f64>,
        exp_rad: f64,
    ) {
        let vec: Vec<_> = input.genrows().into_iter().collect();
        let (l, mid, r) = partition(vec.clone());
        assert_abs_diff_eq!(stack(Axis(0), &l).unwrap(), exp_left);
        assert_abs_diff_eq!(mid.to_owned(), exp_med);
        assert_abs_diff_eq!(stack(Axis(0), &r).unwrap(), exp_right);
        assert_abs_diff_eq!(calc_radius(vec.iter().cloned(), mid, &L2Dist), exp_rad);
    }

    #[test]
    fn partition_test() {
        // partition 2 elements
        assert_partition(
            arr2(&[[0.0, 1.0], [2.0, 3.0]]),
            arr2(&[[0.0, 1.0]]),
            arr1(&[2.0, 3.0]),
            arr2(&[[2.0, 3.0]]),
            8.0f64.sqrt(),
        );
        assert_partition(
            arr2(&[[2.0, 3.0], [0.0, 1.0]]),
            arr2(&[[0.0, 1.0]]),
            arr1(&[2.0, 3.0]),
            arr2(&[[2.0, 3.0]]),
            8.0f64.sqrt(),
        );

        // Partition along the dimension with highest spread
        assert_partition(
            arr2(&[[0.3, 5.0], [4.5, 7.0], [8.1, 1.5]]),
            arr2(&[[0.3, 5.0]]),
            arr1(&[4.5, 7.0]),
            arr2(&[[4.5, 7.0], [8.1, 1.5]]),
            43.21f64.sqrt(),
        );

        // Degenerate data
        assert_partition(
            arr2(&[[1.4, 4.3], [1.4, 4.3], [1.4, 4.3], [1.4, 4.3]]),
            arr2(&[[1.4, 4.3]]),
            arr1(&[1.4, 4.3]),
            arr2(&[[1.4, 4.3], [1.4, 4.3], [1.4, 4.3]]),
            0.0,
        );
    }
}
