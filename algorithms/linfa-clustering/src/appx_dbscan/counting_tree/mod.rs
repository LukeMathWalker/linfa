use crate::appx_dbscan::hyperparameters::AppxDbscanHyperParams;
use linfa::Float;
use ndarray::{Array1, Array2, ArrayView1, Axis};
use ndarray_stats::DeviationExt;
use std::collections::HashMap;

#[derive(PartialEq, Debug)]
pub enum IntersectionType {
    FullyCovered,
    Disjoint,
    Intersecting,
}

#[derive(Clone)]
/// Tree structure that divides the space in nested cells to perform approximate range counting
/// Each member of this structure is a node in the tree
pub struct TreeStructure<F: Float> {
    /// The index of the cell represented by this node
    cell_center: Array1<F>,
    /// The size of the cell
    side_size: F,
    /// The number of points cointained in the cell
    cnt: usize,
    /// The collection of nested sub-cells (bounded by 2^D at max, with D constant)
    children: HashMap<Array1<i64>, TreeStructure<F>>,
}

impl<F: Float> TreeStructure<F> {
    pub fn new(cell_index: &Array1<i64>, side_size: F) -> TreeStructure<F> {
        let structure = TreeStructure {
            cell_center: cell_center_from_cell_index(cell_index.view(), side_size),
            cnt: 0,
            side_size,
            children: HashMap::new(),
        };
        structure
    }

    pub fn new_empty() -> TreeStructure<F> {
        TreeStructure {
            cell_center: Array1::zeros(1),
            cnt: 0,
            side_size: F::cast(0.0),
            children: HashMap::with_capacity(0),
        }
    }

    /// Generates a tree starting from the points given in input. To function correctly, the points in input
    /// must be all and only the core points in a given cell of the approximated DBSCAN algorithm with side size
    /// equal to `tolerance/sqrt(D)`. This is assumed true during the construction.
    pub fn build_structure(
        points: Vec<ArrayView1<F>>,
        params: &AppxDbscanHyperParams<F>,
    ) -> TreeStructure<F> {
        if points.is_empty() {
            panic!("AppxDbscan::build structure internal error: attempting to initialize counting tree with no points");
        }
        let dimensionality = points[0].dim();
        let base_side_size = params.tolerance() / (F::cast(dimensionality)).sqrt();
        let levels_count = F::cast(1.0) + (F::cast(1.0) / params.slack()).log(F::cast(2.0)).ceil();
        let levels_count = if levels_count < F::cast(1.0) {
            1
        } else {
            levels_count.to_i32().unwrap()
        };
        // The approximated DBSCAN algorithm needs one instance of this structure for every core cell.
        // This gives that all the points in input are contained in the cell of side size `epsilon/sqrt(D)`.
        // All the points can then be added to the root and we proceed directly to divide the core cell in its sub-cells
        let mut root = TreeStructure::new(&get_base_cell_index(&points[0], params), base_side_size);
        root.cnt = points.len();

        for point in &points {
            let mut curr_side_size = base_side_size;
            let mut prev_child = &mut root;
            //level 0 is already used by the root
            for _ in 1..=levels_count {
                curr_side_size /= F::cast(2.0);
                let index_arr = get_cell_index(point, curr_side_size);
                let curr_child: &mut TreeStructure<F> = prev_child
                    .children
                    .entry(index_arr.clone())
                    .or_insert_with(|| TreeStructure::new(&index_arr, curr_side_size));
                curr_child.cnt += 1;
                prev_child = curr_child;
            }
        }
        root
    }

    /// Performs the approximated range counting on the tree given the point in input. It stops as soon as the counting
    /// is non zero, so the result is not actually the exact count but rather 0 if there is no point in the tree
    /// in the vicinity of `q`, and a value that is less or equal to the number of points in the vicinity of `q` otherwise.
    /// The points in the vicinity are found for certain if they are at a distance less than equal to `epsilon` from `q` and
    /// are excluded for certain if their distance from `q` is greater than `epsilon(1 + rho)`. All the points in between are
    /// counted in an arbitrary way, depending on what is more efficient.
    pub fn approximate_range_counting(
        &self,
        q: &ArrayView1<F>,
        params: &AppxDbscanHyperParams<F>,
    ) -> usize {
        let mut ans: usize = 0;
        let intersection_type =
            determine_intersection(q, params, &self.cell_center, self.side_size);
        match intersection_type {
            IntersectionType::Disjoint => {}
            IntersectionType::FullyCovered => {
                ans += self.cnt;
            }
            IntersectionType::Intersecting => {
                if !self.children.is_empty() {
                    for child in self.children.values() {
                        ans += child.approximate_range_counting(q, params);
                        // There is no need to know the exact count
                        if ans > 0 {
                            return ans;
                        }
                    }
                } else {
                    ans += self.cnt;
                }
            }
        }
        ans
    }
}

/// Gets the indexes of the intervals of the axes in the `D` dimensional space where lies a Cell with side
/// size equal to `side_size` that contains point `p`
pub fn get_cell_index<F: Float>(p: &ArrayView1<F>, side_size: F) -> Array1<i64> {
    let dimensionality = p.dim();
    let mut new_index = Array1::zeros(dimensionality);
    let half_size = side_size / F::cast(2.0);
    for (i, coord) in p.iter().enumerate() {
        if *coord >= (F::cast(-1.0) * half_size) && *coord < half_size {
            new_index[i] = 0;
        } else if *coord > F::cast(0.0) {
            new_index[i] = ((*coord - half_size) / side_size).ceil().to_i64().unwrap();
        } else {
            new_index[i] = -1 + ((*coord + half_size) / side_size).ceil().to_i64().unwrap();
        }
    }
    new_index
}

/// Gets the indexes of the intervals of the axes in the `D` dimensional space where lies a Cell with side
/// size equal to `epsilon/sqrt(D)` that contains point `p`
pub fn get_base_cell_index<F: Float>(
    p: &ArrayView1<F>,
    params: &AppxDbscanHyperParams<F>,
) -> Array1<i64> {
    let dimensionality = p.dim();
    get_cell_index(p, params.tolerance() / (F::cast(dimensionality)).sqrt())
}

/// Determines the type of intersection between a cell and an approximated ball.
/// The cell is determined by its center and the side of its size.
/// Returns:
///  * IntersectionType::FullyCovered if the cell is completely contained in a ball with center `q` and radius `epsilon(1 + rho)`;
///  * IntersectionType::Disjoint if the cell is completely outside of a ball with center `q` and radius `epsilon`;
///  * IntersectionType::Intersecting otherwise;
pub fn determine_intersection<F: Float>(
    q: &ArrayView1<F>,
    params: &AppxDbscanHyperParams<F>,
    cell_center: &Array1<F>,
    side_size: F,
) -> IntersectionType {
    let dimensionality = q.dim();

    //let appr_dist = (F::cast(1.0) + params.slack()) * params.tolerance();
    let dist_from_center = F::cast(cell_center.l2_dist(&q).unwrap());
    let dist_corner_from_center = (side_size * F::cast(dimensionality).sqrt()) / F::cast(2);
    let min_dist_edge_from_center = side_size / F::cast(2);

    // sufficient condition to be disjoint: shortest possible distance from point q
    // to the closest point of the cell (lying on a corner) still greater than the tolerance
    if dist_from_center - dist_corner_from_center > params.tolerance() {
        // here we have sufficient and necessary conditions to be disjoint so we return that
        return IntersectionType::Disjoint;
    }

    // sufficient condition to be completely covered: distance from point to cell center plus distance from center to a corner
    // (farthest point from center) smaller or equal to the extended radius
    if dist_from_center + dist_corner_from_center <= params.appx_tolerance() {
        return IntersectionType::FullyCovered;
    }

    // sufficient condition to not be disjoint: longest possible distance from point q
    // to the closest point of the cell (lying on the middle of an 'egde') smaller than the tolerance
    if dist_from_center - min_dist_edge_from_center <= params.tolerance() {
        return IntersectionType::Intersecting;
    }

    let corners = get_corners(&cell_center, side_size);

    let mut farthest_corner_distance = F::zero();

    for corner in corners.axis_iter(Axis(0)) {
        let dist = F::cast(corner.l2_dist(&q).unwrap());
        if dist > farthest_corner_distance {
            farthest_corner_distance = dist;
        }
    }

    if farthest_corner_distance < params.appx_tolerance() {
        IntersectionType::FullyCovered
    } else {
        IntersectionType::Intersecting
    }
}

/// Gets the coordinates of all the corners (2^D) of a cell given its center points and its side size.
fn get_corners<F: Float>(cell_center: &Array1<F>, side_size: F) -> Array2<F> {
    let dist = side_size / F::cast(2.0);
    let dimensionality = cell_center.dim();
    // There are 2^d combination of vertices. I can think of each combination as a binary
    // number with d digits. I imagine to associate 0 with quantity -dist and 1 with quantity +dist.
    // Then for each of the 2^d combinations I can sum the associated array of values to the cell
    // center to obtain the corner.
    let top = 2_usize.pow(dimensionality as u32);
    let mut corners = Array2::<F>::zeros((top, dimensionality));
    for bin_rep in 0..top {
        let mut new_corner = corners.row_mut(bin_rep);
        for bit_i in 0..dimensionality {
            let mask = 1 << bit_i;
            if bin_rep & mask == 0 {
                new_corner[bit_i] = cell_center[bit_i] - dist;
            } else {
                new_corner[bit_i] = cell_center[bit_i] + dist;
            }
        }
    }
    corners
}

fn cell_center_from_cell_index<F: Float>(cell_index: ArrayView1<i64>, side_size: F) -> Array1<F> {
    let dimensionality = cell_index.dim();
    let mut cell_center: Array1<F> = Array1::zeros(dimensionality);
    for i in 0..dimensionality {
        cell_center[i] = F::cast(cell_index[i]) * side_size;
    }
    cell_center
}

#[cfg(test)]
mod tests;
