mod cell;

use crate::appx_dbscan::counting_tree::get_base_cell_index;
use crate::AppxDbscanHyperParams;
use linfa::Float;
use linfa_nn::{distance::L2Dist, KdTree, NearestNeighbour};
use ndarray::{Array2, ArrayView1, ArrayView2, Axis};
use partitions::PartitionVec;

use cell::{Cell, CellTable, StatusPoint};

pub type CellVector<F> = PartitionVec<Cell<F>>;

pub struct CellsGrid<F: Float> {
    table: CellTable,
    cells: CellVector<F>,
    labeled: bool,
    dimensionality: usize,
}

impl<F: Float> CellsGrid<F> {
    /// Partitions the euclidean space containing `points` in a grid of
    pub fn new(points: &ArrayView2<F>, params: &AppxDbscanHyperParams<F>) -> CellsGrid<F> {
        let mut grid = CellsGrid {
            table: CellTable::with_capacity(points.dim().0),
            cells: PartitionVec::with_capacity(points.dim().0),
            labeled: false,
            dimensionality: points.dim().1,
        };
        grid.populate(points, params);
        grid
    }

    /// Divides the D dimensional euclidean space in a grid of cells with side length `epsilon\sqrt(D)` and memorizes
    /// the non empty ones in a `CellTable`
    pub fn populate(&mut self, points: &ArrayView2<F>, params: &AppxDbscanHyperParams<F>) {
        for (p_i, curr_point) in points.axis_iter(Axis(0)).enumerate() {
            self.insert_point(&curr_point, p_i, &params);
        }
        self.populate_neighbours();
    }

    /// Function that decides which points from each cell are core points and which cells are core cells.
    ///
    /// # Arguments:
    ///
    /// * `points`: The points to cluster
    /// * `params`: the DBSCAN algorithm parameters
    ///
    /// # Behavior
    /// At the end of this function all the possible unions between neighbouring cells in `self.cells` will have been made.
    pub fn label_points(&mut self, points: &ArrayView2<F>, params: &AppxDbscanHyperParams<F>) {
        for cell_i in self.table.values() {
            let mut cloned_cell = self.cells[*cell_i].clone();
            cloned_cell.label(&self.cells, points, params);
            self.cells[*cell_i] = cloned_cell;
        }
        self.unite_neighbouring_cells(points, params);
        self.labeled = true;
    }

    fn populate_neighbours(&mut self) {
        let nindices = self.table.len();
        // map all cell indices from i64 to F and flatten in order to put them all in an array2
        let all_indices: Vec<F> = self
            .table
            .keys()
            .map(|x| x.mapv(F::cast).to_vec())
            .flatten()
            .collect();
        // construct the matrix of all cell indices
        let indices: Array2<F> =
            Array2::from_shape_vec((nindices, self.dimensionality), all_indices).unwrap();
        // bulk load the kdtree with all cell indices that are actually in the table
        let kd_tree = KdTree::new()
            .from_batch_with_leaf_size(&indices, 4 * self.dimensionality, L2Dist)
            .unwrap();
        for (spatial_index, table_index) in &self.table {
            // the indices of the cell represent their position in space and so the neighboring cells (all the cells that *may* contain a point
            // within the approximated distacne) are the ones that have an index up to the square root of 4 times the dimensionality of the space
            let neighbors = kd_tree
                .within_range(
                    spatial_index.mapv(F::cast).view(),
                    F::cast(4 * self.dimensionality).sqrt(),
                )
                .unwrap();
            // first map the indices of the neighboring cells back to i64 (safe since they came from there) and then use the indices to
            // get the related cell position in the partition vector. Since the tree was constructed from cells in the vector it is safe
            // to unwrap the result of `get`
            let neighbors = neighbors
                .into_iter()
                .map(|i| i.mapv(|x| x.to_i64().unwrap()))
                .map(|i| *self.table.get(&i).unwrap())
                .collect();
            self.cells[*table_index].populate_neighbours(neighbors);
        }
    }

    pub fn labeled(&self) -> &bool {
        &self.labeled
    }

    pub fn cells(&self) -> &PartitionVec<Cell<F>> {
        &self.cells
    }

    pub fn cells_mut(&mut self) -> &mut PartitionVec<Cell<F>> {
        &mut self.cells
    }

    fn insert_point(
        &mut self,
        point: &ArrayView1<F>,
        p_i: usize,
        params: &AppxDbscanHyperParams<F>,
    ) {
        let cell_index = get_base_cell_index(point, params);
        let curr_cell_n = self.cells.len();
        let cell_i = self.table.entry(cell_index.clone()).or_insert(curr_cell_n);
        if *cell_i == curr_cell_n {
            self.cells.push(Cell::new(cell_index));
        }
        self.cells[*cell_i].points_mut().push(StatusPoint::new(p_i));
    }

    fn unite_neighbouring_cells(
        &mut self,
        points: &ArrayView2<F>,
        params: &AppxDbscanHyperParams<F>,
    ) {
        for cell_i in self.table.values() {
            if !self.cells[*cell_i].is_core() {
                continue;
            }
            let curr_cell_points = self.cells[*cell_i].points().clone();
            let neighbours_indexes = self.cells[*cell_i].neighbours_indexes().clone();
            for n_index in neighbours_indexes {
                let neighbour = self.cells.get(n_index).unwrap();
                if !neighbour.is_core() || self.cells.same_set(*cell_i, n_index) {
                    continue;
                }
                for point in curr_cell_points.iter().filter(|p| p.is_core()) {
                    if neighbour.approximate_range_counting(&points.row(point.index()), params) > 0
                    {
                        self.cells.union(*cell_i, n_index);
                        break;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
