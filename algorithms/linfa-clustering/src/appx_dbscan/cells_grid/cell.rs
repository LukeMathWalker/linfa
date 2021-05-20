use crate::appx_dbscan::counting_tree::TreeStructure;
use crate::AppxDbscanHyperParams;
use linfa::Float;
use ndarray::{Array1, ArrayView1, ArrayView2, ArrayViewMut1};
use ndarray_stats::DeviationExt;
use partitions::PartitionVec;
use std::collections::HashMap;

/// A structure that memorizes all non empty cells by their index's hash
pub type CellTable = HashMap<Array1<i64>, usize>;

#[derive(Clone)]
/// A point in a D dimensional euclidean space that memorizes its
/// status: 'core' or 'non core'
pub struct StatusPoint {
    /// memorizes the point index in the input data vector
    point_index: usize,
    is_core: bool,
}

impl StatusPoint {
    pub fn new(point_index: usize) -> StatusPoint {
        StatusPoint {
            point_index,
            is_core: false,
        }
    }

    pub fn is_core(&self) -> bool {
        self.is_core
    }

    pub fn index(&self) -> usize {
        self.point_index
    }
}

#[derive(Clone)]
/// Informations regarding the cell used in various stages of the approximate DBSCAN
/// algorithm if it is a core cell
pub struct CoreCellInfo<F: Float> {
    /// The root of the approximate range counting tree built on the core points of the cell
    root: TreeStructure<F>,
    /// The index of the cluster where the cell belongs
    i_cluster: usize,
}

impl<F: Float> CoreCellInfo<F> {
    fn new() -> CoreCellInfo<F> {
        CoreCellInfo {
            root: TreeStructure::new_empty(),
            i_cluster: 0,
        }
    }
}

#[derive(Clone)]
/// A cell from a grid that partitions the D dimensional euclidean space.
pub struct Cell<F: Float> {
    /// The index of the intervals of the D dimensional axes where this cell lies
    index: Array1<i64>,
    /// The points from the dataset that lie inside this cell
    points: Vec<StatusPoint>,
    /// The list of all the indexes of the cells (in the grid) that might contain poinst at distance at most
    /// `tolerance` from a point in this cell
    neighbour_cell_indexes: Vec<usize>,
    /// Keeps track of wether this cell is a core cell or not
    is_core: bool,
    /// The additional informations that need to be stored if this cell is indeed a core cell
    core_info: CoreCellInfo<F>,
}

impl<F: Float> Cell<F> {
    pub fn new(index_arr: Array1<i64>) -> Cell<F> {
        Cell {
            index: index_arr,
            points: Vec::new(),
            neighbour_cell_indexes: Vec::new(),
            is_core: false,
            core_info: CoreCellInfo::new(),
        }
    }

    pub fn is_core(&self) -> bool {
        self.is_core
    }

    /// Counts the points in `cell` that are at distance at most `epsilon` from `point`.
    /// The distance used is the euclidean one.
    pub fn points_in_range(&self, point_i: usize, points: &ArrayView2<F>, tolerance: F) -> usize {
        self.points
            .iter()
            .filter(|s_p| {
                F::cast(
                    points
                        .row(point_i)
                        .l2_dist(&points.row(s_p.point_index))
                        .unwrap(),
                ) <= tolerance
            })
            .count()
    }

    pub fn points(&self) -> &Vec<StatusPoint> {
        &self.points
    }

    pub fn points_mut(&mut self) -> &mut Vec<StatusPoint> {
        &mut self.points
    }

    pub fn populate_neighbours(&mut self, neighbors: Vec<usize>) {
        self.neighbour_cell_indexes = neighbors;
    }

    pub fn approximate_range_counting(
        &self,
        point: &ArrayView1<F>,
        params: &AppxDbscanHyperParams<F>,
    ) -> usize {
        match self.is_core {
            true => self
                .core_info
                .root
                .approximate_range_counting(point, params),
            false => 0,
        }
    }

    pub fn cluster_i(&self) -> Option<usize> {
        match self.is_core {
            true => Some(self.core_info.i_cluster),
            false => None,
        }
    }

    pub fn assign_to_cluster(
        &mut self,
        cluster_i: usize,
        labels: &mut ArrayViewMut1<Option<usize>>,
    ) {
        match self.is_core {
            true => {
                self.core_info.i_cluster = cluster_i;
                for s_point in self.points.iter().filter(|p| p.is_core) {
                    labels[s_point.point_index] = Some(cluster_i);
                }
            }
            false => {
                panic!("Error: Tried to assign a non core cell to a cluster");
            }
        }
    }

    pub fn neighbours_indexes(&self) -> &Vec<usize> {
        &self.neighbour_cell_indexes
    }

    pub fn label(
        &mut self,
        cells: &PartitionVec<Cell<F>>,
        points: &ArrayView2<F>,
        params: &AppxDbscanHyperParams<F>,
    ) {
        if self.points.len() >= params.minimum_points() {
            self.label_dense(points, params);
        } else {
            self.label_sparse(&cells, points, params);
        }
    }

    /// Function to label as a core cell the cells that have at least 'MinPts' points inside. Sets also the status of
    /// all the points in the cell to 'core'.
    /// An approximate range counting structure is then built on the core points and
    /// memorized in the cell
    fn label_dense(&mut self, points: &ArrayView2<F>, params: &AppxDbscanHyperParams<F>) {
        self.is_core = true;
        let points: Vec<ArrayView1<F>> = self
            .points
            .iter()
            .map(|x| points.row(x.point_index))
            .collect();
        self.core_info.root = TreeStructure::build_structure(points, params);
        for mut s_point in &mut self.points {
            s_point.is_core = true;
        }
    }

    /// Function to decide if a cell with less than 'MinPts' points inside is a core cell. If it is a core cell
    /// then all the core points inside are labeled as such.
    /// An approximate range counting structure is then built on the core points and
    /// memorized in the cell
    fn label_sparse(
        &mut self,
        cells: &PartitionVec<Cell<F>>,
        points: &ArrayView2<F>,
        params: &AppxDbscanHyperParams<F>,
    ) {
        let len = self.points.len();
        let mut core_points: Vec<ArrayView1<F>> = Vec::with_capacity(len);
        for mut s_point in &mut self.points {
            let mut tot_pts = 0;
            for n_index in &self.neighbour_cell_indexes {
                let neighbour = cells.get(*n_index).unwrap();
                tot_pts +=
                    neighbour.points_in_range(s_point.point_index, points, params.tolerance());

                if tot_pts >= params.minimum_points() {
                    break;
                }
            }
            if tot_pts >= params.minimum_points() {
                s_point.is_core = true;
                self.is_core = true;
                core_points.push(points.row(s_point.point_index));
            }
        }
        if self.is_core {
            self.core_info.root = TreeStructure::build_structure(core_points, params);
        }
    }
}
