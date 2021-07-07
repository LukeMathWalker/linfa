use crate::appx_dbscan::cells_grid::CellsGrid;
use crate::appx_dbscan::hyperparameters::AppxDbscanHyperParams;
use linfa::Float;

use ndarray::{Array1, ArrayView2};

/// Struct that labels a set of points according to
/// the Approximated DBSCAN algorithm
pub struct AppxDbscanLabeler {
    labels: Array1<Option<usize>>,
}

impl AppxDbscanLabeler {
    /// Runs the Approximated DBSCAN algorithm on the provided `observations` using the params specified in input.
    /// The `Labeler` struct returned contains the label of every point in `observations`.
    ///
    /// ## Parameters:
    /// * `observations`: the points that you want to cluster according to the approximated DBSCAN rule;
    /// * `params`: the parameters for the approximated DBSCAN algorithm
    ///
    /// ## Return
    ///
    /// Struct of type `Labeler` which contains the label associated with each point in `observations`
    ///
    pub fn new<F: Float>(
        observations: &ArrayView2<F>,
        params: &AppxDbscanHyperParams<F>,
    ) -> AppxDbscanLabeler {
        let mut grid = CellsGrid::new(observations, params);
        AppxDbscanLabeler {
            labels: Self::label(&mut grid, observations, params),
        }
    }

    /// Gives the labels of every point provided in input to the constructor.
    ///
    /// ## Example:
    ///
    /// ```rust
    ///
    /// use ndarray::{array, Axis};
    /// use linfa::prelude::*;
    /// use linfa_clustering::{AppxDbscanLabeler, AppxDbscan};
    ///
    /// // Let's define some observations and set the desired params
    /// let observations = array![[0.,0.], [1., 0.], [0., 1.]];
    /// let params = AppxDbscan::params(2).check().unwrap();
    /// // Now we build the labels for each observation using the Labeler struct
    /// let labeler = AppxDbscanLabeler::new(&observations.view(),&params);
    /// // Here we can access the labels for each point `observations`
    /// for (i, point) in observations.axis_iter(Axis(0)).enumerate() {
    ///     let label_for_point = labeler.labels()[i];
    /// }  
    /// ```
    ///
    pub fn labels(&self) -> &Array1<Option<usize>> {
        &self.labels
    }

    pub(crate) fn into_labels(self) -> Array1<Option<usize>> {
        self.labels
    }

    fn label<F: Float>(
        grid: &mut CellsGrid<F>,
        points: &ArrayView2<F>,
        params: &AppxDbscanHyperParams<F>,
    ) -> Array1<Option<usize>> {
        let mut labels = Self::label_connected_components(grid, points, params);
        Self::label_border_noise_points(grid, points, &mut labels, params);
        labels
    }

    /// Explores the graph of cells contained in `grid` and labels all the core points of core cells in the same connected component
    /// with in the same cluster label, and core points from core cells in different connected components with different cluster labels.
    /// If the points in the input grid were not labeled then they will be inside this method.
    fn label_connected_components<F: Float>(
        grid: &mut CellsGrid<F>,
        observations: &ArrayView2<F>,
        params: &AppxDbscanHyperParams<F>,
    ) -> Array1<Option<usize>> {
        if !grid.labeled() {
            grid.label_points(observations, params);
        }
        let mut labels = Array1::from_elem(observations.dim().0, None);
        let mut current_cluster_i: usize = 0;
        for set in grid.cells_mut().all_sets_mut() {
            let mut core_cells_count = 0;
            for cell in set.filter(|(_, c)| c.is_core()).map(|(_, c)| c) {
                cell.assign_to_cluster(current_cluster_i, &mut labels.view_mut());
                core_cells_count += 1;
            }
            if core_cells_count > 0 {
                current_cluster_i += 1;
            }
        }
        labels
    }

    /// Loops through all non core points of the dataset and labels them with one of the possible cluster labels that they belong to.
    /// If no such cluster is found, the point is given label of `None`.
    fn label_border_noise_points<F: Float>(
        grid: &CellsGrid<F>,
        observations: &ArrayView2<F>,
        clusters: &mut Array1<Option<usize>>,
        params: &AppxDbscanHyperParams<F>,
    ) {
        for cell in grid.cells() {
            for cp_index in cell
                .points()
                .iter()
                .filter(|x| !x.is_core())
                .map(|x| x.index())
            {
                let curr_point = &observations.row(cp_index);
                'nbrs: for neighbour_i in cell.neighbours_indexes() {
                    // indexes are added to neighbours only if tthey are in the table
                    let neighbour = grid.cells().get(*neighbour_i).unwrap();
                    if neighbour.approximate_range_counting(curr_point, params) > 0 {
                        clusters[cp_index] = Some(neighbour.cluster_i().unwrap_or_else(|| {
                            panic!("Attempted to get cluster index of a non core cell")
                        }));
                        // assign only to first matching cluster for compatibility
                        break 'nbrs;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
