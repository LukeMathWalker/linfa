use crate::UncheckedAppxDbscanHyperParams;

use super::*;
use linfa::prelude::UncheckedHyperParams;
use ndarray::Array2;

#[test]
fn find_cells_test() {
    let params = UncheckedAppxDbscanHyperParams::new(2)
        .tolerance(2.0)
        .slack(0.1)
        .check()
        .unwrap();
    let l = params.tolerance / 2_f64.sqrt();
    let points = Array2::from_shape_vec((2, 2), vec![l, -l, -l, l]).unwrap();
    let grid = CellsGrid::new(&points.view(), &params);
    assert_eq!(grid.cells().len(), 2);
}

#[test]
fn label_points_test() {
    let params = UncheckedAppxDbscanHyperParams::new(2)
        .tolerance(2.0)
        .slack(0.1)
        .check()
        .unwrap();
    let l = params.tolerance / 2_f64.sqrt();
    let all_points = vec![2.0 * l, 2.0 * l, 2.0 * l, 2.0 * l, 2.0 * l, 2.0 * l, l, l];
    let points = Array2::from_shape_vec((4, 2), all_points).unwrap();
    assert_eq!(points.row(0).dim(), 2);
    assert_eq!(points.nrows(), 4);
    let mut grid = CellsGrid::new(&points.view(), &params);
    grid.label_points(&points.view(), &params);
    assert_eq!(grid.cells().len(), 2);
    assert_eq!(grid.cells().iter().filter(|x| x.is_core()).count(), 2);
    assert_eq!(grid.cells().all_sets().count(), 1);
    for set in grid.cells().all_sets() {
        assert_eq!(set.count(), 2);
    }
    let all_points = vec![
        2.0 * l,
        2.0 * l,
        2.0 * l,
        2.0 * l,
        2.0 * l,
        2.0 * l,
        -5.0 * l,
        -5.0 * l,
    ];
    let points = Array2::from_shape_vec((4, 2), all_points).unwrap();
    let mut grid = CellsGrid::new(&points.view(), &params);
    grid.label_points(&points.view(), &params);
    assert_eq!(grid.cells().len(), 2);
    assert_eq!(grid.cells().iter().filter(|x| x.is_core()).count(), 1);
    assert_eq!(grid.cells.all_sets().count(), 2);
}
