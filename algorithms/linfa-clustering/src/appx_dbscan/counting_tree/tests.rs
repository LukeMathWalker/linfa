use crate::UncheckedAppxDbscanHyperParams;

use super::*;

use approx::assert_abs_diff_eq;
use linfa::prelude::UncheckedHyperParams;
use ndarray::{arr1, ArrayView};

#[test]
fn counting_test() {
    let params = UncheckedAppxDbscanHyperParams::new(2)
        .tolerance(2.0)
        .slack(0.1)
        .check()
        .unwrap();
    let l = params.tolerance / 2_f64.sqrt();
    let q_fixed = [l, l];
    let q2_fixed = [-l, l];
    let q = ArrayView::from(&q_fixed);
    let q2 = ArrayView::from(&q2_fixed);
    let points_1 = vec![q];
    let points_2 = vec![q2];
    let root1 = TreeStructure::build_structure(points_1, &params);
    let root2 = TreeStructure::build_structure(points_2, &params);
    let central_fixed = [0.0, 0.0];
    let central = ArrayView::from(&central_fixed);
    let far_fixed = [10.0 * l, 10.0 * l];
    let far = ArrayView::from(&far_fixed);
    assert!(root1.approximate_range_counting(&q, &params) > 0);
    assert!(root2.approximate_range_counting(&q2, &params) > 0);
    assert!(root1.approximate_range_counting(&central, &params) > 0);
    assert!(root2.approximate_range_counting(&central, &params) > 0);
    assert_eq!(root1.approximate_range_counting(&far, &params), 0);
    assert_eq!(root2.approximate_range_counting(&far, &params), 0);
    assert_eq!(root1.approximate_range_counting(&q2, &params), 0);
    assert_eq!(root2.approximate_range_counting(&q, &params), 0);
    assert!(root1.approximate_range_counting(&ArrayView::from(&[2.0 * l, 2.0 * l]), &params) > 0);
    assert_eq!(
        root1.approximate_range_counting(&ArrayView::from(&[3.0 * l, 3.0 * l]), &params),
        0
    );
    assert_eq!(
        root1.approximate_range_counting(&ArrayView::from(&[2.5 * l, 2.5 * l]), &params),
        0
    );
    assert_eq!(
        root1.approximate_range_counting(&ArrayView::from(&[2.2 * l, 2.2 * l]), &params),
        0
    );
    assert_eq!(
        root1.approximate_range_counting(&ArrayView::from(&[2.11 * l, 2.11 * l]), &params),
        0
    );
}

#[test]
fn edge_points_counting_test() {
    let epsilon: f64 = 1.0;
    let slack = 0.00001;
    let params = UncheckedAppxDbscanHyperParams::new(2)
        .tolerance(epsilon)
        .slack(slack)
        .check()
        .unwrap();

    let central: Array1<f64> = Array1::from_shape_vec(2, vec![0.39, 0.0]).unwrap();
    let left: Array1<f64> = Array1::from_shape_vec(2, vec![-0.6, 0.0]).unwrap();

    let root = TreeStructure::build_structure(vec![central.view()], &params);
    assert!(root.approximate_range_counting(&left.view(), &params) > 0);

    let central: Array1<f64> =
        Array1::from_shape_vec(6, vec![5.0, 29.0, 4.0, 7.0, 3.0, 1.0]).unwrap();
    let left: Array1<f64> = Array1::from_shape_vec(6, vec![5.0, 29.0, 4.0, 7.0, 3.0, 2.0]).unwrap();

    let root = TreeStructure::build_structure(vec![central.view()], &params);
    assert!(root.approximate_range_counting(&left.view(), &params) > 0);
}

#[test]
fn get_corners_test() {
    let side_size = 2.0;
    let cell_center = arr1(&[0.0, 0.0]);
    let corners = get_corners(&cell_center, side_size);
    //--0
    assert_abs_diff_eq!(corners.row(0)[0], -1.0);
    assert_abs_diff_eq!(corners.row(0)[1], -1.0);
    //--1
    assert_abs_diff_eq!(corners.row(1)[0], 1.0);
    assert_abs_diff_eq!(corners.row(1)[1], -1.0);
    //--2
    assert_abs_diff_eq!(corners.row(2)[0], -1.0);
    assert_abs_diff_eq!(corners.row(2)[1], 1.0);
    //--3
    assert_abs_diff_eq!(corners.row(3)[0], 1.0);
    assert_abs_diff_eq!(corners.row(3)[0], 1.0);

    let cell_center = arr1(&[0.0, 0.0, 0.0]);
    let corners = get_corners(&cell_center, side_size);

    //--0
    assert_abs_diff_eq!(corners.row(0)[0], -1.0);
    assert_abs_diff_eq!(corners.row(0)[1], -1.0);
    assert_abs_diff_eq!(corners.row(0)[2], -1.0);
    //--1
    assert_abs_diff_eq!(corners.row(1)[0], 1.0);
    assert_abs_diff_eq!(corners.row(1)[1], -1.0);
    assert_abs_diff_eq!(corners.row(1)[2], -1.0);
    //--2
    assert_abs_diff_eq!(corners.row(2)[0], -1.0);
    assert_abs_diff_eq!(corners.row(2)[1], 1.0);
    assert_abs_diff_eq!(corners.row(2)[2], -1.0);
    //--3
    assert_abs_diff_eq!(corners.row(3)[0], 1.0);
    assert_abs_diff_eq!(corners.row(3)[1], 1.0);
    assert_abs_diff_eq!(corners.row(3)[2], -1.0);
    //--4
    assert_abs_diff_eq!(corners.row(4)[0], -1.0);
    assert_abs_diff_eq!(corners.row(4)[1], -1.0);
    assert_abs_diff_eq!(corners.row(4)[2], 1.0);
    //--5
    assert_abs_diff_eq!(corners.row(5)[0], 1.0);
    assert_abs_diff_eq!(corners.row(5)[1], -1.0);
    assert_abs_diff_eq!(corners.row(5)[2], 1.0);
    //--6
    assert_abs_diff_eq!(corners.row(6)[0], -1.0);
    assert_abs_diff_eq!(corners.row(6)[1], 1.0);
    assert_abs_diff_eq!(corners.row(6)[2], 1.0);
    //--7
    assert_abs_diff_eq!(corners.row(7)[0], 1.0);
    assert_abs_diff_eq!(corners.row(7)[1], 1.0);
    assert_abs_diff_eq!(corners.row(7)[2], 1.0);
}

#[test]
fn determine_intersection_test() {
    let params = UncheckedAppxDbscanHyperParams::new(2)
        .tolerance(2.0)
        .slack(0.1)
        .check()
        .unwrap();
    let l = params.tolerance / 2.0_f64.sqrt();
    let fixed_point = [l / 2.0, (3.0 / 2.0) * l];
    let q = ArrayView::from(&fixed_point);
    let cell_index_1 = arr1(&[0, 1]);
    let cell_index_2 = arr1(&[1, 1]);
    let cell_index_3 = arr1(&[0, 2]);
    let cell_index_4 = arr1(&[1, 2]);
    let expected_type = IntersectionType::FullyCovered;
    let intersection = determine_intersection(
        &q,
        &params,
        &cell_center_from_cell_index(cell_index_1.view(), l),
        l,
    );
    assert_eq!(intersection, expected_type);
    let intersection = determine_intersection(
        &q,
        &params,
        &cell_center_from_cell_index(cell_index_2.view(), l),
        l,
    );
    assert_eq!(intersection, expected_type);
    let intersection = determine_intersection(
        &q,
        &params,
        &cell_center_from_cell_index(cell_index_3.view(), l),
        l,
    );
    assert_eq!(intersection, expected_type);
    let intersection = determine_intersection(
        &q,
        &params,
        &cell_center_from_cell_index(cell_index_4.view(), l),
        l,
    );
    assert_eq!(intersection, expected_type);
    let cell_index_1 = arr1(&[-1, 1]);
    let cell_index_2 = arr1(&[-1, 2]);
    let cell_index_3 = arr1(&[2, 2]);
    let cell_index_4 = arr1(&[2, 1]);
    let expected_type = IntersectionType::Intersecting;
    let intersection = determine_intersection(
        &q,
        &params,
        &cell_center_from_cell_index(cell_index_1.view(), l),
        l,
    );
    assert_eq!(intersection, expected_type);
    let intersection = determine_intersection(
        &q,
        &params,
        &cell_center_from_cell_index(cell_index_2.view(), l),
        l,
    );
    assert_eq!(intersection, expected_type);
    let intersection = determine_intersection(
        &q,
        &params,
        &cell_center_from_cell_index(cell_index_3.view(), l),
        l,
    );
    assert_eq!(intersection, expected_type);
    let intersection = determine_intersection(
        &q,
        &params,
        &cell_center_from_cell_index(cell_index_4.view(), l),
        l,
    );
    assert_eq!(intersection, expected_type);
    let cell_index_1 = arr1(&[3, 3]);
    let cell_index_2 = arr1(&[3, 2]);
    let cell_index_3 = arr1(&[-2, 2]);
    let cell_index_4 = arr1(&[-2, 1]);
    let expected_type = IntersectionType::Disjoint;
    let intersection = determine_intersection(
        &q,
        &params,
        &cell_center_from_cell_index(cell_index_1.view(), l),
        l,
    );
    assert_eq!(intersection, expected_type);
    let intersection = determine_intersection(
        &q,
        &params,
        &cell_center_from_cell_index(cell_index_2.view(), l),
        l,
    );
    assert_eq!(intersection, expected_type);
    let intersection = determine_intersection(
        &q,
        &params,
        &cell_center_from_cell_index(cell_index_3.view(), l),
        l,
    );
    assert_eq!(intersection, expected_type);
    let intersection = determine_intersection(
        &q,
        &params,
        &cell_center_from_cell_index(cell_index_4.view(), l),
        l,
    );
    assert_eq!(intersection, expected_type);
}
