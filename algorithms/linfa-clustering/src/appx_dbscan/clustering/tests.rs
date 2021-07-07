use crate::UncheckedAppxDbscanHyperParams;

use super::*;
use linfa::prelude::UncheckedHyperParams;
use ndarray::Array2;

#[test]
fn clustering_test() {
    let params = UncheckedAppxDbscanHyperParams::new(2)
        .tolerance(2.0)
        .slack(0.1)
        .check()
        .unwrap();
    let l = params.tolerance / 2_f64.sqrt();
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
    let labeler = AppxDbscanLabeler::new(&points.view(), &params);
    assert_eq!(
        labeler
            .labels()
            .iter()
            .filter(|x| x.is_some())
            .map(|x| x.unwrap() as i64)
            .max()
            .unwrap_or(-1)
            + 1,
        1
    );
    assert_eq!(labeler.labels().iter().filter(|x| x.is_none()).count(), 1);
    assert_eq!(
        labeler
            .labels()
            .iter()
            .filter(|x| x.is_some() && x.unwrap() == 0)
            .count(),
        3
    );
}
