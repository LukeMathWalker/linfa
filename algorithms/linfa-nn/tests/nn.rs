use approx::assert_abs_diff_eq;
use ndarray::{arr1, arr2, aview1, stack, Array2, ArrayView1, Axis};
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use ndarray_stats::DeviationExt;
use noisy_float::{checkers::FiniteChecker, NoisyFloat};
use rand_isaac::Isaac64Rng;

use linfa_nn::{distance::*, CommonNearestNeighbour, LinearSearch, NearestNeighbour};

fn sort_by_dist<'a>(
    mut vec: Vec<(ArrayView1<'a, f64>, usize)>,
    pt: ArrayView1<f64>,
) -> Vec<(ArrayView1<'a, f64>, usize)> {
    vec.sort_by_key(|v| NoisyFloat::<_, FiniteChecker>::new(v.0.sq_l2_dist(&pt).unwrap()));
    vec
}

fn assert_query(
    output: Vec<(ArrayView1<f64>, usize)>,
    input_data: &Array2<f64>,
    exp_pos: Vec<usize>,
) {
    let (pts, pos): (Vec<_>, Vec<_>) = output.into_iter().unzip();
    assert_eq!(pos, exp_pos);
    assert_abs_diff_eq!(
        stack(Axis(0), &pts).unwrap(),
        input_data.select(Axis(0), &exp_pos)
    );
}

fn nn_test_empty(builder: &CommonNearestNeighbour) {
    let points = Array2::zeros((0, 2));
    let nn = builder.from_batch(&points, L2Dist).unwrap();

    let out = nn.k_nearest(aview1(&[0.0, 1.0]), 2).unwrap();
    assert_eq!(out, Vec::<_>::new());

    let out = nn.k_nearest(aview1(&[4.0, 4.0]), 3).unwrap();
    assert_eq!(out, Vec::<_>::new());

    let pt = aview1(&[6.0, 3.0]);
    let out = nn.within_range(pt, 9.0).unwrap();
    assert_eq!(out, Vec::<_>::new());
}

fn nn_test_error(builder: &CommonNearestNeighbour) {
    let points = Array2::<f64>::zeros((4, 0));
    assert!(builder.from_batch(&points, L2Dist).is_err());

    let points = arr2(&[[0.0, 2.0]]);
    assert!(builder
        .from_batch_with_leaf_size(&points, 0, L2Dist)
        .is_err());
    let nn = builder.from_batch(&points, L2Dist).unwrap();
    assert!(nn.k_nearest(aview1(&[]), 2).is_err());
    assert!(nn.within_range(aview1(&[2.2, 4.4, 5.5]), 4.0).is_err());
}

fn nn_test(builder: &CommonNearestNeighbour, sort_within_range: bool) {
    let points = arr2(&[[0.0, 2.0], [10.0, 4.0], [4.0, 5.0], [7.0, 1.0], [1.0, 7.2]]);
    let nn = builder.from_batch(&points, L2Dist).unwrap();

    let out = nn.k_nearest(aview1(&[0.0, 1.0]), 2).unwrap();
    assert_query(out, &points, vec![0, 2]);

    let out = nn.k_nearest(aview1(&[4.0, 4.0]), 3).unwrap();
    assert_query(out, &points, vec![2, 3, 4]);

    let out = nn.k_nearest(aview1(&[4.0, 4.0]), 10).unwrap();
    assert_query(out, &points, vec![2, 3, 4, 0, 1]);

    let pt = aview1(&[6.0, 3.0]);
    let mut out = nn.within_range(pt, 4.3).unwrap();
    if sort_within_range {
        out = sort_by_dist(out, pt.clone());
    }
    assert_query(out, &points, vec![3, 2, 1]);
}

fn nn_test_degenerate(builder: &CommonNearestNeighbour) {
    let points = arr2(&[[0.0, 2.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0]]);
    let nn = builder.from_batch(&points, L2Dist).unwrap();

    let out = nn
        .k_nearest(aview1(&[0.0, 1.0]), 2)
        .unwrap()
        .into_iter()
        .map(|(p, _)| p.reborrow())
        .collect::<Vec<_>>();
    assert_abs_diff_eq!(
        stack(Axis(0), &out).unwrap(),
        arr2(&[[0.0, 2.0], [0.0, 2.0]])
    );

    let out = nn
        .k_nearest(aview1(&[4.0, 4.0]), 3)
        .unwrap()
        .into_iter()
        .map(|(p, _)| p.reborrow())
        .collect::<Vec<_>>();
    assert_abs_diff_eq!(
        stack(Axis(0), &out).unwrap(),
        arr2(&[[0.0, 2.0], [0.0, 2.0], [0.0, 2.0]])
    );

    let pt = aview1(&[3.0, 2.0]);
    let out = nn
        .within_range(pt, 1.0)
        .unwrap()
        .into_iter()
        .map(|(p, _)| p.reborrow())
        .collect::<Vec<_>>();
    assert_eq!(out, Vec::<ArrayView1<_>>::new());
    let out = nn
        .within_range(pt, 20.0)
        .unwrap()
        .into_iter()
        .map(|(p, _)| p.reborrow())
        .collect::<Vec<_>>();
    assert_abs_diff_eq!(stack(Axis(0), &out).unwrap(), points);
}

fn assert_eq_queries(out1: Vec<(ArrayView1<f64>, usize)>, out2: Vec<(ArrayView1<f64>, usize)>) {
    let (pts1, pos1): (Vec<_>, Vec<_>) = out1.into_iter().unzip();
    let (pts2, pos2): (Vec<_>, Vec<_>) = out2.into_iter().unzip();
    assert_eq!(pos1, pos2);
    assert_abs_diff_eq!(
        stack(Axis(0), &pts1).unwrap(),
        stack(Axis(0), &pts2).unwrap(),
    );
}

fn nn_test_random<D: 'static + Distance<f64> + Clone>(
    builder: &CommonNearestNeighbour,
    dist_fn: D,
) {
    let mut rng = Isaac64Rng::seed_from_u64(40);
    let n_points = 50000;
    let n_features = 3;
    let points = Array2::random_using((n_points, n_features), Uniform::new(-50., 50.), &mut rng);

    let linear = LinearSearch::new()
        .from_batch(&points, dist_fn.clone())
        .unwrap();

    let nn = builder.from_batch(&points, dist_fn).unwrap();

    let pt = arr1(&[0., 0., 0.]);
    assert_eq_queries(
        nn.k_nearest(pt.view(), 5).unwrap(),
        linear.k_nearest(pt.view(), 5).unwrap(),
    );
    assert_eq_queries(
        sort_by_dist(nn.within_range(pt.view(), 15.0).unwrap(), pt.view()),
        sort_by_dist(linear.within_range(pt.view(), 15.0).unwrap(), pt.view()),
    );

    let pt = arr1(&[-3.4, 10., 0.95]);
    assert_eq_queries(
        nn.k_nearest(pt.view(), 30).unwrap(),
        linear.k_nearest(pt.view(), 30).unwrap(),
    );
    assert_eq_queries(
        sort_by_dist(nn.within_range(pt.view(), 25.0).unwrap(), pt.view()),
        sort_by_dist(linear.within_range(pt.view(), 25.0).unwrap(), pt.view()),
    );
}

macro_rules! nn_tests {
    ($mod:ident, $builder:ident, $sort:expr $(, $_u:ident)?) => {
        mod $mod {
            use super::*;

            #[test]
            fn empty() {
                nn_test_empty(&CommonNearestNeighbour::$builder);
            }

            #[test]
            fn error() {
                nn_test_error(&CommonNearestNeighbour::$builder);
            }

            #[test]
            fn normal() {
                nn_test(&CommonNearestNeighbour::$builder, $sort);
            }

            #[test]
            fn degenerate() {
                nn_test_degenerate(&CommonNearestNeighbour::$builder);
            }

            $(
                #[test]
                fn random_l2() {
                    let $_u: () = ();
                    nn_test_random(&CommonNearestNeighbour::$builder, L2Dist);
                }

                #[test]
                fn random_l1() {
                    nn_test_random(&CommonNearestNeighbour::$builder, L1Dist);
                }
            )?
        }
    };
}

nn_tests!(linear_search, LinearSearch, true);
nn_tests!(kdtree, KdTree, false, _u);
nn_tests!(balltree, BallTree, false, _u);
