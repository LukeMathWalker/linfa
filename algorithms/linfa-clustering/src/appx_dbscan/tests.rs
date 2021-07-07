use crate::{generate_blobs, AppxDbscanParamsError, Dbscan};
use crate::{AppxDbscan, UncheckedAppxDbscanHyperParams};
use linfa::prelude::UncheckedHyperParams;
use linfa::traits::Predict;
use ndarray::{arr2, s, Array1, Array2};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand_isaac::Isaac64Rng;
use std::collections::HashMap;

#[test]
fn appx_dbscan_test_100() {
    let mut rng = Isaac64Rng::seed_from_u64(40);
    let min_points = 4;
    let n_features = 3;
    let tolerance = 0.8;
    let centroids =
        Array2::random_using((min_points, n_features), Uniform::new(-30., 30.), &mut rng);
    let dataset = generate_blobs(100, &centroids, &mut rng);
    let appx_res = AppxDbscan::params(min_points)
        .tolerance(tolerance)
        .slack(1e-4)
        .predict(&dataset)
        .unwrap();
    let ex_res = Dbscan::params(min_points)
        .tolerance(tolerance)
        .check()
        .unwrap()
        .predict(&dataset);

    // The order of the labels of the clusters in the two algorithms may not be the same
    // but it does not affect the result. We have to create a mapping from the exact labels
    // to the approximate labels.
    let mut ex_appx_correspondence: HashMap<i64, i64> = HashMap::new();
    // For each point in the dataset get the exact and approximate label
    for (ex_label, appx_label) in ex_res.iter().zip(appx_res.iter()) {
        // Get the value of the exact and approximate labels, defaulting to -1 if the
        // point is not in any cluster
        let ex_value = match ex_label {
            Some(value) => *value as i64,
            None => -1,
        };
        let appx_value = match appx_label {
            Some(value) => *value as i64,
            None => -1,
        };
        // assert that every exact noise point is also an approximated noise point
        if ex_value == -1 {
            assert!(appx_value == -1);
        }
        // assert that the approximated label is the one associated to the exact label
        // of the current points
        let expected_appx_val = ex_appx_correspondence.entry(ex_value).or_insert(appx_value);
        assert_eq!(*expected_appx_val, appx_value);
    }
}

#[test]
fn appx_dbscan_test_250() {
    let mut rng = Isaac64Rng::seed_from_u64(15);
    let min_points = 4;
    let n_features = 3;
    let tolerance = 0.8;
    let centroids = Array2::random_using(
        (min_points, n_features),
        Uniform::new(-100., 100.),
        &mut rng,
    );
    //250* 4  = 1000 points
    let dataset = generate_blobs(125, &centroids, &mut rng);
    let appx_res = AppxDbscan::params(min_points)
        .tolerance(tolerance)
        .slack(1e-4)
        .predict(&dataset)
        .unwrap();
    let ex_res = Dbscan::params(min_points)
        .tolerance(tolerance)
        .check()
        .unwrap()
        .predict(&dataset);

    // The order of the labels of the clusters in the two algorithms may not be the same
    // but it does not affect the result. We have to create a mapping from the exact labels
    // to the approximate labels.
    let mut ex_appx_correspondence: HashMap<i64, i64> = HashMap::new();
    // For each point in the dataset get the exact and approximate label
    for (ex_label, appx_label) in ex_res.iter().zip(appx_res.iter()) {
        // Get the value of the exact and approximate labels, defaulting to -1 if the
        // point is not in any cluster
        let ex_value = match ex_label {
            Some(value) => *value as i64,
            None => -1,
        };
        let appx_value = match appx_label {
            Some(value) => *value as i64,
            None => -1,
        };
        // assert that every exact noise point is also an approximated noise point
        if ex_value == -1 {
            assert!(appx_value == -1);
        }
        // assert that the approximated label is the one associated to the exact label
        // of the current points
        let expected_appx_val = ex_appx_correspondence.entry(ex_value).or_insert(appx_value);
        assert_eq!(*expected_appx_val, appx_value);
    }
}

#[test]
fn appx_dbscan_test_500() {
    let mut rng = Isaac64Rng::seed_from_u64(80);
    let min_points = 4;
    let n_features = 3;
    let tolerance = 0.8;
    let centroids =
        Array2::random_using((min_points, n_features), Uniform::new(-50., 50.), &mut rng);
    // 500 * 4 = 2000 points
    let dataset = generate_blobs(250, &centroids, &mut rng);
    let appx_res = AppxDbscan::params(min_points)
        .tolerance(tolerance)
        .slack(1e-4)
        .predict(&dataset)
        .unwrap();
    let ex_res = Dbscan::params(min_points)
        .tolerance(tolerance)
        .check()
        .unwrap()
        .predict(&dataset);

    // The order of the labels of the clusters in the two algorithms may not be the same
    // but it does not affect the result. We have to create a mapping from the exact labels
    // to the approximate labels.
    let mut ex_appx_correspondence: HashMap<i64, i64> = HashMap::new();
    // For each point in the dataset get the exact and approximate label
    for (ex_label, appx_label) in ex_res.iter().zip(appx_res.iter()) {
        // Get the value of the exact and approximate labels, defaulting to -1 if the
        // point is not in any cluster
        let ex_value = match ex_label {
            Some(value) => *value as i64,
            None => -1,
        };
        let appx_value = match appx_label {
            Some(value) => *value as i64,
            None => -1,
        };
        // assert that every exact noise point is also an approximated noise point
        if ex_value == -1 {
            assert!(appx_value == -1);
        }
        // assert that the approximated label is the one associated to the exact label
        // of the current points
        let expected_appx_val = ex_appx_correspondence.entry(ex_value).or_insert(appx_value);
        assert_eq!(*expected_appx_val, appx_value);
    }
}

#[test]
fn test_border() {
    let data: Array2<f64> = arr2(&[
        // Outlier
        [0.0, 2.0],
        // Core point
        [0.0, 0.0],
        // Border points
        [0.0, 1.0],
        [0.0, -1.0],
        [-1.0, 0.0],
        [1.0, 0.0],
    ]);

    // Run the approximate dbscan with tolerance of 1.1, 5 min points for density and
    // a negligible slack
    let labels = AppxDbscan::params(5)
        .tolerance(1.1)
        .slack(1e-5)
        .predict(&data)
        .unwrap();

    assert_eq!(labels[0], None);
    for id in labels.slice(s![1..]).iter() {
        assert_eq!(id, &Some(0));
    }
}

#[test]
fn test_outliers() {
    let mut data: Array2<f64> = Array2::zeros((100, 2));
    // 50 equally spaced values between 0 and 0.8
    let linspace_center = Array1::linspace(0.0, 0.8, 50);
    // Let the first 50 points in data be points with the same value in
    //both dimensions and such values taken from the array above
    data.column_mut(0)
        .slice_mut(s![0..50])
        .assign(&linspace_center);
    data.column_mut(1)
        .slice_mut(s![0..50])
        .assign(&linspace_center);
    // All points have values between 0 and 0.8 in each dimension.

    // Let's now create points with values between -1000 and -5
    //and others with values between 5 and 1000

    // 25 equally spaced values between 5 and 1000
    let linspace_out = Array1::linspace(5., 1000., 25);
    // 25 equally spaced values between -1000 and -5
    let linspace_out_neg = Array1::linspace(-1000., -5., 25);
    // Let the other 50 points in data be points with the same value in
    //both dimensions and such values taken from the arrays above
    data.column_mut(0)
        .slice_mut(s![50..75])
        .assign(&linspace_out);
    data.column_mut(1)
        .slice_mut(s![50..75])
        .assign(&linspace_out);
    data.column_mut(0)
        .slice_mut(s![75..100])
        .assign(&linspace_out_neg);
    data.column_mut(1)
        .slice_mut(s![75..100])
        .assign(&linspace_out_neg);

    // Now let's run the approximate dbscan with tolerance of 1, 2 min points for density and
    // a negligible slack
    let labels = AppxDbscan::params(2)
        .tolerance(1.0)
        .slack(1e-4)
        .predict(&data)
        .unwrap();
    // we should find that the first 50 points are all in the same cluster (cluster 0)
    // and that the other points are so far away from one another that they are all noise points
    for i in 0..50 {
        assert!(labels[i].is_some());
        assert_eq!(labels[i].unwrap(), 0);
        assert!(labels[i + 50].is_none());
    }
}

#[test]
// Identical to the one in the Exact DBSCAN
fn nested_clusters() {
    // Create a circuit of points and then a cluster in the centre
    // and ensure they are identified as two separate clusters
    let mut data: Array2<f64> = Array2::zeros((50, 2));
    let rising = Array1::linspace(0.0, 8.0, 10);
    data.column_mut(0).slice_mut(s![0..10]).assign(&rising);
    data.column_mut(0).slice_mut(s![10..20]).assign(&rising);
    data.column_mut(1).slice_mut(s![20..30]).assign(&rising);
    data.column_mut(1).slice_mut(s![30..40]).assign(&rising);

    data.column_mut(1).slice_mut(s![0..10]).fill(0.0);
    data.column_mut(1).slice_mut(s![10..20]).fill(8.0);
    data.column_mut(0).slice_mut(s![20..30]).fill(0.0);
    data.column_mut(0).slice_mut(s![30..40]).fill(8.0);

    data.column_mut(0).slice_mut(s![40..]).fill(5.0);
    data.column_mut(1).slice_mut(s![40..]).fill(5.0);

    let labels = AppxDbscan::params(2)
        .tolerance(1.0)
        .slack(1e-4)
        .predict(&data)
        .unwrap();

    assert!(labels.slice(s![..40]).iter().all(|x| x == &Some(0)));
    assert!(labels.slice(s![40..]).iter().all(|x| x == &Some(1)));
}

/* ONLY HERE TO CHECK AGAINST PREVIOUS IMPLEMENTATION OF APPX_DBSCAN
#[test]
fn test_exp(){
    let mock_params = params_from_file(&"./e_shop.txt");
    let mock_points = read_points_from_file::<&str,6>(&"./e_shop.txt",&mock_params);
    let points_vec : Vec<f64> = mock_points.iter().map(|x| x.to_vec()).flatten().collect();
    let dataset = Array2::from_shape_vec((mock_params.cardinality, 6), points_vec).unwrap();
    let params = AppxDbscanHyperParams::new(15).tolerance(1.5).slack(0.0001).build();
    let appx_res = AppxDbscan::predict(&params, &dataset);
    let appx_clusters: i64 = appx_res
        .iter()
        .filter(|x| x.is_some())
        .map(|x| x.unwrap() as i64)
        .max()
        .unwrap_or(-1)
        + 1;
    let appx_noise = appx_res.iter().filter(|x| x.is_none()).count();
    assert_eq!(appx_clusters,7);
    assert_eq!(appx_noise, 98);
}*/

#[test]
fn tolerance_cannot_be_zero() {
    let res = UncheckedAppxDbscanHyperParams::new(2)
        .tolerance(0.0)
        .slack(0.1)
        .check();
    assert!(matches!(res, Err(AppxDbscanParamsError::Tolerance)));
}

#[test]
fn slack_cannot_be_zero() {
    let res = UncheckedAppxDbscanHyperParams::new(2)
        .tolerance(0.1)
        .slack(0.0)
        .check();
    assert!(matches!(res, Err(AppxDbscanParamsError::Slack)));
}

#[test]
fn min_points_at_least_2() {
    let res = UncheckedAppxDbscanHyperParams::new(1)
        .tolerance(0.1)
        .slack(0.1)
        .check();
    assert!(matches!(res, Err(AppxDbscanParamsError::MinPoints)));
}

#[test]
fn tolerance_should_be_positive() {
    let res = UncheckedAppxDbscanHyperParams::new(2)
        .tolerance(-1.0)
        .slack(0.1)
        .check();
    assert!(matches!(res, Err(AppxDbscanParamsError::Tolerance)));
}

#[test]
fn slack_should_be_positive() {
    let res = UncheckedAppxDbscanHyperParams::new(2)
        .tolerance(0.1)
        .slack(-1.0)
        .check();
    assert!(matches!(res, Err(AppxDbscanParamsError::Slack)));
}
