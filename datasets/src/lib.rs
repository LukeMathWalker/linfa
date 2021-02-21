//! `linfa-datasets` provides a collection of commonly used datasets ready to be used in tests and examples.
//!
//! ## The Big Picture
//!
//! `linfa-datasets` is a crate in the [`linfa`](https://crates.io/crates/linfa) ecosystem, an effort to create a toolkit for classical Machine Learning implemented in pure Rust, akin to Python's `scikit-learn`.
//!
//! ## Current State
//!
//! Currently the following datasets are provided:
//!
//! * `["iris"]` : iris flower dataset
//! * `["winequality"]` : wine quality dataset
//! * `["diabetes"]` : diabetes dataset
//!
//! along with methods to easily load them. Loaded datasets are returned as a [`linfa::Dataset`](https://docs.rs/linfa/0.3.0/linfa/dataset/type.Dataset.html) structure whith named features.
//!
//! ## Using a dataset
//!
//! To use one of the provided datasets in your project add the crate to your Cargo.toml with the corresponding feature enabled:
//! ```ignore
//! linfa-datasets = { version = "0.3.0", features = ["winequality"] }
//! ```
//! and then use it in your example or tests as
//! ```ignore
//! let (train, valid) = linfa_datasets::winequality()
//! .split_with_ratio(0.8);
//!  /// ...
//! ```

use csv::ReaderBuilder;
use flate2::read::GzDecoder;
use linfa::Dataset;
use ndarray::prelude::*;
use ndarray_csv::Array2Reader;

#[cfg(any(feature = "iris", feature = "diabetes", feature = "winequality"))]
fn array_from_buf(buf: &[u8]) -> Array2<f64> {
    // unzip file
    let file = GzDecoder::new(buf);
    // create a CSV reader with headers and `;` as delimiter
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .delimiter(b',')
        .from_reader(file);

    // extract ndarray
    reader.deserialize_array2_dynamic().unwrap()
}

#[cfg(feature = "iris")]
/// Read in the iris-flower dataset from dataset path.
// The `.csv` data is two dimensional: Axis(0) denotes y-axis (rows), Axis(1) denotes x-axis (columns)
pub fn iris() -> Dataset<f64, usize> {
    let data = include_bytes!("../data/iris.csv.gz");
    let array = array_from_buf(&data[..]);

    let (data, targets) = (
        array.slice(s![.., 0..4]).to_owned(),
        array.column(4).to_owned(),
    );

    let feature_names = vec!["sepal length", "sepal width", "petal length", "petal width"];

    Dataset::new(data, targets)
        .map_targets(|x| *x as usize)
        .with_feature_names(feature_names)
}

#[cfg(feature = "diabetes")]
/// Read in the diabetes dataset from dataset path
pub fn diabetes() -> Dataset<f64, f64> {
    let data = include_bytes!("../data/diabetes_data.csv.gz");
    let data = array_from_buf(&data[..]);

    let targets = include_bytes!("../data/diabetes_target.csv.gz");
    let targets = array_from_buf(&targets[..]).column(0).to_owned();

    let feature_names = vec![
        "age",
        "sex",
        "body mass index",
        "blood pressure",
        "t-cells",
        "low-density lipoproteins",
        "high-density lipoproteins",
        "thyroid stimulating hormone",
        "lamotrigine",
        "blood sugar level",
    ];

    Dataset::new(data, targets).with_feature_names(feature_names)
}

#[cfg(feature = "winequality")]
/// Read in the winequality dataset from dataset path
pub fn winequality() -> Dataset<f64, usize> {
    let data = include_bytes!("../data/winequality-red.csv.gz");
    let array = array_from_buf(&data[..]);

    let (data, targets) = (
        array.slice(s![.., 0..11]).to_owned(),
        array.column(11).to_owned(),
    );

    let feature_names = vec![
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
    ];

    Dataset::new(data, targets)
        .map_targets(|x| *x as usize)
        .with_feature_names(feature_names)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use linfa::prelude::*;

    #[cfg(feature = "iris")]
    #[test]
    fn test_iris() {
        let ds = iris();

        // check that we have the right amount of data
        assert_eq!((ds.nsamples(), ds.nfeatures(), ds.ntargets()), (150, 4, 1));

        // check for feature names
        assert_eq!(
            ds.feature_names(),
            &["sepal length", "sepal width", "petal length", "petal width"]
        );

        // check label frequency
        assert_eq!(
            ds.label_frequencies()
                .into_iter()
                .map(|b| b.1)
                .collect::<Vec<_>>(),
            &[50., 50., 50.]
        );

        // perform correlation analysis and assert that petal length and width are correlated
        let pcc = ds.pearson_correlation_with_p_value(100);
        assert_abs_diff_eq!(pcc.get_p_values().unwrap()[5], 0.04, epsilon = 0.02);

        // get the mean per feature
        let mean_features = ds.records().mean_axis(Axis(0)).unwrap();
        assert_abs_diff_eq!(
            mean_features,
            array![5.84, 3.05, 3.75, 1.20],
            epsilon = 0.01
        );
    }

    #[cfg(feature = "diabetes")]
    #[test]
    fn test_diabetes() {
        let ds = diabetes();

        // check that we have the right amount of data
        assert_eq!((ds.nsamples(), ds.nfeatures(), ds.ntargets()), (441, 10, 1));

        // perform correlation analysis and assert that T-Cells and low-density lipoproteins are
        // correlated
        let pcc = ds.pearson_correlation_with_p_value(100);
        assert_abs_diff_eq!(pcc.get_p_values().unwrap()[30], 0.02, epsilon = 0.02);

        // get the mean per feature, the data should be normalized
        let mean_features = ds.records().mean_axis(Axis(0)).unwrap();
        assert_abs_diff_eq!(mean_features, Array1::zeros(10), epsilon = 0.005);
    }

    #[cfg(feature = "winequality")]
    #[test]
    fn test_winequality() {
        let ds = winequality();

        // check that we have the right amount of data
        assert_eq!(
            (ds.nsamples(), ds.nfeatures(), ds.ntargets()),
            (1599, 11, 1)
        );

        // check for feature names
        let feature_names = vec![
            "fixed acidity",
            "volatile acidity",
            "citric acid",
            "residual sugar",
            "chlorides",
            "free sulfur dioxide",
            "total sulfur dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol",
        ];
        assert_eq!(ds.feature_names(), feature_names);

        // check label frequency
        let compare_to = vec![
            (5, 681.0),
            (7, 199.0),
            (6, 638.0),
            (8, 18.0),
            (3, 10.0),
            (4, 53.0),
        ];

        let freqs = ds.label_frequencies();
        assert!(compare_to
            .into_iter()
            .all(|(key, val)| { freqs.get(&key).map(|x| *x == val).unwrap_or(false) }));

        // perform correlation analysis and assert that fixed acidity and citric acid are
        // correlated
        let pcc = ds.pearson_correlation_with_p_value(100);
        assert_abs_diff_eq!(pcc.get_p_values().unwrap()[1], 0.05, epsilon = 0.05);
    }
}
