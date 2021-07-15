//! Datasets
//!
//! This module implements the dataset struct and various helper traits to extend its
//! functionality.
use ndarray::{
    Array1, Array2, ArrayBase, ArrayView, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2,
    Axis, CowArray, Ix2, Ix3, OwnedRepr, ScalarOperand,
};

#[cfg(feature = "ndarray-linalg")]
use ndarray_linalg::{Lapack, Scalar};

use num_traits::{AsPrimitive, FromPrimitive, NumAssignOps, NumCast, Signed};
use rand::distributions::uniform::SampleUniform;

#[cfg(feature = "polars")]
use polars_core::frame::DataFrame;

use std::cmp::{Ordering, PartialOrd};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::Hash;
use std::iter::Sum;
use std::ops::{AddAssign, Deref, DivAssign, MulAssign, SubAssign};

use crate::error::{Error, Result};

mod impl_dataset;
mod impl_records;
mod impl_targets;

mod iter;

mod lapack_bounds;
pub use lapack_bounds::*;

/// Floating point numbers
///
/// This trait bound multiplexes to the most common assumption of floating point number and
/// implement them for 32bit and 64bit floating points. They are used in records of a dataset and, for
/// regression task, in the targets as well.
pub trait Float:
    FromPrimitive
    + num_traits::Float
    + PartialOrd
    + Sync
    + Send
    + Default
    + fmt::Display
    + fmt::Debug
    + Signed
    + Sum
    + NumAssignOps
    + AsPrimitive<usize>
    + for<'a> AddAssign<&'a Self>
    + for<'a> MulAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
    + for<'a> DivAssign<&'a Self>
    + num_traits::MulAdd<Output = Self>
    + SampleUniform
    + ScalarOperand
    + approx::AbsDiffEq
{
    #[cfg(feature = "ndarray-linalg")]
    type Lapack: Float + Scalar + Lapack;
    #[cfg(not(feature = "ndarray-linalg"))]
    type Lapack: Float;

    fn cast<T: NumCast>(x: T) -> Self {
        NumCast::from(x).unwrap()
    }
}

impl Float for f32 {
    type Lapack = f32;
}

impl Float for f64 {
    type Lapack = f64;
}

/// Discrete labels
///
/// Labels are countable, comparable and hashable. Currently null-type (no targets),
/// boolean (binary task) and usize, strings (multi-label tasks) are supported.
pub trait Label: PartialEq + Eq + Hash + Clone {}

impl Label for bool {}
impl Label for usize {}
impl Label for String {}
impl Label for () {}
impl Label for &str {}
impl Label for Option<usize> {}

/// Probability types
///
/// This helper struct exists to distinguish probabilities from floating points. For example SVM
/// selects regression or classification training, based on the target type, and could not
/// distinguish them without a new-type definition.
#[derive(Debug, Copy, Clone)]
pub struct Pr(pub f32);

impl Pr {
    pub fn even() -> Pr {
        Pr(0.5)
    }
}

impl PartialEq for Pr {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl PartialOrd for Pr {
    fn partial_cmp(&self, other: &Pr) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Deref for Pr {
    type Target = f32;

    fn deref(&self) -> &f32 {
        &self.0
    }
}

/// DatasetBase
///
/// This is the fundamental structure of a dataset. It contains a number of records about the data
/// and may contain targets, weights and feature names. In order to keep the type complexity low
/// the dataset base is only generic over the records and targets and introduces a trait bound on
/// the records. `weights` and `feature_names`, on the other hand, are always assumed to be owned
/// and copied when views are created.
///
/// # Fields
///
/// * `records`: a two-dimensional matrix with dimensionality (nsamples, nfeatures), in case of
/// kernel methods a quadratic matrix with dimensionality (nsamples, nsamples), which may be sparse
/// * `targets`: a two-/one-dimension matrix with dimensionality (nsamples, ntargets)
/// * `weights`: optional weights for each sample with dimensionality (nsamples)
/// * `feature_names`: optional descriptive feature names with dimensionality (nfeatures)
///
/// # Trait bounds
///
/// * `R: Records`: generic over feature matrices or kernel matrices
/// * `T`: generic over any `ndarray` matrix which can be used as targets. The `AsTargets` trait
/// bound is omitted here to avoid some repetition in implementation `src/dataset/impl_dataset.rs`
pub struct DatasetBase<R, T>
where
    R: Records,
{
    pub records: R,
    pub targets: T,

    pub weights: Array1<f32>,
    feature_names: Vec<String>,
}

/// Targets with precomputed, counted labels
///
/// This extends plain targets with pre-counted labels. The label map is useful when, for example,
/// a prior probability is estimated (e.g. in Naive Bayesian implementation) or the samples are
/// weighted inverse to their occurence.
///
/// # Fields
///
/// * `targets`: wrapped target field
/// * `labels`: counted labels with label-count association
pub struct CountedTargets<L: Label, P> {
    targets: P,
    labels: Vec<HashMap<L, usize>>,
}

/// Dataset
///
/// The most commonly used typed of dataset. It contains a number of records
/// stored as an `Array2` and each record may correspond to multiple targets. The
/// targets are stored as an `Array2`.
pub type Dataset<D, T> = DatasetBase<ArrayBase<OwnedRepr<D>, Ix2>, ArrayBase<OwnedRepr<T>, Ix2>>;

/// DatasetView
///
/// A read only view of a Dataset
pub type DatasetView<'a, D, T> = DatasetBase<ArrayView<'a, D, Ix2>, ArrayView<'a, T, Ix2>>;

/// DatasetPr
///
/// Dataset with probabilities as targets. Useful for multiclass probabilities.
/// It stores records as an `Array2` of elements of type `D`, and targets as an `Array1`
/// of elements of type `Pr`
pub type DatasetPr<D, L> =
    DatasetBase<ArrayBase<OwnedRepr<D>, Ix2>, CountedTargets<L, ArrayBase<OwnedRepr<Pr>, Ix3>>>;

#[cfg(feature = "polars")]
/// Dataframe
pub type Dataframe<T> = DatasetBase<DataFrame, ArrayBase<OwnedRepr<T>, Ix2>>;

/// Record trait
pub trait Records: Sized {
    type Elem;

    fn nsamples(&self) -> usize;
    fn nfeatures(&self) -> usize;
}

/// Return a reference to single or multiple target variables
pub trait AsTargets {
    type Elem;

    /// Returns a view on targets as two-dimensional array
    fn as_multi_targets(&self) -> ArrayView2<Self::Elem>;

    /// Convert to single target, fails for more than one target
    ///
    /// # Returns
    ///
    /// May return a single target with the same label type, but returns an
    /// `Error::MultipleTargets` in case that there are more than a single target.
    fn try_single_target(&self) -> Result<ArrayView1<Self::Elem>> {
        let multi_targets = self.as_multi_targets();

        if multi_targets.len_of(Axis(1)) > 1 {
            return Err(Error::MultipleTargets);
        }

        Ok(multi_targets.index_axis_move(Axis(1), 0))
    }
}

/// Helper trait to construct counted labels
///
/// This is implemented for objects which can act as targets and created from a target matrix. For
/// targets represented as `ndarray` matrix this is identity, for counted labels, i.e.
/// `TargetsWithLabels`, it creates the corresponding wrapper struct.
pub trait FromTargetArray<'a, F> {
    type Owned;
    type View;

    /// Create self object from new target array
    fn new_targets(targets: Array2<F>) -> Self::Owned;
    fn new_targets_view(targets: ArrayView2<'a, F>) -> Self::View;
}

pub trait AsTargetsMut {
    type Elem;

    /// Returns a mutable view on targets as two-dimensional array
    fn as_multi_targets_mut(&mut self) -> ArrayViewMut2<Self::Elem>;

    /// Convert to single target, fails for more than one target
    fn try_single_target_mut(&mut self) -> Result<ArrayViewMut1<Self::Elem>> {
        let multi_targets = self.as_multi_targets_mut();

        if multi_targets.len_of(Axis(1)) > 1 {
            return Err(Error::MultipleTargets);
        }

        Ok(multi_targets.index_axis_move(Axis(1), 0))
    }
}

/// Convert to probability matrix
///
/// Some algorithms are working with probabilities. Targets which allow an implicit conversion into
/// probabilities can implement this trait.
pub trait AsProbabilities {
    fn as_multi_target_probabilities(&self) -> CowArray<Pr, Ix3>;
}

/// Get the labels in all targets
///
pub trait Labels {
    type Elem: Label;

    fn label_count(&self) -> Vec<HashMap<Self::Elem, usize>>;

    fn label_set(&self) -> Vec<HashSet<Self::Elem>> {
        self.label_count()
            .iter()
            .map(|x| x.keys().cloned().collect::<HashSet<_>>())
            .collect()
    }

    fn labels(&self) -> Vec<Self::Elem> {
        self.label_set().into_iter().flatten().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1, Array2};
    use rand::{rngs::SmallRng, SeedableRng};

    #[test]
    fn dataset_implements_required_methods() {
        let mut rng = SmallRng::seed_from_u64(42);

        // ------ Targets ------

        // New
        let mut dataset = Dataset::new(array![[1., 2.], [1., 2.]], array![0., 1.]);

        // Shuffle
        dataset = dataset.shuffle(&mut rng);

        // Bootstrap samples
        {
            let mut iter = dataset.bootstrap_samples(3, &mut rng);
            for _ in 1..5 {
                let b_dataset = iter.next().unwrap();
                assert_eq!(b_dataset.records().dim().0, 3);
            }
        }

        // Bootstrap features
        {
            let mut iter = dataset.bootstrap_features(3, &mut rng);
            for _ in 1..5 {
                let dataset = iter.next().unwrap();
                assert_eq!(dataset.records().dim(), (2, 3));
            }
        }

        // Bootstrap both
        {
            let mut iter = dataset.bootstrap((10, 10), &mut rng);
            for _ in 1..5 {
                let dataset = iter.next().unwrap();
                assert_eq!(dataset.records().dim(), (10, 10));
            }
        }

        let linspace: Array1<f64> = Array1::linspace(0.0, 0.8, 100);
        let records = Array2::from_shape_vec((50, 2), linspace.to_vec()).unwrap();
        let targets: Array1<f64> = Array1::linspace(0.0, 0.8, 50);
        let dataset = Dataset::from((records, targets));

        //Split with ratio view
        let dataset_view = dataset.view();
        let (train, val) = dataset_view.split_with_ratio(0.5);
        assert_eq!(train.nsamples(), 25);
        assert_eq!(val.nsamples(), 25);

        // Split with ratio
        let (train, val) = dataset.split_with_ratio(0.25);
        assert_eq!(train.targets().dim().0, 13);
        assert_eq!(val.targets().dim().0, 37);
        assert_eq!(train.records().dim().0, 13);
        assert_eq!(val.records().dim().0, 37);

        // ------ Labels ------
        let dataset_multiclass =
            Dataset::from((array![[1., 2.], [2., 1.], [0., 0.]], array![0usize, 1, 2]));

        // One Vs All
        let datasets_one_vs_all = dataset_multiclass.one_vs_all().unwrap();

        assert_eq!(datasets_one_vs_all.len(), 3);

        for (_, dataset) in datasets_one_vs_all.iter() {
            assert_eq!(dataset.labels().iter().filter(|x| **x).count(), 1);
        }

        let dataset_multiclass = Dataset::from((
            array![[1., 2.], [2., 1.], [0., 0.], [2., 2.]],
            array![0, 1, 2, 2],
        ));

        // Frequencies with mask
        let freqs = dataset_multiclass.label_frequencies_with_mask(&[true, true, true, true]);
        assert_eq!(*freqs.get(&0).unwrap() as usize, 1);
        assert_eq!(*freqs.get(&1).unwrap() as usize, 1);
        assert_eq!(*freqs.get(&2).unwrap() as usize, 2);

        let freqs = dataset_multiclass.label_frequencies_with_mask(&[true, true, true, false]);
        assert_eq!(*freqs.get(&0).unwrap() as usize, 1);
        assert_eq!(*freqs.get(&1).unwrap() as usize, 1);
        assert_eq!(*freqs.get(&2).unwrap() as usize, 1);
    }

    #[test]
    fn dataset_view_implements_required_methods() -> Result<()> {
        let mut rng = SmallRng::seed_from_u64(42);
        let records = array![[1., 2.], [1., 2.]];
        let targets = array![0., 1.];

        // ------ Targets ------

        // New
        let dataset_view = DatasetView::from((records.view(), targets.view()));

        // Shuffle
        let _shuffled_owned = dataset_view.shuffle(&mut rng);

        // Bootstrap
        let mut iter = dataset_view.bootstrap_samples(3, &mut rng);
        for _ in 1..5 {
            let b_dataset = iter.next().unwrap();
            assert_eq!(b_dataset.records().dim().0, 3);
        }

        let linspace: Array1<f64> = Array1::linspace(0.0, 0.8, 100);
        let records = Array2::from_shape_vec((50, 2), linspace.to_vec()).unwrap();
        let targets: Array1<f64> = Array1::linspace(0.0, 0.8, 50);
        let dataset = Dataset::from((records, targets));

        // view ,Split with ratio view
        let view: DatasetView<f64, f64> = dataset.view();

        let (train, val) = view.split_with_ratio(0.5);
        assert_eq!(train.targets().len(), 25);
        assert_eq!(val.targets().len(), 25);
        assert_eq!(train.nsamples(), 25);
        assert_eq!(val.nsamples(), 25);

        // ------ Labels ------
        let dataset_multiclass =
            Dataset::from((array![[1., 2.], [2., 1.], [0., 0.]], array![0, 1, 2]));
        let view: DatasetView<f64, usize> = dataset_multiclass.view();

        // One Vs All
        let datasets_one_vs_all = view.one_vs_all()?;
        assert_eq!(datasets_one_vs_all.len(), 3);

        for (_, dataset) in datasets_one_vs_all.iter() {
            assert_eq!(dataset.labels().iter().filter(|x| **x).count(), 1);
        }

        let dataset_multiclass = Dataset::from((
            array![[1., 2.], [2., 1.], [0., 0.], [2., 2.]],
            array![0, 1, 2, 2],
        ));

        let view: DatasetView<f64, usize> = dataset_multiclass.view();

        // Frequencies with mask
        let freqs = view.label_frequencies_with_mask(&[true, true, true, true]);
        assert_eq!(*freqs.get(&0).unwrap() as usize, 1);
        assert_eq!(*freqs.get(&1).unwrap() as usize, 1);
        assert_eq!(*freqs.get(&2).unwrap() as usize, 2);

        let freqs = view.label_frequencies_with_mask(&[true, true, true, false]);
        assert_eq!(*freqs.get(&0).unwrap() as usize, 1);
        assert_eq!(*freqs.get(&1).unwrap() as usize, 1);
        assert_eq!(*freqs.get(&2).unwrap() as usize, 1);

        Ok(())
    }

    #[test]
    fn datasets_have_k_fold() {
        let linspace: Array1<f64> = Array1::linspace(0.0, 0.8, 100);
        let records = Array2::from_shape_vec((50, 2), linspace.to_vec()).unwrap();
        let targets: Array1<f64> = Array1::linspace(0.0, 0.8, 50);
        for (train, val) in DatasetView::from((records.view(), targets.view()))
            .fold(2)
            .into_iter()
        {
            assert_eq!(train.records().dim(), (25, 2));
            assert_eq!(val.records().dim(), (25, 2));
            assert_eq!(train.targets().dim(), (25, 1));
            assert_eq!(val.targets().dim(), (25, 1));
        }
        assert_eq!(Dataset::from((records, targets)).fold(10).len(), 10);

        let records =
            Array2::from_shape_vec((5, 2), vec![1., 1., 2., 2., 3., 3., 4., 4., 5., 5.]).unwrap();
        let targets = Array1::from_shape_vec(5, vec![1., 2., 3., 4., 5.]).unwrap();
        for (i, (train, val)) in Dataset::from((records, targets))
            .fold(5)
            .into_iter()
            .enumerate()
        {
            assert_eq!(val.records.row(0)[0] as usize, (i + 1));
            assert_eq!(val.records.row(0)[1] as usize, (i + 1));
            assert_eq!(val.targets.column(0)[0] as usize, (i + 1));

            for j in 0..4 {
                assert!(train.records.row(j)[0] as usize != (i + 1));
                assert!(train.records.row(j)[1] as usize != (i + 1));
                assert!(train.targets.column(0)[j] as usize != (i + 1));
            }
        }
    }

    #[test]
    fn check_iteration() {
        let dataset = Dataset::new(
            array![[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]],
            array![[1, 2], [3, 4], [5, 6]],
        );

        let res = dataset
            .target_iter()
            .map(|x| x.try_single_target().unwrap().to_owned())
            .collect::<Vec<_>>();

        assert_eq!(res, &[array![1, 3, 5], array![2, 4, 6]]);

        let res = dataset
            .feature_iter()
            .map(|x| x.records)
            .collect::<Vec<_>>();

        assert_eq!(
            res,
            &[
                array![[1.], [5.], [9.]],
                array![[2.], [6.], [10.]],
                array![[3.], [7.], [11.]],
                array![[4.], [8.], [12.]],
            ]
        );

        let res = dataset
            .sample_iter()
            .map(|(a, b)| (a.to_owned(), b.to_owned()))
            .collect::<Vec<_>>();

        assert_eq!(
            res,
            &[
                (array![1., 2., 3., 4.], array![1, 2]),
                (array![5., 6., 7., 8.], array![3, 4]),
                (array![9., 10., 11., 12.], array![5, 6]),
            ]
        );
    }

    use crate::traits::{Fit, PredictRef};
    use ndarray::ArrayView2;
    use thiserror::Error;

    struct MockFittable {
        mock_var: usize,
    }

    struct MockFittableResult {
        mock_var: usize,
    }

    #[derive(Error, Debug)]
    enum MockError {
        #[error(transparent)]
        LinfaError(#[from] crate::error::Error),
    }

    type MockResult<T> = std::result::Result<T, MockError>;

    impl<'a> Fit<ArrayView2<'a, f64>, ArrayView2<'a, f64>, MockError> for MockFittable {
        type Object = MockFittableResult;

        fn fit(
            &self,
            training_data: &DatasetView<f64, f64>,
        ) -> std::result::Result<Self::Object, MockError> {
            if self.mock_var == 0 {
                Err(MockError::LinfaError(Error::Parameters("0".to_string())))
            } else {
                Ok(MockFittableResult {
                    mock_var: training_data.nsamples(),
                })
            }
        }
    }

    impl<'b> PredictRef<ArrayView2<'b, f64>, Array1<f64>> for MockFittableResult {
        fn predict_ref<'a>(&'a self, _x: &'a ArrayView2<'b, f64>) -> Array1<f64> {
            array![0.]
        }
    }

    impl<'b> PredictRef<ArrayView2<'b, f64>, Array2<f64>> for MockFittableResult {
        fn predict_ref<'a>(&'a self, _x: &'a ArrayView2<'b, f64>) -> Array2<f64> {
            array![[0., 0.]]
        }
    }

    #[test]
    fn test_iter_fold() {
        let records =
            Array2::from_shape_vec((5, 2), vec![1., 1., 2., 2., 3., 3., 4., 4., 5., 5.]).unwrap();
        let targets = Array1::from_shape_vec(5, vec![1., 2., 3., 4., 5.]).unwrap();
        let mut dataset: Dataset<f64, f64> = (records, targets).into();
        let params = MockFittable { mock_var: 1 };

        for (i, (model, validation_set)) in
            dataset.iter_fold(5, |v| params.fit(v).unwrap()).enumerate()
        {
            assert_eq!(model.mock_var, 4);
            assert_eq!(validation_set.records().row(0)[0] as usize, i + 1);
            assert_eq!(validation_set.records().row(0)[1] as usize, i + 1);
            assert_eq!(validation_set.targets().column(0)[0] as usize, i + 1);
            assert_eq!(validation_set.records().dim(), (1, 2));
            assert_eq!(validation_set.targets().dim(), (1, 1));
        }
    }

    #[test]
    fn test_iter_fold_uneven_folds() {
        let records =
            Array2::from_shape_vec((5, 2), vec![1., 1., 2., 2., 3., 3., 4., 4., 5., 5.]).unwrap();
        let targets = Array1::from_shape_vec(5, vec![1., 2., 3., 4., 5.]).unwrap();
        let mut dataset: Dataset<f64, f64> = (records, targets).into();
        let params = MockFittable { mock_var: 1 };

        // If we request three folds from a dataset with 5 samples it will cut the
        // last two samples from the folds and always add them as a tail of the training
        // data
        for (i, (model, validation_set)) in
            dataset.iter_fold(3, |v| params.fit(v).unwrap()).enumerate()
        {
            assert_eq!(model.mock_var, 4);
            assert_eq!(validation_set.records().row(0)[0] as usize, i + 1);
            assert_eq!(validation_set.records().row(0)[1] as usize, i + 1);
            assert_eq!(validation_set.targets().column(0)[0] as usize, i + 1);
            assert_eq!(validation_set.records().dim(), (1, 2));
            assert_eq!(validation_set.targets().dim(), (1, 1));
            assert!(i < 3);
        }

        // the same goes for the last sample if we choose 4 folds
        for (i, (model, validation_set)) in
            dataset.iter_fold(4, |v| params.fit(v).unwrap()).enumerate()
        {
            assert_eq!(model.mock_var, 4);
            assert_eq!(validation_set.records().row(0)[0] as usize, i + 1);
            assert_eq!(validation_set.records().row(0)[1] as usize, i + 1);
            assert_eq!(validation_set.targets().column(0)[0] as usize, i + 1);
            assert_eq!(validation_set.records().dim(), (1, 2));
            assert_eq!(validation_set.targets().dim(), (1, 1));
            assert!(i < 4);
        }

        // if we choose 2 folds then again the last sample will be only
        // used for trainig
        for (i, (model, validation_set)) in
            dataset.iter_fold(2, |v| params.fit(v).unwrap()).enumerate()
        {
            assert_eq!(model.mock_var, 3);
            assert_eq!(validation_set.targets().dim(), (2, 1));
            assert!(i < 2);
        }
    }

    #[test]
    #[should_panic]
    fn iter_fold_panics_k_0() {
        let records =
            Array2::from_shape_vec((5, 2), vec![1., 1., 2., 2., 3., 3., 4., 4., 5., 5.]).unwrap();
        let targets = Array1::from_shape_vec(5, vec![1., 2., 3., 4., 5.]).unwrap();
        let mut dataset: Dataset<f64, f64> = (records, targets).into();
        let params = MockFittable { mock_var: 1 };
        let _ = dataset.iter_fold(0, |v| params.fit(v)).enumerate();
    }

    #[test]
    #[should_panic]
    fn iter_fold_panics_k_more_than_samples() {
        let records =
            Array2::from_shape_vec((5, 2), vec![1., 1., 2., 2., 3., 3., 4., 4., 5., 5.]).unwrap();
        let targets = Array1::from_shape_vec(5, vec![1., 2., 3., 4., 5.]).unwrap();
        let mut dataset: Dataset<f64, f64> = (records, targets).into();
        let params = MockFittable { mock_var: 1 };
        let _ = dataset.iter_fold(6, |v| params.fit(v)).enumerate();
    }

    #[test]
    fn test_st_cv_all_correct() {
        let records =
            Array2::from_shape_vec((5, 2), vec![1., 1., 2., 2., 3., 3., 4., 4., 5., 5.]).unwrap();
        let targets = Array1::from_shape_vec(5, vec![1., 2., 3., 4., 5.]).unwrap();
        let mut dataset: Dataset<f64, f64> = (records, targets).into();
        let params = vec![MockFittable { mock_var: 1 }, MockFittable { mock_var: 2 }];
        let acc = dataset
            .cross_validate(5, &params, |_pred, _truth| Ok(3.))
            .unwrap();
        assert_eq!(acc, array![3., 3.]);

        let mut dataset: Dataset<f64, f64> =
            (array![[1., 1.], [2., 2.]], array![[1., 2.], [3., 4.]]).into();

        let params = vec![MockFittable { mock_var: 1 }, MockFittable { mock_var: 2 }];
        let acc = dataset
            .cross_validate(2, &params, |_pred, _truth| Ok(3.))
            .unwrap();
        assert_eq!(acc, array![[3., 3.], [3., 3.]]);
    }
    #[test]
    #[should_panic(
        expected = "called `Result::unwrap()` on an `Err` value: LinfaError(Parameters(\"0\"))"
    )]
    fn test_st_cv_one_incorrect() {
        let records =
            Array2::from_shape_vec((5, 2), vec![1., 1., 2., 2., 3., 3., 4., 4., 5., 5.]).unwrap();
        let targets = Array1::from_shape_vec(5, vec![1., 2., 3., 4., 5.]).unwrap();
        let mut dataset: Dataset<f64, f64> = (records, targets).into();
        // second one should throw an error
        let params = vec![MockFittable { mock_var: 1 }, MockFittable { mock_var: 0 }];
        let acc: MockResult<Array1<_>> = dataset.cross_validate(5, &params, |_pred, _truth| Ok(0.));

        acc.unwrap();
    }

    #[test]
    #[should_panic(
        expected = "called `Result::unwrap()` on an `Err` value: LinfaError(Parameters(\"eval\"))"
    )]
    fn test_st_cv_incorrect_eval() {
        let records =
            Array2::from_shape_vec((5, 2), vec![1., 1., 2., 2., 3., 3., 4., 4., 5., 5.]).unwrap();
        let targets = Array1::from_shape_vec(5, vec![1., 2., 3., 4., 5.]).unwrap();
        let mut dataset: Dataset<f64, f64> = (records, targets).into();
        // second one should throw an error
        let params = vec![MockFittable { mock_var: 1 }, MockFittable { mock_var: 1 }];
        let err: MockResult<Array1<_>> = dataset.cross_validate(5, &params, |_pred, _truth| {
            if false {
                Ok(0f32)
            } else {
                Err(Error::Parameters("eval".to_string()))
            }
        });

        err.unwrap();
    }

    #[test]
    fn test_st_cv_mt_all_correct() {
        let records =
            Array2::from_shape_vec((5, 2), vec![1., 1., 2., 2., 3., 3., 4., 4., 5., 5.]).unwrap();
        let targets = array![[1., 1.], [2., 2.], [3., 3.], [4., 4.], [5., 5.]];
        let mut dataset: Dataset<f64, f64> = (records, targets).into();
        let params = vec![MockFittable { mock_var: 1 }, MockFittable { mock_var: 2 }];
        let acc = dataset
            .cross_validate_multi(5, &params, |_pred, _truth| Ok(array![5., 6.]))
            .unwrap();
        assert_eq!(acc.dim(), (params.len(), dataset.ntargets()));
        assert_eq!(acc, array![[5., 6.], [5., 6.]])
    }
    #[test]
    fn test_st_cv_mt_one_incorrect() {
        let records =
            Array2::from_shape_vec((5, 2), vec![1., 1., 2., 2., 3., 3., 4., 4., 5., 5.]).unwrap();
        let targets = Array1::from_shape_vec(5, vec![1., 2., 3., 4., 5.]).unwrap();
        let mut dataset: Dataset<f64, f64> = (records, targets).into();
        // second one should throw an error
        let params = vec![MockFittable { mock_var: 1 }, MockFittable { mock_var: 0 }];
        let err = dataset
            .cross_validate_multi(5, &params, |_pred, _truth| Ok(array![5.]))
            .unwrap_err();
        assert_eq!(err.to_string(), "invalid parameter 0".to_string());
    }

    #[test]
    fn test_st_cv_mt_incorrect_eval() {
        let records =
            Array2::from_shape_vec((5, 2), vec![1., 1., 2., 2., 3., 3., 4., 4., 5., 5.]).unwrap();
        let targets = Array1::from_shape_vec(5, vec![1., 2., 3., 4., 5.]).unwrap();
        let mut dataset: Dataset<f64, f64> = (records, targets).into();
        // second one should throw an error
        let params = vec![MockFittable { mock_var: 1 }, MockFittable { mock_var: 1 }];
        let err = dataset
            .cross_validate_multi(5, &params, |_pred, _truth| {
                if false {
                    Ok(array![0f32])
                } else {
                    Err(Error::Parameters("eval".to_string()))
                }
            })
            .unwrap_err();
        assert_eq!(err.to_string(), "invalid parameter eval".to_string());
    }

    #[test]
    fn test_with_labels_st() {
        let records = array![
            [0., 1.],
            [1., 2.],
            [2., 3.],
            [0., 4.],
            [1., 5.],
            [2., 6.],
            [0., 7.],
            [1., 8.],
            [2., 9.],
            [0., 10.]
        ];
        let targets = array![0, 1, 2, 0, 1, 2, 0, 1, 2, 0].insert_axis(Axis(1));
        let dataset = DatasetBase::from((records, targets));
        assert_eq!(dataset.nsamples(), 10);
        assert_eq!(dataset.ntargets(), 1);
        let dataset_no_0 = dataset.with_labels(&[1, 2]);
        assert_eq!(dataset_no_0.nsamples(), 6);
        assert_eq!(dataset_no_0.ntargets(), 1);
        assert_abs_diff_eq!(
            dataset_no_0.records,
            array![[1., 2.], [2., 3.], [1., 5.], [2., 6.], [1., 8.], [2., 9.]]
        );
        assert_abs_diff_eq!(
            dataset_no_0.try_single_target().unwrap(),
            array![1, 2, 1, 2, 1, 2]
        );
        let dataset_no_1 = dataset.with_labels(&[0, 2]);
        assert_eq!(dataset_no_1.nsamples(), 7);
        assert_eq!(dataset_no_1.ntargets(), 1);
        assert_abs_diff_eq!(
            dataset_no_1.records,
            array![
                [0., 1.],
                [2., 3.],
                [0., 4.],
                [2., 6.],
                [0., 7.],
                [2., 9.],
                [0., 10.]
            ]
        );
        assert_abs_diff_eq!(
            dataset_no_1.try_single_target().unwrap(),
            array![0, 2, 0, 2, 0, 2, 0]
        );
        let dataset_no_2 = dataset.with_labels(&[0, 1]);
        assert_eq!(dataset_no_2.nsamples(), 7);
        assert_eq!(dataset_no_2.ntargets(), 1);
        assert_abs_diff_eq!(
            dataset_no_2.records,
            array![
                [0., 1.],
                [1., 2.],
                [0., 4.],
                [1., 5.],
                [0., 7.],
                [1., 8.],
                [0., 10.]
            ]
        );
        assert_abs_diff_eq!(
            dataset_no_2.try_single_target().unwrap(),
            array![0, 1, 0, 1, 0, 1, 0]
        );
    }

    #[test]
    fn test_with_labels_mt() {
        let records = array![
            [0., 1.],
            [1., 2.],
            [2., 3.],
            [0., 4.],
            [1., 5.],
            [2., 6.],
            [0., 7.],
            [1., 8.],
            [2., 9.],
            [0., 10.]
        ];
        let targets = array![
            [0, 7],
            [1, 8],
            [2, 9],
            [0, 7],
            [1, 8],
            [2, 9],
            [0, 7],
            [1, 8],
            [2, 9],
            [0, 7]
        ];
        let dataset = DatasetBase::from((records, targets));
        assert_eq!(dataset.nsamples(), 10);
        assert_eq!(dataset.ntargets(), 2);
        // remove 0 from target 1 and 7 from target 2
        let dataset_no_07 = dataset.with_labels(&[1, 2, 8, 9]);
        assert_eq!(dataset_no_07.nsamples(), 6);
        assert_eq!(dataset_no_07.ntargets(), 2);
        assert_abs_diff_eq!(
            dataset_no_07.records,
            array![[1., 2.], [2., 3.], [1., 5.], [2., 6.], [1., 8.], [2., 9.]]
        );
        assert_abs_diff_eq!(
            dataset_no_07.as_multi_targets(),
            array![[1, 8], [2, 9], [1, 8], [2, 9], [1, 8], [2, 9]]
        );
        // remove label 1 from target 1 and label 7 from target 2: with labels is an "any" so all targets should be kept
        let dataset_no_17 = dataset.with_labels(&[0, 2, 8, 9]);
        assert_eq!(dataset_no_17.nsamples(), 10);
        assert_eq!(dataset_no_17.ntargets(), 2);
    }
}
