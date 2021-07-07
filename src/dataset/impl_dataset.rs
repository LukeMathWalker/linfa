use super::{
    super::traits::{Predict, PredictRef},
    iter::{ChunksIter, DatasetIter, Iter},
    AsTargets, AsTargetsMut, CountedTargets, Dataset, DatasetBase, DatasetView, Float,
    FromTargetArray, Label, Labels, Records, Result,
};
use crate::traits::Fit;
use ndarray::{
    concatenate, s, Array, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, ArrayViewMut2, Axis,
    Data, DataMut, Dimension, Ix1, Ix2, OwnedRepr,
};
use rand::{seq::SliceRandom, Rng};
use std::collections::HashMap;
use std::ops::AddAssign;

/// Implementation without constraints on records and targets
///
/// This implementation block provides methods for the creation and mutation of datasets. This
/// includes swapping the targets, return the records etc.
impl<R: Records, S> DatasetBase<R, S> {
    /// Create a new dataset from records and targets
    ///
    /// # Example
    ///
    /// ```ignore
    /// let dataset = Dataset::new(records, targets);
    /// ```
    pub fn new<T: IntoTargets<S>>(records: R, targets: T) -> DatasetBase<R, S> {
        let targets = targets.into();

        DatasetBase {
            records,
            targets,
            weights: Array1::zeros(0),
            feature_names: Vec::new(),
        }
    }

    /// Returns reference to targets
    pub fn targets(&self) -> &S {
        &self.targets
    }

    /// Returns optionally weights
    pub fn weights(&self) -> Option<&[f32]> {
        if !self.weights.is_empty() {
            Some(self.weights.as_slice().unwrap())
        } else {
            None
        }
    }

    /// Return a single weight
    ///
    /// The weight of the `idx`th observation is returned. If no weight is specified, then all
    /// observations are unweighted with default value `1.0`.
    pub fn weight_for(&self, idx: usize) -> f32 {
        self.weights.get(idx).copied().unwrap_or(1.0)
    }

    /// Returns feature names
    ///
    /// A feature name gives a human-readable string describing the purpose of a single feature.
    /// This allow the reader to understand its purpose while analysing results, for example
    /// correlation analysis or feature importance.
    pub fn feature_names(&self) -> Vec<String> {
        if !self.feature_names.is_empty() {
            self.feature_names.clone()
        } else {
            (0..self.records.nfeatures())
                .map(|idx| format!("feature-{}", idx))
                .collect()
        }
    }

    /// Return records of a dataset
    ///
    /// The records are data points from which predictions are made. This functions returns a
    /// reference to the record field.
    pub fn records(&self) -> &R {
        &self.records
    }

    /// Updates the records of a dataset
    ///
    /// This function overwrites the records in a dataset. It also invalidates the weights and
    /// feature names.
    pub fn with_records<T: Records>(self, records: T) -> DatasetBase<T, S> {
        DatasetBase {
            records,
            targets: self.targets,
            weights: Array1::zeros(0),
            feature_names: Vec::new(),
        }
    }

    /// Updates the targets of a dataset
    ///
    /// This function overwrites the targets in a dataset.
    pub fn with_targets<T>(self, targets: T) -> DatasetBase<R, T> {
        DatasetBase {
            records: self.records,
            targets,
            weights: self.weights,
            feature_names: self.feature_names,
        }
    }

    /// Updates the weights of a dataset
    pub fn with_weights(mut self, weights: Array1<f32>) -> DatasetBase<R, S> {
        self.weights = weights;

        self
    }

    /// Updates the feature names of a dataset
    pub fn with_feature_names<I: Into<String>>(mut self, names: Vec<I>) -> DatasetBase<R, S> {
        let feature_names = names.into_iter().map(|x| x.into()).collect();

        self.feature_names = feature_names;

        self
    }
}

impl<L, R: Records, T: AsTargets<Elem = L>> DatasetBase<R, T> {
    /// Map targets with a function `f`
    ///
    /// # Example
    ///
    /// ```
    /// let dataset = linfa_datasets::winequality()
    ///     .map_targets(|x| *x > 6);
    ///
    /// // dataset has now boolean targets
    /// println!("{:?}", dataset.targets());
    /// ```
    ///
    /// # Returns
    ///
    /// A modified dataset with new target type.
    ///
    pub fn map_targets<S, G: FnMut(&L) -> S>(self, fnc: G) -> DatasetBase<R, Array2<S>> {
        let DatasetBase {
            records,
            targets,
            weights,
            feature_names,
            ..
        } = self;

        let targets = targets.as_multi_targets();

        DatasetBase {
            records,
            targets: targets.map(fnc),
            weights,
            feature_names,
        }
    }

    /// Return the number of targets in the dataset
    ///
    /// # Example
    ///
    /// ```
    /// let dataset = linfa_datasets::winequality();
    ///
    /// println!("#targets {}", dataset.ntargets());
    /// ```
    ///
    pub fn ntargets(&self) -> usize {
        self.targets.as_multi_targets().len_of(Axis(1))
    }
}

impl<'a, F: Float, L, D, T> DatasetBase<ArrayBase<D, Ix2>, T>
where
    D: Data<Elem = F>,
    T: AsTargets<Elem = L>,
{
    /// Iterate over observations
    ///
    /// This function creates an iterator which produces tuples of data points and target value. The
    /// iterator runs once for each data point and, while doing so, holds an reference to the owned
    /// dataset.
    ///
    /// # Example
    /// ```
    /// let dataset = linfa_datasets::iris();
    ///
    /// for (x, y) in dataset.sample_iter() {
    ///     println!("{} => {}", x, y);
    /// }
    /// ```
    pub fn sample_iter(&'a self) -> Iter<'a, '_, F, T::Elem> {
        Iter::new(self.records.view(), self.targets.as_multi_targets())
    }
}

impl<'a, F: Float, L: 'a, D, T> DatasetBase<ArrayBase<D, Ix2>, T>
where
    D: Data<Elem = F>,
    T: AsTargets<Elem = L> + FromTargetArray<'a, L>,
{
    /// Creates a view of a dataset
    pub fn view(&'a self) -> DatasetBase<ArrayView2<'a, F>, T::View> {
        let records = self.records().view();
        let targets = T::new_targets_view(self.as_multi_targets());

        DatasetBase::new(records, targets)
            .with_feature_names(self.feature_names.clone())
            .with_weights(self.weights.clone())
    }

    /// Iterate over features
    ///
    /// This iterator produces dataset views with only a single feature, while the set of targets remain
    /// complete. It can be useful to compare each feature individual to all targets.
    pub fn feature_iter(&'a self) -> DatasetIter<'a, '_, ArrayBase<D, Ix2>, T> {
        DatasetIter::new(self, true)
    }

    /// Iterate over targets
    ///
    /// This functions creates an iterator which produces dataset views complete records, but only
    /// a single target each. Useful to train multiple single target models for a multi-target
    /// dataset.
    ///
    pub fn target_iter(&'a self) -> DatasetIter<'a, '_, ArrayBase<D, Ix2>, T> {
        DatasetIter::new(self, false)
    }
}

impl<L, R: Records, T: AsTargets<Elem = L>> AsTargets for DatasetBase<R, T> {
    type Elem = L;

    fn as_multi_targets(&self) -> ArrayView2<'_, Self::Elem> {
        self.targets.as_multi_targets()
    }
}

impl<L, R: Records, T: AsTargetsMut<Elem = L>> AsTargetsMut for DatasetBase<R, T> {
    type Elem = L;

    fn as_multi_targets_mut(&mut self) -> ArrayViewMut2<'_, Self::Elem> {
        self.targets.as_multi_targets_mut()
    }
}

#[allow(clippy::type_complexity)]
impl<'a, L: 'a, F: Float, T> DatasetBase<ArrayView2<'a, F>, T>
where
    T: AsTargets<Elem = L> + FromTargetArray<'a, L>,
{
    /// Split dataset into two disjoint chunks
    ///
    /// This function splits the observations in a dataset into two disjoint chunks. The splitting
    /// threshold is calculated with the `ratio`. For example a ratio of `0.9` allocates 90% to the
    /// first chunks and 9% to the second. This is often used in training, validation splitting
    /// procedures.
    pub fn split_with_ratio(
        &'a self,
        ratio: f32,
    ) -> (
        DatasetBase<ArrayView2<'a, F>, T::View>,
        DatasetBase<ArrayView2<'a, F>, T::View>,
    ) {
        let n = (self.nsamples() as f32 * ratio).ceil() as usize;
        let (records_first, records_second) = self.records.view().split_at(Axis(0), n);
        let (targets_first, targets_second) = self.targets.as_multi_targets().split_at(Axis(0), n);

        let targets_first = T::new_targets_view(targets_first);
        let targets_second = T::new_targets_view(targets_second);

        let (first_weights, second_weights) = if self.weights.len() == self.nsamples() {
            let a = self.weights.slice(s![..n]).to_vec();
            let b = self.weights.slice(s![n..]).to_vec();

            (Array1::from(a), Array1::from(b))
        } else {
            (Array1::zeros(0), Array1::zeros(0))
        };
        let dataset1 = DatasetBase::new(records_first, targets_first)
            .with_weights(first_weights)
            .with_feature_names(self.feature_names.clone());

        let dataset2 = DatasetBase::new(records_second, targets_second)
            .with_weights(second_weights)
            .with_feature_names(self.feature_names.clone());

        (dataset1, dataset2)
    }
}

impl<L: Label, T: Labels<Elem = L>, R: Records> Labels for DatasetBase<R, T> {
    type Elem = L;

    fn label_count(&self) -> Vec<HashMap<L, usize>> {
        self.targets().label_count()
    }
}

#[allow(clippy::type_complexity)]
impl<'a, 'b: 'a, F: Float, L: Label, T, D> DatasetBase<ArrayBase<D, Ix2>, T>
where
    D: Data<Elem = F>,
    T: AsTargets<Elem = L> + Labels<Elem = L>,
{
    /// Produce N boolean targets from multi-class targets
    ///
    /// Some algorithms (like SVM) don't support multi-class targets. This function splits a
    /// dataset into multiple binary target view of the same dataset.
    pub fn one_vs_all(
        &self,
    ) -> Result<
        Vec<(
            L,
            DatasetBase<ArrayView2<'_, F>, CountedTargets<bool, Array2<bool>>>,
        )>,
    > {
        let targets = self.targets().try_single_target()?;

        Ok(self
            .labels()
            .into_iter()
            .map(|label| {
                let targets = targets
                    .iter()
                    .map(|x| x == &label)
                    .collect::<Array1<_>>()
                    .insert_axis(Axis(1));

                let targets = CountedTargets::new(targets);

                (
                    label,
                    DatasetBase::new(self.records().view(), targets)
                        .with_feature_names(self.feature_names.clone())
                        .with_weights(self.weights.clone()),
                )
            })
            .collect())
    }
}

impl<L: Label, R: Records, S: AsTargets<Elem = L>> DatasetBase<R, S> {
    /// Calculates label frequencies from a dataset while masking certain samples.
    ///
    /// ### Parameters
    ///
    /// * `mask`: a boolean array that specifies which samples to include in the count
    ///
    /// ### Returns
    ///
    /// A mapping of the Dataset's samples to their frequencies
    pub fn label_frequencies_with_mask(&self, mask: &[bool]) -> HashMap<L, f32> {
        let mut freqs = HashMap::new();

        for (elms, val) in self
            .targets
            .as_multi_targets()
            .axis_iter(Axis(0))
            .enumerate()
            .filter(|(i, _)| *mask.get(*i).unwrap_or(&true))
            .map(|(i, x)| (x, self.weight_for(i)))
        {
            for elm in elms {
                if !freqs.contains_key(elm) {
                    freqs.insert(elm.clone(), 0.0);
                }

                *freqs.get_mut(elm).unwrap() += val;
            }
        }

        freqs
    }

    /// Calculates label frequencies from a dataset
    pub fn label_frequencies(&self) -> HashMap<L, f32> {
        self.label_frequencies_with_mask(&[])
    }
}

impl<F: Float, D: Data<Elem = F>, I: Dimension> From<ArrayBase<D, I>>
    for DatasetBase<ArrayBase<D, I>, Array2<()>>
{
    fn from(records: ArrayBase<D, I>) -> Self {
        let empty_targets = Array2::default((records.len_of(Axis(0)), 1));
        DatasetBase {
            records,
            targets: empty_targets,
            weights: Array1::zeros(0),
            feature_names: Vec::new(),
        }
    }
}

impl<F: Float, E, D, S> From<(ArrayBase<D, Ix2>, ArrayBase<S, Ix2>)>
    for DatasetBase<ArrayBase<D, Ix2>, ArrayBase<S, Ix2>>
where
    D: Data<Elem = F>,
    S: Data<Elem = E>,
{
    fn from(rec_tar: (ArrayBase<D, Ix2>, ArrayBase<S, Ix2>)) -> Self {
        DatasetBase {
            records: rec_tar.0,
            targets: rec_tar.1,
            weights: Array1::zeros(0),
            feature_names: Vec::new(),
        }
    }
}

impl<F: Float, E, D, S> From<(ArrayBase<D, Ix2>, ArrayBase<S, Ix1>)>
    for DatasetBase<ArrayBase<D, Ix2>, ArrayBase<S, Ix2>>
where
    D: Data<Elem = F>,
    S: Data<Elem = E>,
{
    fn from(rec_tar: (ArrayBase<D, Ix2>, ArrayBase<S, Ix1>)) -> Self {
        DatasetBase {
            records: rec_tar.0,
            targets: rec_tar.1.insert_axis(Axis(1)),
            weights: Array1::zeros(0),
            feature_names: Vec::new(),
        }
    }
}

impl<'b, F: Float, E: Copy + 'b, D, T> DatasetBase<ArrayBase<D, Ix2>, T>
where
    D: Data<Elem = F>,
    T: AsTargets<Elem = E> + FromTargetArray<'b, E>,
    T::Owned: AsTargets,
{
    /// Apply bootstrapping for samples and features
    ///
    /// Bootstrap aggregating is used for sub-sample generation and improves the accuracy and
    /// stability of machine learning algorithms. It samples data uniformly with replacement and
    /// generates datasets where elements may be shared. This selects a subset of observations as
    /// well as features.
    ///
    /// # Parameters
    ///
    ///  * `sample_feature_size`: The number of samples and features per bootstrap
    ///  * `rng`: The random number generator used in the sampling procedure
    ///
    ///  # Returns
    ///
    ///  An infinite Iterator yielding at each step a new bootstrapped dataset
    ///
    pub fn bootstrap<R: Rng>(
        &'b self,
        sample_feature_size: (usize, usize),
        rng: &'b mut R,
    ) -> impl Iterator<Item = DatasetBase<Array2<F>, <T as FromTargetArray<'b, E>>::Owned>> + 'b
    {
        std::iter::repeat(()).map(move |_| {
            // sample with replacement
            let indices = (0..sample_feature_size.0)
                .map(|_| rng.gen_range(0..self.nsamples()))
                .collect::<Vec<_>>();

            let records = self.records().select(Axis(0), &indices);
            let targets = T::new_targets(self.as_multi_targets().select(Axis(0), &indices));

            let indices = (0..sample_feature_size.1)
                .map(|_| rng.gen_range(0..self.nfeatures()))
                .collect::<Vec<_>>();

            let records = records.select(Axis(1), &indices);

            DatasetBase::new(records, targets)
        })
    }

    /// Apply sample bootstrapping
    ///
    /// Bootstrap aggregating is used for sub-sample generation and improves the accuracy and
    /// stability of machine learning algorithms. It samples data uniformly with replacement and
    /// generates datasets where elements may be shared. Only a sample subset is selected which
    /// retains all features and targets.
    ///
    /// # Parameters
    ///
    ///  * `num_samples`: The number of samples per bootstrap
    ///  * `rng`: The random number generator used in the sampling procedure
    ///
    ///  # Returns
    ///
    ///  An infinite Iterator yielding at each step a new bootstrapped dataset
    ///
    pub fn bootstrap_samples<R: Rng>(
        &'b self,
        num_samples: usize,
        rng: &'b mut R,
    ) -> impl Iterator<Item = DatasetBase<Array2<F>, <T as FromTargetArray<'b, E>>::Owned>> + 'b
    {
        std::iter::repeat(()).map(move |_| {
            // sample with replacement
            let indices = (0..num_samples)
                .map(|_| rng.gen_range(0..self.nsamples()))
                .collect::<Vec<_>>();

            let records = self.records().select(Axis(0), &indices);
            let targets = T::new_targets(self.as_multi_targets().select(Axis(0), &indices));

            DatasetBase::new(records, targets)
        })
    }

    /// Apply feature bootstrapping
    ///
    /// Bootstrap aggregating is used for sub-sample generation and improves the accuracy and
    /// stability of machine learning algorithms. It samples data uniformly with replacement and
    /// generates datasets where elements may be shared. Only a feature subset is selected while
    /// retaining all samples and targets.
    ///
    /// # Parameters
    ///
    ///  * `num_features`: The number of features per bootstrap
    ///  * `rng`: The random number generator used in the sampling procedure
    ///
    ///  # Returns
    ///
    ///  An infinite Iterator yielding at each step a new bootstrapped dataset
    ///
    pub fn bootstrap_features<R: Rng>(
        &'b self,
        num_features: usize,
        rng: &'b mut R,
    ) -> impl Iterator<Item = DatasetBase<Array2<F>, <T as FromTargetArray<'b, E>>::Owned>> + 'b
    {
        std::iter::repeat(()).map(move |_| {
            let targets = T::new_targets(self.as_multi_targets().to_owned());

            let indices = (0..num_features)
                .map(|_| rng.gen_range(0..self.nfeatures()))
                .collect::<Vec<_>>();

            let records = self.records.select(Axis(1), &indices);

            DatasetBase::new(records, targets)
        })
    }

    /// Produces a shuffled version of the current Dataset.
    ///
    /// ### Parameters
    ///
    /// * `rng`: the random number generator that will be used to shuffle the samples
    ///
    /// ### Returns
    ///
    /// A new shuffled version of the current Dataset
    pub fn shuffle<R: Rng>(&self, rng: &mut R) -> DatasetBase<Array2<F>, T::Owned> {
        let mut indices = (0..self.nsamples()).collect::<Vec<_>>();
        indices.shuffle(rng);

        let records = self.records().select(Axis(0), &indices);
        let targets = self.as_multi_targets().select(Axis(0), &indices);
        let targets = T::new_targets(targets);

        DatasetBase::new(records, targets)
    }

    #[allow(clippy::type_complexity)]
    /// Performs K-folding on the dataset.
    /// The dataset is divided into `k` "fold", each containing
    /// `(dataset size)/k` samples, used to generate `k` training-validation
    /// dataset pairs. Each pair contains a validation `Dataset` with `k` samples,
    ///  the ones contained in the i-th fold, and a training `Dataset` composed by the
    /// union of all the samples in the remaining folds.
    ///
    /// ### Parameters
    ///
    /// * `k`: the number of folds to apply
    ///
    /// ### Returns
    ///
    /// A vector of `k` training-validation Dataset pairs.
    ///
    /// ### Example
    ///
    /// ```rust
    /// use linfa::dataset::DatasetView;
    /// use ndarray::array;
    ///
    /// let records = array![[1.,1.], [2.,1.], [3.,2.], [4.,1.],[5., 3.], [6.,2.]];
    /// let targets = array![1, 1, 0, 1, 0, 0];
    ///
    /// let dataset : DatasetView<f64, usize> = (records.view(), targets.view()).into();
    /// let accuracies = dataset.fold(3).into_iter().map(|(train, valid)| {
    ///     // Here you can train your model and perform validation
    ///     
    ///     // let model = params.fit(&dataset);
    ///     // let predi = model.predict(&valid);
    ///     // predi.confusion_matrix(&valid).accuracy()  
    /// });
    /// ```
    ///  
    pub fn fold(
        &self,
        k: usize,
    ) -> Vec<(
        DatasetBase<Array2<F>, T::Owned>,
        DatasetBase<Array2<F>, T::Owned>,
    )> {
        let targets = self.as_multi_targets();
        let fold_size = targets.len() / k;
        let mut res = Vec::new();

        // Generates all k folds of records and targets
        let mut records_chunks: Vec<_> =
            self.records.axis_chunks_iter(Axis(0), fold_size).collect();
        let mut targets_chunks: Vec<_> = targets.axis_chunks_iter(Axis(0), fold_size).collect();

        // For each iteration, take the first chunk for both records and targets as the validation set and
        // concatenate all the other chunks to create the training set. In the end swap the first chunk with the
        // one in the next index so that it is ready for the next iteration
        for i in 0..k {
            let remaining_records = concatenate(Axis(0), &records_chunks.as_slice()[1..]).unwrap();
            let remaining_targets = concatenate(Axis(0), &targets_chunks.as_slice()[1..]).unwrap();

            res.push((
                // training
                DatasetBase::new(remaining_records, T::new_targets(remaining_targets)),
                // validation
                DatasetBase::new(
                    records_chunks[0].into_owned(),
                    T::new_targets(targets_chunks[0].into_owned()),
                ),
            ));

            // swap
            if i < k - 1 {
                records_chunks.swap(0, i + 1);
                targets_chunks.swap(0, i + 1);
            }
        }
        res
    }

    pub fn sample_chunks<'a: 'b>(&'b self, chunk_size: usize) -> ChunksIter<'b, 'a, F, T> {
        ChunksIter::new(self.records().view(), &self.targets, chunk_size, Axis(0))
    }

    pub fn to_owned(&self) -> DatasetBase<Array2<F>, T::Owned> {
        DatasetBase::new(
            self.records().to_owned(),
            T::new_targets(self.as_multi_targets().to_owned()),
        )
    }
}

macro_rules! assist_swap_array2 {
    ($slice: expr, $index: expr, $fold_size: expr, $features: expr) => {
        if $index != 0 {
            let adj_fold_size = $fold_size * $features;
            let start = adj_fold_size * $index;
            let (first_s, second_s) = $slice.split_at_mut(start);
            let (mut fold, _) = second_s.split_at_mut(adj_fold_size);
            first_s[..$fold_size * $features].swap_with_slice(&mut fold);
        }
    };
}

impl<'a, F: Float, E: Copy + 'a, D, S> DatasetBase<ArrayBase<D, Ix2>, ArrayBase<S, Ix2>>
where
    D: DataMut<Elem = F>,
    S: DataMut<Elem = E>,
{
    /// Allows to perform k-folding cross validation on fittable algorithms.
    ///
    /// Given in input a dataset, a value of k and the desired params for the fittable
    /// algorithm, returns an iterator over the k trained models and the
    /// associated validation set.
    ///
    /// The models are trained according to a closure specified
    /// as an input.
    ///
    /// ## Parameters
    ///
    /// - `k`: the number of folds to apply to the dataset
    /// - `params`: the desired parameters for the fittable algorithm at hand
    /// - `fit_closure`: a closure of the type `(params, training_data) -> fitted_model`
    /// that will be used to produce the trained model for each fold. The training data given in input
    /// won't outlive the closure.
    ///
    /// ## Returns
    ///
    /// An iterator over couples `(trained_model, validation_set)`.
    ///
    /// ## Panics
    ///
    /// This method will panic for any of the following three reasons:
    ///
    /// - The value of `k` provided is not positive;
    /// - The value of `k` provided is greater than the total number of samples in the dataset;
    /// - The dataset's data is not stored contiguously and in standard order;
    ///
    /// ## Example
    /// ```rust
    /// use linfa::traits::Fit;
    /// use linfa::dataset::{Dataset, DatasetView, Records};
    /// use ndarray::{array, ArrayView1, ArrayView2};
    /// use linfa::Error;
    ///
    /// struct MockFittable {}
    ///
    /// struct MockFittableResult {
    ///    mock_var: usize,
    /// }
    ///
    ///
    /// impl<'a> Fit<ArrayView2<'a,f64>, ArrayView2<'a, f64>, linfa::error::Error> for MockFittable {
    ///     type Object = MockFittableResult;
    ///
    ///     fn fit(&self, training_data: &DatasetView<f64, f64>) -> Result<Self::Object, linfa::error::Error> {
    ///         Ok(MockFittableResult {
    ///             mock_var: training_data.nsamples(),
    ///         })
    ///     }
    /// }
    ///
    /// let records = array![[1.,1.], [2.,2.], [3.,3.], [4.,4.], [5.,5.]];
    /// let targets = array![1.,2.,3.,4.,5.];
    /// let mut dataset: Dataset<f64, f64> = (records, targets).into();
    /// let params = MockFittable {};
    ///
    ///for (model,validation_set) in dataset.iter_fold(5, |v| params.fit(&v).unwrap()){
    ///     // Here you can use `model` and `validation_set` to
    ///     // assert the performance of the chosen algorithm
    /// }
    /// ```
    pub fn iter_fold<O, C: Fn(&DatasetView<F, E>) -> O>(
        &'a mut self,
        k: usize,
        fit_closure: C,
    ) -> impl Iterator<Item = (O, DatasetBase<ArrayView2<F>, ArrayView2<E>>)> {
        assert!(k > 0);
        assert!(k <= self.nsamples());
        let samples_count = self.nsamples();
        let fold_size = samples_count / k;

        let features = self.nfeatures();
        let targets = self.ntargets();

        let mut objs: Vec<O> = Vec::new();

        {
            let records_sl = self.records.as_slice_mut().unwrap();
            let mut targets_sl2 = self.targets.as_multi_targets_mut();
            let targets_sl = targets_sl2.as_slice_mut().unwrap();

            for i in 0..k {
                assist_swap_array2!(records_sl, i, fold_size, features);
                assist_swap_array2!(targets_sl, i, fold_size, targets);

                {
                    let train = DatasetBase::new(
                        ArrayView2::from_shape(
                            (samples_count - fold_size, features),
                            records_sl.split_at(fold_size * features).1,
                        )
                        .unwrap(),
                        ArrayView2::from_shape(
                            (samples_count - fold_size, targets),
                            targets_sl.split_at(fold_size * targets).1,
                        )
                        .unwrap(),
                    );

                    let obj = fit_closure(&train);
                    objs.push(obj);
                }

                assist_swap_array2!(records_sl, i, fold_size, features);
                assist_swap_array2!(targets_sl, i, fold_size, targets);
            }
        }

        objs.into_iter().zip(self.sample_chunks(fold_size))
    }

    /// Cross validation for multi-target algorithms
    ///
    /// Given a list of fittable models, cross validation
    /// is used to compare their performance according to some
    /// performance metric. To do so, k-folding is applied to the
    /// dataset and, for each fold, each model is trained on the training set
    /// and its performance is evaluated on the validation set. The performances
    /// collected for each model are then averaged over the number of folds.
    ///
    /// ### Parameters:
    ///
    /// - `k`: the number of folds to apply
    /// - `parameters`: a list of models to compare
    /// - `eval`: closure used to evaluate the performance of each trained model
    ///
    /// ### Returns
    ///
    /// An array of model performances, in the same order as the models in input, if no errors occur.
    /// The performance of each model is given as an array of performances, one for each target.
    /// Otherwise, it might return an Error in one of the following cases:
    ///
    /// - An error occurred during the fitting of one model
    /// - An error occurred inside the evaluation closure
    ///
    /// ### Example
    ///
    /// ```rust, ignore
    ///
    /// use linfa::prelude::*;
    ///
    /// // mutability needed for fast cross validation
    /// let mut dataset = linfa_datasets::diabetes();
    ///
    /// let models = vec![model1, model2, ... ];
    ///
    /// let r2_scores = dataset.cross_validate_multi(5,&models, |prediction, truth| prediction.r2(truth))?;
    ///
    /// ```
    pub fn cross_validate_multi<O, ER, M, FACC, C>(
        &'a mut self,
        k: usize,
        parameters: &[M],
        eval: C,
    ) -> std::result::Result<Array2<FACC>, ER>
    where
        ER: std::error::Error + std::convert::From<crate::error::Error>,
        M: for<'c> Fit<ArrayView2<'c, F>, ArrayView2<'c, E>, ER, Object = O>,
        O: for<'d> PredictRef<ArrayView2<'a, F>, Array2<E>>,
        FACC: Float,
        C: Fn(&Array2<E>, &ArrayView2<E>) -> std::result::Result<Array1<FACC>, crate::error::Error>,
    {
        let mut evaluations = Array2::from_elem((parameters.len(), self.ntargets()), FACC::zero());
        let folds_evaluations: std::result::Result<Vec<_>, ER> = self
            .iter_fold(k, |train| {
                let fit_result: std::result::Result<Vec<_>, ER> =
                    parameters.iter().map(|p| p.fit(train)).collect();
                fit_result
            })
            .map(|(models, valid)| {
                let targets = valid.targets();
                let models = models?;
                let mut eval_predictions =
                    Array2::from_elem((models.len(), targets.len()), FACC::zero());
                for (i, model) in models.iter().enumerate() {
                    let predicted = model.predict(valid.records());
                    let eval_pred = match eval(&predicted, targets) {
                        Err(e) => Err(ER::from(e)),
                        Ok(res) => Ok(res),
                    }?;
                    eval_predictions.row_mut(i).add_assign(&eval_pred);
                }
                Ok(eval_predictions)
            })
            .collect();

        for fold_evaluation in folds_evaluations? {
            evaluations.add_assign(&fold_evaluation)
        }
        Ok(evaluations / FACC::from(k).unwrap())
    }

    /// Cross validation for single target algorithms
    ///
    /// Given a list of fittable models, cross validation
    /// is used to compare their performance according to some
    /// performance metric. To do so, k-folding is applied to the
    /// dataset and, for each fold, each model is trained on the training set
    /// and its performance is evaluated on the validation set. The performances
    /// collected for each model are then averaged over the number of folds.
    ///
    /// ### Parameters:
    ///
    /// - `k`: the number of folds to apply
    /// - `parameters`: a list of models to compare
    /// - `eval`: closure used to evaluate the performance of each trained model. For single target
    ///    datasets, this closure is called once for each fold.
    ///    For multi-target datasets the closure is called, in each fold, once for every different target.
    ///    If there is the need to use different evaluations for each target, take a look at the
    ///    [`cross_validate_multi`](struct.DatasetBase.html#method.cross_validate_multi) method.
    ///
    /// ### Returns
    ///
    /// On succesful evalutation it returns an array of model performances, in the same order as the models in input.
    ///
    /// It returns an Error in one of the following cases:
    ///
    /// - An error occurred during the fitting of one model
    /// - An error occurred inside the evaluation closure
    ///
    /// ### Example
    ///
    /// ```rust, ignore
    ///
    /// use linfa::prelude::*;
    ///
    /// // mutability needed for fast cross validation
    /// let mut dataset = linfa_datasets::diabetes();
    ///
    /// let models = vec![model1, model2, ... ];
    ///
    /// let r2_scores = dataset.cross_validate(5,&models, |prediction, truth| prediction.r2(truth))?;
    ///
    /// ```
    pub fn cross_validate<O, ER, M, FACC, C, I>(
        &'a mut self,
        k: usize,
        parameters: &[M],
        eval: C,
    ) -> std::result::Result<ArrayBase<OwnedRepr<FACC>, I>, ER>
    where
        ER: std::error::Error + std::convert::From<crate::error::Error>,
        M: for<'c> Fit<ArrayView2<'c, F>, ArrayView2<'c, E>, ER, Object = O>,
        O: for<'d> PredictRef<ArrayView2<'a, F>, ArrayBase<OwnedRepr<E>, I>>,
        FACC: Float,
        C: Fn(&ArrayView1<E>, &ArrayView1<E>) -> std::result::Result<FACC, crate::error::Error>,
        I: Dimension,
    {
        // construct shape as either vector or matrix
        let mut shape = match I::NDIM {
            Some(1) | Some(2) => Ok(I::zeros(I::NDIM.unwrap())),
            _ => Err(crate::Error::NdShape(ndarray::ShapeError::from_kind(
                ndarray::ErrorKind::IncompatibleShape,
            ))),
        }?;

        // assign shape form of output
        let mut tmp = shape.as_array_view_mut();
        tmp[0] = parameters.len();
        if tmp.len() == 2 {
            tmp[1] = self.ntargets();
        }

        let folds_evaluations = self
            .iter_fold(k, |train| {
                let fit_result: std::result::Result<Vec<_>, ER> =
                    parameters.iter().map(|p| p.fit(train)).collect();
                fit_result
            })
            .map(|(models, valid)| {
                let targets = valid.as_multi_targets();
                let models = models?;

                let eval_predictions = models
                    .iter()
                    .map(|m| {
                        let nsamples = valid.nsamples();
                        let predicted = m.predict(valid.records());

                        // reshape to ensure that matrix has two dimensions
                        let ntargets = if predicted.ndim() == 1 {
                            1
                        } else {
                            predicted.len_of(Axis(1))
                        };

                        let predicted: Array2<_> =
                            predicted.into_shape((nsamples, ntargets)).unwrap();

                        predicted
                            .gencolumns()
                            .into_iter()
                            .zip(targets.gencolumns().into_iter())
                            .map(|(p, t)| eval(&p.view(), &t).map_err(ER::from))
                            .collect()
                    })
                    .collect::<std::result::Result<Vec<Vec<FACC>>, ER>>()?
                    .into_iter()
                    .flatten()
                    .collect();

                Ok(Array::from_shape_vec(shape.clone(), eval_predictions).unwrap())
            })
            .collect::<std::result::Result<Vec<_>, ER>>();

        let res = folds_evaluations?
            .into_iter()
            .fold(Array::<FACC, _>::zeros(shape.clone()), std::ops::Add::add);

        Ok(res / FACC::cast(k))
    }
}

impl<F: Float, E> Dataset<F, E> {
    /// Split dataset into two disjoint chunks
    ///
    /// This function splits the observations in a dataset into two disjoint chunks. The splitting
    /// threshold is calculated with the `ratio`. If the input Dataset contains `n` samples then the
    /// two new Datasets will have respectively `n * ratio` and `n - (n*ratio)` samples.
    /// For example a ratio of `0.9` allocates 90% to the
    /// first chunks and 10% to the second. This is often used in training, validation splitting
    /// procedures.
    ///
    /// ### Parameters
    ///
    /// * `ratio`: the ratio of samples in the input Dataset to include in the first output one
    ///
    /// ### Returns
    ///  
    /// The input Dataset split into two according to the input ratio.
    pub fn split_with_ratio(mut self, ratio: f32) -> (Self, Self) {
        let (nfeatures, ntargets) = (self.nfeatures(), self.ntargets());

        let n1 = (self.nsamples() as f32 * ratio).ceil() as usize;
        let n2 = self.nsamples() - n1;

        let feature_names = self.feature_names();

        // split records into two disjoint arrays
        let mut array_buf = self.records.into_raw_vec();
        let second_array_buf = array_buf.split_off(n1 * nfeatures);

        let first = Array2::from_shape_vec((n1, nfeatures), array_buf).unwrap();
        let second = Array2::from_shape_vec((n2, nfeatures), second_array_buf).unwrap();

        // split targets into two disjoint Vec
        let mut array_buf = self.targets.into_raw_vec();
        let second_array_buf = array_buf.split_off(n1 * ntargets);

        let first_targets = Array2::from_shape_vec((n1, ntargets), array_buf).unwrap();
        let second_targets = Array2::from_shape_vec((n2, ntargets), second_array_buf).unwrap();

        // split weights into two disjoint Vec
        let second_weights = if self.weights.len() == n1 + n2 {
            let mut weights = self.weights.into_raw_vec();

            let weights2 = weights.split_off(n1);
            self.weights = Array1::from(weights);

            Array1::from(weights2)
        } else {
            Array1::zeros(0)
        };

        // create new datasets with attached weights
        let dataset1 = Dataset::new(first, first_targets)
            .with_weights(self.weights)
            .with_feature_names(feature_names.clone());
        let dataset2 = Dataset::new(second, second_targets)
            .with_weights(second_weights)
            .with_feature_names(feature_names);

        (dataset1, dataset2)
    }
}

impl<F: Float, D, E, T, O> Predict<ArrayBase<D, Ix2>, DatasetBase<ArrayBase<D, Ix2>, T>> for O
where
    D: Data<Elem = F>,
    T: AsTargets<Elem = E>,
    O: PredictRef<ArrayBase<D, Ix2>, T>,
{
    fn predict(&self, records: ArrayBase<D, Ix2>) -> DatasetBase<ArrayBase<D, Ix2>, T> {
        let new_targets = self.predict_ref(&records);
        DatasetBase::new(records, new_targets)
    }
}

impl<F: Float, D, E, T, O>
    Predict<ArrayBase<D, Ix2>, std::result::Result<DatasetBase<ArrayBase<D, Ix2>, T>, E>> for O
where
    D: Data<Elem = F>,
    O: PredictRef<ArrayBase<D, Ix2>, std::result::Result<T, E>>,
{
    fn predict(
        &self,
        records: ArrayBase<D, Ix2>,
    ) -> std::result::Result<DatasetBase<ArrayBase<D, Ix2>, T>, E> {
        let new_targets = self.predict_ref(&records)?;
        Ok(DatasetBase::new(records, new_targets))
    }
}

impl<F: Float, R, T, E, S, O> Predict<DatasetBase<R, T>, DatasetBase<R, S>> for O
where
    R: Records<Elem = F>,
    S: AsTargets<Elem = E>,
    O: PredictRef<R, S>,
{
    fn predict(&self, ds: DatasetBase<R, T>) -> DatasetBase<R, S> {
        let new_targets = self.predict_ref(&ds.records);
        DatasetBase::new(ds.records, new_targets)
    }
}

impl<F: Float, R, T, E, S, O> Predict<DatasetBase<R, T>, std::result::Result<DatasetBase<R, S>, E>>
    for O
where
    R: Records<Elem = F>,
    O: PredictRef<R, std::result::Result<S, E>>,
{
    fn predict(&self, ds: DatasetBase<R, T>) -> std::result::Result<DatasetBase<R, S>, E> {
        let new_targets = self.predict_ref(&ds.records)?;
        Ok(DatasetBase::new(ds.records, new_targets))
    }
}

impl<'a, F: Float, R, T, S, O> Predict<&'a DatasetBase<R, T>, S> for O
where
    R: Records<Elem = F>,
    O: PredictRef<R, S>,
{
    fn predict(&self, ds: &'a DatasetBase<R, T>) -> S {
        self.predict_ref(&ds.records)
    }
}

impl<'a, F: Float, D, T, O> Predict<&'a ArrayBase<D, Ix2>, T> for O
where
    D: Data<Elem = F>,
    O: PredictRef<ArrayBase<D, Ix2>, T>,
{
    fn predict(&self, records: &'a ArrayBase<D, Ix2>) -> T {
        self.predict_ref(records)
    }
}

impl<L: Label, S: Labels<Elem = L>> CountedTargets<L, S> {
    pub fn new(targets: S) -> Self {
        let labels = targets.label_count();

        CountedTargets { targets, labels }
    }
}

pub trait IntoTargets<T> {
    fn into(self) -> T;
}

impl<F, D: Data<Elem = F>> IntoTargets<ArrayBase<D, Ix2>> for ArrayBase<D, Ix1> {
    fn into(self) -> ArrayBase<D, Ix2> {
        self.insert_axis(Axis(1))
    }
}

impl<T> IntoTargets<T> for T {
    fn into(self) -> T {
        self
    }
}
