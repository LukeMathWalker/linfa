use std::cmp::Ordering;

use crate::k_means::hyperparameters::{KMeansHyperParams, KMeansHyperParamsBuilder};
use crate::{
    k_means::errors::{KMeansError, Result},
    KMeansInit,
};
use linfa::{prelude::*, DatasetBase, Float};
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, DataMut, Ix1, Ix2, Zip};
use ndarray_rand::rand::Rng;
use ndarray_rand::rand::SeedableRng;
use ndarray_stats::DeviationExt;
use rand_isaac::Isaac64Rng;

#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Debug, PartialEq)]
/// K-means clustering aims to partition a set of unlabeled observations into clusters,
/// where each observation belongs to the cluster with the nearest mean.
///
/// The mean of the points within a cluster is called *centroid*.
///
/// Given the set of centroids, you can assign an observation to a cluster
/// choosing the nearest centroid.
///
/// We provide a modified version of the _standard algorithm_ (also known as Lloyd's Algorithm),
/// called m_k-means, which uses a slightly modified update step to avoid problems with empty
/// clusters. We also provide an incremental version of the algorithm that runs on smaller batches
/// of input data.
///
/// More details on the algorithm can be found in the next section or
/// [here](https://en.wikipedia.org/wiki/K-means_clustering). Details on m_k-means can be found
/// [here](https://www.researchgate.net/publication/228414762_A_Modified_k-means_Algorithm_to_Avoid_Empty_Clusters).
///
/// ## Standard algorithm
///
/// K-means is an iterative algorithm: it progressively refines the choice of centroids.
///
/// It's guaranteed to converge, even though it might not find the optimal set of centroids
/// (unfortunately it can get stuck in a local minimum, finding the optimal minimum if NP-hard!).
///
/// There are three steps in the standard algorithm:
/// - initialisation step: select initial centroids using one of our provided algorithms.
/// - assignment step: assign each observation to the nearest cluster
///                    (minimum distance between the observation and the cluster's centroid);
/// - update step: recompute the centroid of each cluster.
///
/// The initialisation step is a one-off, done at the very beginning.
/// Assignment and update are repeated in a loop until convergence is reached (either the
/// euclidean distance between the old and the new clusters is below `tolerance` or
/// we exceed the `max_n_iterations`).
///
/// ## Incremental Algorithm
///
/// In addition to the standard algorithm, we also provide an incremental version of K-means known
/// as Mini-Batch K-means. In this algorithm, the dataset is divided into small batches, and the
/// assignment and update steps are performed on each batch instead of the entire dataset. The
/// update step also takes previous update steps into account when updating the centroids.
///
/// Due to using smaller batches, Mini-Batch K-means takes significantly less time to execute than
/// the standard K-means algorithm, although it may yield slightly worse centroids.
///
/// More details on Mini-Batch K-means can be found [here](https://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf).
///
/// ## Parallelisation
///
/// The work performed by the assignment step does not require any coordination:
/// the closest centroid for each point can be computed independently from the
/// closest centroid for any of the remaining points.
///
/// This makes it a good candidate for parallel execution: `KMeans::fit` parallelises the
/// assignment step thanks to the `rayon` feature in `ndarray`.
///
/// The update step requires a bit more coordination (computing a rolling mean in
/// parallel) but it is still parallelisable.
/// Nonetheless, our first attempts have not improved performance
/// (most likely due to our strategy used to split work between threads), hence
/// the update step is currently executed on a single thread.
///
/// ## Tutorial
///
/// Let's do a walkthrough of a training-predict-save example.
///
/// ```
/// use linfa::DatasetBase;
/// use linfa::traits::{Fit, IncrementalFit, Predict};
/// use linfa_clustering::{KMeansHyperParams, KMeans, generate_blobs};
/// use ndarray::{Axis, array, s};
/// use ndarray_rand::rand::SeedableRng;
/// use rand_isaac::Isaac64Rng;
/// use approx::assert_abs_diff_eq;
///
/// // Our random number generator, seeded for reproducibility
/// let seed = 42;
/// let mut rng = Isaac64Rng::seed_from_u64(seed);
///
/// // `expected_centroids` has shape `(n_centroids, n_features)`
/// // i.e. three points in the 2-dimensional plane
/// let expected_centroids = array![[0., 1.], [-10., 20.], [-1., 10.]];
/// // Let's generate a synthetic dataset: three blobs of observations
/// // (100 points each) centered around our `expected_centroids`
/// let data = generate_blobs(100, &expected_centroids, &mut rng);
/// let n_clusters = expected_centroids.len_of(Axis(0));
///
/// // Standard K-means
/// {
///     let observations = DatasetBase::from(data.clone());
///     // Let's configure and run our K-means algorithm
///     // We use the builder pattern to specify the hyperparameters
///     // `n_clusters` is the only mandatory parameter.
///     // If you don't specify the others (e.g. `n_runs`, `tolerance`, `max_n_iterations`)
///     // default values will be used.
///     let model = KMeans::params_with_rng(n_clusters, rng.clone())
///         .tolerance(1e-2)
///         .fit(&observations)
///         .expect("KMeans fitted");
///
///     // Once we found our set of centroids, we can also assign new points to the nearest cluster
///     let new_observation = DatasetBase::from(array![[-9., 20.5]]);
///     // Predict returns the **index** of the nearest cluster
///     let dataset = model.predict(new_observation);
///     // We can retrieve the actual centroid of the closest cluster using `.centroids()`
///     let closest_centroid = &model.centroids().index_axis(Axis(0), dataset.targets()[0]);
///     assert_abs_diff_eq!(closest_centroid.to_owned(), &array![-10., 20.], epsilon = 1e-1);
/// }
///
/// // Incremental K-means
/// {
///     let batch_size = 100;
///     // Shuffling the dataset is one way of ensuring that the batches contain random points from
///     // the dataset, which is required for the algorithm to work properly
///     let observations = DatasetBase::from(data.clone()).shuffle(&mut rng);
///
///     let n_clusters = expected_centroids.nrows();
///     let clf = KMeans::params_with_rng(n_clusters, rng.clone())
///         .tolerance(1e-3)
///         .build();
///
///     // Repeatedly run fit_with on every batch in the dataset until we have converged
///     let model = observations
///         .sample_chunks(batch_size)
///         .cycle()
///         .try_fold(None, |current, batch| {
///             let (model, converged) = clf.fit_with(current, &batch);
///             if converged {
///                 // Once we have converged, raise an error to break from the iterator
///                 Err(model)
///             } else {
///                 Ok(Some(model))
///             }
///         })
///         .unwrap_err();
///
///     let new_observation = DatasetBase::from(array![[-9., 20.5]]);
///     let dataset = model.predict(new_observation);
///     let closest_centroid = &model.centroids().index_axis(Axis(0), dataset.targets()[0]);
///     assert_abs_diff_eq!(closest_centroid.to_owned(), &array![-10., 20.], epsilon = 1e-1);
/// }
/// ```
///
/*///
/// // The model can be serialised (and deserialised) to disk using serde
/// // We'll use the JSON format here for simplicity
/// let filename = "k_means_model.json";
/// let writer = std::fs::File::create(filename).expect("Failed to open file.");
/// serde_json::to_writer(writer, &model).expect("Failed to serialise model.");
///
/// let reader = std::fs::File::open(filename).expect("Failed to open file.");
/// let loaded_model: KMeans<f64> = serde_json::from_reader(reader).expect("Failed to deserialise model");
///
/// assert_abs_diff_eq!(model.centroids(), loaded_model.centroids(), epsilon = 1e-10);
/// assert_eq!(model.hyperparameters(), loaded_model.hyperparameters());
/// ```
*/
pub struct KMeans<F: Float> {
    centroids: Array2<F>,
    cluster_count: Array1<F>,
    inertia: F,
}

impl<F: Float> KMeans<F> {
    pub fn params(nclusters: usize) -> KMeansHyperParamsBuilder<F, Isaac64Rng> {
        KMeansHyperParams::new(nclusters)
    }

    pub fn params_with_rng<R: Rng + Clone>(
        nclusters: usize,
        rng: R,
    ) -> KMeansHyperParamsBuilder<F, R> {
        KMeansHyperParams::new_with_rng(nclusters, rng)
    }

    /// Return the set of centroids as a 2-dimensional matrix with shape
    /// `(n_centroids, n_features)`.
    pub fn centroids(&self) -> &Array2<F> {
        &self.centroids
    }

    /// Return the number of training points belonging to each cluster
    pub fn cluster_count(&self) -> &Array1<F> {
        &self.cluster_count
    }

    /// Return the sum of distances between each training point and its closest centroid, averaged
    /// across all training points.  When training incrementally, this value is computed on the
    /// most recent batch.
    pub fn inertia(&self) -> F {
        self.inertia
    }
}

impl<F: Float, R: Rng + Clone + SeedableRng, D: Data<Elem = F>, T>
    Fit<ArrayBase<D, Ix2>, T, KMeansError> for KMeansHyperParams<F, R>
{
    type Object = KMeans<F>;

    /// Given an input matrix `observations`, with shape `(n_observations, n_features)`,
    /// `fit` identifies `n_clusters` centroids based on the training data distribution.
    ///
    /// An instance of `KMeans` is returned.
    ///
    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object> {
        let mut rng = self.rng();
        let observations = dataset.records().view();
        let n_samples = dataset.nsamples();

        let mut min_inertia = F::infinity();
        let mut best_centroids = None;
        let mut best_iter = None;
        let mut memberships = Array1::zeros(n_samples);
        let mut dists = Array1::zeros(n_samples);

        let n_runs = self.n_runs();

        for _ in 0..n_runs {
            let mut inertia = min_inertia;
            let mut centroids = self
                .init_method()
                .run(self.n_clusters(), observations, &mut rng);
            let mut converged_iter: Option<u64> = None;
            for n_iter in 0..self.max_n_iterations() {
                update_memberships_and_dists(
                    &centroids,
                    &observations,
                    &mut memberships,
                    &mut dists,
                );
                let new_centroids = compute_centroids(&centroids, &observations, &memberships);
                inertia = dists.sum();
                let distance = centroids
                    .sq_l2_dist(&new_centroids)
                    .expect("Failed to compute distance");
                centroids = new_centroids;
                if distance < self.tolerance() {
                    converged_iter = Some(n_iter);
                    break;
                }
            }

            // We keep the centroids which minimize the inertia (defined as the sum of
            // the squared distances of the closest centroid for all observations)
            // over the n runs of the KMeans algorithm.
            if inertia < min_inertia {
                min_inertia = inertia;
                best_centroids = Some(centroids.clone());
                best_iter = converged_iter;
            }
        }

        match best_iter {
            Some(_n_iter) => match best_centroids {
                Some(centroids) => {
                    let mut cluster_count = Array1::zeros(self.n_clusters());
                    memberships
                        .iter()
                        .for_each(|&c| cluster_count[c] += F::one());
                    Ok(KMeans {
                        centroids,
                        cluster_count,
                        inertia: min_inertia / F::cast(dataset.nsamples()),
                    })
                }
                _ => Err(KMeansError::InertiaError(
                    "No inertia improvement (-inf)".to_string(),
                )),
            },
            None => Err(KMeansError::NotConverged(format!(
                "KMeans fitting algorithm {} did not converge. Try different init parameters, \
                or increase max_n_iterations, tolerance or check for degenerate data.",
                (n_runs + 1)
            ))),
        }
    }
}

impl<'a, F: Float, R: Rng + Clone + SeedableRng, D: Data<Elem = F>, T>
    IncrementalFit<'a, ArrayBase<D, Ix2>, T> for KMeansHyperParams<F, R>
{
    type ObjectIn = Option<KMeans<F>>;
    type ObjectOut = (KMeans<F>, bool);

    /// Performs a single batch update of the Mini-Batch K-means algorithm.
    ///
    /// Given an input matrix `observations`, with shape `(n_batch, n_features)` and a previous
    /// `KMeans` model, the model's centroids are updated with the input matrix. If `model` is
    /// `None`, then it's initialized using the specified initialization algorithm. The return
    /// value consists of the updated model and a `bool` value that indicates whether the algorithm
    /// has converged.
    fn fit_with(
        &self,
        model: Self::ObjectIn,
        dataset: &'a DatasetBase<ArrayBase<D, Ix2>, T>,
    ) -> Self::ObjectOut {
        let mut rng = self.rng();
        let observations = dataset.records().view();
        let n_samples = dataset.nsamples();

        let mut model = match model {
            Some(model) => model,
            None => {
                let centroids = if let KMeansInit::Precomputed(centroids) = self.init_method() {
                    // If using precomputed centroids, don't run the init algorithm multiple times
                    // since it's pointless
                    centroids.clone()
                } else {
                    let mut dists = Array1::zeros(n_samples);
                    // Initial centroids derived from the first batch by running the init algorithm
                    // n_runs times and taking the centroids with the lowest inertia
                    (0..self.n_runs())
                        .map(|_| {
                            let centroids =
                                self.init_method()
                                    .run(self.n_clusters(), observations, &mut rng);
                            update_min_dists(&centroids, &observations, &mut dists);
                            (centroids, dists.sum())
                        })
                        .min_by(|(_, d1), (_, d2)| {
                            if d1 < d2 {
                                Ordering::Less
                            } else {
                                Ordering::Greater
                            }
                        })
                        .unwrap()
                        .0
                };
                KMeans {
                    centroids,
                    cluster_count: Array1::zeros(self.n_clusters()),
                    inertia: F::zero(),
                }
            }
        };

        let mut memberships = Array1::zeros(n_samples);
        let mut dists = Array1::zeros(n_samples);
        update_memberships_and_dists(
            &model.centroids,
            &observations,
            &mut memberships,
            &mut dists,
        );
        let new_centroids = compute_centroids_incremental(
            &observations,
            &memberships,
            &model.centroids,
            &mut model.cluster_count,
        );
        model.inertia = dists.sum() / F::cast(n_samples);
        let dist = model.centroids.sq_l2_dist(&new_centroids).unwrap();
        model.centroids = new_centroids;

        (model, dist < self.tolerance())
    }
}

impl<'a, F: Float, R: Rng + SeedableRng + Clone> KMeansHyperParamsBuilder<F, R> {
    /// Shortcut for `.build().fit()`
    pub fn fit<D: Data<Elem = F>, T>(
        self,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, T>,
    ) -> Result<KMeans<F>> {
        self.build().fit(dataset)
    }
}

impl<F: Float, D: Data<Elem = F>> Transformer<&ArrayBase<D, Ix2>, Array1<F>> for KMeans<F> {
    /// Given an input matrix `observations`, with shape `(n_observations, n_features)`,
    /// `transform` returns, for each observation, its squared distance to its centroid.
    fn transform(&self, observations: &ArrayBase<D, Ix2>) -> Array1<F> {
        let mut dists = Array1::zeros(observations.nrows());
        update_min_dists(&self.centroids, &observations.view(), &mut dists);
        dists
    }
}

impl<F: Float, D: Data<Elem = F>> PredictRef<ArrayBase<D, Ix2>, Array1<usize>> for KMeans<F> {
    /// Given an input matrix `observations`, with shape `(n_observations, n_features)`,
    /// `predict` returns, for each observation, the index of the closest cluster/centroid.
    ///
    /// You can retrieve the centroid associated to an index using the
    /// [`centroids` method](#method.centroids).
    fn predict_ref<'a>(&'_ self, observations: &ArrayBase<D, Ix2>) -> Array1<usize> {
        compute_cluster_memberships(&self.centroids, &observations.view())
    }
}

impl<F: Float, D: Data<Elem = F>> PredictRef<ArrayBase<D, Ix1>, usize> for KMeans<F> {
    /// Given one input observation, return the index of its closest cluster
    ///
    /// You can retrieve the centroid associated to an index using the
    /// [`centroids` method](#method.centroids).
    fn predict_ref<'a>(&'_ self, observation: &ArrayBase<D, Ix1>) -> usize {
        closest_centroid(&self.centroids, &observation).0
    }
}

/// We compute inertia defined as the sum of the squared distances
/// of the closest centroid for all observations.
pub fn compute_inertia<F: Float>(
    centroids: &ArrayBase<impl Data<Elem = F> + Sync, Ix2>,
    observations: &ArrayBase<impl Data<Elem = F>, Ix2>,
    cluster_memberships: &ArrayBase<impl Data<Elem = usize>, Ix1>,
) -> F {
    let mut dists = Array1::<F>::zeros(observations.nrows());
    Zip::from(observations.genrows())
        .and(cluster_memberships)
        .and(&mut dists)
        .par_apply(|observation, &cluster_membership, d| {
            *d = centroids
                .row(cluster_membership)
                .sq_l2_dist(&observation)
                .expect("Failed to compute distance");
        });
    dists.sum()
}

/// K-means is an iterative algorithm.
/// We will perform the assignment and update steps until we are satisfied
/// (according to our convergence criteria).
///
/// If you check the `compute_cluster_memberships` function,
/// you can see that it expects to receive centroids as a 2-dimensional array.
///
/// `compute_centroids` returns a 2-dimensional array,
/// where the i-th row corresponds to the i-th cluster.
fn compute_centroids<F: Float>(
    old_centroids: &Array2<F>,
    // (n_observations, n_features)
    observations: &ArrayBase<impl Data<Elem = F>, Ix2>,
    // (n_observations,)
    cluster_memberships: &ArrayBase<impl Data<Elem = usize>, Ix1>,
) -> Array2<F> {
    let n_clusters = old_centroids.nrows();
    let mut counts: Array1<usize> = Array1::ones(n_clusters);
    let mut centroids = Array2::zeros((n_clusters, observations.ncols()));

    Zip::from(observations.genrows())
        .and(cluster_memberships)
        .apply(|observation, &cluster_membership| {
            let mut centroid = centroids.row_mut(cluster_membership);
            centroid += &observation;
            counts[cluster_membership] += 1;
        });
    // m_k-means: Treat the old centroid like another point in the cluster
    centroids += old_centroids;

    Zip::from(centroids.genrows_mut())
        .and(&counts)
        .apply(|mut centroid, &cnt| centroid /= F::cast(cnt));
    centroids
}

/// Returns new centroids which has the moving average of all observations in each cluster added to
/// the old centroids.
/// Updates `counts` with the number of observations in each cluster.
fn compute_centroids_incremental<F: Float>(
    observations: &ArrayBase<impl Data<Elem = F>, Ix2>,
    cluster_memberships: &ArrayBase<impl Data<Elem = usize>, Ix1>,
    old_centroids: &ArrayBase<impl Data<Elem = F>, Ix2>,
    counts: &mut ArrayBase<impl DataMut<Elem = F>, Ix1>,
) -> Array2<F> {
    let mut centroids = old_centroids.to_owned();
    // We can parallelize this
    Zip::from(observations.genrows())
        .and(cluster_memberships)
        .apply(|obs, &c| {
            // Computes centroids[c] += (observation - centroids[c]) / counts[c]
            // If cluster is empty for this batch, then this wouldn't even be called, so no
            // chance of getting NaN.
            counts[c] += F::one();
            let shift = (&obs - &centroids.row(c)) / counts[c];
            let mut centroid = centroids.row_mut(c);
            centroid += &shift;
        });
    centroids
}

// Update `cluster_memberships` with the index of the cluster each observation belongs to.
pub(crate) fn update_cluster_memberships<F: Float>(
    centroids: &ArrayBase<impl Data<Elem = F> + Sync, Ix2>,
    observations: &ArrayBase<impl Data<Elem = F> + Sync, Ix2>,
    cluster_memberships: &mut ArrayBase<impl DataMut<Elem = usize>, Ix1>,
) {
    Zip::from(observations.axis_iter(Axis(0)))
        .and(cluster_memberships)
        .par_apply(|observation, cluster_membership| {
            *cluster_membership = closest_centroid(&centroids, &observation).0
        });
}

// Updates `dists` with the distance of each observation from its closest centroid.
pub(crate) fn update_min_dists<F: Float>(
    centroids: &ArrayBase<impl Data<Elem = F> + Sync, Ix2>,
    observations: &ArrayBase<impl Data<Elem = F> + Sync, Ix2>,
    dists: &mut ArrayBase<impl DataMut<Elem = F>, Ix1>,
) {
    Zip::from(observations.axis_iter(Axis(0)))
        .and(dists)
        .par_apply(|observation, dist| *dist = closest_centroid(&centroids, &observation).1);
}

// Efficient combination of `update_cluster_memberships` and `update_min_dists`.
pub(crate) fn update_memberships_and_dists<F: Float>(
    centroids: &ArrayBase<impl Data<Elem = F> + Sync, Ix2>,
    observations: &ArrayBase<impl Data<Elem = F> + Sync, Ix2>,
    cluster_memberships: &mut ArrayBase<impl DataMut<Elem = usize>, Ix1>,
    dists: &mut ArrayBase<impl DataMut<Elem = F>, Ix1>,
) {
    Zip::from(observations.axis_iter(Axis(0)))
        .and(cluster_memberships)
        .and(dists)
        .par_apply(|observation, cluster_membership, dist| {
            let (m, d) = closest_centroid(&centroids, &observation);
            *cluster_membership = m;
            *dist = d;
        });
}

/// Given a matrix of centroids with shape (n_centroids, n_features)
/// and a matrix of observations with shape (n_observations, n_features),
/// return a 1-dimensional `membership` array such that:
///
/// membership[i] == closest_centroid(&centroids, &observations.slice(s![i, ..])
///
fn compute_cluster_memberships<F: Float>(
    // (n_centroids, n_features)
    centroids: &ArrayBase<impl Data<Elem = F> + Sync, Ix2>,
    // (n_observations, n_features)
    observations: &ArrayBase<impl Data<Elem = F> + Sync, Ix2>,
) -> Array1<usize> {
    let mut memberships = Array1::zeros(observations.nrows());
    update_cluster_memberships(&centroids, &observations, &mut memberships);
    memberships
}

/// Given a matrix of centroids with shape (n_centroids, n_features) and an observation,
/// return the index of the closest centroid (the index of the corresponding row in `centroids`).
pub(crate) fn closest_centroid<F: Float>(
    // (n_centroids, n_features)
    centroids: &ArrayBase<impl Data<Elem = F>, Ix2>,
    // (n_features)
    observation: &ArrayBase<impl Data<Elem = F>, Ix1>,
) -> (usize, F) {
    let iterator = centroids.genrows().into_iter();

    let first_centroid = centroids.row(0);
    let (mut closest_index, mut minimum_distance) = (
        0,
        first_centroid
            .sq_l2_dist(&observation)
            .expect("Failed to compute distance"),
    );

    for (centroid_index, centroid) in iterator.enumerate() {
        let distance = centroid
            .sq_l2_dist(&observation)
            .expect("Failed to compute distance");
        if distance < minimum_distance {
            closest_index = centroid_index;
            minimum_distance = distance;
        }
    }
    (closest_index, minimum_distance)
}

#[cfg(test)]
mod tests {
    use super::super::KMeansInit;
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, concatenate, Array, Array1, Array2, Axis};
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    fn function_test_1d(x: &Array2<f64>) -> Array2<f64> {
        let mut y = Array2::zeros(x.dim());
        Zip::from(&mut y).and(x).apply(|yi, &xi| {
            if xi < 0.4 {
                *yi = xi * xi;
            } else if xi >= 0.4 && xi < 0.8 {
                *yi = 3. * xi + 1.;
            } else {
                *yi = f64::sin(10. * xi);
            }
        });
        y
    }

    #[test]
    fn test_n_runs() {
        let mut rng = Isaac64Rng::seed_from_u64(42);
        let xt = Array::random_using(100, Uniform::new(0., 1.0), &mut rng).insert_axis(Axis(1));
        let yt = function_test_1d(&xt);
        let data = concatenate(Axis(1), &[xt.view(), yt.view()]).unwrap();

        for init in &[
            KMeansInit::Random,
            KMeansInit::KMeansPlusPlus,
            KMeansInit::KMeansPara,
        ] {
            // First clustering with one iteration
            let dataset = DatasetBase::from(data.clone());
            let model = KMeans::params_with_rng(3, rng.clone())
                .n_runs(1)
                .init_method(init.clone())
                .fit(&dataset)
                .expect("KMeans fitted");
            let clusters = model.predict(dataset);
            let inertia = compute_inertia(model.centroids(), &clusters.records, &clusters.targets);
            let total_dist = model.transform(&clusters.records.view()).sum();
            assert_abs_diff_eq!(inertia, total_dist);

            // Second clustering with 10 iterations (default)
            let dataset2 = DatasetBase::from(clusters.records().clone());
            let model2 = KMeans::params_with_rng(3, rng.clone())
                .init_method(init.clone())
                .fit(&dataset2)
                .expect("KMeans fitted");
            let clusters2 = model2.predict(dataset2);
            let inertia2 =
                compute_inertia(model2.centroids(), &clusters2.records, &clusters2.targets);
            let total_dist2 = model2.transform(&clusters2.records.view()).sum();
            assert_abs_diff_eq!(inertia2, total_dist2);

            // Check we improve inertia (only really makes a difference for random init)
            if *init == KMeansInit::Random {
                assert!(inertia2 <= inertia);
            }
        }
    }

    #[test]
    fn compute_centroids_works() {
        let cluster_size = 100;
        let n_features = 4;

        // Let's setup a synthetic set of observations, composed of two clusters with known means
        let cluster_1: Array2<f64> =
            Array::random((cluster_size, n_features), Uniform::new(-100., 100.));
        let memberships_1 = Array1::zeros(cluster_size);
        let expected_centroid_1 = cluster_1.sum_axis(Axis(0)) / (cluster_size + 1) as f64;

        let cluster_2: Array2<f64> =
            Array::random((cluster_size, n_features), Uniform::new(-100., 100.));
        let memberships_2 = Array1::ones(cluster_size);
        let expected_centroid_2 = cluster_2.sum_axis(Axis(0)) / (cluster_size + 1) as f64;

        // `concatenate` combines arrays along a given axis: https://docs.rs/ndarray/0.13.0/ndarray/fn.concatenate.html
        let observations = concatenate(Axis(0), &[cluster_1.view(), cluster_2.view()]).unwrap();
        let memberships =
            concatenate(Axis(0), &[memberships_1.view(), memberships_2.view()]).unwrap();

        // Does it work?
        let old_centroids = Array2::zeros((2, n_features));
        let centroids = compute_centroids(&old_centroids, &observations, &memberships);
        assert_abs_diff_eq!(
            centroids.index_axis(Axis(0), 0),
            expected_centroid_1,
            epsilon = 1e-5
        );
        assert_abs_diff_eq!(
            centroids.index_axis(Axis(0), 1),
            expected_centroid_2,
            epsilon = 1e-5
        );

        assert_eq!(centroids.len_of(Axis(0)), 2);
    }

    #[test]
    fn test_compute_extra_centroids() {
        let observations = array![[1.0, 2.0]];
        let memberships = array![0];
        // Should return an average of 0 for empty clusters
        let old_centroids = Array2::ones((2, 2));
        let centroids = compute_centroids(&old_centroids, &observations, &memberships);
        assert_abs_diff_eq!(centroids, array![[1.0, 1.5], [1.0, 1.0]]);
    }

    #[test]
    // An observation is closest to itself.
    fn nothing_is_closer_than_self() {
        let n_centroids = 20;
        let n_features = 5;
        let mut rng = Isaac64Rng::seed_from_u64(42);
        let centroids: Array2<f64> = Array::random_using(
            (n_centroids, n_features),
            Uniform::new(-100., 100.),
            &mut rng,
        );

        let expected_memberships: Vec<usize> = (0..n_centroids).into_iter().collect();
        assert_eq!(
            compute_cluster_memberships(&centroids, &centroids),
            Array1::from(expected_memberships)
        );
    }

    #[test]
    fn oracle_test_for_closest_centroid() {
        let centroids = array![[0., 0.], [1., 2.], [20., 0.], [0., 20.],];
        let observations = array![[1., 0.5], [20., 2.], [20., 0.], [7., 20.],];
        let memberships = array![0, 2, 2, 3];

        assert_eq!(
            compute_cluster_memberships(&centroids, &observations),
            memberships
        );
    }

    #[test]
    fn test_compute_centroids_incremental() {
        let observations = array![[-1.0, -3.0], [0., 0.], [3., 5.], [5., 5.]];
        let memberships = array![0, 0, 1, 1];
        let centroids = array![[-1., -1.], [3., 4.], [7., 8.]];
        let mut counts = array![3.0, 0.0, 1.0];
        let centroids =
            compute_centroids_incremental(&observations, &memberships, &centroids, &mut counts);

        assert_abs_diff_eq!(centroids, array![[-4. / 5., -6. / 5.], [4., 5.], [7., 8.]]);
        assert_abs_diff_eq!(counts, array![5., 2., 1.]);
    }

    #[test]
    fn test_incremental_kmeans() {
        let dataset1 = DatasetBase::from(array![[-1.0, -3.0], [0., 0.], [3., 5.], [5., 5.]]);
        let dataset2 = DatasetBase::from(array![[-5.0, -5.0], [0., 0.], [10., 10.]]);
        let model = KMeans {
            centroids: array![[-1., -1.], [3., 4.], [7., 8.]],
            cluster_count: array![0., 0., 0.],
            inertia: 0.0,
        };
        let rng = Isaac64Rng::seed_from_u64(45);
        let params = KMeans::params_with_rng(3, rng).tolerance(100.0).build();

        let (model, converged) = params.fit_with(Some(model), &dataset1);
        assert_abs_diff_eq!(model.centroids(), &array![[-0.5, -1.5], [4., 5.], [7., 8.]]);
        assert!(converged);

        let (model, converged) = params.fit_with(Some(model), &dataset2);
        assert_abs_diff_eq!(
            model.centroids(),
            &array![[-6. / 4., -8. / 4.], [4., 5.], [10., 10.]]
        );
        assert!(converged);
    }
}
