use crate::gaussian_mixture::hyperparameters::{
    GmmCovarType, GmmHyperParams, GmmHyperParamsBuilder, GmmInitMethod,
};
use crate::k_means::KMeans;
use linfa::{
    dataset::{Dataset, Targets},
    traits::*,
    Float,
};
use ndarray::{s, Array, Array1, Array2, Array3, ArrayBase, Axis, Data, Ix2, Ix3, Zip};
use ndarray_linalg::{cholesky::*, triangular::*};
use ndarray_rand::rand::Rng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use ndarray_stats::QuantileExt;
use rand_isaac::Isaac64Rng;
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
/// Gaussian Mixture Model (GMM) aims at clustering a dataset by finding normally
/// distributed sub datasets (hence the Gaussian Mixture name) .
///
/// GMM assumes all the data points are generated from a mixture of a number K
/// of Gaussian distributions with certain parameters.
/// Expectation-maximization (EM) algorithm is used to fit the GMM to the dataset
/// by parameterizing the weight, mean, and covariance of each cluster distribution.
///
/// This implementation is a port of the [scikit-learn 0.23.2 Gaussian Mixture](https://scikit-learn.org)
/// implementation.
///
/// ## The algorithm  
///
/// The general idea is to maximize the likelihood (equivalently the log likelihood)
/// that is maximising the probability that the dataset is drawn from our mixture of normal distributions.
///
/// After an initialization step which can be either from random distribution or from the result
/// of the [KMeans](struct.KMeans.html) algorithm (which is the default value of the `init_method` parameter).
/// The core EM iterative algorithm for Gaussian Mixture is a fixed-point two-step algorithm:
///
/// 1. Expectation step: compute the expectation of the likelihood of the current gaussian mixture model wrt the dataset.
/// 2. Maximization step: update the gaussian parameters (weigths, means and covariances) to maximize the likelihood.
///
/// We stop iterating when there is no significant gaussian parameters change (controlled by the `tolerance` parameter) or
/// if we reach a max number of iterations (controlled by `max_n_iterations` parameter)
/// As the initialization of the algorithm is subject to randomness, several initializations are performed (controlled by
/// the `n_init` parameter).   
///
/// ## Tutorial
///
/// Let's do a walkthrough of a training-predict-save example.
///
/// ```rust
/// extern crate openblas_src;
/// use linfa::Dataset;
/// use linfa::traits::{Fit, Predict};
/// use linfa_clustering::{GmmHyperParams, GaussianMixtureModel, generate_blobs};
/// use ndarray::{Axis, array, s, Zip};
/// use ndarray_rand::rand::SeedableRng;
/// use rand_isaac::Isaac64Rng;
/// use approx::assert_abs_diff_eq;
///
/// let mut rng = Isaac64Rng::seed_from_u64(42);
/// let expected_centroids = array![[0., 1.], [-10., 20.], [-1., 10.]];
/// let n = 200;
///
/// // We generate a dataset from points normally distributed around some distant centroids.  
/// let dataset = Dataset::from(generate_blobs(n, &expected_centroids, &mut rng));
///
/// // Our GMM is expected to have a number of clusters equals the number of centroids
/// // used to generate the dataset
/// let n_clusters = expected_centroids.len_of(Axis(0));
///
/// // We fit the model from the dataset setting some options
/// let gmm = GaussianMixtureModel::params_with_rng(n_clusters, rng)
///             .n_init(10)
///             .tolerance(1e-4)
///             .build()
///             .fit(&dataset);
///
/// // We can get predicted centroids (ie means of learnt gaussian distributions) from the model
/// let gmm_centroids = gmm.centroids();
///
/// // We can check that centroids used to generate test dataset are close to GMM centroids
/// let memberships = gmm.predict(&expected_centroids);
/// for (i, expected_c) in expected_centroids.outer_iter().enumerate() {
/// let closest_c = gmm_centroids.index_axis(Axis(0), memberships[i]);
/// Zip::from(&closest_c)
///     .and(&expected_c)
///     .apply(|a, b| assert_abs_diff_eq!(a, b, epsilon = 1.))
/// }
/// ```
pub struct GaussianMixtureModel<F: Float> {
    covar_type: GmmCovarType,
    weights: Array1<F>,
    means: Array2<F>,
    precisions: Array3<F>,
}

impl<F: Float> Clone for GaussianMixtureModel<F> {
    fn clone(&self) -> Self {
        Self {
            covar_type: self.covar_type,
            weights: self.weights.to_owned(),
            means: self.means.to_owned(),
            precisions: self.precisions.to_owned(),
        }
    }
}

impl<F: Float + Into<f64>> GaussianMixtureModel<F> {
    pub fn params(n_clusters: usize) -> GmmHyperParamsBuilder<F, Isaac64Rng> {
        GmmHyperParams::new(n_clusters)
    }

    pub fn params_with_rng<R: Rng + Clone>(
        n_clusters: usize,
        rng: R,
    ) -> GmmHyperParamsBuilder<F, R> {
        GmmHyperParams::new_with_rng(n_clusters, rng)
    }

    pub fn weights(&self) -> &Array1<F> {
        &self.weights
    }

    pub fn means(&self) -> &Array2<F> {
        &self.means
    }

    pub fn precisions(&self) -> &Array3<F> {
        &self.precisions
    }

    pub fn centroids(&self) -> &Array2<F> {
        self.means()
    }

    fn new<D: Data<Elem = F>, R: Rng + Clone, T: Targets>(
        hyperparameters: &GmmHyperParams<F, R>,
        dataset: &Dataset<ArrayBase<D, Ix2>, T>,
        mut rng: R,
    ) -> GaussianMixtureModel<F> {
        let observations = dataset.records().view();
        let n_samples = observations.nrows();

        let resp = match hyperparameters.init_method() {
            GmmInitMethod::KMeans => {
                let model = KMeans::params_with_rng(hyperparameters.n_clusters(), rng)
                    .build()
                    .fit(&dataset);
                let mut resp = Array::<F, Ix2>::zeros((n_samples, hyperparameters.n_clusters()));
                for (k, idx) in model.predict(dataset.records()).iter().enumerate() {
                    resp[[k, *idx]] = F::from(1.).unwrap();
                }
                resp
            }
            GmmInitMethod::Random => {
                let mut resp = Array2::<f64>::random_using(
                    (n_samples, hyperparameters.n_clusters()),
                    Uniform::new(0., 1.),
                    &mut rng,
                );
                let totals = &resp.sum_axis(Axis(1)).insert_axis(Axis(0));
                resp = (resp.reversed_axes() / totals).reversed_axes();
                resp.mapv(|v| F::from(v).unwrap())
            }
        };

        let (mut weights, means, covariances) = Self::estimate_gaussian_parameters(
            &observations,
            &resp,
            hyperparameters.covariance_type(),
            hyperparameters.reg_covariance(),
        );
        weights = weights / F::from(n_samples).unwrap();

        // GmmCovarType = full
        let precisions = Self::compute_precision_cholesky_full(&covariances);

        GaussianMixtureModel {
            covar_type: *hyperparameters.covariance_type(),
            weights,
            means,
            precisions,
        }
    }

    fn estimate_gaussian_parameters<D: Data<Elem = F>>(
        observations: &ArrayBase<D, Ix2>,
        resp: &Array2<F>,
        _covar_type: &GmmCovarType,
        reg_covar: F,
    ) -> (Array1<F>, Array2<F>, Array3<F>) {
        let nk = resp.sum_axis(Axis(0)) + F::from(10.).unwrap() * F::epsilon();
        let nk2 = nk.to_owned().insert_axis(Axis(1));
        let means = resp.t().dot(observations) / nk2;
        // GmmCovarType = Full
        let covariances =
            Self::estimate_gaussian_covariances_full(&observations, resp, &nk, &means, reg_covar);
        (nk, means, covariances)
    }

    fn estimate_gaussian_covariances_full<D: Data<Elem = F>>(
        observations: &ArrayBase<D, Ix2>,
        resp: &Array2<F>,
        nk: &Array1<F>,
        means: &Array2<F>,
        reg_covar: F,
    ) -> Array3<F> {
        let n_clusters = means.nrows();
        let n_features = means.ncols();
        let mut covariances = Array::zeros((n_clusters, n_features, n_features));
        for k in 0..n_clusters {
            let diff = observations - &means.slice(s![k..k + 1, ..]);
            let m = diff.t().to_owned() * resp.slice(s![.., k]);
            let mut cov_k = m.dot(&diff) / nk[k];
            let diag = cov_k.diag().to_owned() + reg_covar;
            cov_k.diag_mut().assign(&diag);
            covariances.slice_mut(s![k, .., ..]).assign(&cov_k);
        }
        covariances
    }

    fn compute_precision_cholesky_full<D: Data<Elem = F>>(
        covariances: &ArrayBase<D, Ix3>,
    ) -> Array3<F> {
        let n_clusters = covariances.shape()[0];
        let n_features = covariances.shape()[1];
        let mut precisions_chol = Array::zeros((n_clusters, n_features, n_features));
        for (k, covariance) in covariances.outer_iter().enumerate() {
            let cov: Array2<f64> = covariance.mapv(|v| v.into());
            match cov.cholesky(UPLO::Lower) {
                Ok(cov_chol) => {
                    let sol = cov_chol
                        .solve_triangular(UPLO::Lower, Diag::NonUnit, &Array::eye(n_features))
                        .unwrap()
                        .to_owned();
                    precisions_chol.slice_mut(s![k, .., ..]).assign(&sol.t());
                }
                Err(_) => panic!(
                    "Fitting the mixture model failed because some components have \
                ill-defined empirical covariance (for instance caused by singleton \
                or collapsed samples). Try to decrease the number of components, \
                or increase reg_covar."
                ),
            };
        }
        precisions_chol.mapv(|v| F::from(v).unwrap())
    }

    fn e_step<D: Data<Elem = F>>(&self, observations: &ArrayBase<D, Ix2>) -> (F, Array2<F>) {
        let (log_prob_norm, log_resp) = self.estimate_log_prob_resp(&observations);
        let log_mean = log_prob_norm.sum() / F::from(log_prob_norm.len()).unwrap();
        (log_mean, log_resp)
    }

    fn m_step<D: Data<Elem = F>>(
        &mut self,
        reg_covar: F,
        observations: &ArrayBase<D, Ix2>,
        log_resp: &Array2<F>,
    ) {
        let n_samples = observations.nrows();
        let (weights, means, covariances) = Self::estimate_gaussian_parameters(
            &observations,
            &log_resp.mapv(F::exp),
            &self.covar_type,
            reg_covar,
        );
        self.means = means;
        self.weights = weights / F::from(n_samples).unwrap();
        // GmmCovarType = Full()
        self.precisions = Self::compute_precision_cholesky_full(&covariances);
    }

    fn compute_lower_bound<D: Data<Elem = F>>(
        _log_resp: &ArrayBase<D, Ix2>,
        log_prob_norm: F,
    ) -> F {
        log_prob_norm
    }

    fn estimate_log_prob_resp<D: Data<Elem = F>>(
        &self,
        observations: &ArrayBase<D, Ix2>,
    ) -> (Array1<F>, Array2<F>) {
        let weighted_log_prob = self.estimate_weighted_log_prob(&observations);
        let log_prob_norm = weighted_log_prob
            .mapv(|v| v.exp())
            .sum_axis(Axis(1))
            .mapv(|v| v.ln());
        let log_resp = weighted_log_prob - log_prob_norm.to_owned().insert_axis(Axis(1));
        (log_prob_norm, log_resp)
    }

    fn estimate_weighted_log_prob<D: Data<Elem = F>>(
        &self,
        observations: &ArrayBase<D, Ix2>,
    ) -> Array2<F> {
        self.estimate_log_prob(&observations) + self.estimate_log_weights()
    }

    fn estimate_log_prob<D: Data<Elem = F>>(&self, observations: &ArrayBase<D, Ix2>) -> Array2<F> {
        self.estimate_log_gaussian_prob(&observations)
    }

    fn estimate_log_gaussian_prob<D: Data<Elem = F>>(
        &self,
        observations: &ArrayBase<D, Ix2>,
    ) -> Array2<F> {
        let n_samples = observations.nrows();
        let n_features = observations.ncols();
        let means = self.means();
        let precisions_chol = self.precisions();
        let n_clusters = means.nrows();
        // GmmCovarType = full
        let log_det = Self::compute_log_det_cholesky_full(&precisions_chol, n_features);
        let mut log_prob: Array2<F> = Array::zeros((n_samples, n_clusters));
        Zip::indexed(means.genrows())
            .and(precisions_chol.outer_iter())
            .apply(|k, mu, prec_chol| {
                let diff = (&observations.to_owned() - &mu).dot(&prec_chol);
                log_prob
                    .slice_mut(s![.., k])
                    .assign(&diff.mapv(|v| v * v).sum_axis(Axis(1)))
            });
        log_prob.mapv(|v| {
            F::from(-0.5).unwrap()
                * (v + F::from(n_features as f64 * f64::ln(2. * std::f64::consts::PI)).unwrap())
        }) + log_det
    }

    fn compute_log_det_cholesky_full<D: Data<Elem = F>>(
        matrix_chol: &ArrayBase<D, Ix3>,
        n_features: usize,
    ) -> Array1<F> {
        let n_clusters = matrix_chol.shape()[0];
        let log_diags = &matrix_chol
            .to_owned()
            .into_shape((n_clusters, n_features * n_features))
            .unwrap()
            .slice(s![.., ..; n_features+1])
            .to_owned()
            .mapv(|v| v.ln());
        let log_det_chol = log_diags.sum_axis(Axis(1));
        log_det_chol
    }

    fn estimate_log_weights(&self) -> Array1<F> {
        self.weights().mapv(|v| v.ln())
    }
}

impl<'a, F: Float + Into<f64>, R: Rng + Clone, D: Data<Elem = F>, T: Targets>
    Fit<'a, ArrayBase<D, Ix2>, T> for GmmHyperParams<F, R>
{
    type Object = GaussianMixtureModel<F>;

    fn fit(&self, dataset: &Dataset<ArrayBase<D, Ix2>, T>) -> Self::Object {
        let observations = dataset.records().view();
        let mut gmm = GaussianMixtureModel::<F>::new(self, dataset, self.rng());

        let mut max_lower_bound = -F::infinity();
        let mut best_params = None;
        let mut best_iter = None;

        let n_init = self.n_init();

        for _ in 0..n_init {
            let mut lower_bound = -F::infinity();

            let mut converged_iter: Option<u64> = None;
            for n_iter in 0..self.max_n_iterations() {
                let prev_lower_bound = lower_bound;
                let (log_prob_norm, log_resp) = gmm.e_step(&observations);
                gmm.m_step(self.reg_covariance(), &observations, &log_resp);
                lower_bound =
                    GaussianMixtureModel::<F>::compute_lower_bound(&log_resp, log_prob_norm);
                let change = lower_bound - prev_lower_bound;
                if num_traits::sign::Signed::abs(&change) < self.tolerance() {
                    converged_iter = Some(n_iter);
                    break;
                }
            }

            if lower_bound > max_lower_bound {
                max_lower_bound = lower_bound;
                best_params = Some(gmm.clone());
                best_iter = converged_iter;
            }
        }

        match best_iter {
            Some(_n_iter) => match best_params {
                Some(gmm) => gmm,
                _ => panic!("No lower bound improvement. GMM fit fail!"),
            },
            None => {
                panic!(
                    "Initialization {} did not converge. Try different init parameters, \
                         or increase max_n_iterations, tolerance or check for degenerate data.",
                    (n_init + 1)
                );
            }
        }
    }
}

impl<F: Float + Into<f64>, D: Data<Elem = F>> Predict<&ArrayBase<D, Ix2>, Array1<usize>>
    for GaussianMixtureModel<F>
{
    fn predict(&self, observations: &ArrayBase<D, Ix2>) -> Array1<usize> {
        let (_, log_resp) = self.estimate_log_prob_resp(&observations);
        return log_resp
            .mapv(|v| v.exp())
            .map_axis(Axis(1), |row| row.argmax().unwrap());
    }
}

impl<F: Float + Into<f64>, D: Data<Elem = F>, T: Targets>
    Predict<Dataset<ArrayBase<D, Ix2>, T>, Dataset<ArrayBase<D, Ix2>, Array1<usize>>>
    for GaussianMixtureModel<F>
{
    fn predict(
        &self,
        dataset: Dataset<ArrayBase<D, Ix2>, T>,
    ) -> Dataset<ArrayBase<D, Ix2>, Array1<usize>> {
        let predicted = self.predict(dataset.records());
        dataset.with_targets(predicted)
    }
}

#[cfg(test)]
mod tests {
    extern crate openblas_src;
    use super::*;
    use crate::generate_blobs;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Axis};
    // use ndarray_npy::write_npy;
    use ndarray_rand::rand::SeedableRng;

    #[test]
    fn test_centroids_prediction() {
        let mut rng = Isaac64Rng::seed_from_u64(42);
        let expected_centroids = array![[0., 1.], [-10., 20.], [-1., 10.]];
        let n = 200;
        let blobs = Dataset::from(generate_blobs(n, &expected_centroids, &mut rng));

        let n_clusters = expected_centroids.len_of(Axis(0));
        let gmm = GaussianMixtureModel::params_with_rng(n_clusters, rng)
            .build()
            .fit(&blobs);

        let gmm_centroids = gmm.centroids();
        let memberships = gmm.predict(&expected_centroids);

        // check that centroids used to generate test dataset belongs to the right predicted cluster
        for (i, expected_c) in expected_centroids.outer_iter().enumerate() {
            let closest_c = gmm_centroids.index_axis(Axis(0), memberships[i]);
            Zip::from(&closest_c)
                .and(&expected_c)
                .apply(|a, b| assert_abs_diff_eq!(a, b, epsilon = 1.))
        }

        let blobs_dataset = gmm.predict(blobs);
        let Dataset {
            records: _blobs_records,
            targets: _blobs_targets,
            ..
        } = blobs_dataset;
        // write_npy("linfa_blobs.npy", blobs_records).expect("Failed to write .npy file");
        // write_npy(
        //     "linfa_memberships_blobs.npy",
        //     blobs_targets.map(|&v| v as u64),
        // )
        // .expect("Failed to write .npy file");

        // write_npy("linfa_blobgen_centroids.npy", expected_centroids.view())
        //     .expect("Failed to write .npy file");
        // write_npy("linfa_pred_centroids.npy", gmm_centroids.view())
        //     .expect("Failed to write .npy file");
        // write_npy(
        //     "linfa_blobgen_memberships.npy",
        //     memberships.map(|&x| x as u64),
        // )
        // .expect("Failed to write .npy file");
    }
}