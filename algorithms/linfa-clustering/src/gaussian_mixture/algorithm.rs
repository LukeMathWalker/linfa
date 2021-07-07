use crate::gaussian_mixture::errors::{GmmError, Result};
use crate::gaussian_mixture::hyperparameters::{GmmCovarType, GmmHyperParams, GmmInitMethod};
use crate::k_means::KMeans;
use linfa::{
    dataset::{WithLapack, WithoutLapack},
    prelude::*,
    DatasetBase, Float,
};
use ndarray::{s, Array, Array1, Array2, Array3, ArrayBase, Axis, Data, Ix2, Ix3, Zip};
use ndarray_linalg::{cholesky::*, triangular::*, Lapack, Scalar};
use ndarray_rand::rand::Rng;
use ndarray_rand::rand::SeedableRng;
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
/// the `n_runs` parameter).   
///
/// ## Tutorial
///
/// Let's do a walkthrough of a training-predict-save example.
///
/// ```rust, ignore
/// use linfa::DatasetBase;
/// use linfa::traits::{Fit, PredictRef};
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
/// let dataset = DatasetBase::from(generate_blobs(n, &expected_centroids, &mut rng));
///
/// // Our GMM is expected to have a number of clusters equals the number of centroids
/// // used to generate the dataset
/// let n_clusters = expected_centroids.len_of(Axis(0));
///
/// // We fit the model from the dataset setting some options
/// let gmm = GaussianMixtureModel::params(n_clusters)
///             .with_n_runs(10)
///             .with_tolerance(1e-4)
///             .with_rng(rng)
///             .fit(&dataset).expect("GMM fitting");
///
/// // Then we can get dataset membership information, targets contain **cluster indexes**
/// // corresponding to the cluster infos in the list of GMM means and covariances
/// let blobs_dataset = gmm.predict(dataset);
/// let DatasetBase {
///     records: _blobs_records,
///     targets: blobs_targets,
///     ..
/// } = blobs_dataset;
/// println!("GMM means = {:?}", gmm.means());
/// println!("GMM covariances = {:?}", gmm.covariances());
/// println!("GMM membership = {:?}", blobs_targets);
///
/// // We can also get the nearest cluster for a new point
/// let new_observation = DatasetBase::from(array![[-9., 20.5]]);
/// // Predict returns the **index** of the nearest cluster
/// let dataset = gmm.predict(new_observation);
/// // We can retrieve the actual centroid of the closest cluster using `.centroids()` (alias of .means())
/// let closest_centroid = &gmm.centroids().index_axis(Axis(0), dataset.targets()[0]);
/// ```
#[derive(Debug, PartialEq)]
pub struct GaussianMixtureModel<F: Float> {
    covar_type: GmmCovarType,
    weights: Array1<F>,
    means: Array2<F>,
    covariances: Array3<F>,
    precisions: Array3<F>,
    precisions_chol: Array3<F>,
}

impl<F: Float> Clone for GaussianMixtureModel<F> {
    fn clone(&self) -> Self {
        Self {
            covar_type: self.covar_type,
            weights: self.weights.to_owned(),
            means: self.means.to_owned(),
            covariances: self.covariances.to_owned(),
            precisions: self.precisions.to_owned(),
            precisions_chol: self.precisions_chol.to_owned(),
        }
    }
}

impl<F: Float> GaussianMixtureModel<F> {
    fn new<D: Data<Elem = F>, R: Rng + SeedableRng + Clone, T>(
        hyperparameters: &GmmHyperParams<F, R>,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, T>,
        mut rng: R,
    ) -> Result<GaussianMixtureModel<F>> {
        let observations = dataset.records().view();
        let n_samples = observations.nrows();

        // We initialize responsabilities (n_samples, n_clusters) of each clusters
        // that is, given a sample, the probabilities of a cluster being the source.
        // Responsabilities can be initialized either from a KMeans result or randomly.
        let resp = match hyperparameters.init_method() {
            GmmInitMethod::KMeans => {
                let model = KMeans::params_with_rng(hyperparameters.n_clusters(), rng)
                    .check()
                    .unwrap()
                    .fit(dataset)?;
                let mut resp = Array::<F, Ix2>::zeros((n_samples, hyperparameters.n_clusters()));
                for (k, idx) in model.predict(dataset.records()).iter().enumerate() {
                    resp[[k, *idx]] = F::cast(1.);
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
                resp.mapv(F::cast)
            }
        };

        // We compute an initial GMM model from dataset and initial responsabilities wrt
        // to covariance specification.
        let (mut weights, means, covariances) = Self::estimate_gaussian_parameters(
            &observations,
            &resp,
            hyperparameters.covariance_type(),
            hyperparameters.reg_covariance(),
        )?;
        weights /= F::cast(n_samples);

        // GmmCovarType = full
        let precisions_chol = Self::compute_precisions_cholesky_full(&covariances)?;
        let precisions = Self::compute_precisions_full(&precisions_chol);

        Ok(GaussianMixtureModel {
            covar_type: *hyperparameters.covariance_type(),
            weights,
            means,
            covariances,
            precisions,
            precisions_chol,
        })
    }
}

impl<F: Float> GaussianMixtureModel<F> {
    pub fn params(n_clusters: usize) -> GmmHyperParams<F, Isaac64Rng> {
        GmmHyperParams::new(n_clusters)
    }

    pub fn weights(&self) -> &Array1<F> {
        &self.weights
    }

    pub fn means(&self) -> &Array2<F> {
        &self.means
    }

    pub fn covariances(&self) -> &Array3<F> {
        &self.covariances
    }

    pub fn precisions(&self) -> &Array3<F> {
        &self.precisions
    }

    pub fn centroids(&self) -> &Array2<F> {
        self.means()
    }

    fn estimate_gaussian_parameters<D: Data<Elem = F>>(
        observations: &ArrayBase<D, Ix2>,
        resp: &Array2<F>,
        _covar_type: &GmmCovarType,
        reg_covar: F,
    ) -> Result<(Array1<F>, Array2<F>, Array3<F>)> {
        let nk = resp.sum_axis(Axis(0));
        if nk.min()? < &(F::cast(10.) * F::epsilon()) {
            return Err(GmmError::EmptyCluster(format!(
                "Cluster #{} has no more point. Consider decreasing number of clusters or change initialization.",
                nk.argmin()? + 1
            )));
        }

        let nk2 = nk.to_owned().insert_axis(Axis(1));
        let means = resp.t().dot(observations) / nk2;
        // GmmCovarType = Full
        let covariances =
            Self::estimate_gaussian_covariances_full(observations, resp, &nk, &means, reg_covar);
        Ok((nk, means, covariances))
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
            let diff = observations - &means.row(k);
            let m = &diff.t() * &resp.index_axis(Axis(1), k);
            let mut cov_k = m.dot(&diff) / nk[k];
            cov_k.diag_mut().mapv_inplace(|x| x + reg_covar);
            covariances.slice_mut(s![k, .., ..]).assign(&cov_k);
        }
        covariances
    }

    fn compute_precisions_cholesky_full<D: Data<Elem = F>>(
        covariances: &ArrayBase<D, Ix3>,
    ) -> Result<Array3<F>> {
        let n_clusters = covariances.shape()[0];
        let n_features = covariances.shape()[1];
        let mut precisions_chol = Array::zeros((n_clusters, n_features, n_features));
        for (k, covariance) in covariances.outer_iter().enumerate() {
            let decomp = covariance.with_lapack().cholesky(UPLO::Lower)?;
            let sol = decomp
                .solve_triangular(UPLO::Lower, Diag::NonUnit, &Array::eye(n_features))?
                .without_lapack();

            precisions_chol.slice_mut(s![k, .., ..]).assign(&sol.t());
        }
        Ok(precisions_chol)
    }

    fn compute_precisions_full<D: Data<Elem = F>>(
        precisions_chol: &ArrayBase<D, Ix3>,
    ) -> Array3<F> {
        let mut precisions = Array3::zeros(precisions_chol.dim());
        for (k, prec_chol) in precisions_chol.outer_iter().enumerate() {
            precisions
                .slice_mut(s![k, .., ..])
                .assign(&prec_chol.dot(&prec_chol.t()));
        }
        precisions
    }

    // Refresh precisions value only at the end of the fitting procedure
    fn refresh_precisions_full(&mut self) {
        self.precisions = Self::compute_precisions_full(&self.precisions_chol);
    }

    fn e_step<D: Data<Elem = F>>(
        &self,
        observations: &ArrayBase<D, Ix2>,
    ) -> Result<(F, Array2<F>)> {
        let (log_prob_norm, log_resp) = self.estimate_log_prob_resp(observations);
        let log_mean = log_prob_norm.mean().unwrap();
        Ok((log_mean, log_resp))
    }

    fn m_step<D: Data<Elem = F>>(
        &mut self,
        reg_covar: F,
        observations: &ArrayBase<D, Ix2>,
        log_resp: &Array2<F>,
    ) -> Result<()> {
        let n_samples = observations.nrows();
        let (weights, means, covariances) = Self::estimate_gaussian_parameters(
            observations,
            &log_resp.mapv(|x| x.exp()),
            &self.covar_type,
            reg_covar,
        )?;
        self.means = means;
        self.weights = weights / F::cast(n_samples);
        // GmmCovarType = Full()
        self.precisions_chol = Self::compute_precisions_cholesky_full(&covariances)?;
        Ok(())
    }

    // We keep methods names and method boundaries from scikit-learn implementation
    // which handles also Bayesian mixture hence below the _log_resp argument which is not used.
    fn compute_lower_bound<D: Data<Elem = F>>(
        _log_resp: &ArrayBase<D, Ix2>,
        log_prob_norm: F,
    ) -> F {
        log_prob_norm
    }

    // Estimate log probabilities (log P(X)) and responsibilities for each sample.
    // Compute weighted log probabilities per component (log P(X)) and responsibilities
    // for each sample in X with respect to the current state of the model.
    fn estimate_log_prob_resp<D: Data<Elem = F>>(
        &self,
        observations: &ArrayBase<D, Ix2>,
    ) -> (Array1<F>, Array2<F>) {
        let weighted_log_prob = self.estimate_weighted_log_prob(observations);
        let log_prob_norm = weighted_log_prob
            .mapv(|x| x.exp())
            .sum_axis(Axis(1))
            .mapv(|x| x.ln());
        let log_resp = weighted_log_prob - log_prob_norm.to_owned().insert_axis(Axis(1));
        (log_prob_norm, log_resp)
    }

    // Estimate weighted log probabilities for each samples wrt to the model
    fn estimate_weighted_log_prob<D: Data<Elem = F>>(
        &self,
        observations: &ArrayBase<D, Ix2>,
    ) -> Array2<F> {
        self.estimate_log_prob(observations) + self.estimate_log_weights()
    }

    // Compute log probabilities for each samples wrt to the model which is gaussian
    fn estimate_log_prob<D: Data<Elem = F>>(&self, observations: &ArrayBase<D, Ix2>) -> Array2<F> {
        self.estimate_log_gaussian_prob(observations)
    }

    // Compute the log likelihood in case of the gaussian probabilities
    // log(P(X|Mean, Precision)) = -0.5*(d*ln(2*PI)-ln(det(Precision))-(X-Mean)^t.Precision.(X-Mean)
    fn estimate_log_gaussian_prob<D: Data<Elem = F>>(
        &self,
        observations: &ArrayBase<D, Ix2>,
    ) -> Array2<F> {
        let n_samples = observations.nrows();
        let n_features = observations.ncols();
        let means = self.means();
        let n_clusters = means.nrows();
        // GmmCovarType = full
        // det(precision_chol) is half of det(precision)
        let log_det = Self::compute_log_det_cholesky_full(&self.precisions_chol, n_features);
        let mut log_prob: Array2<F> = Array::zeros((n_samples, n_clusters));
        Zip::indexed(means.genrows())
            .and(self.precisions_chol.outer_iter())
            .apply(|k, mu, prec_chol| {
                let diff = (&observations.to_owned() - &mu).dot(&prec_chol);
                log_prob
                    .slice_mut(s![.., k])
                    .assign(&diff.mapv(|v| v * v).sum_axis(Axis(1)))
            });
        log_prob.mapv(|v| {
            F::cast(-0.5) * (v + F::cast(n_features as f64 * f64::ln(2. * std::f64::consts::PI)))
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
            .mapv(|x| x.ln());
        log_diags.sum_axis(Axis(1))
    }

    fn estimate_log_weights(&self) -> Array1<F> {
        self.weights().mapv(|x| x.ln())
    }
}

impl<F: Float, R: Rng + SeedableRng + Clone, D: Data<Elem = F>, T>
    Fit<ArrayBase<D, Ix2>, T, GmmError> for GmmHyperParams<F, R>
{
    type Object = GaussianMixtureModel<F>;

    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object> {
        self.validate()?;
        let observations = dataset.records().view();
        let mut gmm = GaussianMixtureModel::<F>::new(self, dataset, self.rng())?;

        let mut max_lower_bound = -F::infinity();
        let mut best_params = None;
        let mut best_iter = None;

        let n_runs = self.n_runs();

        for _ in 0..n_runs {
            let mut lower_bound = -F::infinity();

            let mut converged_iter: Option<u64> = None;
            for n_iter in 0..self.max_n_iterations() {
                let prev_lower_bound = lower_bound;
                let (log_prob_norm, log_resp) = gmm.e_step(&observations)?;
                gmm.m_step(self.reg_covariance(), &observations, &log_resp)?;
                lower_bound =
                    GaussianMixtureModel::<F>::compute_lower_bound(&log_resp, log_prob_norm);
                let change = lower_bound - prev_lower_bound;
                if change.abs() < self.tolerance() {
                    converged_iter = Some(n_iter);
                    break;
                }
            }

            if lower_bound > max_lower_bound {
                max_lower_bound = lower_bound;
                gmm.refresh_precisions_full();
                best_params = Some(gmm.clone());
                best_iter = converged_iter;
            }
        }

        match best_iter {
            Some(_n_iter) => match best_params {
                Some(gmm) => Ok(gmm),
                _ => Err(GmmError::LowerBoundError(
                    "No lower bound improvement (-inf)".to_string(),
                )),
            },
            None => Err(GmmError::NotConverged(format!(
                "EM fitting algorithm {} did not converge. Try different init parameters, \
                            or increase max_n_iterations, tolerance or check for degenerate data.",
                (n_runs + 1)
            ))),
        }
    }
}

impl<F: Float + Lapack + Scalar, D: Data<Elem = F>> PredictRef<ArrayBase<D, Ix2>, Array1<usize>>
    for GaussianMixtureModel<F>
{
    fn predict_ref(&self, observations: &ArrayBase<D, Ix2>) -> Array1<usize> {
        let (_, log_resp) = self.estimate_log_prob_resp(observations);
        log_resp
            .mapv(Scalar::exp)
            .map_axis(Axis(1), |row| row.argmax().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate_blobs;
    use approx::{abs_diff_eq, assert_abs_diff_eq};
    use lax::error::Error;
    use ndarray::{array, concatenate, ArrayView1, ArrayView2, Axis};
    use ndarray_linalg::error::LinalgError;
    use ndarray_linalg::error::Result as LAResult;
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::{Distribution, StandardNormal};

    pub struct MultivariateNormal {
        pub mean: Array1<f64>,
        pub covariance: Array2<f64>,
        /// Lower triangular matrix (Cholesky decomposition of the coviariance matrix)
        lower: Array2<f64>,
    }
    impl MultivariateNormal {
        pub fn new(mean: &ArrayView1<f64>, covariance: &ArrayView2<f64>) -> LAResult<Self> {
            let lower = covariance.cholesky(UPLO::Lower)?;
            Ok(MultivariateNormal {
                mean: mean.to_owned(),
                covariance: covariance.to_owned(),
                lower,
            })
        }
    }
    impl Distribution<Array1<f64>> for MultivariateNormal {
        fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Array1<f64> {
            // standard normal distribution
            let res = Array1::random_using(self.mean.shape()[0], StandardNormal, rng);
            // use Cholesky decomposition to obtain a sample of our general multivariate normal
            self.mean.clone() + self.lower.view().dot(&res)
        }
    }

    #[test]
    fn test_gmm_fit() {
        let mut rng = Isaac64Rng::seed_from_u64(42);
        let weights = array![0.5, 0.5];
        let means = array![[0., 0.], [5., 5.]];
        let covars = array![[[1., 0.8], [0.8, 1.]], [[1.0, -0.6], [-0.6, 1.0]]];
        let mvn1 =
            MultivariateNormal::new(&means.slice(s![0, ..]), &covars.slice(s![0, .., ..])).unwrap();
        let mvn2 =
            MultivariateNormal::new(&means.slice(s![1, ..]), &covars.slice(s![1, .., ..])).unwrap();

        let n = 500;
        let mut observations = Array2::zeros((2 * n, means.ncols()));
        for (i, mut row) in observations.genrows_mut().into_iter().enumerate() {
            let sample = if i < n {
                mvn1.sample(&mut rng)
            } else {
                mvn2.sample(&mut rng)
            };
            row.assign(&sample);
        }
        let dataset = DatasetBase::from(observations);
        let gmm = GaussianMixtureModel::params(2)
            .with_rng(rng)
            .fit(&dataset)
            .expect("GMM fitting");

        // check weights
        let w = gmm.weights();
        assert_abs_diff_eq!(w, &weights, epsilon = 1e-1);
        // check means (since kmeans centroids are ordered randomly, we try matching both orderings)
        let m = gmm.means();
        assert!(
            abs_diff_eq!(means, &m, epsilon = 1e-1)
                || abs_diff_eq!(means, m.slice(s![..;-1, ..]), epsilon = 1e-1)
        );
        // check covariances
        let c = gmm.covariances();
        assert!(
            abs_diff_eq!(covars, &c, epsilon = 1e-1)
                || abs_diff_eq!(covars, c.slice(s![..;-1, .., ..]), epsilon = 1e-1)
        );
    }

    fn function_test_1d(x: &Array2<f64>) -> Array2<f64> {
        let mut y = Array2::zeros(x.dim());
        Zip::from(&mut y).and(x).apply(|yi, &xi| {
            if xi < 0.4 {
                *yi = xi * xi;
            } else if (0.4..0.8).contains(&xi) {
                *yi = 3. * xi + 1.;
            } else {
                *yi = f64::sin(10. * xi);
            }
        });
        y
    }

    #[test]
    fn test_zeroed_reg_covar_failure() {
        let mut rng = Isaac64Rng::seed_from_u64(42);
        let xt = Array2::random_using((50, 1), Uniform::new(0., 1.), &mut rng);
        let yt = function_test_1d(&xt);
        let data = concatenate(Axis(1), &[xt.view(), yt.view()]).unwrap();
        let dataset = DatasetBase::from(data);

        // Test that cholesky decomposition fails when reg_covariance is zero
        let gmm = GaussianMixtureModel::params(3)
            .with_reg_covariance(0.)
            .with_rng(rng.clone())
            .fit(&dataset);
        assert!(
            match gmm.expect_err("should generate an error with reg_covar being nul") {
                GmmError::LinalgError(e) => match e {
                    LinalgError::Lapack(Error::LapackComputationalFailure { return_code: 2 }) =>
                        true,
                    _ => panic!("should be a lapack error 2"),
                },
                _ => panic!("should be a linear algebra error"),
            }
        );
        // Test it passes when default value is used
        assert!(GaussianMixtureModel::params(3)
            .with_rng(rng)
            .fit(&dataset)
            .is_ok());
    }

    #[test]
    fn test_zeroed_reg_covar_const_failure() {
        // repeat values such that covariance is zero
        let xt = Array2::ones((50, 1));
        let data = concatenate(Axis(1), &[xt.view(), xt.view()]).unwrap();
        let dataset = DatasetBase::from(data);

        // Test that cholesky decomposition fails when reg_covariance is zero
        let gmm = GaussianMixtureModel::params(1)
            .with_reg_covariance(0.)
            .fit(&dataset);

        assert!(
            match gmm.expect_err("should generate an error with reg_covar being nul") {
                GmmError::LinalgError(e) => match e {
                    LinalgError::Lapack(Error::LapackComputationalFailure { return_code: 1 }) =>
                        true,
                    _ => panic!("should be a lapack error 1"),
                },
                _ => panic!("should be a linear algebra error"),
            }
        );

        // Test it passes when default value is used
        assert!(GaussianMixtureModel::params(1).fit(&dataset).is_ok());
    }

    #[test]
    fn test_centroids_prediction() {
        let mut rng = Isaac64Rng::seed_from_u64(42);
        let expected_centroids = array![[0., 1.], [-10., 20.], [-1., 10.]];
        let n = 1000;
        let blobs = DatasetBase::from(generate_blobs(n, &expected_centroids, &mut rng));

        let n_clusters = expected_centroids.len_of(Axis(0));
        let gmm = GaussianMixtureModel::params(n_clusters)
            .with_rng(rng)
            .fit(&blobs)
            .expect("GMM fitting");

        let gmm_centroids = gmm.centroids();
        let memberships = gmm.predict(&expected_centroids);

        // check that centroids used to generate test dataset belongs to the right predicted cluster
        for (i, expected_c) in expected_centroids.outer_iter().enumerate() {
            let closest_c = gmm_centroids.index_axis(Axis(0), memberships[i]);
            Zip::from(&closest_c)
                .and(&expected_c)
                .apply(|a, b| assert_abs_diff_eq!(a, b, epsilon = 1.))
        }
    }

    #[test]
    fn test_invalid_n_runs() {
        assert!(
            GaussianMixtureModel::params(1)
                .with_n_runs(0)
                .fit(&DatasetBase::from(array![[0.]]))
                .is_err(),
            "n_runs must be strictly positive"
        );
    }

    #[test]
    fn test_invalid_tolerance() {
        assert!(
            GaussianMixtureModel::params(1)
                .with_tolerance(0.)
                .fit(&DatasetBase::from(array![[0.]]))
                .is_err(),
            "tolerance must be strictly positive"
        );
    }

    #[test]
    fn test_invalid_n_clusters() {
        assert!(
            GaussianMixtureModel::params(0)
                .fit(&DatasetBase::from(array![[0., 0.]]))
                .is_err(),
            "n_clusters must be strictly positive"
        );
    }

    #[test]
    fn test_invalid_reg_covariance() {
        assert!(
            GaussianMixtureModel::params(1)
                .with_reg_covariance(-1e-6)
                .fit(&DatasetBase::from(array![[0.]]))
                .is_err(),
            "reg_covariance must be positive"
        );
    }

    #[test]
    fn test_invalid_max_n_iterations() {
        assert!(
            GaussianMixtureModel::params(1)
                .with_max_n_iterations(0)
                .fit(&DatasetBase::from(array![[0.]]))
                .is_err(),
            "max_n_iterations must be stricly positive"
        );
    }
}
