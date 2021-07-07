use linfa::traits::Fit;
use linfa::traits::Predict;
use linfa::DatasetBase;
use linfa_clustering::{generate_blobs, KMeans};
use ndarray::{array, Axis};
use ndarray_npy::write_npy;
use ndarray_rand::rand::SeedableRng;
use rand_isaac::Isaac64Rng;

// A routine K-means task: build a synthetic dataset, fit the algorithm on it
// and save both training data and predictions to disk.
fn main() {
    // Our random number generator, seeded for reproducibility
    let mut rng = Isaac64Rng::seed_from_u64(42);

    // For each our expected centroids, generate `n` data points around it (a "blob")
    let expected_centroids = array![[10., 10.], [1., 12.], [20., 30.], [-20., 30.],];
    let n = 10000;
    let dataset = DatasetBase::from(generate_blobs(n, &expected_centroids, &mut rng));

    // Configure our training algorithm
    let n_clusters = expected_centroids.len_of(Axis(0));
    let model = KMeans::params_with_rng(n_clusters, rng)
        .max_n_iterations(200)
        .tolerance(1e-5)
        .fit(&dataset)
        .expect("KMeans fitted");

    // Assign each point to a cluster using the set of centroids found using `fit`
    let dataset = model.predict(dataset);
    let DatasetBase {
        records, targets, ..
    } = dataset;

    // Save to disk our dataset (and the cluster label assigned to each observation)
    // We use the `npy` format for compatibility with NumPy
    write_npy("clustered_dataset.npy", &records).expect("Failed to write .npy file");
    write_npy("clustered_memberships.npy", &targets.map(|&x| x as u64))
        .expect("Failed to write .npy file");
}
