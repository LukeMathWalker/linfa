use linfa::dataset::{DatasetBase, Labels, Records};
use linfa::metrics::SilhouetteScore;
use linfa::traits::Predict;
use linfa_clustering::{generate_blobs, Dbscan};
use ndarray::array;
use ndarray_npy::write_npy;
use ndarray_rand::rand::SeedableRng;
use rand_isaac::Isaac64Rng;

// A routine DBScan task: build a synthetic dataset, predict clusters for it
// and save both training data and predictions to disk.
fn main() {
    // Our random number generator, seeded for reproducibility
    let mut rng = Isaac64Rng::seed_from_u64(42);

    // For each our expected centroids, generate `n` data points around it (a "blob")
    let expected_centroids = array![[10., 10.], [1., 12.], [20., 30.], [-20., 30.],];
    let n = 100;
    let dataset: DatasetBase<_, _> = generate_blobs(n, &expected_centroids, &mut rng).into();

    // Configure our training algorithm
    let min_points = 3;

    println!(
        "Clustering #{} data points grouped in 4 clusters of {} points each",
        dataset.nsamples(),
        n
    );

    // Infer an optimal set of centroids based on the training data distribution
    let cluster_memberships: Result<_, _> =
        Dbscan::params(min_points).tolerance(1.).predict(dataset);
    let cluster_memberships = cluster_memberships.unwrap();

    // sigle target dataset
    let label_count = cluster_memberships.label_count().remove(0);

    println!();
    println!("Result: ");
    for (label, count) in label_count {
        match label {
            None => println!(" - {} noise points", count),
            Some(i) => println!(" - {} points in cluster {}", count, i),
        }
    }
    println!();

    let silhouette_score = cluster_memberships.silhouette_score().unwrap();

    println!("Silhouette score: {}", silhouette_score);

    let (records, cluster_memberships) = (cluster_memberships.records, cluster_memberships.targets);

    // Save to disk our dataset (and the cluster label assigned to each observation)
    // We use the `npy` format for compatibility with NumPy
    write_npy("clustered_dataset.npy", &records).expect("Failed to write .npy file");
    write_npy(
        "clustered_memberships.npy",
        &cluster_memberships.map(|&x| x.map(|c| c as i64).unwrap_or(-1)),
    )
    .expect("Failed to write .npy file");
}
