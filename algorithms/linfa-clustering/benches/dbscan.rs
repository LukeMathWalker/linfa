use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion,
    PlotConfiguration,
};
use linfa::prelude::UncheckedHyperParams;
use linfa::traits::Predict;
use linfa_clustering::{generate_blobs, Dbscan};
use ndarray::Array2;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand_isaac::Isaac64Rng;

fn dbscan_bench(c: &mut Criterion) {
    let mut rng = Isaac64Rng::seed_from_u64(40);
    let cluster_sizes = vec![10, 100, 1000, 10000];

    let mut benchmark = c.benchmark_group("dbscan");
    benchmark.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for cluster_size in cluster_sizes {
        let rng = &mut rng;
        benchmark.bench_with_input(
            BenchmarkId::new("dbscan", cluster_size),
            &cluster_size,
            move |bencher, &cluster_size| {
                let min_points = 4;
                let n_features = 3;
                let tolerance = 0.3;
                let centroids =
                    Array2::random_using((min_points, n_features), Uniform::new(-30., 30.), rng);
                let dataset = generate_blobs(cluster_size, &centroids, rng);
                let model = Dbscan::params(min_points)
                    .tolerance(tolerance)
                    .check()
                    .unwrap();

                bencher.iter(|| black_box(model.predict(&dataset)));
            },
        );
    }
    benchmark.finish()
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = dbscan_bench
}
criterion_main!(benches);
