use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion,
    PlotConfiguration,
};
use linfa::prelude::*;
use linfa::DatasetBase;
use linfa_clustering::{generate_blobs, IncrKMeansError, KMeans, KMeansInit};
use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform};
use rand_isaac::Isaac64Rng;

#[derive(Default)]
struct Stats {
    total_cost: f64,
    runs: usize,
}

impl Stats {
    fn add(&mut self, cost: f64) {
        self.total_cost += cost;
        self.runs += 1;
    }
}

impl Drop for Stats {
    fn drop(&mut self) {
        if self.runs != 0 {
            println!("Average cost = {}", self.total_cost / self.runs as f64);
        }
    }
}

fn k_means_bench(c: &mut Criterion) {
    let mut rng = Isaac64Rng::seed_from_u64(40);
    let cluster_sizes = [(100, 4), (400, 10), (3000, 10)];
    let n_features = 3;

    let mut benchmark = c.benchmark_group("naive_k_means");
    benchmark.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for &(cluster_size, n_clusters) in &cluster_sizes {
        let rng = &mut rng;
        let centroids =
            Array2::random_using((n_clusters, n_features), Uniform::new(-30., 30.), rng);
        let dataset = DatasetBase::from(generate_blobs(cluster_size, &centroids, rng));
        let mut stats = Stats::default();

        benchmark.bench_function(
            BenchmarkId::new("naive_k_means", format!("{}x{}", n_clusters, cluster_size)),
            |bencher| {
                bencher.iter(|| {
                    let m = KMeans::params_with_rng(black_box(n_clusters), black_box(rng.clone()))
                        .init_method(KMeansInit::KMeansPlusPlus)
                        .max_n_iterations(black_box(1000))
                        .tolerance(black_box(1e-3))
                        .fit(&dataset)
                        .unwrap();
                    stats.add(m.inertia());
                });
            },
        );
    }

    benchmark.finish();
}

fn k_means_incr_bench(c: &mut Criterion) {
    let mut rng = Isaac64Rng::seed_from_u64(40);
    let cluster_sizes = [(100, 4), (400, 10), (3000, 10)];
    let n_features = 3;

    let mut benchmark = c.benchmark_group("incremental_k_means");
    benchmark.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for &(cluster_size, n_clusters) in &cluster_sizes {
        let rng = &mut rng;
        let centroids =
            Array2::random_using((n_clusters, n_features), Uniform::new(-30., 30.), rng);
        let dataset = DatasetBase::from(generate_blobs(cluster_size, &centroids, rng)).shuffle(rng);
        let mut stats = Stats::default();

        benchmark.bench_function(
            BenchmarkId::new(
                "incremental_k_means",
                format!("{}x{}", n_clusters, cluster_size),
            ),
            |bencher| {
                bencher.iter(|| {
                    let clf =
                        KMeans::params_with_rng(black_box(n_clusters), black_box(rng.clone()))
                            .init_method(KMeansInit::KMeansPlusPlus)
                            .tolerance(black_box(1e-3))
                            .check()
                            .unwrap();
                    let model = dataset
                        .sample_chunks(200)
                        .cycle()
                        .try_fold(None, |current, batch| {
                            match clf.fit_with(current, &batch) {
                                // Early stop condition for the kmeans loop
                                Ok(model) => Err(model),
                                Err(IncrKMeansError::NotConverged(model)) => Ok(Some(model)),
                                Err(err) => panic!("unexpected kmeans error: {}", err),
                            }
                        })
                        .unwrap_err();
                    // Evaluate how well the model performs on the test dataset
                    stats.add(model.transform(dataset.records()).mean().unwrap());
                });
            },
        );
    }

    benchmark.finish();
}

fn k_means_init_bench(c: &mut Criterion) {
    let mut rng = Isaac64Rng::seed_from_u64(40);
    let init_methods = [KMeansInit::KMeansPlusPlus, KMeansInit::KMeansPara];
    let cluster_sizes = [(100, 10), (3000, 10), (400, 30), (500, 100)];
    let n_features = 3;

    let mut benchmark = c.benchmark_group("k_means_init");
    benchmark.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for init in &init_methods {
        for &(cluster_size, n_clusters) in &cluster_sizes {
            let rng = &mut rng;
            let centroids =
                Array2::random_using((n_clusters, n_features), Uniform::new(-30., 30.), rng);
            let dataset = DatasetBase::from(generate_blobs(cluster_size, &centroids, rng));
            let mut stats = Stats::default();

            benchmark.bench_function(
                BenchmarkId::new(
                    "k_means_init",
                    format!("{:?}:{}x{}", init, n_clusters, cluster_size),
                ),
                |bencher| {
                    bencher.iter(|| {
                        // Do 1 run of KMeans with 1 iterations, so it's mostly just the init
                        // algorithm
                        let m = KMeans::params_with_rng(black_box(n_clusters), rng.clone())
                            .init_method(init.clone())
                            .max_n_iterations(1)
                            .n_runs(1)
                            .tolerance(1000.0) // Guaranteed convergence
                            .fit(&dataset)
                            .unwrap();
                        stats.add(m.inertia());
                    });
                },
            );
        }
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = k_means_bench, k_means_init_bench, k_means_incr_bench
}
criterion_main!(benches);
