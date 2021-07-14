use crate::optics::analysis::*;
use crate::optics::errors::Result;
use crate::optics::hyperparameters::OpticsHyperParams;
use hnsw::{Hnsw, Params, Searcher};
use linfa::traits::Transformer;
use linfa::Float;
use ndarray::{ArrayView, ArrayView1, Ix1, Ix2};
use ndarray_stats::DeviationExt;
use noisy_float::prelude::*;
use rand_pcg::Pcg64;
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};
use space::MetricPoint;
use std::cmp::{max, Ordering};
use std::collections::BTreeSet;

/// Implementation of euclidean distance for ndarray
struct Euclidean<'a, F>(ArrayView1<'a, F>);

impl<F: Float> MetricPoint for Euclidean<'_, F> {
    fn distance(&self, rhs: &Self) -> u64 {
        let val = self.0.l2_dist(&rhs.0).unwrap();
        space::f64_metric(val)
    }
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
/// OPTICS (Ordering Points To Identify Clustering Structure) is a clustering algorithm that
/// doesn't explicitly cluster the data but instead creates an "augmented ordering" of the dataset
/// representing it's density-based clustering structure. This ordering contains information which
/// is equivalent to the density-based clusterings and can then be used for automatic and
/// interactive cluster analysis.
///
/// OPTICS cluster analysis can be used to derive clusters equivalent to the output of other
/// clustering algorithms such as DBSCAN. However, due to it's more complicated neighborhood
/// queries it typically has a higher computational cost than other more specific algorithms.
pub struct Optics;

#[derive(Clone)]
struct Neighbor {
    /// Index of the observation in the dataset
    index: usize,
    /// The core distance, named so to avoid name clash with `Sample::core_distance`
    c_distance: Option<N64>,
    /// The reachability distance
    r_distance: Option<N64>,
}

impl Neighbor {
    /// Create a new neighbor
    fn new(index: usize) -> Self {
        Self {
            index,
            c_distance: None,
            r_distance: None,
        }
    }

    /// Set the core distance given the minimum points in a cluster and the points neighbors
    fn set_core_distance<F: Float>(
        &mut self,
        min_pts: usize,
        neighbors: &[Neighbor],
        dataset: ArrayView<F, Ix2>,
    ) {
        let observation = dataset.row(self.index);
        self.c_distance = neighbors
            .get(min_pts - 1)
            .map(|x| dataset.row(x.index))
            .map(|x| observation.l2_dist(&x).unwrap())
            .map(|x| n64(x));
    }

    /// Convert the neighbor to a sample for the user
    fn sample(&self) -> Sample {
        Sample {
            index: self.index,
            reachability_distance: self.r_distance.map(|x| x.raw()),
            core_distance: self.c_distance.map(|x| x.raw()),
        }
    }
}

impl Eq for Neighbor {}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.r_distance == other.r_distance
    }

    fn ne(&self, other: &Self) -> bool {
        !self.eq(other)
    }
}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.r_distance.partial_cmp(&other.r_distance)
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.r_distance.cmp(&other.r_distance)
    }
}

impl Optics {
    pub fn params(min_points: usize) -> OpticsHyperParams {
        OpticsHyperParams::new(min_points)
    }
}

impl<F: Float> Transformer<ArrayView<'_, F, Ix2>, Result<OpticsAnalysis>> for OpticsHyperParams {
    fn transform(&self, observations: ArrayView<F, Ix2>) -> Result<OpticsAnalysis> {
        self.validate()?;
        let mut result = OpticsAnalysis { orderings: vec![] };

        let mut points = (0..observations.nrows())
            .map(|x| Neighbor::new(x))
            .collect::<Vec<_>>();

        // The BTreeSet is used so that the indexes are ordered to make it easy to find next
        // index
        let mut processed = BTreeSet::new();
        let mut index = 0;
        let mut seeds = Vec::new();
        loop {
            if index == points.len() {
                break;
            } else if processed.contains(&index) {
                index += 1;
                continue;
            }
            let mut expected = if processed.is_empty() { 0 } else { index };
            let mut points_index = index;
            // Look for next point to process starting from lowest possible unprocessed index
            for index in processed.range(index..) {
                if expected != *index {
                    points_index = expected;
                    break;
                }
                expected += 1;
            }
            index += 1;
            let neighbors = find_neighbors(
                observations.row(points_index),
                observations,
                self.get_tolerance(),
            );
            let n = &mut points[points_index];
            n.set_core_distance(self.get_minimum_points(), &neighbors, observations);
            if n.c_distance.is_some() {
                seeds.clear();
                // Here we get a list of "density reachable" samples that haven't been processed
                // and sort them by reachability so we can process the closest ones first.
                get_seeds(
                    observations,
                    n.clone(),
                    &neighbors,
                    &mut points,
                    &processed,
                    &mut seeds,
                );
                while !seeds.is_empty() {
                    seeds.sort_unstable_by(|a, b| b.cmp(a));
                    let (i, min_point) = seeds
                        .iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| points[**a].cmp(&points[**b]))
                        .unwrap();
                    let n = &mut points[*min_point];
                    seeds.remove(i);
                    processed.insert(n.index);
                    let neighbors = find_neighbors(
                        observations.row(n.index),
                        observations,
                        self.get_tolerance(),
                    );

                    n.set_core_distance(self.get_minimum_points(), &neighbors, observations);
                    result.orderings.push(n.sample());
                    if n.c_distance.is_some() {
                        get_seeds(
                            observations,
                            n.clone(),
                            &neighbors,
                            &mut points,
                            &processed,
                            &mut seeds,
                        );
                    }
                }
            } else {
                // Ensure whole dataset is included so we can see the points with undefined core or
                // reachability distance
                result.orderings.push(n.sample());
                processed.insert(n.index);
            }
        }
        Ok(result)
    }
}

/// For a sample find the points which are directly density reachable which have not
/// yet been processed
fn get_seeds<F: Float>(
    observations: ArrayView<F, Ix2>,
    sample: Neighbor,
    neighbors: &[Neighbor],
    points: &mut [Neighbor],
    processed: &BTreeSet<usize>,
    seeds: &mut Vec<usize>,
) {
    for n in neighbors.iter().filter(|x| !processed.contains(&x.index)) {
        let dist = n64(observations
            .row(n.index)
            .l2_dist(&observations.row(sample.index))
            .unwrap());
        let r_dist = max(sample.c_distance.unwrap(), dist);
        match points[n.index].r_distance {
            None => {
                points[n.index].r_distance = Some(r_dist);
                seeds.push(n.index);
            }
            Some(s) if r_dist < s => points[n.index].r_distance = Some(r_dist),
            _ => {}
        }
    }
}

/// Given a candidate point, a list of observations, epsilon and list of already
/// assigned cluster IDs return a list of observations that neighbor the candidate. This function
/// uses euclidean distance and the neighbours are returned in sorted order.
fn find_neighbors<F: Float>(
    candidate: ArrayView<F, Ix1>,
    observations: ArrayView<F, Ix2>,
    eps: f64,
) -> Vec<Neighbor> {
    let params = Params::new().ef_construction(observations.nrows());
    let mut searcher = Searcher::default();
    let mut hnsw: Hnsw<Euclidean<F>, Pcg64, 12, 24> = Hnsw::new_params(params);

    // insert all rows as data points into HNSW graph
    for feature in observations.genrows().into_iter() {
        hnsw.insert(Euclidean(feature), &mut searcher);
    }
    let mut neighbours = vec![space::Neighbor::invalid(); observations.nrows()];
    hnsw.nearest(
        &Euclidean(candidate.view()),
        observations.nrows(),
        &mut searcher,
        &mut neighbours,
    );
    let eps = space::f64_metric(eps);
    let out_of_bounds = neighbours
        .iter()
        .enumerate()
        .find(|(_, x)| x.distance > eps)
        .map(|(i, _)| i);
    if let Some(i) = out_of_bounds {
        // once shrink_to is stablised can switch to that
        neighbours.resize_with(i, space::Neighbor::invalid);
    }

    neighbours
        .into_iter()
        .map(|x| x.index)
        .map(|idx| {
            let observation = observations.row(idx);
            let distance = candidate.l2_dist(&observation).unwrap();
            Neighbor {
                index: idx,
                r_distance: Some(n64(distance)),
                c_distance: None,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use std::collections::BTreeSet;

    #[test]
    fn optics_consistency() {
        let params = Optics::params(3);
        let data = vec![1.0, 2.0, 3.0, 8.0, 8.0, 7.0, 2.0, 5.0, 6.0, 7.0, 8.0, 3.0];
        let data: Array2<f64> = Array2::from_shape_vec((data.len(), 1), data.clone()).unwrap();

        let samples = params.transform(data.view()).unwrap();

        // Make sure whole dataset is present:
        let indexes = samples
            .orderings
            .iter()
            .map(|x| x.index)
            .collect::<BTreeSet<_>>();
        assert!((0..data.len()).all(|x| indexes.contains(&x)));

        // As we haven't set a tolerance every point should have a core distance
        assert!(samples.orderings.iter().all(|x| x.core_distance.is_some()));
    }

    #[test]
    fn simple_dataset() {
        let params = Optics::params(3).tolerance(4.0);
        //               0    1   2    3     4     5     6     7     8    9     10    11     12
        let data = vec![
            1.0, 2.0, 3.0, 10.0, 18.0, 18.0, 15.0, 2.0, 15.0, 18.0, 3.0, 100.0, 101.0,
        ];
        let data: Array2<f64> = Array2::from_shape_vec((data.len(), 1), data.clone()).unwrap();

        // indexes of groupings of points in the dataset. These will end up with an outlier value
        // in between them to help separate things
        let first_grouping = [0, 1, 2, 7, 10].iter().collect::<BTreeSet<_>>();
        let second_grouping = [4, 5, 6, 8, 9].iter().collect::<BTreeSet<_>>();

        let samples = params.transform(data.view()).unwrap();

        let indexes = samples
            .orderings
            .iter()
            .map(|x| x.index)
            .collect::<BTreeSet<_>>();
        assert!((0..data.len()).all(|x| indexes.contains(&x)));

        assert!(samples
            .orderings
            .iter()
            .take(first_grouping.len())
            .all(|x| first_grouping.contains(&x.index)));
        let skip_len = first_grouping.len() + 1;
        assert!(samples
            .orderings
            .iter()
            .skip(skip_len)
            .take(first_grouping.len())
            .all(|x| second_grouping.contains(&x.index)));

        let anomaly = samples.orderings.iter().find(|x| x.index == 3).unwrap();
        assert!(anomaly.core_distance.is_none());
        assert!(anomaly.reachability_distance.is_none());

        let anomaly = samples.orderings.iter().find(|x| x.index == 11).unwrap();
        assert!(anomaly.core_distance.is_none());
        assert!(anomaly.reachability_distance.is_none());

        let anomaly = samples.orderings.iter().find(|x| x.index == 12).unwrap();
        assert!(anomaly.core_distance.is_none());
        assert!(anomaly.reachability_distance.is_none());
    }

    #[test]
    fn dataset_too_small() {
        let params = Optics::params(4);
        let data = vec![1.0, 2.0, 3.0];
        let data: Array2<f64> = Array2::from_shape_vec((data.len(), 1), data).unwrap();

        let samples = params.transform(data.view()).unwrap();

        assert!(samples
            .orderings
            .iter()
            .all(|x| x.core_distance.is_none() && x.reachability_distance.is_none()));
    }

    #[test]
    fn invalid_params() {
        let params = Optics::params(1);
        let data = vec![1.0, 2.0, 3.0];
        let data: Array2<f64> = Array2::from_shape_vec((data.len(), 1), data).unwrap();
        assert!(params.transform(data.view()).is_err());

        let params = Optics::params(2);
        assert!(params.transform(data.view()).is_ok());

        let params = params.tolerance(0.0);
        assert!(params.transform(data.view()).is_err());
    }

    #[test]
    fn find_neighbors_test() {
        let data = vec![1.0, 2.0, 10.0, 15.0, 13.0];
        let data: Array2<f64> = Array2::from_shape_vec((data.len(), 1), data.clone()).unwrap();

        let neighbors = find_neighbors(data.row(0), data.view(), 6.0);
        assert_eq!(neighbors.len(), 2);
        assert_eq!(
            vec![0, 1],
            neighbors
                .iter()
                .map(|x| x.r_distance.unwrap().raw() as u32)
                .collect::<Vec<u32>>()
        );
        assert!(neighbors.iter().all(|x| x.c_distance.is_none()));

        let neighbors = find_neighbors(data.row(4), data.view(), 6.0);
        assert_eq!(neighbors.len(), 3);
        assert!(neighbors.iter().all(|x| x.c_distance.is_none()));
        assert_eq!(
            vec![0, 2, 3],
            neighbors
                .iter()
                .map(|x| x.r_distance.unwrap().raw() as u32)
                .collect::<Vec<u32>>()
        );
    }

    #[test]
    fn get_seeds_test() {
        let data = vec![1.0, 2.0, 10.0, 15.0, 13.0];
        let data: Array2<f64> = Array2::from_shape_vec((data.len(), 1), data.clone()).unwrap();

        let mut points = (0..data.nrows())
            .map(|i| Neighbor::new(i))
            .collect::<Vec<_>>();

        let neighbors = find_neighbors(data.row(0), data.view(), 6.0);
        // set core distance and make sure it's set correctly given number of neghobrs restriction

        points[0].set_core_distance(3, &neighbors, data.view());
        assert!(points[0].c_distance.is_none());

        let neighbors = find_neighbors(data.row(4), data.view(), 6.0);
        points[4].set_core_distance(3, &neighbors, data.view());
        assert!(points[4].c_distance.is_some());

        let mut seeds = vec![];
        let mut processed = BTreeSet::new();
        // With a valid core distance make sure neighbours to point are returned in order if
        // unprocessed

        get_seeds(
            data.view(),
            points[4].clone(),
            &neighbors,
            &mut points,
            &processed,
            &mut seeds,
        );

        assert_eq!(seeds, vec![4, 3, 2]);

        let mut points = (0..data.nrows())
            .map(|i| Neighbor::new(i))
            .collect::<Vec<_>>();

        // if one of the neighbours has been processed make sure it's not in the seed list

        points[4].set_core_distance(3, &neighbors, data.view());
        processed.insert(3);
        seeds.clear();

        get_seeds(
            data.view(),
            points[4].clone(),
            &neighbors,
            &mut points,
            &processed,
            &mut seeds,
        );

        assert_eq!(seeds, vec![4, 2]);

        let mut points = (0..data.nrows())
            .map(|i| Neighbor::new(i))
            .collect::<Vec<_>>();

        // If one of the neighbours has a smaller R distance than it has to the core point make
        // sure it's not added to the seed list

        processed.clear();
        points[4].set_core_distance(3, &neighbors, data.view());
        points[2].r_distance = Some(n64(0.001));
        seeds.clear();

        get_seeds(
            data.view(),
            points[4].clone(),
            &neighbors,
            &mut points,
            &processed,
            &mut seeds,
        );

        assert_eq!(seeds, vec![4, 3]);
    }
}
