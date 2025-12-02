use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_distr::{Distribution, Normal, Uniform};
use sprs::{CsMat, TriMat};

/// Generate multiple Gaussian cliques with clear separation for motif detection.
///
/// Returns:
/// - points: Vec<Vec<f64>> (n_points x dims)
/// - adjacency: CsMat<f64> sparse symmetric 0/1 adjacency matrix
/// - norms: Vec<f64> L2 norm of each point
pub fn make_gaussian_cliques_multi(
    n_points: usize,
    noise: f64,
    n_cliques: usize,
    dims: usize,
    seed: u64,
) -> (Vec<Vec<f64>>, CsMat<f64>, Vec<f64>) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut rows = Vec::with_capacity(n_points);

    // Add some outliers (5% of data).
    let n_outliers = (n_points as f64 * 0.05).round() as usize;
    let n_cluster_points = n_points - n_outliers;

    // Distribute points evenly across cliques.
    let base = n_cluster_points / n_cliques;
    let rem = n_cluster_points % n_cliques;

    // Generate clique centers with maximum separation.
    let grid_size = (n_cliques as f64).sqrt().ceil() as usize;
    let spacing = 20.0;

    // Use up to 4 dims for separation, at least 2.
    let separation_dims = dims.min(4).max(2);

    let mut clique_centers = Vec::new();
    for i in 0..n_cliques {
        let mut center = vec![0.0; dims];

        // Grid layout in first 2 dimensions.
        let grid_x = (i % grid_size) as f64;
        let grid_y = (i / grid_size) as f64;
        center[0] = grid_x * spacing;
        if dims > 1 {
            center[1] = grid_y * spacing;
        }

        // For at least 2/3 of clusters, add distinct patterns in extra dims.
        if i < (n_cliques * 2 / 3).max(1) {
            for d in 2..separation_dims {
                let pattern_offset = match d {
                    2 => (i % 3) as f64 * spacing * 0.8,
                    3 => ((i / 3) % 3) as f64 * spacing * 0.6,
                    _ => ((i / 9) % 2) as f64 * spacing * 0.4,
                };
                center[d] = pattern_offset;
            }

            let small_offset = Uniform::new(-spacing * 0.2, spacing * 0.2).unwrap();
            for d in separation_dims..dims {
                center[d] = small_offset.sample(&mut rng);
            }
        } else {
            let medium_offset = Uniform::new(-spacing * 0.3, spacing * 0.3).unwrap();
            for d in 2..dims {
                center[d] = medium_offset.sample(&mut rng);
            }
        }

        clique_centers.push(center);
    }

    // Verify centroid distinctiveness (debug only).
    let min_distinct_distance = spacing * 0.5;
    let mut distinct_count = 0;
    for i in 0..n_cliques.min(n_cliques * 2 / 3) {
        let mut is_distinct = false;
        for j in (i + 1)..n_cliques {
            let dist: f64 = clique_centers[i]
                .iter()
                .zip(clique_centers[j].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            if dist > min_distinct_distance {
                is_distinct = true;
                break;
            }
        }
        if is_distinct {
            distinct_count += 1;
        }
    }

    debug_assert!(
        distinct_count >= (n_cliques * 2 / 3).saturating_sub(1),
        "Expected at least 2/3 distinct centroids, got {}/{}",
        distinct_count,
        n_cliques
    );

    // Track clique membership to build adjacency.
    let mut memberships = Vec::with_capacity(n_points);

    // Generate points for each clique.
    for (clique_idx, center) in clique_centers.iter().enumerate() {
        let n_for_clique = base + if clique_idx < rem { 1 } else { 0 };

        for _ in 0..n_for_clique {
            let mut point = Vec::with_capacity(dims);
            for &c in center {
                let normal = Normal::new(c, noise).unwrap();
                point.push(normal.sample(&mut rng));
            }
            rows.push(point);
            memberships.push(Some(clique_idx));
        }
    }

    // Generate outliers uniformly.
    let outlier_dist = Uniform::new(-10.0, (grid_size as f64) * spacing + 10.0).unwrap();
    for _ in 0..n_outliers {
        let mut point = Vec::with_capacity(dims);
        for _ in 0..dims {
            point.push(outlier_dist.sample(&mut rng));
        }
        rows.push(point);
        memberships.push(None);
    }

    // Ensure exact count.
    if rows.len() > n_points {
        rows.truncate(n_points);
        memberships.truncate(n_points);
    }
    while rows.len() < n_points {
        let mut point = Vec::with_capacity(dims);
        for _ in 0..dims {
            point.push(outlier_dist.sample(&mut rng));
        }
        rows.push(point);
        memberships.push(None);
    }

    // Shuffle points and memberships consistently.
    let mut indices: Vec<usize> = (0..n_points).collect();
    indices.shuffle(&mut rng);

    let mut shuffled_rows = Vec::with_capacity(n_points);
    let mut shuffled_memberships = Vec::with_capacity(n_points);
    for idx in indices {
        shuffled_rows.push(rows[idx].clone());
        shuffled_memberships.push(memberships[idx]);
    }

    // Compute L2 norms for each row.
    let mut norms = Vec::with_capacity(n_points);
    for row in &shuffled_rows {
        let sq_sum: f64 = row.iter().map(|v| v * v).sum();
        norms.push(sq_sum.sqrt());
    }

    // Build sparse adjacency as CsMat<f64>, symmetric 0/1.
    let mut triplets = TriMat::<f64>::new((n_points, n_points));

    for i in 0..n_points {
        if let Some(ci) = shuffled_memberships[i] {
            for j in (i + 1)..n_points {
                if shuffled_memberships[j] == Some(ci) {
                    // Undirected edge (i,j) and (j,i) with weight 1.0.
                    triplets.add_triplet(i, j, 1.0);
                    triplets.add_triplet(j, i, 1.0);
                }
            }
        }
    }

    let adj = triplets.to_csr();

    (shuffled_rows, adj, norms)
}
