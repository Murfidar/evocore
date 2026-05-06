use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

use crate::gene_spec::GeneKind;
use crate::operators::{binary_ops, float_ops};
use crate::selection;
use crate::utils::{derive_seed, OP_CROSSOVER_PROB, OP_INIT, OP_MUTATION};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CrossoverType {
    Sbx,
    Blx,
    OnePoint,
    TwoPoint,
    UniformXO,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MutationType {
    Gaussian,
    Uniform,
    BitFlip,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SelectionType {
    Tournament,
    Roulette,
    Rank,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReproduceConfig {
    pub crossover_type: CrossoverType,
    pub crossover_prob: f64,
    pub crossover_eta: f64,
    pub crossover_alpha: f64,
    pub mutation_type: MutationType,
    pub mutation_prob: f64,
    pub mutation_individual_prob: f64,
    pub mutation_sigmas: Vec<f64>,
    pub gene_bounds: Vec<(f64, f64)>,
    pub gene_kinds: Vec<GeneKind>,
    pub selection_type: SelectionType,
    pub tournament_size: usize,
    pub population_size: usize,
}

pub fn clamp_and_round(genes: &[f64], bounds: &[(f64, f64)], kinds: &[GeneKind]) -> Vec<f64> {
    assert_eq!(
        genes.len(),
        bounds.len(),
        "clamp_and_round: genes/bounds mismatch"
    );
    assert_eq!(
        genes.len(),
        kinds.len(),
        "clamp_and_round: genes/kinds mismatch"
    );

    genes
        .iter()
        .enumerate()
        .map(|(idx, &gene)| {
            let (low, high) = bounds[idx];
            match kinds[idx] {
                GeneKind::Float => gene.clamp(low, high),
                GeneKind::Int => gene.round().clamp(low, high),
                GeneKind::Bool => {
                    if gene >= 0.5 {
                        1.0
                    } else {
                        0.0
                    }
                }
            }
        })
        .collect()
}

fn apply_crossover(
    a: &[f64],
    b: &[f64],
    config: &ReproduceConfig,
    master_seed: u64,
    generation: u64,
    pair_idx: u64,
) -> (Vec<f64>, Vec<f64>) {
    match config.crossover_type {
        CrossoverType::Sbx => float_ops::simulated_binary_crossover(
            a,
            b,
            config.crossover_eta,
            master_seed,
            generation,
            pair_idx,
        ),
        CrossoverType::Blx => float_ops::blend_crossover(
            a,
            b,
            config.crossover_alpha,
            master_seed,
            generation,
            pair_idx,
        ),
        CrossoverType::OnePoint => {
            binary_ops::one_point_crossover(a, b, master_seed, generation, pair_idx)
        }
        CrossoverType::TwoPoint => {
            binary_ops::two_point_crossover(a, b, master_seed, generation, pair_idx)
        }
        CrossoverType::UniformXO => {
            binary_ops::uniform_crossover(a, b, 0.5, master_seed, generation, pair_idx)
        }
    }
}

fn apply_mutation(
    genes: &[f64],
    config: &ReproduceConfig,
    master_seed: u64,
    generation: u64,
    individual_idx: u64,
) -> Vec<f64> {
    if config.mutation_individual_prob <= 0.0 {
        return genes.to_vec();
    }

    let mut rng = StdRng::seed_from_u64(derive_seed(
        master_seed,
        generation,
        individual_idx,
        OP_MUTATION,
    ));
    if config.mutation_individual_prob < 1.0 && rng.gen::<f64>() >= config.mutation_individual_prob
    {
        return genes.to_vec();
    }

    genes
        .iter()
        .enumerate()
        .map(
            |(idx, &gene)| match (&config.mutation_type, &config.gene_kinds[idx]) {
                (MutationType::Gaussian, GeneKind::Float) => {
                    if rng.gen::<f64>() < config.mutation_prob {
                        let sigma = config.mutation_sigmas[idx].max(1e-20);
                        gene + Normal::new(0.0, sigma).unwrap().sample(&mut rng)
                    } else {
                        gene
                    }
                }
                (MutationType::Gaussian, GeneKind::Int) => {
                    if rng.gen::<f64>() < config.mutation_prob {
                        let sigma = config.mutation_sigmas[idx].max(1e-20);
                        (gene + Normal::new(0.0, sigma).unwrap().sample(&mut rng)).round()
                    } else {
                        gene
                    }
                }
                (MutationType::Uniform, GeneKind::Float) => {
                    if rng.gen::<f64>() < config.mutation_prob {
                        let (low, high) = config.gene_bounds[idx];
                        if low == high {
                            low
                        } else {
                            rng.gen_range(low..high)
                        }
                    } else {
                        gene
                    }
                }
                (MutationType::Uniform, GeneKind::Int) => {
                    if rng.gen::<f64>() < config.mutation_prob {
                        let low = config.gene_bounds[idx].0.round() as i64;
                        let high = config.gene_bounds[idx].1.round() as i64;
                        rng.gen_range(low..=high) as f64
                    } else {
                        gene
                    }
                }
                (MutationType::BitFlip, GeneKind::Bool) | (_, GeneKind::Bool) => {
                    if rng.gen::<f64>() < config.mutation_prob {
                        if gene >= 0.5 {
                            0.0
                        } else {
                            1.0
                        }
                    } else {
                        gene
                    }
                }
                (MutationType::BitFlip, _) => gene,
            },
        )
        .collect()
}

pub fn init_population(
    gene_bounds: &[(f64, f64)],
    gene_kinds: &[GeneKind],
    population_size: usize,
    master_seed: u64,
) -> Vec<Vec<f64>> {
    let gene_len = gene_bounds.len();
    assert_eq!(
        gene_len,
        gene_kinds.len(),
        "init_population: bounds and kinds length mismatch"
    );

    (0..population_size)
        .map(|individual_idx| {
            let mut rng =
                StdRng::seed_from_u64(derive_seed(master_seed, 0, individual_idx as u64, OP_INIT));

            (0..gene_len)
                .map(|gene_idx| {
                    let (low, high) = gene_bounds[gene_idx];
                    match gene_kinds[gene_idx] {
                        GeneKind::Float => {
                            if low == high {
                                low
                            } else {
                                rng.gen_range(low..high)
                            }
                        }
                        GeneKind::Int => {
                            let low = low.round() as i64;
                            let high = high.round() as i64;
                            rng.gen_range(low..=high) as f64
                        }
                        GeneKind::Bool => {
                            if rng.gen::<bool>() {
                                1.0
                            } else {
                                0.0
                            }
                        }
                    }
                })
                .collect()
        })
        .collect()
}

pub fn reproduce(
    population: &[Vec<f64>],
    fitnesses: &[f64],
    config: &ReproduceConfig,
    master_seed: u64,
    generation: u64,
) -> Vec<Vec<f64>> {
    assert_eq!(
        population.len(),
        fitnesses.len(),
        "reproduce: population and fitness length mismatch"
    );
    assert_eq!(
        config.gene_bounds.len(),
        config.gene_kinds.len(),
        "reproduce: bounds and kinds length mismatch"
    );
    assert_eq!(
        config.gene_bounds.len(),
        config.mutation_sigmas.len(),
        "reproduce: bounds and mutation_sigmas length mismatch"
    );

    let pop_size = config.population_size;
    if pop_size == 0 {
        return Vec::new();
    }

    let pairs_needed = pop_size.div_ceil(2);
    let parent_count = pairs_needed * 2;
    let parent_indices = match config.selection_type {
        SelectionType::Tournament => selection::tournament_selection(
            fitnesses,
            parent_count,
            config.tournament_size,
            master_seed,
            generation,
        ),
        SelectionType::Roulette => {
            selection::roulette_selection(fitnesses, parent_count, master_seed, generation)
        }
        SelectionType::Rank => {
            selection::rank_selection(fitnesses, parent_count, master_seed, generation)
        }
    };

    let mut new_population = Vec::with_capacity(pop_size);
    let mut offspring_idx = 0_u64;

    for pair_idx in 0..pairs_needed {
        let parent_a = &population[parent_indices[pair_idx * 2]];
        let parent_b = &population[parent_indices[pair_idx * 2 + 1]];

        let apply_xo = {
            let mut rng = StdRng::seed_from_u64(derive_seed(
                master_seed,
                generation,
                pair_idx as u64,
                OP_CROSSOVER_PROB,
            ));
            rng.gen::<f64>() < config.crossover_prob
        };

        let (child_a, child_b) = if apply_xo {
            apply_crossover(
                parent_a,
                parent_b,
                config,
                master_seed,
                generation,
                pair_idx as u64,
            )
        } else {
            (parent_a.clone(), parent_b.clone())
        };

        let child_a = clamp_and_round(
            &apply_mutation(&child_a, config, master_seed, generation, offspring_idx),
            &config.gene_bounds,
            &config.gene_kinds,
        );
        offspring_idx += 1;

        let child_b = clamp_and_round(
            &apply_mutation(&child_b, config, master_seed, generation, offspring_idx),
            &config.gene_bounds,
            &config.gene_kinds,
        );
        offspring_idx += 1;

        new_population.push(child_a);
        if new_population.len() < pop_size {
            new_population.push(child_b);
        }
    }

    new_population
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gene_spec::GeneKind;

    fn float_bounds(n: usize) -> Vec<(f64, f64)> {
        vec![(-5.0, 5.0); n]
    }

    fn float_kinds(n: usize) -> Vec<GeneKind> {
        vec![GeneKind::Float; n]
    }

    #[test]
    fn test_init_population_correct_size() {
        let pop = init_population(&float_bounds(5), &float_kinds(5), 20, 42);
        assert_eq!(pop.len(), 20, "population size must equal requested size");
    }

    #[test]
    fn test_init_population_correct_gene_length() {
        let pop = init_population(&float_bounds(7), &float_kinds(7), 10, 42);
        assert!(pop.iter().all(|ind| ind.len() == 7));
    }

    #[test]
    fn test_init_population_genes_within_bounds() {
        let bounds = vec![(-2.0_f64, 2.0), (0.0, 10.0), (-1.0, 1.0)];
        let kinds = vec![GeneKind::Float; 3];
        let pop = init_population(&bounds, &kinds, 50, 42);
        for ind in &pop {
            for (i, &g) in ind.iter().enumerate() {
                assert!(
                    g >= bounds[i].0 && g < bounds[i].1,
                    "gene[{}]={} outside [{}, {})",
                    i,
                    g,
                    bounds[i].0,
                    bounds[i].1
                );
            }
        }
    }

    #[test]
    fn test_init_population_int_genes_are_integers() {
        let bounds = vec![(5.0_f64, 200.0), (10.0, 500.0)];
        let kinds = vec![GeneKind::Int, GeneKind::Int];
        let pop = init_population(&bounds, &kinds, 20, 42);
        for ind in &pop {
            for &g in ind {
                assert_eq!(g, g.round(), "int gene {} is not an integer", g);
            }
        }
    }

    #[test]
    fn test_init_population_bool_genes_are_binary() {
        let bounds = vec![(0.0_f64, 1.0); 10];
        let kinds = vec![GeneKind::Bool; 10];
        let pop = init_population(&bounds, &kinds, 20, 42);
        for ind in &pop {
            for &g in ind {
                assert!(
                    g == 0.0 || g == 1.0,
                    "bool gene {} is neither 0.0 nor 1.0",
                    g
                );
            }
        }
    }

    #[test]
    fn test_init_population_deterministic() {
        let bounds = float_bounds(4);
        let kinds = float_kinds(4);
        let p1 = init_population(&bounds, &kinds, 10, 7);
        let p2 = init_population(&bounds, &kinds, 10, 7);
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_init_population_different_seeds_diverge() {
        let bounds = float_bounds(4);
        let kinds = float_kinds(4);
        let p1 = init_population(&bounds, &kinds, 10, 1);
        let p2 = init_population(&bounds, &kinds, 10, 2);
        assert_ne!(p1[0], p2[0], "different seeds must produce different genes");
    }

    #[test]
    fn test_init_population_fixed_float_bounds_return_fixed_value() {
        let bounds = vec![(1.25_f64, 1.25), (-5.0, 5.0)];
        let kinds = vec![GeneKind::Float, GeneKind::Float];
        let pop = init_population(&bounds, &kinds, 12, 42);

        assert!(pop.iter().all(|ind| ind[0] == 1.25));
        assert!(pop.iter().all(|ind| ind[1] >= -5.0 && ind[1] < 5.0));
    }

    #[test]
    fn test_reproduce_fixed_numeric_bounds_are_preserved() {
        let bounds = vec![(1.25_f64, 1.25), (2.0, 2.0), (-5.0, 5.0)];
        let kinds = vec![GeneKind::Float, GeneKind::Int, GeneKind::Float];
        let pop = init_population(&bounds, &kinds, 20, 42);
        let fitnesses = (0..20).map(|value| value as f64).collect::<Vec<_>>();
        let config = ReproduceConfig {
            crossover_type: CrossoverType::Sbx,
            crossover_prob: 1.0,
            crossover_eta: 2.0,
            crossover_alpha: 0.5,
            mutation_type: MutationType::Uniform,
            mutation_prob: 1.0,
            mutation_individual_prob: 1.0,
            mutation_sigmas: vec![0.0, 0.0, 2.0],
            gene_bounds: bounds.clone(),
            gene_kinds: kinds.clone(),
            selection_type: SelectionType::Tournament,
            tournament_size: 3,
            population_size: 20,
        };

        let new_pop = reproduce(&pop, &fitnesses, &config, 42, 0);

        assert!(new_pop.iter().all(|ind| ind[0] == 1.25));
        assert!(new_pop.iter().all(|ind| ind[1] == 2.0));
        assert!(new_pop.iter().all(|ind| ind[2] >= -5.0 && ind[2] <= 5.0));
    }

    #[test]
    fn test_reproduce_mutation_individual_probability_zero_skips_all_mutation() {
        let pop = vec![
            vec![0.0_f64, 5.0],
            vec![1.0_f64, 10.0],
            vec![2.0_f64, 15.0],
            vec![3.0_f64, 20.0],
        ];
        let fitnesses = (0..pop.len()).map(|value| value as f64).collect::<Vec<_>>();
        let config = ReproduceConfig {
            crossover_type: CrossoverType::UniformXO,
            crossover_prob: 0.0,
            crossover_eta: 2.0,
            crossover_alpha: 0.5,
            mutation_type: MutationType::Uniform,
            mutation_prob: 1.0,
            mutation_individual_prob: 0.0,
            mutation_sigmas: vec![1.0, 1.0],
            gene_bounds: vec![(0.0, 100.0), (0.0, 100.0)],
            gene_kinds: vec![GeneKind::Float, GeneKind::Int],
            selection_type: SelectionType::Tournament,
            tournament_size: 2,
            population_size: 8,
        };

        let new_pop = reproduce(&pop, &fitnesses, &config, 42, 0);

        assert_eq!(new_pop.len(), 8);
        assert!(new_pop.iter().all(|ind| pop.contains(ind)));
    }

    #[test]
    fn test_clamp_and_round_float_clamps_to_bounds() {
        let genes = vec![-10.0_f64, 10.0];
        let bounds = vec![(-5.0_f64, 5.0), (-5.0, 5.0)];
        let kinds = vec![GeneKind::Float, GeneKind::Float];
        let result = clamp_and_round(&genes, &bounds, &kinds);
        assert_eq!(result, vec![-5.0, 5.0]);
    }

    #[test]
    fn test_clamp_and_round_int_rounds_and_clamps() {
        let genes = vec![3.7_f64, -1.3, 200.9];
        let bounds = vec![(1.0_f64, 100.0), (0.0, 50.0), (0.0, 100.0)];
        let kinds = vec![GeneKind::Int; 3];
        let result = clamp_and_round(&genes, &bounds, &kinds);
        assert_eq!(result[0], 4.0);
        assert_eq!(result[1], 0.0);
        assert_eq!(result[2], 100.0);
    }

    #[test]
    fn test_clamp_and_round_bool_thresholds_at_half() {
        let genes = vec![0.3_f64, 0.7, 0.5, 0.49];
        let bounds = vec![(0.0_f64, 1.0); 4];
        let kinds = vec![GeneKind::Bool; 4];
        let result = clamp_and_round(&genes, &bounds, &kinds);
        assert_eq!(result, vec![0.0, 1.0, 1.0, 0.0]);
    }

    fn make_float_config(pop_size: usize, gene_len: usize) -> ReproduceConfig {
        ReproduceConfig {
            crossover_type: CrossoverType::Sbx,
            crossover_prob: 0.9,
            crossover_eta: 2.0,
            crossover_alpha: 0.5,
            mutation_type: MutationType::Gaussian,
            mutation_prob: 0.1,
            mutation_individual_prob: 1.0,
            mutation_sigmas: vec![0.5; gene_len],
            gene_bounds: vec![(-5.0, 5.0); gene_len],
            gene_kinds: vec![GeneKind::Float; gene_len],
            selection_type: SelectionType::Tournament,
            tournament_size: 3,
            population_size: pop_size,
        }
    }

    #[test]
    fn test_reproduce_returns_correct_population_size() {
        let pop: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.1; 5]).collect();
        let fitnesses: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let config = make_float_config(20, 5);
        let new_pop = reproduce(&pop, &fitnesses, &config, 42, 0);
        assert_eq!(new_pop.len(), 20);
    }

    #[test]
    fn test_reproduce_returns_correct_gene_length() {
        let pop: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64; 6]).collect();
        let fitnesses: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let config = make_float_config(10, 6);
        let new_pop = reproduce(&pop, &fitnesses, &config, 42, 0);
        assert!(new_pop.iter().all(|ind| ind.len() == 6));
    }

    #[test]
    fn test_reproduce_float_genes_within_bounds() {
        let pop: Vec<Vec<f64>> = (0..30).map(|i| vec![i as f64 * 0.1 - 1.5; 4]).collect();
        let fitnesses: Vec<f64> = (0..30).map(|i| -(i as f64 * 0.1)).collect();
        let config = ReproduceConfig {
            gene_bounds: vec![(-5.0, 5.0); 4],
            ..make_float_config(30, 4)
        };
        let new_pop = reproduce(&pop, &fitnesses, &config, 42, 0);
        for ind in &new_pop {
            for &g in ind {
                assert!(
                    (-5.0..=5.0).contains(&g),
                    "float gene {} outside [-5.0, 5.0]",
                    g
                );
            }
        }
    }

    #[test]
    fn test_reproduce_int_genes_are_always_integers() {
        let pop: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64; 3]).collect();
        let fitnesses: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let config = ReproduceConfig {
            crossover_type: CrossoverType::Sbx,
            crossover_prob: 0.9,
            crossover_eta: 2.0,
            crossover_alpha: 0.5,
            mutation_type: MutationType::Gaussian,
            mutation_prob: 0.5,
            mutation_individual_prob: 1.0,
            mutation_sigmas: vec![2.0; 3],
            gene_bounds: vec![(0.0, 20.0); 3],
            gene_kinds: vec![GeneKind::Int; 3],
            selection_type: SelectionType::Tournament,
            tournament_size: 2,
            population_size: 20,
        };
        let new_pop = reproduce(&pop, &fitnesses, &config, 42, 0);
        for ind in &new_pop {
            for &g in ind {
                assert_eq!(g, g.round(), "int gene {} is not integer-valued", g);
            }
        }
    }

    #[test]
    fn test_reproduce_bool_genes_are_always_binary() {
        let pop: Vec<Vec<f64>> = (0..20)
            .map(|i| {
                (0..8)
                    .map(|j| if (i + j) % 2 == 0 { 1.0 } else { 0.0 })
                    .collect()
            })
            .collect();
        let fitnesses: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let config = ReproduceConfig {
            crossover_type: CrossoverType::OnePoint,
            crossover_prob: 0.8,
            crossover_eta: 2.0,
            crossover_alpha: 0.5,
            mutation_type: MutationType::BitFlip,
            mutation_prob: 0.1,
            mutation_individual_prob: 1.0,
            mutation_sigmas: vec![0.0; 8],
            gene_bounds: vec![(0.0, 1.0); 8],
            gene_kinds: vec![GeneKind::Bool; 8],
            selection_type: SelectionType::Tournament,
            tournament_size: 2,
            population_size: 20,
        };
        let new_pop = reproduce(&pop, &fitnesses, &config, 42, 0);
        for ind in &new_pop {
            for &g in ind {
                assert!(g == 0.0 || g == 1.0, "bool gene {} is not 0.0 or 1.0", g);
            }
        }
    }

    #[test]
    fn test_reproduce_deterministic() {
        let pop: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.5; 4]).collect();
        let fitnesses: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let config = make_float_config(10, 4);
        let r1 = reproduce(&pop, &fitnesses, &config, 77, 5);
        let r2 = reproduce(&pop, &fitnesses, &config, 77, 5);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_reproduce_different_generations_diverge() {
        let pop: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.5; 4]).collect();
        let fitnesses: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let config = make_float_config(10, 4);
        let r1 = reproduce(&pop, &fitnesses, &config, 42, 0);
        let r2 = reproduce(&pop, &fitnesses, &config, 42, 1);
        assert_ne!(
            r1, r2,
            "different generations must produce different offspring"
        );
    }
}
