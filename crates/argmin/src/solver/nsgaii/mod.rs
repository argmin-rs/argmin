// Copyright 2018-2025 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # NSGA-II (Non-dominated Sorting Genetic Algorithm II)
//!
//! NSGA-II is a popular multi-objective genetic algorithm proposed by Deb et al. in 2002.
//! It uses fast non-dominated sorting and crowding distance to maintain diversity in
//! the Pareto front.
//!
//! ## Reference
//!
//! Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. A. M. T. (2002).
//! A fast and elitist multiobjective genetic algorithm: NSGA-II.
//! IEEE transactions on evolutionary computation, 6(2), 182-197.
//! <https://doi.org/10.1109/4235.996017>
//!
//! ## Example
//!
//! ```
//! use argmin::core::{Error, Executor, MultiObjectiveCostFunction, PopulationState};
//! use argmin::solver::nsgaii::NsgaII;
//!
//! #[derive(Clone)]
//! struct MyProblem;
//!
//! impl MultiObjectiveCostFunction for MyProblem {
//!     type Param = Vec<f64>;
//!     type Output = Vec<f64>;
//!
//!     fn objectives(&self, param: &Self::Param) -> Result<Self::Output, Error> {
//!         // Example: Two objectives (minimize both)
//!         let f1 = param[0].powi(2) + param[1].powi(2);
//!         let f2 = (param[0] - 1.0).powi(2) + (param[1] - 1.0).powi(2);
//!         Ok(vec![f1, f2])
//!     }
//!
//!     fn num_objectives(&self) -> usize { 2 }
//! }
//!
//! fn run() -> Result<(), Error> {
//!     let problem = MyProblem;
//!     let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
//!
//!     let solver = NsgaII::new(bounds, 100)?
//!         .with_crossover_probability(0.9)
//!         .with_mutation_probability(0.1);
//!
//!     let res = Executor::new(problem, solver)
//!         .configure(|state| state.max_iters(100))
//!         .run()?;
//!
//!     Ok(())
//! }
//! ```

use crate::core::{
    ArgminFloat, Error, MultiObjectiveCostFunction, PopulationState, Problem, Solver, KV,
};
use rand::prelude::*;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// Individual in the NSGA-II population
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct Individual<P, F> {
    /// Decision variables
    pub position: P,
    /// Objective function values
    pub objectives: Vec<F>,
    /// Non-domination rank (0 is best, lower is better)
    pub rank: usize,
    /// Crowding distance (higher is better for diversity)
    pub crowding_distance: F,
}

impl<P, F: ArgminFloat> Individual<P, F> {
    /// Create a new individual
    pub fn new(position: P, objectives: Vec<F>) -> Self {
        Individual {
            position,
            objectives,
            rank: 0,
            crowding_distance: F::zero(),
        }
    }
}

/// NSGA-II solver
///
/// Implements the NSGA-II algorithm for multi-objective optimization.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct NsgaII<F, R = StdRng> {
    /// Variable bounds (min, max) for each dimension
    bounds: Vec<(F, F)>,
    /// Population size
    population_size: usize,
    /// Crossover probability
    crossover_probability: F,
    /// Mutation probability
    mutation_probability: F,
    /// Distribution index for crossover
    crossover_eta: F,
    /// Distribution index for mutation
    mutation_eta: F,
    /// Random number generator
    #[cfg_attr(feature = "serde1", serde(skip))]
    rng: R,
}

impl<F> NsgaII<F, StdRng>
where
    F: ArgminFloat,
{
    /// Create a new NSGA-II solver
    ///
    /// # Arguments
    ///
    /// * `bounds` - Variable bounds (min, max) for each dimension
    /// * `population_size` - Size of the population (should be even)
    ///
    /// # Errors
    ///
    /// Returns an error if bounds are invalid or population size is < 4
    pub fn new(bounds: Vec<(F, F)>, population_size: usize) -> Result<Self, Error> {
        if bounds.is_empty() {
            return Err(Error::msg("NSGA-II: bounds cannot be empty"));
        }

        if population_size < 4 {
            return Err(Error::msg("NSGA-II: population size must be at least 4"));
        }

        if population_size % 2 != 0 {
            return Err(Error::msg("NSGA-II: population size must be even"));
        }

        for (i, &(min, max)) in bounds.iter().enumerate() {
            if min >= max {
                return Err(Error::msg(format!(
                    "NSGA-II: invalid bounds at dimension {}: min ({}) must be < max ({})",
                    i, min, max
                )));
            }
        }

        let n_vars = bounds.len();
        let mutation_probability = F::one() / F::from_usize(n_vars).unwrap();

        Ok(NsgaII {
            bounds,
            population_size,
            crossover_probability: F::from_f64(0.9).unwrap(),
            mutation_probability,
            crossover_eta: F::from_f64(20.0).unwrap(),
            mutation_eta: F::from_f64(20.0).unwrap(),
            rng: StdRng::from_os_rng(),
        })
    }
}

impl<F, R> NsgaII<F, R>
where
    F: ArgminFloat,
    R: Rng,
{
    /// Set a custom random number generator
    pub fn with_rng<R2: Rng>(self, rng: R2) -> NsgaII<F, R2> {
        NsgaII {
            bounds: self.bounds,
            population_size: self.population_size,
            crossover_probability: self.crossover_probability,
            mutation_probability: self.mutation_probability,
            crossover_eta: self.crossover_eta,
            mutation_eta: self.mutation_eta,
            rng,
        }
    }

    /// Set crossover probability (default: 0.9)
    pub fn with_crossover_probability(mut self, prob: F) -> Self {
        self.crossover_probability = prob;
        self
    }

    /// Set mutation probability per variable (default: 1/n)
    pub fn with_mutation_probability(mut self, prob: F) -> Self {
        self.mutation_probability = prob;
        self
    }

    /// Set distribution index for SBX crossover (default: 20.0)
    ///
    /// Higher values produce offspring closer to parents.
    pub fn with_crossover_eta(mut self, eta: F) -> Self {
        self.crossover_eta = eta;
        self
    }

    /// Set distribution index for polynomial mutation (default: 20.0)
    ///
    /// Higher values produce smaller mutations.
    pub fn with_mutation_eta(mut self, eta: F) -> Self {
        self.mutation_eta = eta;
        self
    }
}

impl<F, R> NsgaII<F, R>
where
    F: ArgminFloat,
    R: Rng,
{
    /// Generate random position within bounds
    fn random_position(&mut self) -> Vec<F> {
        self.bounds
            .iter()
            .map(|&(min, max)| {
                let val = self.rng.random::<f64>();
                let min_f64 = min.to_f64().unwrap();
                let max_f64 = max.to_f64().unwrap();
                let range = max_f64 - min_f64;
                F::from_f64(min_f64 + val * range).unwrap()
            })
            .collect()
    }

    /// Simulated Binary Crossover (SBX)
    fn crossover(&mut self, parent1: &[F], parent2: &[F]) -> (Vec<F>, Vec<F>) {
        let n = parent1.len();
        let mut child1 = parent1.to_vec();
        let mut child2 = parent2.to_vec();

        if self.rng.random::<f64>() > self.crossover_probability.to_f64().unwrap() {
            return (child1, child2);
        }

        let eta = self.crossover_eta.to_f64().unwrap();

        for i in 0..n {
            if self.rng.random::<f64>() > 0.5 {
                continue;
            }

            let y1 = parent1[i].to_f64().unwrap();
            let y2 = parent2[i].to_f64().unwrap();

            if (y1 - y2).abs() < 1e-14 {
                child1[i] = parent1[i];
                child2[i] = parent2[i];
                continue;
            }

            let (y_min, y_max) = if y1 < y2 { (y1, y2) } else { (y2, y1) };

            let rand = self.rng.random::<f64>();

            let beta = if rand <= 0.5 {
                (2.0 * rand).powf(1.0 / (eta + 1.0))
            } else {
                (1.0 / (2.0 * (1.0 - rand))).powf(1.0 / (eta + 1.0))
            };

            let c1 = 0.5 * ((y1 + y2) - beta * (y_max - y_min));
            let c2 = 0.5 * ((y1 + y2) + beta * (y_max - y_min));

            let (lb, ub) = self.bounds[i];
            let lb = lb.to_f64().unwrap();
            let ub = ub.to_f64().unwrap();

            child1[i] = F::from_f64(c1.max(lb).min(ub)).unwrap();
            child2[i] = F::from_f64(c2.max(lb).min(ub)).unwrap();
        }

        (child1, child2)
    }

    /// Polynomial mutation
    fn mutate(&mut self, individual: &[F]) -> Vec<F> {
        let n = individual.len();
        let mut mutated = individual.to_vec();
        let eta = self.mutation_eta.to_f64().unwrap();

        for i in 0..n {
            if self.rng.random::<f64>() > self.mutation_probability.to_f64().unwrap() {
                continue;
            }

            let y = individual[i].to_f64().unwrap();
            let (lb, ub) = self.bounds[i];
            let lb = lb.to_f64().unwrap();
            let ub = ub.to_f64().unwrap();

            let delta1 = (y - lb) / (ub - lb);
            let delta2 = (ub - y) / (ub - lb);

            let rand = self.rng.random::<f64>();
            let mut_pow = 1.0 / (eta + 1.0);

            let deltaq = if rand < 0.5 {
                let xy = 1.0 - delta1;
                let val = 2.0 * rand + (1.0 - 2.0 * rand) * xy.powf(eta + 1.0);
                val.powf(mut_pow) - 1.0
            } else {
                let xy = 1.0 - delta2;
                let val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy.powf(eta + 1.0);
                1.0 - val.powf(mut_pow)
            };

            let new_val = y + deltaq * (ub - lb);
            mutated[i] = F::from_f64(new_val.max(lb).min(ub)).unwrap();
        }

        mutated
    }

    /// Check if individual a dominates individual b
    fn dominates(a: &[F], b: &[F]) -> bool {
        let mut at_least_one_better = false;

        for (ai, bi) in a.iter().zip(b.iter()) {
            if ai > bi {
                return false;
            }
            if ai < bi {
                at_least_one_better = true;
            }
        }
        at_least_one_better
    }

    /// Fast non-dominated sorting
    fn fast_non_dominated_sort(&self, population: &mut Vec<Individual<Vec<F>, F>>) {
        let n = population.len();
        let mut domination_count = vec![0usize; n];
        let mut dominated_solutions: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut fronts: Vec<Vec<usize>> = Vec::new();
        let mut current_front = Vec::new();

        // Find domination relationships
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }

                if Self::dominates(&population[i].objectives, &population[j].objectives) {
                    dominated_solutions[i].push(j);
                } else if Self::dominates(&population[j].objectives, &population[i].objectives) {
                    domination_count[i] += 1;
                }
            }

            if domination_count[i] == 0 {
                population[i].rank = 0;
                current_front.push(i);
            }
        }

        let mut rank = 0;
        while !current_front.is_empty() {
            fronts.push(current_front.clone());
            let mut next_front = Vec::new();

            for &i in &current_front {
                for &j in &dominated_solutions[i] {
                    domination_count[j] -= 1;
                    if domination_count[j] == 0 {
                        population[j].rank = rank + 1;
                        next_front.push(j);
                    }
                }
            }

            rank += 1;
            current_front = next_front;
        }
    }

    /// Calculate crowding distance for a front
    fn calculate_crowding_distance(
        &self,
        population: &mut [Individual<Vec<F>, F>],
        front: &[usize],
    ) {
        let n = front.len();
        if n == 0 {
            return;
        }

        let num_objectives = population[front[0]].objectives.len();

        for &i in front {
            population[i].crowding_distance = F::zero();
        }

        for m in 0..num_objectives {
            let mut sorted_indices = front.to_vec();
            sorted_indices.sort_by(|&a, &b| {
                population[a].objectives[m]
                    .partial_cmp(&population[b].objectives[m])
                    .unwrap()
            });

            population[sorted_indices[0]].crowding_distance = F::infinity();
            population[sorted_indices[n - 1]].crowding_distance = F::infinity();

            if n <= 2 {
                continue;
            }

            let obj_min = population[sorted_indices[0]].objectives[m];
            let obj_max = population[sorted_indices[n - 1]].objectives[m];
            let range = obj_max - obj_min;

            if range.abs() < F::epsilon() {
                for i in 1..n - 1 {
                    let idx = sorted_indices[i];
                    if !population[idx].crowding_distance.is_infinite() {
                        population[idx].crowding_distance =
                            population[idx].crowding_distance + F::from_f64(1e-10).unwrap();
                    }
                }
                continue;
            }

            for i in 1..n - 1 {
                let idx = sorted_indices[i];
                if population[idx].crowding_distance.is_infinite() {
                    continue;
                }

                let prev_obj = population[sorted_indices[i - 1]].objectives[m];
                let next_obj = population[sorted_indices[i + 1]].objectives[m];

                population[idx].crowding_distance =
                    population[idx].crowding_distance + (next_obj - prev_obj) / range;
            }
        }
    }

    /// Crowded comparison operator
    fn crowded_compare(pop: &[Individual<Vec<F>, F>], i: usize, j: usize) -> bool {
        if pop[i].rank < pop[j].rank {
            true
        } else if pop[i].rank > pop[j].rank {
            false
        } else {
            pop[i].crowding_distance > pop[j].crowding_distance
        }
    }

    /// Binary tournament selection
    fn tournament_selection(&mut self, population: &[Individual<Vec<F>, F>]) -> usize {
        let i = self.rng.random_range(0..population.len());
        let mut j = self.rng.random_range(0..population.len());

        while i == j && population.len() > 1 {
            j = self.rng.random_range(0..population.len());
        }

        if Self::crowded_compare(population, i, j) {
            i
        } else {
            j
        }
    }

    /// Select next generation using elitism
    fn select_next_generation(
        &mut self,
        combined: &mut Vec<Individual<Vec<F>, F>>,
    ) -> Vec<Individual<Vec<F>, F>> {
        self.fast_non_dominated_sort(combined);

        let max_rank = combined.iter().map(|ind| ind.rank).max().unwrap_or(0);
        let mut fronts: Vec<Vec<usize>> = vec![Vec::new(); max_rank + 1];

        for (i, ind) in combined.iter().enumerate() {
            fronts[ind.rank].push(i);
        }

        let mut next_gen = Vec::new();

        for front in &fronts {
            if next_gen.len() + front.len() <= self.population_size {
                for &i in front {
                    next_gen.push(combined[i].clone());
                }

                if next_gen.len() == self.population_size {
                    break;
                }
            } else {
                self.calculate_crowding_distance(combined, front);

                let mut sorted_front = front.clone();
                sorted_front.sort_by(|&a, &b| {
                    combined[b]
                        .crowding_distance
                        .partial_cmp(&combined[a].crowding_distance)
                        .unwrap()
                });

                for &i in &sorted_front {
                    if next_gen.len() >= self.population_size {
                        break;
                    }
                    next_gen.push(combined[i].clone());
                }
                break;
            }
        }

        next_gen
    }
}

impl<O, F, R> Solver<O, PopulationState<Individual<Vec<F>, F>, F>> for NsgaII<F, R>
where
    O: MultiObjectiveCostFunction<Param = Vec<F>, Output = Vec<F>>,
    F: ArgminFloat,
    R: Rng,
{
    fn name(&self) -> &str {
        "NSGA-II"
    }

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        mut state: PopulationState<Individual<Vec<F>, F>, F>,
    ) -> Result<(PopulationState<Individual<Vec<F>, F>, F>, Option<KV>), Error> {
        let population = if let Some(pop) = state.take_population() {
            if pop.len() != self.population_size {
                return Err(Error::msg(format!(
                    "NSGA-II: Provided population size ({}) does not match expected size ({})",
                    pop.len(),
                    self.population_size
                )));
            }

            let problem_ref = problem
                .problem
                .as_ref()
                .ok_or_else(|| Error::msg("NSGA-II: Problem not set"))?;
            let num_objectives = problem_ref.num_objectives();

            for (idx, individual) in pop.iter().enumerate() {
                if individual.objectives.len() != num_objectives {
                    return Err(Error::msg(format!(
                        "NSGA-II: Individual {} has {} objectives, expected {}",
                        idx,
                        individual.objectives.len(),
                        num_objectives
                    )));
                }

                for (obj_idx, &obj) in individual.objectives.iter().enumerate() {
                    if !obj.is_finite() {
                        return Err(Error::msg(format!(
                            "NSGA-II: Individual {} has non-finite objective {} (NaN or infinite)",
                            idx, obj_idx
                        )));
                    }
                }

                if individual.position.len() != self.bounds.len() {
                    return Err(Error::msg(format!(
                        "NSGA-II: Individual {} has {} dimensions, expected {}",
                        idx,
                        individual.position.len(),
                        self.bounds.len()
                    )));
                }

                for (dim, (&val, &(min, max))) in individual
                    .position
                    .iter()
                    .zip(self.bounds.iter())
                    .enumerate()
                {
                    if val < min || val > max {
                        return Err(Error::msg(format!(
                            "NSGA-II: Individual {} dimension {} value {} is outside bounds [{}, {}]",
                            idx, dim, val, min, max
                        )));
                    }
                }
            }

            pop
        } else {
            let problem_ref = problem
                .problem
                .as_ref()
                .ok_or_else(|| Error::msg("NSGA-II: Problem not set"))?;

            let num_objectives = problem_ref.num_objectives();

            let mut pop = Vec::with_capacity(self.population_size);
            for _ in 0..self.population_size {
                let position = self.random_position();
                let objectives =
                    problem.problem("objectives_count", |p| p.objectives(&position))?;

                if objectives.len() != num_objectives {
                    return Err(Error::msg(format!(
                        "NSGA-II: Expected {} objectives, got {}",
                        num_objectives,
                        objectives.len()
                    )));
                }

                for (i, &obj) in objectives.iter().enumerate() {
                    if !obj.is_finite() {
                        return Err(Error::msg(format!(
                            "NSGA-II: Objective {} is not finite (NaN or infinite)",
                            i
                        )));
                    }
                }

                pop.push(Individual::new(position, objectives));
            }
            pop
        };

        Ok((state.population(population), None))
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        mut state: PopulationState<Individual<Vec<F>, F>, F>,
    ) -> Result<(PopulationState<Individual<Vec<F>, F>, F>, Option<KV>), Error> {
        let population = state
            .take_population()
            .ok_or_else(|| Error::msg("NSGA-II: No population in state"))?;

        let mut offspring = Vec::with_capacity(self.population_size);

        while offspring.len() < self.population_size {
            let p1_idx = self.tournament_selection(&population);
            let p2_idx = self.tournament_selection(&population);

            let parent1 = &population[p1_idx].position;
            let parent2 = &population[p2_idx].position;

            let (child1_pos, child2_pos) = self.crossover(parent1, parent2);

            let child1_pos = self.mutate(&child1_pos);
            let child2_pos = self.mutate(&child2_pos);

            let obj1 = problem.problem("objectives_count", |p| p.objectives(&child1_pos))?;
            let obj2 = problem.problem("objectives_count", |p| p.objectives(&child2_pos))?;

            for (i, &obj) in obj1.iter().enumerate() {
                if !obj.is_finite() {
                    return Err(Error::msg(format!(
                        "NSGA-II: Objective {} is not finite (NaN or infinite) in offspring",
                        i
                    )));
                }
            }
            for (i, &obj) in obj2.iter().enumerate() {
                if !obj.is_finite() {
                    return Err(Error::msg(format!(
                        "NSGA-II: Objective {} is not finite (NaN or infinite) in offspring",
                        i
                    )));
                }
            }

            offspring.push(Individual::new(child1_pos, obj1));
            if offspring.len() < self.population_size {
                offspring.push(Individual::new(child2_pos, obj2));
            }
        }

        let mut combined = population;
        combined.append(&mut offspring);

        let next_population = self.select_next_generation(&mut combined);

        Ok((state.population(next_population), None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct TestProblem;

    impl MultiObjectiveCostFunction for TestProblem {
        type Param = Vec<f64>;
        type Output = Vec<f64>;

        fn objectives(&self, param: &Self::Param) -> Result<Self::Output, Error> {
            let f1 = param[0].powi(2) + param[1].powi(2);
            let f2 = (param[0] - 1.0).powi(2) + (param[1] - 1.0).powi(2);
            Ok(vec![f1, f2])
        }

        fn num_objectives(&self) -> usize {
            2
        }
    }

    #[test]
    fn test_objectives() {
        let problem = TestProblem;
        let param = vec![1.0, 1.0];
        let result = problem.objectives(&param).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 2.0);
        assert_eq!(result[1], 0.0);
    }

    #[test]
    fn test_new() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let solver = NsgaII::<f64>::new(bounds, 100);
        assert!(solver.is_ok());
    }

    #[test]
    fn test_new_invalid_bounds() {
        let bounds = vec![(5.0, -5.0)];
        let solver = NsgaII::<f64>::new(bounds, 100);
        let err = solver.unwrap_err();
        assert!(
            err.to_string().contains("NSGA-II: invalid bounds at dimension 0: min (5) must be < max (-5)"),
            "Expected 'NSGA-II: invalid bounds at dimension 0: min (5) must be < max (-5)' error, 
            got: {}",
            err
        );
    }

    #[test]
    fn test_new_odd_population() {
        let bounds = vec![(-5.0, 5.0)];
        let solver = NsgaII::<f64>::new(bounds, 101);
        let err = solver.unwrap_err();
        assert!(
            err.to_string()
                .contains("NSGA-II: population size must be even"),
            "Expected 'NSGA-II: population size must be even' error, got: {}",
            err
        );
    }

    #[test]
    fn test_dominates() {
        // a dominates b (a is better in all objectives)
        let a = vec![1.0, 2.0];
        let b = vec![2.0, 3.0];
        assert!(NsgaII::<f64, StdRng>::dominates(&a, &b));
        assert!(!NsgaII::<f64, StdRng>::dominates(&b, &a));
    }

    #[test]
    fn test_dominates_partial() {
        // Neither dominates (a is better in first, b is better in second)
        let a = vec![1.0, 4.0];
        let b = vec![2.0, 3.0];
        assert!(!NsgaII::<f64, StdRng>::dominates(&a, &b));
        assert!(!NsgaII::<f64, StdRng>::dominates(&b, &a));
    }

    #[test]
    fn test_dominates_equal() {
        // Equal solutions - neither dominates
        let a = vec![2.0, 3.0];
        let b = vec![2.0, 3.0];
        assert!(!NsgaII::<f64, StdRng>::dominates(&a, &b));
        assert!(!NsgaII::<f64, StdRng>::dominates(&b, &a));
    }

    #[test]
    fn test_dominates_one_equal() {
        // a dominates b (equal in first, better in second)
        let a = vec![2.0, 2.0];
        let b = vec![2.0, 3.0];
        assert!(NsgaII::<f64, StdRng>::dominates(&a, &b));
        assert!(!NsgaII::<f64, StdRng>::dominates(&b, &a));
    }

    #[test]
    fn test_individual_new() {
        let ind: Individual<Vec<f64>, f64> = Individual::new(vec![1.0, 2.0], vec![3.0, 4.0]);
        assert_eq!(ind.position, vec![1.0, 2.0]);
        assert_eq!(ind.objectives, vec![3.0, 4.0]);
        assert_eq!(ind.rank, 0);
        assert_eq!(ind.crowding_distance.to_ne_bytes(), 0.0_f64.to_ne_bytes());
    }

    #[test]
    fn test_crossover() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let mut solver = NsgaII::<f64>::new(bounds, 10).unwrap();

        let parent1 = vec![1.0, 2.0];
        let parent2 = vec![3.0, 4.0];

        let (child1, child2) = solver.crossover(&parent1, &parent2);

        assert_eq!(child1.len(), 2);
        assert_eq!(child2.len(), 2);

        for &val in &child1 {
            assert!(val >= -5.0 && val <= 5.0);
        }
        for &val in &child2 {
            assert!(val >= -5.0 && val <= 5.0);
        }
    }

    #[test]
    fn test_mutation() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let mut solver = NsgaII::<f64>::new(bounds, 10).unwrap();

        let individual = vec![1.0, 2.0];
        let mutated = solver.mutate(&individual);

        assert_eq!(mutated.len(), 2);

        for &val in &mutated {
            assert!(val >= -5.0 && val <= 5.0);
        }
    }

    #[test]
    fn test_fast_non_dominated_sort() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let solver = NsgaII::<f64>::new(bounds, 10).unwrap();

        let mut population = vec![
            Individual::new(vec![0.0, 0.0], vec![1.0, 5.0]),
            Individual::new(vec![1.0, 1.0], vec![2.0, 4.0]),
            Individual::new(vec![2.0, 2.0], vec![3.0, 3.0]),
            Individual::new(vec![3.0, 3.0], vec![4.0, 2.0]),
            Individual::new(vec![4.0, 4.0], vec![5.0, 1.0]),
            Individual::new(vec![5.0, 5.0], vec![5.5, 5.5]), // Dominated by all except [6]
            Individual::new(vec![6.0, 6.0], vec![6.0, 6.0]), // Dominated by all
        ];

        solver.fast_non_dominated_sort(&mut population);

        assert_eq!(population[0].rank, 0);
        assert_eq!(population[1].rank, 0);
        assert_eq!(population[2].rank, 0);
        assert_eq!(population[3].rank, 0);
        assert_eq!(population[4].rank, 0);
        assert_eq!(population[5].rank, 1);
        assert_eq!(population[6].rank, 2);
    }

    #[test]
    fn test_crowding_distance() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let solver = NsgaII::<f64>::new(bounds, 10).unwrap();

        let mut population = vec![
            Individual::new(vec![0.0, 0.0], vec![1.0, 5.0]),
            Individual::new(vec![1.0, 1.0], vec![1.5, 4.8]),
            Individual::new(vec![2.0, 2.0], vec![2.0, 4.5]),
            Individual::new(vec![3.0, 3.0], vec![4.5, 1.5]),
            Individual::new(vec![4.0, 4.0], vec![5.0, 1.0]),
        ];

        let front: Vec<usize> = (0..5).collect();
        solver.calculate_crowding_distance(&mut population, &front);

        assert!(population[0].crowding_distance.is_infinite());
        assert_eq!(population[1].crowding_distance, 0.375);
        assert_eq!(population[2].crowding_distance, 1.575);
        assert_eq!(population[3].crowding_distance, 1.625);
        assert!(population[4].crowding_distance.is_infinite());
    }

    #[test]
    fn test_crowding_distance_degenerate_case() {
        let bounds = vec![(-5.0, 5.0)];
        let solver = NsgaII::<f64>::new(bounds, 10).unwrap();

        let mut population = vec![
            Individual::new(vec![0.0], vec![1.0]),
            Individual::new(vec![1.0], vec![1.0]),
            Individual::new(vec![2.0], vec![1.0]),
        ];

        let front: Vec<usize> = (0..3).collect();
        solver.calculate_crowding_distance(&mut population, &front);

        assert!(population[0].crowding_distance.is_infinite());
        assert_eq!(population[1].crowding_distance, 1e-10);
        assert!(population[2].crowding_distance.is_infinite());
    }

    #[test]
    fn test_tournament_selection_different_individuals() {
        let bounds = vec![(-5.0, 5.0)];
        let mut solver = NsgaII::<f64>::new(bounds, 10).unwrap();

        let population = vec![
            Individual::new(vec![0.0], vec![1.0]),
            Individual::new(vec![1.0], vec![2.0]),
            Individual::new(vec![2.0], vec![3.0]),
        ];

        for _ in 0..20 {
            let selected = solver.tournament_selection(&population);
            assert!(selected < population.len());
        }
    }

    #[test]
    fn test_crowded_compare() {
        let pop = vec![
            Individual {
                position: vec![0.0],
                objectives: vec![1.0],
                rank: 0,
                crowding_distance: 5.0,
            },
            Individual {
                position: vec![1.0],
                objectives: vec![2.0],
                rank: 1,
                crowding_distance: 10.0,
            },
            Individual {
                position: vec![2.0],
                objectives: vec![3.0],
                rank: 0,
                crowding_distance: 3.0,
            },
        ];

        assert!(NsgaII::<f64, StdRng>::crowded_compare(&pop, 0, 1));
        assert!(!NsgaII::<f64, StdRng>::crowded_compare(&pop, 1, 0));

        assert!(NsgaII::<f64, StdRng>::crowded_compare(&pop, 0, 2));
        assert!(!NsgaII::<f64, StdRng>::crowded_compare(&pop, 2, 0));
    }

    #[test]
    fn test_invalid_objectives_nan() {
        use crate::core::Executor;

        #[derive(Clone, Debug)]
        struct BadProblem;

        impl MultiObjectiveCostFunction for BadProblem {
            type Param = Vec<f64>;
            type Output = Vec<f64>;

            fn objectives(&self, _param: &Self::Param) -> Result<Self::Output, Error> {
                Ok(vec![f64::NAN, 1.0])
            }

            fn num_objectives(&self) -> usize {
                2
            }
        }

        let problem = BadProblem;
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let solver = NsgaII::new(bounds, 10).unwrap();

        let result = Executor::new(problem, solver)
            .configure(|state| state.max_iters(1))
            .run();
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("NSGA-II: Objective 0 is not finite (NaN or infinite)"),
            "Expected 'NSGA-II: Objective 0 is not finite (NaN or infinite)' error, got: {}",
            err
        );
    }

    #[test]
    fn test_provided_population_validation() {
        use crate::core::{PopulationState, Problem, State};

        let problem = TestProblem;
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let mut solver = NsgaII::new(bounds, 4).unwrap();

        let population = vec![
            Individual::new(vec![10.0, 10.0], vec![1.0, 2.0]),
            Individual::new(vec![1.0, 1.0], vec![1.0, 2.0]),
            Individual::new(vec![2.0, 2.0], vec![2.0, 3.0]),
            Individual::new(vec![3.0, 3.0], vec![3.0, 4.0]),
        ];

        let state: PopulationState<Individual<Vec<f64>, f64>, f64> =
            PopulationState::new().population(population);

        let mut problem_wrapper = Problem::new(problem);
        let result = solver.init(&mut problem_wrapper, state);
        
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("NSGA-II: Individual 0 dimension 0 value 10 is outside bounds [-5, 5]"),
            "Expected 'NSGA-II: Individual 0 dimension 0 value 10 is outside bounds [-5, 5]' error, got: {}",
            err
        );
    }
}
