// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{ArgminFloat, Problem, State, TerminationReason, TerminationStatus};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use web_time::Duration;

/// Maintains the state from iteration to iteration of a population-based solver
///
/// This struct is passed from one iteration of an algorithm to the next.
///
/// Keeps track of
///
/// * individual of current and previous iteration
/// * best individual of current and previous iteration
/// * current and previous best cost function value
/// * target cost function value
/// * population (for population based algorithms)
/// * current iteration number
/// * iteration number where the last best individual was found
/// * maximum number of iterations that will be executed
/// * problem function evaluation counts
/// * elapsed time
/// * termination status
#[derive(Clone, Default, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct PopulationState<P, F> {
    /// Current individual vector
    pub individual: Option<P>,
    /// Previous individual vector
    pub prev_individual: Option<P>,
    /// Current best individual vector
    pub best_individual: Option<P>,
    /// Previous best individual vector
    pub prev_best_individual: Option<P>,
    /// Current cost function value
    pub cost: F,
    /// Previous cost function value
    pub prev_cost: F,
    /// Current best cost function value
    pub best_cost: F,
    /// Previous best cost function value
    pub prev_best_cost: F,
    /// Target cost function value
    pub target_cost: F,
    /// All members of the population
    pub population: Option<Vec<P>>,
    /// Current iteration
    pub iter: u64,
    /// Iteration number of last best cost
    pub last_best_iter: u64,
    /// Maximum number of iterations
    pub max_iters: u64,
    /// Evaluation counts
    pub counts: HashMap<String, u64>,
    /// Update evaluation counts?
    pub counting_enabled: bool,
    /// Time required so far
    pub time: Option<Duration>,
    /// Status of optimization execution
    pub termination_status: TerminationStatus,
}

impl<P, F> PopulationState<P, F>
where
    Self: State<Float = F>,
    F: ArgminFloat,
{
    /// Set best individual of current iteration. This shifts the stored individual to the
    /// previous individual.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{PopulationState, State};
    /// # let state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    /// # let individual_old = vec![1.0f64, 2.0f64];
    /// # let state = state.individual(individual_old);
    /// # assert!(state.prev_individual.is_none());
    /// # assert_eq!(state.individual.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.individual.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # let individual = vec![0.0f64, 3.0f64];
    /// let state = state.individual(individual);
    /// # assert_eq!(state.prev_individual.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_individual.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.individual.as_ref().unwrap()[0].to_ne_bytes(), 0.0f64.to_ne_bytes());
    /// # assert_eq!(state.individual.as_ref().unwrap()[1].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// ```
    #[must_use]
    pub fn individual(mut self, individual: P) -> Self {
        std::mem::swap(&mut self.prev_individual, &mut self.individual);
        self.individual = Some(individual);
        self
    }
    /// Set the current cost function value. This shifts the stored cost function value to the
    /// previous cost function value.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{PopulationState, State};
    /// # let state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    /// # let cost_old = 1.0f64;
    /// # let state = state.cost(cost_old);
    /// # assert_eq!(state.prev_cost.to_ne_bytes(), f64::INFINITY.to_ne_bytes());
    /// # assert_eq!(state.cost.to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # let cost = 0.0f64;
    /// let state = state.cost(cost);
    /// # assert_eq!(state.prev_cost.to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.cost.to_ne_bytes(), 0.0f64.to_ne_bytes());
    /// ```
    #[must_use]
    pub fn cost(mut self, cost: F) -> Self {
        std::mem::swap(&mut self.prev_cost, &mut self.cost);
        self.cost = cost;
        self
    }

    /// Set target cost.
    ///
    /// When this cost is reached, the algorithm will stop. The default is
    /// `Self::Float::NEG_INFINITY`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{PopulationState, State, ArgminFloat};
    /// # let state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    /// # assert_eq!(state.target_cost.to_ne_bytes(), f64::NEG_INFINITY.to_ne_bytes());
    /// let state = state.target_cost(0.0);
    /// # assert_eq!(state.target_cost.to_ne_bytes(), 0.0f64.to_ne_bytes());
    /// ```
    #[must_use]
    pub fn target_cost(mut self, target_cost: F) -> Self {
        self.target_cost = target_cost;
        self
    }

    /// Set population.
    ///
    /// A population is a `Vec` of individuals.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{PopulationState, State};
    /// # let state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    /// # assert!(state.population.is_none());
    /// # let individual1 = vec![0.0f64, 1.0f64];
    /// # let individual2 = vec![2.0f64, 3.0f64];
    /// let state = state.population(vec![individual1, individual2]);
    /// # assert_eq!(state.population.as_ref().unwrap()[0][0].to_ne_bytes(), 0.0f64.to_ne_bytes());
    /// # assert_eq!(state.population.as_ref().unwrap()[0][1].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.population.as_ref().unwrap()[1][0].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.population.as_ref().unwrap()[1][1].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// ```
    #[must_use]
    pub fn population(mut self, population: Vec<P>) -> Self {
        self.population = Some(population);
        self
    }

    /// Set maximum number of iterations
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{PopulationState, State, ArgminFloat};
    /// # let state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    /// # assert_eq!(state.max_iters, u64::MAX);
    /// let state = state.max_iters(1000);
    /// # assert_eq!(state.max_iters, 1000);
    /// ```
    #[must_use]
    pub fn max_iters(mut self, iters: u64) -> Self {
        self.max_iters = iters;
        self
    }

    /// Returns the current cost function value
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{PopulationState, State, ArgminFloat};
    /// # let state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    /// # let state = state.cost(2.0);
    /// let cost = state.get_cost();
    /// # assert_eq!(cost.to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn get_cost(&self) -> F {
        self.cost
    }

    /// Returns the previous cost function value
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{PopulationState, State, ArgminFloat};
    /// # let mut state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    /// # state.prev_cost = 2.0;
    /// let prev_cost = state.get_prev_cost();
    /// # assert_eq!(prev_cost.to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn get_prev_cost(&self) -> F {
        self.prev_cost
    }

    /// Returns the current best cost function value
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{PopulationState, State, ArgminFloat};
    /// # let mut state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    /// # state.best_cost = 2.0;
    /// let best_cost = state.get_best_cost();
    /// # assert_eq!(best_cost.to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn get_best_cost(&self) -> F {
        self.best_cost
    }

    /// Returns the previous best cost function value
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{PopulationState, State, ArgminFloat};
    /// # let mut state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    /// # state.prev_best_cost = 2.0;
    /// let prev_best_cost = state.get_prev_best_cost();
    /// # assert_eq!(prev_best_cost.to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn get_prev_best_cost(&self) -> F {
        self.prev_best_cost
    }

    /// Returns the target cost function value
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{PopulationState, State, ArgminFloat};
    /// # let mut state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    /// # assert_eq!(state.target_cost.to_ne_bytes(), f64::NEG_INFINITY.to_ne_bytes());
    /// # state.target_cost = 0.0;
    /// let target_cost = state.get_target_cost();
    /// # assert_eq!(target_cost.to_ne_bytes(), 0.0f64.to_ne_bytes());
    /// ```
    pub fn get_target_cost(&self) -> F {
        self.target_cost
    }

    /// Moves the current individual out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{PopulationState, State, ArgminFloat};
    /// # let mut state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    /// # assert!(state.take_individual().is_none());
    /// # let mut state = state.individual(vec![1.0, 2.0]);
    /// # assert_eq!(state.individual.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.individual.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let individual = state.take_individual();  // Option<P>
    /// # assert!(state.take_individual().is_none());
    /// # assert!(state.individual.is_none());
    /// # assert_eq!(individual.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(individual.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn take_individual(&mut self) -> Option<P> {
        self.individual.take()
    }

    /// Returns a reference to previous individual
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{PopulationState, State, ArgminFloat};
    /// # let mut state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    /// # assert!(state.prev_individual.is_none());
    /// # state.prev_individual = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.prev_individual.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_individual.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let prev_individual = state.get_prev_individual();  // Option<&P>
    /// # assert_eq!(prev_individual.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(prev_individual.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn get_prev_individual(&self) -> Option<&P> {
        self.prev_individual.as_ref()
    }

    /// Moves the previous individual out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{PopulationState, State, ArgminFloat};
    /// # let mut state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    /// # assert!(state.take_prev_individual().is_none());
    /// # state.prev_individual = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.prev_individual.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_individual.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let prev_individual = state.take_prev_individual();  // Option<P>
    /// # assert!(state.take_prev_individual().is_none());
    /// # assert!(state.prev_individual.is_none());
    /// # assert_eq!(prev_individual.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(prev_individual.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn take_prev_individual(&mut self) -> Option<P> {
        self.prev_individual.take()
    }

    /// Returns a reference to previous best individual
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{PopulationState, State, ArgminFloat};
    /// # let mut state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    /// # assert!(state.prev_best_individual.is_none());
    /// # state.prev_best_individual = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.prev_best_individual.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_best_individual.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let prev_best_individual = state.get_prev_best_individual();  // Option<&P>
    /// # assert_eq!(prev_best_individual.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(prev_best_individual.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn get_prev_best_individual(&self) -> Option<&P> {
        self.prev_best_individual.as_ref()
    }

    /// Moves the best individual out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{PopulationState, State, ArgminFloat};
    /// # let mut state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    /// # assert!(state.take_best_individual().is_none());
    /// # state.best_individual = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.best_individual.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.best_individual.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let best_individual = state.take_best_individual();  // Option<P>
    /// # assert!(state.take_best_individual().is_none());
    /// # assert!(state.best_individual.is_none());
    /// # assert_eq!(best_individual.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(best_individual.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn take_best_individual(&mut self) -> Option<P> {
        self.best_individual.take()
    }

    /// Moves the previous best individual out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{PopulationState, State, ArgminFloat};
    /// # let mut state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    /// # assert!(state.take_prev_best_individual().is_none());
    /// # state.prev_best_individual = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.prev_best_individual.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_best_individual.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let prev_best_individual = state.take_prev_best_individual();  // Option<P>
    /// # assert!(state.take_prev_best_individual().is_none());
    /// # assert!(state.prev_best_individual.is_none());
    /// # assert_eq!(prev_best_individual.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(prev_best_individual.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn take_prev_best_individual(&mut self) -> Option<P> {
        self.prev_best_individual.take()
    }

    /// Returns a reference to the population
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{PopulationState, State, ArgminFloat};
    /// # let state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    /// # assert!(state.population.is_none());
    /// # assert!(state.get_population().is_none());
    /// # let individual1 = vec![0.0f64, 1.0f64];
    /// # let individual2 = vec![2.0f64, 3.0f64];
    /// # let state = state.population(vec![individual1, individual2]);
    /// # assert_eq!(state.population.as_ref().unwrap()[0][0].to_ne_bytes(), 0.0f64.to_ne_bytes());
    /// # assert_eq!(state.population.as_ref().unwrap()[0][1].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.population.as_ref().unwrap()[1][0].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.population.as_ref().unwrap()[1][1].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// let population = state.get_population();
    /// # assert_eq!(population.unwrap()[0][0].to_ne_bytes(), 0.0f64.to_ne_bytes());
    /// # assert_eq!(population.unwrap()[0][1].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(population.unwrap()[1][0].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(population.unwrap()[1][1].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// ```
    pub fn get_population(&self) -> Option<&Vec<P>> {
        self.population.as_ref()
    }

    /// Takes population and replaces it internally with `None`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{PopulationState, State, ArgminFloat};
    /// # let state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    /// # assert!(state.population.is_none());
    /// # assert!(state.get_population().is_none());
    /// # let individual1 = vec![0.0f64, 1.0f64];
    /// # let individual2 = vec![2.0f64, 3.0f64];
    /// # let state = state.population(vec![individual1, individual2]);
    /// # assert_eq!(state.population.as_ref().unwrap()[0][0].to_ne_bytes(), 0.0f64.to_ne_bytes());
    /// # assert_eq!(state.population.as_ref().unwrap()[0][1].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.population.as_ref().unwrap()[1][0].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.population.as_ref().unwrap()[1][1].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// let population = state.get_population();
    /// # assert_eq!(population.unwrap()[0][0].to_ne_bytes(), 0.0f64.to_ne_bytes());
    /// # assert_eq!(population.unwrap()[0][1].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(population.unwrap()[1][0].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(population.unwrap()[1][1].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// ```
    pub fn take_population(&mut self) -> Option<Vec<P>> {
        self.population.take()
    }

    /// Overrides state of counting function executions (default: false)
    /// ```
    /// # use argmin::core::{State, PopulationState};
    /// # let mut state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    /// # assert!(!state.counting_enabled);
    /// let state = state.counting(true);
    /// # assert!(state.counting_enabled);
    /// ```
    #[must_use]
    pub fn counting(mut self, mode: bool) -> Self {
        self.counting_enabled = mode;
        self
    }
}

impl<P, F> State for PopulationState<P, F>
where
    P: Clone,
    F: ArgminFloat,
{
    /// Type of an individual
    type Param = P;
    /// Floating point precision
    type Float = F;

    /// Create a new PopulationState instance
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate web_time;
    /// # use web_time::Duration;
    /// # use argmin::core::{PopulationState, State, ArgminFloat, TerminationStatus};
    /// let state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    /// # assert!(state.individual.is_none());
    /// # assert!(state.prev_individual.is_none());
    /// # assert!(state.best_individual.is_none());
    /// # assert!(state.prev_best_individual.is_none());
    /// # assert_eq!(state.cost.to_ne_bytes(), f64::INFINITY.to_ne_bytes());
    /// # assert_eq!(state.prev_cost.to_ne_bytes(), f64::INFINITY.to_ne_bytes());
    /// # assert_eq!(state.best_cost.to_ne_bytes(), f64::INFINITY.to_ne_bytes());
    /// # assert_eq!(state.prev_best_cost.to_ne_bytes(), f64::INFINITY.to_ne_bytes());
    /// # assert_eq!(state.target_cost.to_ne_bytes(), f64::NEG_INFINITY.to_ne_bytes());
    /// # assert!(state.population.is_none());
    /// # assert_eq!(state.iter, 0);
    /// # assert_eq!(state.last_best_iter, 0);
    /// # assert_eq!(state.max_iters, u64::MAX);
    /// # assert_eq!(state.counts.len(), 0);
    /// # assert_eq!(state.time.unwrap(), Duration::ZERO);
    /// # assert_eq!(state.termination_status, TerminationStatus::NotTerminated);
    /// ```
    fn new() -> Self {
        PopulationState {
            individual: None,
            prev_individual: None,
            best_individual: None,
            prev_best_individual: None,
            cost: F::infinity(),
            prev_cost: F::infinity(),
            best_cost: F::infinity(),
            prev_best_cost: F::infinity(),
            target_cost: F::neg_infinity(),
            population: None,
            iter: 0,
            last_best_iter: 0,
            max_iters: u64::MAX,
            counts: HashMap::new(),
            counting_enabled: false,
            time: Some(Duration::ZERO),
            termination_status: TerminationStatus::NotTerminated,
        }
    }

    /// Checks if the current individual is better than the previous best individual. If
    /// a new best individual was found, the state is updated accordingly.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{PopulationState, State, ArgminFloat};
    /// let mut state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    ///
    /// // Simulating a new, better individual
    /// state.best_individual = Some(vec![1.0f64]);
    /// state.best_cost = 10.0;
    /// state.individual = Some(vec![2.0f64]);
    /// state.cost = 5.0;
    ///
    /// // Calling update
    /// state.update();
    ///
    /// // Check if update was successful
    /// assert_eq!(state.best_individual.as_ref().unwrap()[0], 2.0f64);
    /// assert_eq!(state.best_cost.to_ne_bytes(), state.best_cost.to_ne_bytes());
    /// assert!(state.is_best());
    /// ```
    ///
    /// For algorithms which do not compute the cost function, every new individual will be
    /// the new best:
    ///
    /// ```
    /// # use argmin::core::{PopulationState, State, ArgminFloat};
    /// let mut state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    ///
    /// // Simulating a new, better individual
    /// state.best_individual = Some(vec![1.0f64]);
    /// state.individual = Some(vec![2.0f64]);
    ///
    /// // Calling update
    /// state.update();
    ///
    /// // Check if update was successful
    /// assert_eq!(state.best_individual.as_ref().unwrap()[0], 2.0f64);
    /// assert_eq!(state.best_cost.to_ne_bytes(), state.best_cost.to_ne_bytes());
    /// assert!(state.is_best());
    /// ```
    fn update(&mut self) {
        // check if individual is the best so far
        // Comparison is done using `<` to avoid new solutions with the same cost function value as
        // the current best to be accepted. However, some solvers to not compute the cost function
        // value. Those will always have `Inf` cost. Therefore if both the new value and the
        // previous best value are `Inf`, the solution is also accepted. Care is taken that both
        // `Inf` also have the same sign.
        if self.cost < self.best_cost
            || (self.cost.is_infinite()
                && self.best_cost.is_infinite()
                && self.cost.is_sign_positive() == self.best_cost.is_sign_positive())
        {
            // If there is no individual, then also don't set the best individual.
            if let Some(individual) = self.individual.as_ref().cloned() {
                std::mem::swap(&mut self.prev_best_individual, &mut self.best_individual);
                self.best_individual = Some(individual);
            }
            std::mem::swap(&mut self.prev_best_cost, &mut self.best_cost);
            self.best_cost = self.cost;
            self.last_best_iter = self.iter;
        }
    }

    /// Returns a reference to the current individual
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{PopulationState, State, ArgminFloat};
    /// # let mut state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    /// # assert!(state.individual.is_none());
    /// # state.individual = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.individual.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.individual.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let individual = state.get_param();  // Option<&P>
    /// # assert_eq!(individual.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(individual.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    fn get_param(&self) -> Option<&P> {
        self.individual.as_ref()
    }

    /// Returns a reference to the current best individual
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{PopulationState, State, ArgminFloat};
    /// # let mut state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    /// # assert!(state.best_individual.is_none());
    /// # state.best_individual = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.best_individual.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.best_individual.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let best_individual = state.get_best_param();  // Option<&P>
    /// # assert_eq!(best_individual.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(best_individual.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    fn get_best_param(&self) -> Option<&P> {
        self.best_individual.as_ref()
    }

    /// Sets the termination status to [`Terminated`](`TerminationStatus::Terminated`) with the given reason
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{PopulationState, State, ArgminFloat, TerminationReason, TerminationStatus};
    /// # let mut state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    /// # assert_eq!(state.termination_status, TerminationStatus::NotTerminated);
    /// let state = state.terminate_with(TerminationReason::MaxItersReached);
    /// # assert_eq!(state.termination_status, TerminationStatus::Terminated(TerminationReason::MaxItersReached));
    /// ```
    fn terminate_with(mut self, reason: TerminationReason) -> Self {
        self.termination_status = TerminationStatus::Terminated(reason);
        self
    }

    /// Sets the time required so far.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate web_time;
    /// # use web_time::Duration;
    /// # use argmin::core::{PopulationState, State, ArgminFloat, TerminationReason};
    /// # let mut state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    /// let state = state.time(Some(Duration::from_nanos(12)));
    /// # assert_eq!(state.time.unwrap(), Duration::from_nanos(12));
    /// ```
    fn time(&mut self, time: Option<Duration>) -> &mut Self {
        self.time = time;
        self
    }

    /// Returns current cost function value.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{PopulationState, State, ArgminFloat};
    /// # let mut state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    /// # state.cost = 12.0;
    /// let cost = state.get_cost();
    /// # assert_eq!(cost.to_ne_bytes(), 12.0f64.to_ne_bytes());
    /// ```
    fn get_cost(&self) -> Self::Float {
        self.cost
    }

    /// Returns current best cost function value.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{PopulationState, State, ArgminFloat};
    /// # let mut state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    /// # state.best_cost = 12.0;
    /// let best_cost = state.get_best_cost();
    /// # assert_eq!(best_cost.to_ne_bytes(), 12.0f64.to_ne_bytes());
    /// ```
    fn get_best_cost(&self) -> Self::Float {
        self.best_cost
    }

    /// Returns target cost function value.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{PopulationState, State, ArgminFloat};
    /// # let mut state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    /// # state.target_cost = 12.0;
    /// let target_cost = state.get_target_cost();
    /// # assert_eq!(target_cost.to_ne_bytes(), 12.0f64.to_ne_bytes());
    /// ```
    fn get_target_cost(&self) -> Self::Float {
        self.target_cost
    }

    /// Returns current number of iterations.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{PopulationState, State, ArgminFloat};
    /// # let mut state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    /// # state.iter = 12;
    /// let iter = state.get_iter();
    /// # assert_eq!(iter, 12);
    /// ```
    fn get_iter(&self) -> u64 {
        self.iter
    }

    /// Returns iteration number of last best individual
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{PopulationState, State, ArgminFloat};
    /// # let mut state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    /// # state.last_best_iter = 12;
    /// let last_best_iter = state.get_last_best_iter();
    /// # assert_eq!(last_best_iter, 12);
    /// ```
    fn get_last_best_iter(&self) -> u64 {
        self.last_best_iter
    }

    /// Returns the maximum number of iterations.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{PopulationState, State, ArgminFloat};
    /// # let mut state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    /// # state.max_iters = 12;
    /// let max_iters = state.get_max_iters();
    /// # assert_eq!(max_iters, 12);
    /// ```
    fn get_max_iters(&self) -> u64 {
        self.max_iters
    }

    /// Returns the termination reason.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{PopulationState, State, ArgminFloat, TerminationStatus};
    /// # let mut state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    /// let termination_status = state.get_termination_status();
    /// # assert_eq!(*termination_status, TerminationStatus::NotTerminated);
    /// ```
    fn get_termination_status(&self) -> &TerminationStatus {
        &self.termination_status
    }

    /// Returns the termination reason if terminated, otherwise None.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{PopulationState, State, ArgminFloat, TerminationReason};
    /// # let mut state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    /// let termination_reason = state.get_termination_reason();
    /// # assert_eq!(termination_reason, None);
    /// ```
    fn get_termination_reason(&self) -> Option<&TerminationReason> {
        match &self.termination_status {
            TerminationStatus::Terminated(reason) => Some(reason),
            TerminationStatus::NotTerminated => None,
        }
    }

    /// Returns the time elapsed since the start of the optimization.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate web_time;
    /// # use web_time::Duration;
    /// # use argmin::core::{PopulationState, State, ArgminFloat};
    /// # let mut state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    /// let time = state.get_time();
    /// # assert_eq!(time.unwrap(), Duration::ZERO);
    /// ```
    fn get_time(&self) -> Option<Duration> {
        self.time
    }

    /// Increments the number of iterations by one
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{PopulationState, State, ArgminFloat};
    /// # let mut state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    /// # assert_eq!(state.iter, 0);
    /// state.increment_iter();
    /// # assert_eq!(state.iter, 1);
    /// ```
    fn increment_iter(&mut self) {
        self.iter += 1;
    }

    /// Set all function evaluation counts to the evaluation counts of another `Problem`.
    ///
    /// ```
    /// # use std::collections::HashMap;
    /// # use argmin::core::{Problem, PopulationState, State, ArgminFloat};
    /// # let mut state: PopulationState<Vec<f64>, f64> = PopulationState::new().counting(true);
    /// # assert_eq!(state.counts, HashMap::new());
    /// # state.counts.insert("test2".to_string(), 10u64);
    /// #
    /// # #[derive(Eq, PartialEq, Debug)]
    /// # struct UserDefinedProblem {};
    /// #
    /// # let mut problem = Problem::new(UserDefinedProblem {});
    /// # problem.counts.insert("test1", 10u64);
    /// # problem.counts.insert("test2", 2);
    /// state.func_counts(&problem);
    /// # let mut hm = HashMap::new();
    /// # hm.insert("test1".to_string(), 10u64);
    /// # hm.insert("test2".to_string(), 2u64);
    /// # assert_eq!(state.counts, hm);
    /// ```
    fn func_counts<O>(&mut self, problem: &Problem<O>) {
        if self.counting_enabled {
            for (k, &v) in problem.counts.iter() {
                let count = self.counts.entry(k.to_string()).or_insert(0);
                *count = v
            }
        }
    }

    /// Returns function evaluation counts
    ///
    /// # Example
    ///
    /// ```
    /// # use std::collections::HashMap;
    /// # use argmin::core::{PopulationState, State, ArgminFloat};
    /// # let mut state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    /// # assert_eq!(state.counts, HashMap::new());
    /// # state.counts.insert("test2".to_string(), 10u64);
    /// let counts = state.get_func_counts();
    /// # let mut hm = HashMap::new();
    /// # hm.insert("test2".to_string(), 10u64);
    /// # assert_eq!(*counts, hm);
    /// ```
    fn get_func_counts(&self) -> &HashMap<String, u64> {
        &self.counts
    }

    /// Returns whether the current individual is also the best individual found so
    /// far.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{PopulationState, State, ArgminFloat};
    /// # let mut state: PopulationState<Vec<f64>, f64> = PopulationState::new();
    /// # state.last_best_iter = 12;
    /// # state.iter = 12;
    /// let is_best = state.is_best();
    /// # assert!(is_best);
    /// # state.last_best_iter = 12;
    /// # state.iter = 21;
    /// # let is_best = state.is_best();
    /// # assert!(!is_best);
    /// ```
    fn is_best(&self) -> bool {
        self.last_best_iter == self.iter
    }
}

// TODO: Tests? Actually doc tests should already cover everything.
