// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Particle Swarm Optimization (PSO)
//!
//! Canonical implementation of the particle swarm optimization method as outlined in \[0\] in
//! chapter II, section A.
//!
//! For details see [`ParticleSwarm`].
//!
//! ## References
//!
//! \[0\] Zambrano-Bigiarini, M. et.al. (2013): Standard Particle Swarm Optimisation 2011 at
//! CEC-2013: A baseline for future PSO improvements. 2013 IEEE Congress on Evolutionary
//! Computation. <https://doi.org/10.1109/CEC.2013.6557848>
//!
//! \[1\] <https://en.wikipedia.org/wiki/Particle_swarm_optimization>

use std::fmt::Debug;

use crate::core::{
    ArgminFloat, CostFunction, Error, PopulationState, Problem, SerializeAlias, Solver, SyncAlias,
    KV,
};
use argmin_math::{ArgminAdd, ArgminMinMax, ArgminMul, ArgminRandom, ArgminSub, ArgminZeroLike};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// # Particle Swarm Optimization (PSO)
///
/// Canonical implementation of the particle swarm optimization method as outlined in \[0\] in
/// chapter II, section A.
///
/// The `rayon` feature enables parallel computation of the cost function. This can be beneficial
/// for expensive cost functions, but may cause a drop in performance for cheap cost functions. Be
/// sure to benchmark both parallel and sequential computation.
///
/// ## Requirements on the optimization problem
///
/// The optimization problem is required to implement [`CostFunction`].
///
/// ## References
///
/// \[0\] Zambrano-Bigiarini, M. et.al. (2013): Standard Particle Swarm Optimisation 2011 at
/// CEC-2013: A baseline for future PSO improvements. 2013 IEEE Congress on Evolutionary
/// Computation. <https://doi.org/10.1109/CEC.2013.6557848>
///
/// \[1\] <https://en.wikipedia.org/wiki/Particle_swarm_optimization>
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct ParticleSwarm<P, F> {
    /// Inertia weight
    weight_inertia: F,
    /// Cognitive acceleration coefficient
    weight_cognitive: F,
    /// Social acceleration coefficient
    weight_social: F,
    /// Bounds on parameter space
    bounds: (P, P),
    /// Number of particles
    num_particles: usize,
}

impl<P, F> ParticleSwarm<P, F>
where
    P: Clone + SyncAlias + ArgminSub<P, P> + ArgminMul<F, P> + ArgminRandom + ArgminZeroLike,
    F: ArgminFloat,
{
    /// Construct a new instance of `ParticleSwarm`
    ///
    /// Takes the number of particles and bounds on the search space as inputs. `bounds` is a tuple
    /// `(lower_bound, upper_bound)`, where `lower_bound` and `upper_bound` are of the same type as
    /// the position of a particle (`P`) and of the same length as the problem as dimensions.
    ///
    /// The inertia weight on velocity and the social and cognitive acceleration factors can be
    /// adapted with [`with_inertia_factor`](`ParticleSwarm::with_inertia_factor`),
    /// [`with_cognitive_factor`](`ParticleSwarm::with_cognitive_factor`) and
    /// [`with_social_factor`](`ParticleSwarm::with_social_factor`), respectively.
    ///
    /// The weights and acceleration factors default to:
    ///
    /// * inertia: `1/(2 * ln(2))`
    /// * cognitive: `0.5 + ln(2)`
    /// * social: `0.5 + ln(2)`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::particleswarm::ParticleSwarm;
    /// # let lower_bound: Vec<f64> = vec![-1.0, -1.0];
    /// # let upper_bound: Vec<f64> = vec![1.0, 1.0];
    /// let pso: ParticleSwarm<_, f64> = ParticleSwarm::new((lower_bound, upper_bound), 40);
    /// ```
    pub fn new(bounds: (P, P), num_particles: usize) -> Self {
        ParticleSwarm {
            weight_inertia: float!(1.0f64 / (2.0 * 2.0f64.ln())),
            weight_cognitive: float!(0.5 + 2.0f64.ln()),
            weight_social: float!(0.5 + 2.0f64.ln()),
            bounds,
            num_particles,
        }
    }

    /// Set inertia factor on particle velocity
    ///
    /// Defaults to `1/(2 * ln(2))`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::particleswarm::ParticleSwarm;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// # let lower_bound: Vec<f64> = vec![-1.0, -1.0];
    /// # let upper_bound: Vec<f64> = vec![1.0, 1.0];
    /// let pso: ParticleSwarm<_, f64> =
    ///     ParticleSwarm::new((lower_bound, upper_bound), 40).with_inertia_factor(0.5)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_inertia_factor(mut self, factor: F) -> Result<Self, Error> {
        if factor < float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`ParticleSwarm`: inertia factor must be >=0."
            ));
        }
        self.weight_inertia = factor;
        Ok(self)
    }

    /// Set cognitive acceleration factor
    ///
    /// Defaults to `0.5 + ln(2)`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::particleswarm::ParticleSwarm;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// # let lower_bound: Vec<f64> = vec![-1.0, -1.0];
    /// # let upper_bound: Vec<f64> = vec![1.0, 1.0];
    /// let pso: ParticleSwarm<_, f64> =
    ///     ParticleSwarm::new((lower_bound, upper_bound), 40).with_cognitive_factor(1.1)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_cognitive_factor(mut self, factor: F) -> Result<Self, Error> {
        if factor < float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`ParticleSwarm`: cognitive factor must be >=0."
            ));
        }
        self.weight_cognitive = factor;
        Ok(self)
    }

    /// Set social acceleration factor
    ///
    /// Defaults to `0.5 + ln(2)`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::particleswarm::ParticleSwarm;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// # let lower_bound: Vec<f64> = vec![-1.0, -1.0];
    /// # let upper_bound: Vec<f64> = vec![1.0, 1.0];
    /// let pso: ParticleSwarm<_, f64> =
    ///     ParticleSwarm::new((lower_bound, upper_bound), 40).with_social_factor(1.1)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_social_factor(mut self, factor: F) -> Result<Self, Error> {
        if factor < float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`ParticleSwarm`: social factor must be >=0."
            ));
        }
        self.weight_social = factor;
        Ok(self)
    }

    /// Initializes all particles randomly and sorts them by their cost function values
    fn initialize_particles<O: CostFunction<Param = P, Output = F> + SyncAlias>(
        &mut self,
        problem: &mut Problem<O>,
    ) -> Result<Vec<Particle<P, F>>, Error> {
        let (positions, velocities) = self.initialize_positions_and_velocities();

        let costs = problem.bulk_cost(&positions)?;

        let mut particles = positions
            .into_iter()
            .zip(velocities.into_iter())
            .zip(costs.into_iter())
            .map(|((p, v), c)| Particle::new(p, c, v))
            .collect::<Vec<_>>();

        // sort them, such that the first one is the best one
        particles.sort_by(|a, b| {
            a.cost
                .partial_cmp(&b.cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(particles)
    }

    /// Initializes positions and velocities for all particles
    fn initialize_positions_and_velocities(&self) -> (Vec<P>, Vec<P>) {
        let (min, max) = &self.bounds;
        let delta = max.sub(min);
        let delta_neg = delta.mul(&float!(-1.0));

        (
            (0..self.num_particles)
                .map(|_| P::rand_from_range(min, max))
                .collect(),
            (0..self.num_particles)
                .map(|_| P::rand_from_range(&delta_neg, &delta))
                .collect(),
        )
    }
}

impl<O, P, F> Solver<O, PopulationState<Particle<P, F>, F>> for ParticleSwarm<P, F>
where
    O: CostFunction<Param = P, Output = F> + SyncAlias,
    P: SerializeAlias
        + Clone
        + Debug
        + SyncAlias
        + ArgminAdd<P, P>
        + ArgminSub<P, P>
        + ArgminMul<F, P>
        + ArgminZeroLike
        + ArgminRandom
        + ArgminMinMax,
    F: ArgminFloat,
{
    const NAME: &'static str = "Particle Swarm Optimization";

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        mut state: PopulationState<Particle<P, F>, F>,
    ) -> Result<(PopulationState<Particle<P, F>, F>, Option<KV>), Error> {
        // Users can provide a population or it will be randomly created.
        let particles = match state.take_population() {
            Some(mut particles) if particles.len() == self.num_particles => {
                // sort them first
                particles.sort_by(|a, b| {
                    a.cost
                        .partial_cmp(&b.cost)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                particles
            }
            Some(particles) => {
                return Err(argmin_error!(
                    InvalidParameter,
                    format!(
                        "`ParticleSwarm`: Provided list of particles is of length {}, expected {}",
                        particles.len(),
                        self.num_particles
                    )
                ))
            }
            None => self.initialize_particles(problem)?,
        };

        Ok((
            state
                .individual(particles[0].clone())
                .cost(particles[0].cost)
                .population(particles),
            None,
        ))
    }

    /// Perform one iteration of algorithm
    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        mut state: PopulationState<Particle<P, F>, F>,
    ) -> Result<(PopulationState<Particle<P, F>, F>, Option<KV>), Error> {
        let mut best_particle = state.take_individual().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`ParticleSwarm`: No current best individual in state."
        ))?;
        let mut best_cost = state.get_cost();
        let mut particles = state.take_population().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`ParticleSwarm`: No population in state."
        ))?;

        let zero = P::zero_like(&best_particle.position);

        let positions: Vec<_> = particles
            .iter_mut()
            .map(|p| {
                // New velocity is composed of
                // 1) previous velocity (momentum),
                // 2) motion toward particle optimum and
                // 3) motion toward global optimum.

                // ad 1)
                let momentum = p.velocity.mul(&self.weight_inertia);

                // ad 2)
                let to_optimum = p.best_position.sub(&p.position);
                let pull_to_optimum = P::rand_from_range(&zero, &to_optimum);
                let pull_to_optimum = pull_to_optimum.mul(&self.weight_cognitive);

                // ad 3)
                let to_global_optimum = best_particle.position.sub(&p.position);
                let pull_to_global_optimum =
                    P::rand_from_range(&zero, &to_global_optimum).mul(&self.weight_social);

                p.velocity = momentum.add(&pull_to_optimum).add(&pull_to_global_optimum);
                let new_position = p.position.add(&p.velocity);

                // Limit to search window
                p.position = P::min(&P::max(&new_position, &self.bounds.0), &self.bounds.1);
                &p.position
            })
            .collect();

        let costs = problem.bulk_cost(&positions)?;

        for (p, c) in particles.iter_mut().zip(costs.into_iter()) {
            p.cost = c;

            if p.cost < p.best_cost {
                p.best_position = p.position.clone();
                p.best_cost = p.cost;

                if p.cost < best_cost {
                    best_particle.position = p.position.clone();
                    best_particle.best_position = p.position.clone();
                    best_particle.cost = p.cost;
                    best_particle.best_cost = p.cost;
                    best_cost = p.cost;
                }
            }
        }

        Ok((
            state
                .individual(best_particle)
                .cost(best_cost)
                .population(particles),
            None,
        ))
    }
}

/// A single particle
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct Particle<T, F> {
    /// Position of particle
    pub position: T,
    /// Velocity of particle
    velocity: T,
    /// Cost of particle
    pub cost: F,
    /// Best position of particle so far
    best_position: T,
    /// Best cost of particle so far
    best_cost: F,
}

impl<T, F> Particle<T, F>
where
    T: Clone,
    F: ArgminFloat,
{
    /// Create a new particle with a given position, cost and velocity.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::particleswarm::Particle;
    /// let particle: Particle<Vec<f64>, f64> = Particle::new(vec![0.0, 1.4], 12.0, vec![0.1, 0.5]);
    /// ```
    pub fn new(position: T, cost: F, velocity: T) -> Particle<T, F> {
        Particle {
            position: position.clone(),
            velocity,
            cost,
            best_position: position,
            best_cost: cost,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{test_utils::TestProblem, ArgminError, State};
    use crate::test_trait_impl;
    use approx::assert_relative_eq;

    test_trait_impl!(particleswarm, ParticleSwarm<Vec<f64>, f64>);

    #[test]
    fn test_new() {
        let lower_bound: Vec<f64> = vec![-1.0, -1.0];
        let upper_bound: Vec<f64> = vec![1.0, 1.0];
        let pso: ParticleSwarm<_, f64> =
            ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 40);
        let ParticleSwarm {
            weight_inertia,
            weight_cognitive,
            weight_social,
            bounds,
            num_particles,
        } = pso;

        assert_relative_eq!(
            weight_inertia,
            (1.0f64 / (2.0 * 2.0f64.ln())),
            epsilon = f64::EPSILON
        );
        assert_relative_eq!(
            weight_cognitive,
            (0.5f64 + 2.0f64.ln()),
            epsilon = f64::EPSILON
        );
        assert_relative_eq!(
            weight_social,
            (0.5f64 + 2.0f64.ln()),
            epsilon = f64::EPSILON
        );
        assert_eq!(lower_bound[0].to_ne_bytes(), bounds.0[0].to_ne_bytes());
        assert_eq!(lower_bound[1].to_ne_bytes(), bounds.0[1].to_ne_bytes());
        assert_eq!(upper_bound[0].to_ne_bytes(), bounds.1[0].to_ne_bytes());
        assert_eq!(upper_bound[1].to_ne_bytes(), bounds.1[1].to_ne_bytes());
        assert_eq!(num_particles, 40);
    }

    #[test]
    fn test_with_inertia_factor() {
        let lower_bound: Vec<f64> = vec![-1.0, -1.0];
        let upper_bound: Vec<f64> = vec![1.0, 1.0];

        for inertia in [0.0, f64::EPSILON, 0.5, 1.0, 1.2, 3.0] {
            let res = ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 40)
                .with_inertia_factor(inertia);
            assert!(res.is_ok());
            assert_eq!(
                res.unwrap().weight_inertia.to_ne_bytes(),
                inertia.to_ne_bytes()
            );
        }

        for inertia in [-f64::EPSILON, -0.5, -1.0, -1.2, -3.0] {
            let res = ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 40)
                .with_inertia_factor(inertia);
            assert_error!(
                res,
                ArgminError,
                concat!(
                    "Invalid parameter: \"`ParticleSwarm`: ",
                    "inertia factor must be >=0.\""
                )
            );
        }
    }

    #[test]
    fn test_with_cognitive_factor() {
        let lower_bound: Vec<f64> = vec![-1.0, -1.0];
        let upper_bound: Vec<f64> = vec![1.0, 1.0];

        for cognitive in [0.0, f64::EPSILON, 0.5, 1.0, 1.2, 3.0] {
            let res = ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 40)
                .with_cognitive_factor(cognitive);
            assert!(res.is_ok());
            assert_eq!(
                res.unwrap().weight_cognitive.to_ne_bytes(),
                cognitive.to_ne_bytes()
            );
        }

        for cognitive in [-f64::EPSILON, -0.5, -1.0, -1.2, -3.0] {
            let res = ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 40)
                .with_cognitive_factor(cognitive);
            assert_error!(
                res,
                ArgminError,
                concat!(
                    "Invalid parameter: \"`ParticleSwarm`: ",
                    "cognitive factor must be >=0.\""
                )
            );
        }
    }

    #[test]
    fn test_with_social_factor() {
        let lower_bound: Vec<f64> = vec![-1.0, -1.0];
        let upper_bound: Vec<f64> = vec![1.0, 1.0];

        for social in [0.0, f64::EPSILON, 0.5, 1.0, 1.2, 3.0] {
            let res = ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 40)
                .with_social_factor(social);
            assert!(res.is_ok());
            assert_eq!(
                res.unwrap().weight_social.to_ne_bytes(),
                social.to_ne_bytes()
            );
        }

        for social in [-f64::EPSILON, -0.5, -1.0, -1.2, -3.0] {
            let res = ParticleSwarm::new((lower_bound.clone(), upper_bound.clone()), 40)
                .with_social_factor(social);
            assert_error!(
                res,
                ArgminError,
                concat!(
                    "Invalid parameter: \"`ParticleSwarm`: ",
                    "social factor must be >=0.\""
                )
            );
        }
    }

    #[test]
    fn test_initialize_positions_and_velocities() {
        let lower_bound: Vec<f64> = vec![-1.0, -1.0];
        let upper_bound: Vec<f64> = vec![1.0, 1.0];
        let num_particles = 100;
        let pso: ParticleSwarm<_, f64> =
            ParticleSwarm::new((lower_bound, upper_bound), num_particles);

        let (positions, velocities) = pso.initialize_positions_and_velocities();
        assert_eq!(positions.len(), num_particles);
        assert_eq!(velocities.len(), num_particles);

        for pos in positions {
            for elem in pos {
                assert!(elem <= 1.0f64);
                assert!(elem >= -1.0f64);
            }
        }

        for velo in velocities {
            for elem in velo {
                assert!(elem <= 2.0f64);
                assert!(elem >= -2.0f64);
            }
        }
    }

    #[test]
    fn test_initialize_particles() {
        let lower_bound: Vec<f64> = vec![-1.0, -1.0];
        let upper_bound: Vec<f64> = vec![1.0, 1.0];
        let num_particles = 10;
        let mut pso: ParticleSwarm<_, f64> =
            ParticleSwarm::new((lower_bound, upper_bound), num_particles);

        struct PsoProblem {
            counter: std::sync::Arc<std::sync::Mutex<usize>>,
            values: [f64; 10],
        }

        impl CostFunction for PsoProblem {
            type Param = Vec<f64>;
            type Output = f64;

            fn cost(&self, _param: &Self::Param) -> Result<Self::Output, Error> {
                let mut counter = self.counter.lock().unwrap();
                let cost = self.values[*counter];
                *counter += 1;
                Ok(cost)
            }
        }

        let mut values = [1.0, 4.0, 10.0, 2.0, -3.0, 8.0, 4.4, 8.1, 6.4, 4.5];

        let mut problem = Problem::new(PsoProblem {
            counter: std::sync::Arc::new(std::sync::Mutex::new(0)),
            values,
        });

        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let particles = pso.initialize_particles(&mut problem).unwrap();
        assert_eq!(particles.len(), num_particles);

        // at least assure that they are ordered correctly and have the correct cost.
        for (particle, cost) in particles.iter().zip(values.iter()) {
            assert_eq!(particle.cost.to_ne_bytes(), cost.to_ne_bytes());
        }
    }

    #[test]
    fn test_particle_new() {
        let init_position = vec![0.2, 3.0];
        let init_cost = 12.0;
        let init_velocity = vec![1.2, -1.3];

        let particle: Particle<Vec<f64>, f64> =
            Particle::new(init_position.clone(), init_cost, init_velocity.clone());
        let Particle {
            position,
            velocity,
            cost,
            best_position,
            best_cost,
        } = particle;

        assert_eq!(init_position, position);
        assert_eq!(init_position, best_position);
        assert_eq!(init_cost.to_ne_bytes(), cost.to_ne_bytes());
        assert_eq!(init_cost.to_ne_bytes(), best_cost.to_ne_bytes());
        assert_eq!(init_velocity, velocity);
    }

    #[test]
    fn test_init_provided_population_wrong_size() {
        let lower_bound: Vec<f64> = vec![-1.0, -1.0];
        let upper_bound: Vec<f64> = vec![1.0, 1.0];
        let mut pso: ParticleSwarm<_, f64> = ParticleSwarm::new((lower_bound, upper_bound), 40);
        let state: PopulationState<Particle<Vec<f64>, f64>, f64> = PopulationState::new()
            .population(vec![Particle::new(vec![1.0, 2.0], 12.0, vec![0.1, 0.3])]);
        let res = pso.init(&mut Problem::new(TestProblem::new()), state);
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Invalid parameter: \"`ParticleSwarm`: ",
                "Provided list of particles is of length 1, expected 40\"",
            )
        );
    }

    #[test]
    fn test_init_provided_population_correct_size() {
        let lower_bound: Vec<f64> = vec![-1.0, -1.0];
        let upper_bound: Vec<f64> = vec![1.0, 1.0];
        let particle_a = Particle::new(vec![1.0, 2.0], 12.0, vec![0.1, 0.3]);
        let particle_b = Particle::new(vec![2.0, 3.0], 10.0, vec![0.2, 0.4]);
        let mut pso: ParticleSwarm<_, f64> = ParticleSwarm::new((lower_bound, upper_bound), 2);
        let state: PopulationState<Particle<Vec<f64>, f64>, f64> =
            PopulationState::new().population(vec![particle_a.clone(), particle_b.clone()]);
        let res = pso.init(&mut Problem::new(TestProblem::new()), state);
        assert!(res.is_ok());
        let (mut state, kv) = res.unwrap();
        assert!(kv.is_none());
        assert_eq!(*state.get_param().unwrap(), particle_b);
        let population = state.take_population().unwrap();
        // assert that it was sorted!
        assert_eq!(population[0], particle_b);
        assert_eq!(population[1], particle_a);
    }

    #[test]
    fn test_init_random_population() {
        let lower_bound: Vec<f64> = vec![-1.0, -1.0];
        let upper_bound: Vec<f64> = vec![1.0, 1.0];
        let mut pso: ParticleSwarm<_, f64> = ParticleSwarm::new((lower_bound, upper_bound), 40);
        let state: PopulationState<Particle<Vec<f64>, f64>, f64> = PopulationState::new();
        let res = pso.init(&mut Problem::new(TestProblem::new()), state);
        assert!(res.is_ok());
        let (mut state, kv) = res.unwrap();
        assert!(kv.is_none());
        assert!(state.get_param().is_some());
        let population = state.take_population().unwrap();
        assert_eq!(population.len(), 40);
    }

    #[test]
    fn test_next_iter() {
        struct PsoProblem {
            counter: std::sync::Mutex<usize>,
            values: [f64; 10],
        }

        impl CostFunction for PsoProblem {
            type Param = Vec<f64>;
            type Output = f64;

            fn cost(&self, _param: &Self::Param) -> Result<Self::Output, Error> {
                let cost = self.values[*self.counter.lock().unwrap() % 10];
                *self.counter.lock().unwrap() += 1;
                Ok(cost)
            }
        }

        let values = [1.0, 4.0, 10.0, 2.0, -3.0, 8.0, 4.4, 8.1, 6.4, 4.4];

        let mut problem = Problem::new(PsoProblem {
            counter: std::sync::Mutex::new(0),
            values,
        });

        // setup
        let lower_bound: Vec<f64> = vec![-1.0, -1.0];
        let upper_bound: Vec<f64> = vec![1.0, 1.0];
        let mut pso: ParticleSwarm<_, f64> = ParticleSwarm::new((lower_bound, upper_bound), 100);
        let state: PopulationState<Particle<Vec<f64>, f64>, f64> = PopulationState::new();

        // init
        let (mut state, _) = pso.init(&mut problem, state).unwrap();

        // next_iter
        for _ in 0..200 {
            (state, _) = pso.next_iter(&mut problem, state).unwrap();
            let population = state.get_population().unwrap();
            assert_eq!(population.len(), 100);
            for particle in population {
                for x in particle.position.iter() {
                    assert!(*x <= 1.0);
                    assert!(*x >= -1.0);
                }
            }
            assert_eq!(state.get_cost().to_ne_bytes(), (-3.0f64).to_ne_bytes());
        }
    }
}
