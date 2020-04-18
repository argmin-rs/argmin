// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # References:
//!
//! TODO

use crate::prelude::*;
use serde::Serialize;
use std;
use std::default::Default;

/// Particle Swarm Optimization (PSO)
///
/// [Example](https://github.com/argmin-rs/argmin/blob/master/examples/particleswarm.rs)
///
/// # References:
///
/// TODO
#[derive(Serialize)]
pub struct ParticleSwarm<O, F>
where
    O: ArgminOp<Output = F>,
    <O as ArgminOp>::Param: Position<F>,
    F: ArgminFloat,
{
    particles: Vec<Particle<O::Param, F>>,
    best_position: O::Param,
    best_cost: F,

    // Weights for particle updates
    weight_momentum: F,
    weight_particle: F,
    weight_swarm: F,

    search_region: (O::Param, O::Param),
    num_particles: usize,
}

impl<O, F> ParticleSwarm<O, F>
where
    O: ArgminOp<Output = F>,
    <O as ArgminOp>::Param: Position<F>,
    F: ArgminFloat,
{
    /// Constructor
    ///
    /// Parameters:
    ///
    /// * `cost_function`: cost function
    /// * `init_temp`: initial temperature
    pub fn new(
        search_region: (O::Param, O::Param),
        num_particles: usize,
        weight_momentum: F,
        weight_particle: F,
        weight_swarm: F,
    ) -> Result<Self, Error> {
        let particle_swarm = ParticleSwarm {
            particles: vec![],
            best_position: O::Param::rand_from_range(
                // FIXME: random smart?
                &search_region.0,
                &search_region.1,
            ),
            best_cost: F::infinity(),
            weight_momentum,
            weight_particle,
            weight_swarm,
            search_region,
            num_particles,
        };

        Ok(particle_swarm)
    }

    fn initialize_particles(&mut self, op: &mut OpWrapper<O>) {
        self.particles = (0..self.num_particles)
            .map(|_| self.initialize_particle(op))
            .collect();

        self.best_position = self.get_best_position();
        self.best_cost = op.apply(&self.best_position).unwrap();
        // TODO unwrap evil
    }

    fn initialize_particle(&mut self, op: &mut OpWrapper<O>) -> Particle<O::Param, F> {
        let (min, max) = &self.search_region;
        let delta = max.sub(min);
        let delta_neg = delta.mul(&F::from_f64(-1.0).unwrap());

        let initial_position = O::Param::rand_from_range(min, max);
        let initial_cost = op.apply(&initial_position).unwrap(); // FIXME do not unwrap

        Particle {
            position: initial_position.clone(),
            velocity: O::Param::rand_from_range(&delta_neg, &delta),
            cost: initial_cost,
            best_position: initial_position,
            best_cost: initial_cost,
        }
    }

    fn get_best_position(&self) -> O::Param {
        let mut best: Option<(&O::Param, F)> = None;

        for p in &self.particles {
            match best {
                Some(best_sofar) => {
                    if p.cost < best_sofar.1 {
                        best = Some((&p.position, p.cost))
                    }
                }
                None => best = Some((&p.position, p.cost)),
            }
        }

        match best {
            Some(best_sofar) => best_sofar.0.clone(),
            None => panic!("Particles not initialized"),
        }
    }
}

impl<O, F> Solver<O, F> for ParticleSwarm<O, F>
where
    O: ArgminOp<Output = F>,
    <O as ArgminOp>::Param: Position<F>,
    <O as ArgminOp>::Hessian: Clone + Default,
    F: ArgminFloat,
{
    const NAME: &'static str = "Particle Swarm Optimization";

    fn init(
        &mut self,
        _op: &mut OpWrapper<O>,
        _state: &IterState<O, F>,
    ) -> Result<Option<ArgminIterData<O, F>>, Error> {
        self.initialize_particles(_op);

        Ok(None)
    }

    /// Perform one iteration of algorithm
    fn next_iter(
        &mut self,
        _op: &mut OpWrapper<O>,
        _state: &IterState<O, F>,
    ) -> Result<ArgminIterData<O, F>, Error> {
        let zero = O::Param::zero_like(&self.best_position);

        for p in self.particles.iter_mut() {
            // New velocity is composed of
            // 1) previous velocity (momentum),
            // 2) motion toward particle optimum and
            // 3) motion toward global optimum.

            // ad 1)
            let momentum = p.velocity.mul(&self.weight_momentum);

            // ad 2)
            let to_optimum = p.best_position.sub(&p.position);
            let pull_to_optimum = O::Param::rand_from_range(&zero, &to_optimum);
            let pull_to_optimum = pull_to_optimum.mul(&self.weight_particle);

            // ad 3)
            let to_global_optimum = self.best_position.sub(&p.position);
            let pull_to_global_optimum =
                O::Param::rand_from_range(&zero, &to_global_optimum).mul(&self.weight_swarm);

            p.velocity = momentum.add(&pull_to_optimum).add(&pull_to_global_optimum);
            let new_position = p.position.add(&p.velocity);

            // Limit to search window:
            p.position = O::Param::min(
                &O::Param::max(&new_position, &self.search_region.0),
                &self.search_region.1,
            );

            p.cost = _op.apply(&p.position)?;
            if p.cost < p.best_cost {
                p.best_position = p.position.clone();
                p.best_cost = p.cost;

                if p.cost < self.best_cost {
                    self.best_position = p.position.clone();
                    self.best_cost = p.cost;
                }
            }
        }

        // Store particles as population
        let population = self
            .particles
            .iter()
            .map(|particle| (particle.position.clone(), particle.cost))
            .collect();

        let out = ArgminIterData::new()
            .param(self.best_position.clone())
            .cost(self.best_cost)
            .population(population)
            .kv(make_kv!(
                "particles" => &self.particles;
            ));

        Ok(out)
    }
}

/// Position
pub trait Position<F: ArgminFloat>:
    Clone
    + Default
    + ArgminAdd<Self, Self>
    + ArgminSub<Self, Self>
    + ArgminMul<F, Self>
    + ArgminZeroLike
    + ArgminRandom
    + ArgminMinMax
    + std::fmt::Debug
{
}
impl<T, F: ArgminFloat> Position<F> for T
where
    T: Clone
        + Default
        + ArgminAdd<Self, Self>
        + ArgminSub<Self, Self>
        + ArgminMul<F, Self>
        + ArgminZeroLike
        + ArgminRandom
        + ArgminMinMax
        + std::fmt::Debug,
    F: ArgminFloat,
{
}

// trait_bound!(Position<F>
// ; Clone
// , Default
// , ArgminAdd<Self, Self>
// , ArgminSub<Self, Self>
// , ArgminMul<F, Self>
// , ArgminZeroLike
// , ArgminRandom
// , ArgminMinMax
// , std::fmt::Debug
// );

/// A single particle
#[derive(Clone, Serialize, Debug)]
pub struct Particle<T, F>
where
    T: Position<F>,
    F: ArgminFloat,
{
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
