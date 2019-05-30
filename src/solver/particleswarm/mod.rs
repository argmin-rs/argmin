// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # References:
//!
//! TODO

use crate::prelude::*;
use argmin_core::ArgminAdd;
use argmin_core::ArgminOp;
use serde::{Deserialize, Serialize};
use std;
use std::default::Default;
use std::f64;

/// Particle Swarm Optimization (PSO)
///
/// [Example](https://github.com/argmin-rs/argmin/blob/master/examples/particleswarm.rs)
///
/// # References:
///
/// TODO
#[derive(Serialize, Deserialize)]
pub struct ParticleSwarm<O>
where
    O: ArgminOp<Output = f64>,
    <O as ArgminOp>::Param: Position,
{
    particles: Vec<Particle<O::Param>>,
    best_position: O::Param,
    best_cost: f64,

    // Weights for particle updates
    weight_momentum: f64,
    weight_particle: f64,
    weight_swarm: f64,

    search_region: (O::Param, O::Param),
    num_particles: usize,
}

impl<O> ParticleSwarm<O>
where
    O: ArgminOp<Output = f64>,
    <O as ArgminOp>::Param: Position,
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
        weight_momentum: f64,
        weight_particle: f64,
        weight_swarm: f64,
    ) -> Result<Self, Error> {
        let particle_swarm = ParticleSwarm {
            particles: vec![],
            best_position: O::Param::rand_from_range(
                // FIXME: random smart?
                &search_region.0,
                &search_region.1,
            ),
            best_cost: f64::INFINITY,
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

    fn initialize_particle(&mut self, op: &mut OpWrapper<O>) -> Particle<O::Param> {
        let (min, max) = &self.search_region;
        let delta = max.sub(min);
        let delta_neg = delta.mul(&-1.0);

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
        let mut best: Option<(&O::Param, f64)> = None;

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

impl<O> Solver<O> for ParticleSwarm<O>
where
    O: ArgminOp<Output = f64>,
    <O as ArgminOp>::Param: Position,
    <O as ArgminOp>::Hessian: Clone + Default,
{
    const NAME: &'static str = "Particle Swarm Optimization";

    fn init(
        &mut self,
        _op: &mut OpWrapper<O>,
        _state: &IterState<O>,
    ) -> Result<Option<ArgminIterData<O>>, Error> {
        self.initialize_particles(_op);

        Ok(None)
    }

    /// Perform one iteration of algorithm
    fn next_iter(
        &mut self,
        _op: &mut OpWrapper<O>,
        _state: &IterState<O>,
    ) -> Result<ArgminIterData<O>, Error> {
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

        let out = ArgminIterData::new()
            .param(self.best_position.clone())
            .cost(self.best_cost)
            .kv(make_kv!(
                "particles" => &self.particles;
            ));
        // out.add_kv(make_kv!(
        //     "t" => self.cur_temp;

        // ));

        Ok(out)
    }
}

trait_bound!(Position
; Clone
, Default
, ArgminAdd<Self, Self>
, ArgminSub<Self, Self>
, ArgminMul<f64, Self>
, ArgminZeroLike
, ArgminRandom
, ArgminMinMax
, std::fmt::Debug
);

/// A single particle
#[derive(Serialize, Deserialize, Debug)]
pub struct Particle<T: Position> {
    /// Position of particle
    pub position: T,
    /// Velocity of particle
    velocity: T,
    /// Cost of particle
    pub cost: f64,
    /// Best position of particle so far
    best_position: T,
    /// Best cost of particle so far
    best_cost: f64,
}
