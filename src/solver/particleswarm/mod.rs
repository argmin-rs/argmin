// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.


use crate::prelude::*;
use argmin_codegen::ArgminSolver;
// use rand;
use rand::Rng;
use rand::distributions::uniform::SampleUniform;
use std;
use std::default::Default;
use argmin_core::ArgminAdd;


// TODO: pass particles by reference
type Callback<T> = FnMut(&T, f64, Vec<Particle<T>>) -> ();


// #[log("initial_temperature" => "self.init_temp")]
// #[log("stall_iter_accepted_limit" => "self.stall_iter_accepted_limit")]
// #[log("stall_iter_best_limit" => "self.stall_iter_best_limit")]
// #[log("reanneal_fixed" => "self.reanneal_fixed")]
// #[log("reanneal_accepted" => "self.reanneal_accepted")]
// #[log("reanneal_best" => "self.reanneal_best")]
#[derive(ArgminSolver)]
pub struct ParticleSwarm<'a, T, H>
where
    T: Position,
    H: Clone + Default,
{
    base: ArgminBase<'a, T, f64, H>,
    rng: rand::prelude::ThreadRng,
    iter_callback: Option<&'a mut Callback<T>>,
    particles: Vec<Particle<T>>
}

impl<'a, T, H> ParticleSwarm<'a, T, H>
where
    T: Position,
    H: Clone + Default,
{
    /// Constructor
    ///
    /// Parameters:
    ///
    /// * `cost_function`: cost function
    /// * `init_param`: initial parameter vector
    /// * `init_temp`: initial temperature
    pub fn new(
        cost_function: &'a ArgminOperator<Parameters = T, OperatorOutput = f64, Hessian = H>,
        init_param: T,
        search_region: (T, T),
        num_particles: usize,
    ) -> Result<Self, Error> {

        let rng = rand::thread_rng();

        let mut particle_swarm = ParticleSwarm {
            base: ArgminBase::new(cost_function, init_param),
            rng: rng.clone(),
            iter_callback: None,
            particles: vec![]
        };

        particle_swarm.initialize_particles(num_particles, &search_region);

        Ok(particle_swarm)
    }

    pub fn set_iter_callback(&mut self, callback: &'a mut Callback<T>) {
        self.iter_callback = Some(callback);
    }

    fn initialize_particles(&mut self, num_particles: usize, search_region: &(T, T)) {
        self.particles = (0..num_particles).map(
                |_| self.initialize_particle(search_region)
        ).collect();
    }

    fn initialize_particle(&mut self, search_region: &(T, T)) -> Particle<T> {
        let (min, max) = search_region;
        let delta = max.sub(min);
        let delta_neg = delta.mul(&-1.0);

        let initial_position = T::rand_from_range(&mut self.rng, min, max);
        let initial_cost = self.apply(&initial_position).unwrap(); // TODO: unwrap evil?

        Particle {
            position: initial_position.clone(),
            velocity: T::rand_from_range(&mut self.rng, &delta_neg, &delta),
            cost: initial_cost,
            best_position: initial_position,
        }
    }
}


impl<'a, T, H> ArgminNextIter for ParticleSwarm<'a, T, H>
where
    T: Position,
    H: Clone + Default,
{
    type Parameters = T;
    type OperatorOutput = f64;
    type Hessian = H;

    /// Perform one iteration of algorithm
    fn next_iter(&mut self) -> Result<ArgminIterationData<Self::Parameters>, Error> {

        // FIXME: replace by actual parameter search
        let new_param = self.cur_param().add(&self.cur_param());

        let new_cost = self.apply(&new_param)?;

        // // TODO: this must be possible with less code
        // let mut costs = vec![];
        // for particle in self.particles.clone() {
        //     let cost: f64 = self.apply(&particle.position)?;
        //     costs.push(cost);
        // }
        // for i in 0..self.particles.len() {
        //     self.particles[i].cost = costs[i];
        // }

        // TODO: move callback to ArgminBase
        // TODO: accept &self, not new_param, new_cost
        //       as callback parameters
        match &mut self.iter_callback {
            Some(callback) => (*callback)(&new_param, new_cost, self.particles.clone()),
            None => ()
        };


        let out = ArgminIterationData::new(new_param, new_cost);
        // out.add_kv(make_kv!(
        //     "t" => self.cur_temp;

        // ));

        Ok(out)
    }
}


// TODO: use a generic function
pub trait RandFromRange
{
    fn rand_from_range(rng: &mut rand::prelude::ThreadRng,
                       min: &Self, max: &Self) -> Self;
}

impl<Scalar> RandFromRange for Vec<Scalar>
    where Scalar: SampleUniform
{
    fn rand_from_range(rng: &mut rand::prelude::ThreadRng,
                       min: &Self, max: &Self) -> Self
    {
        return min.iter().zip(max.iter()).map(|(a, b)| rng.gen_range(a, b)).collect();
    }
}


pub trait Position
: Clone
+ Default
+ ArgminAdd<Self, Self>
+ ArgminSub<Self, Self>
+ ArgminMul<f64, Self>
+ RandFromRange
{}

impl<T> Position for T where T
: Clone
+ Default
+ ArgminAdd<Self, Self>
+ ArgminSub<Self, Self>
+ ArgminMul<f64, Self>
+ RandFromRange
{}



#[derive(Clone)]
pub struct Particle<T: Position> {
    pub position: T,
    velocity: T,
    pub cost: f64,
    best_position: T,
}

