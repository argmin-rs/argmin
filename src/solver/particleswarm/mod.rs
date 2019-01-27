// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.


use crate::prelude::*;
use argmin_codegen::ArgminSolver;
// use rand;
// use rand::Rng;
use std;
use std::default::Default;


// #[log("initial_temperature" => "self.init_temp")]
// #[log("stall_iter_accepted_limit" => "self.stall_iter_accepted_limit")]
// #[log("stall_iter_best_limit" => "self.stall_iter_best_limit")]
// #[log("reanneal_fixed" => "self.reanneal_fixed")]
// #[log("reanneal_accepted" => "self.reanneal_accepted")]
// #[log("reanneal_best" => "self.reanneal_best")]
#[derive(ArgminSolver)]
pub struct ParticleSwarm<'a, T, H>
where
    T: Clone + Default,
    H: Clone + Default,
{
    // /// Initial temperature
    // init_temp: f64,
    // /// which temperature function?
    // temp_func: SATempFunc,
    // /// Number of iterations used for the caluclation of temperature. This is needed for
    // /// reannealing!
    // temp_iter: u64,
    // /// Iterations since the last accepted solution
    // stall_iter_accepted: u64,
    // /// Stop if stall_iter_accepted exceedes this number
    // stall_iter_accepted_limit: u64,
    // /// Iterations since the last best solution was found
    // stall_iter_best: u64,
    // /// Stop if stall_iter_best exceedes this number
    // stall_iter_best_limit: u64,
    // /// Reanneal after this number of iterations is reached
    // reanneal_fixed: u64,
    // /// Similar to `iter`, but will be reset to 0 when reannealing is performed
    // reanneal_iter_fixed: u64,
    // /// Reanneal after no accepted solution has been found for `reanneal_accepted` iterations
    // reanneal_accepted: u64,
    // /// Similar to `stall_iter_accepted`, but will be reset to 0 when reannealing  is performed
    // reanneal_iter_accepted: u64,
    // /// Reanneal after no new best solution has been found for `reanneal_best` iterations
    // reanneal_best: u64,
    // /// Similar to `stall_iter_best`, but will be reset to 0 when reannealing is performed
    // reanneal_iter_best: u64,
    // /// current temperature
    // cur_temp: f64,
    // /// previous cost
    // prev_cost: f64,
    // /// random number generator
    // rng: rand::prelude::ThreadRng,
    /// base
    base: ArgminBase<'a, T, f64, H>,
}

impl<'a, T, H> ParticleSwarm<'a, T, H>
where
    T: Clone + Default,
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
    ) -> Result<Self, Error> {
        Ok(ParticleSwarm {
            // init_temp,
            // temp_func: SATempFunc::TemperatureFast,
            // temp_iter: 0u64,
            // stall_iter_accepted: 0u64,
            // stall_iter_accepted_limit: std::u64::MAX,
            // stall_iter_best: 0u64,
            // stall_iter_best_limit: std::u64::MAX,
            // reanneal_fixed: std::u64::MAX,
            // reanneal_iter_fixed: 0,
            // reanneal_accepted: std::u64::MAX,
            // reanneal_iter_accepted: 0,
            // reanneal_best: std::u64::MAX,
            // reanneal_iter_best: 0,
            // cur_temp: init_temp,
            // prev_cost,
            // rng: rand::thread_rng(),
            base: ArgminBase::new(cost_function, init_param),
        })
    }
}


impl<'a, T, H> ArgminNextIter for ParticleSwarm<'a, T, H>
where
    T: Clone + Default,
    H: Clone + Default,
{
    type Parameters = T;
    type OperatorOutput = f64;
    type Hessian = H;

    /// Perform one iteration of algorithm
    fn next_iter(&mut self) -> Result<ArgminIterationData<Self::Parameters>, Error> {

        let new_param = self.cur_param(); // FIXME
        let new_cost = self.apply(&new_param)?;

        let out = ArgminIterationData::new(new_param, new_cost);
        // out.add_kv(make_kv!(
        //     "t" => self.cur_temp;
        //     "new_be" => new_best;
        //     "acc" => accepted;
        //     "st_i_be" => self.stall_iter_best;
        //     "st_i_ac" => self.stall_iter_accepted;
        //     "ra_i_fi" => self.reanneal_iter_fixed;
        //     "ra_i_be" => self.reanneal_iter_best;
        //     "ra_i_ac" => self.reanneal_iter_accepted;
        //     "ra_fi" => r_fixed;
        //     "ra_be" => r_best;
        //     "ra_ac" => r_accepted;
        // ));
        Ok(out)
    }
}
