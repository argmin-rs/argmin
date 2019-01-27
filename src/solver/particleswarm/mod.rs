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
use argmin_core::ArgminAdd;


// #[log("initial_temperature" => "self.init_temp")]
// #[log("stall_iter_accepted_limit" => "self.stall_iter_accepted_limit")]
// #[log("stall_iter_best_limit" => "self.stall_iter_best_limit")]
// #[log("reanneal_fixed" => "self.reanneal_fixed")]
// #[log("reanneal_accepted" => "self.reanneal_accepted")]
// #[log("reanneal_best" => "self.reanneal_best")]
#[derive(ArgminSolver)]
pub struct ParticleSwarm<'a, T, H>
where
    T: Clone + Default + ArgminAdd<T>,
    H: Clone + Default,
{
    base: ArgminBase<'a, T, f64, H>,
    rng: rand::prelude::ThreadRng,
    iter_callback: Option<&'a mut (FnMut(&T, f64) -> ())>,
}

fn default_callback<T>(p: &T, cost: f64) {}

impl<'a, T, H> ParticleSwarm<'a, T, H>
where
    T: Clone + Default + ArgminAdd<T>,
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
            base: ArgminBase::new(cost_function, init_param),
            rng: rand::thread_rng(),
            iter_callback: None,
        })
    }

    pub fn set_iter_callback(&mut self, callback: &'a mut FnMut(&T, f64) -> ()) {
        self.iter_callback = Some(callback);
    }
}


impl<'a, T, H> ArgminNextIter for ParticleSwarm<'a, T, H>
where
    T: Clone + Default + ArgminAdd<T>,
    H: Clone + Default,
{
    type Parameters = T;
    type OperatorOutput = f64;
    type Hessian = H;

    /// Perform one iteration of algorithm
    fn next_iter(&mut self) -> Result<ArgminIterationData<Self::Parameters>, Error> {

        let new_param = self.cur_param().add(&self.cur_param());
        let new_cost = self.apply(&new_param)?;

        // TODO: move callback to ArgminBase
        // TODO: accept &self, not new_param, new_cost
        //       as callback parameters
        match &mut self.iter_callback {
            Some(callback) => (*callback)(&new_param, new_cost),
            None => ()
        };


        let out = ArgminIterationData::new(new_param, new_cost);
        // out.add_kv(make_kv!(
        //     "t" => self.cur_temp;

        // ));

        Ok(out)
    }
}
