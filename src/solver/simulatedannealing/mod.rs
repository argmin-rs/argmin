// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! * [Simulated Annealing](struct.SimulatedAnnealing.html)
//!
//! # References
//!
//! \[0\] [Wikipedia](https://en.wikipedia.org/wiki/Simulated_annealing)
//!
//! \[1\] S Kirkpatrick, CD Gelatt Jr, MP Vecchi. (1983). "Optimization by Simulated Annealing".
//! Science 13 May 1983, Vol. 220, Issue 4598, pp. 671-680
//! DOI: 10.1126/science.220.4598.671

use crate::prelude::*;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// Temperature functions for Simulated Annealing.
///
/// Given the initial temperature `t_init` and the iteration number `i`, the current temperature
/// `t_i` is given as follows:
///
/// * `SATempFunc::TemperatureFast`: `t_i = t_init / i`
/// * `SATempFunc::Boltzmann`: `t_i = t_init / ln(i)`
/// * `SATempFunc::Exponential`: `t_i = t_init * 0.95^i`
#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub enum SATempFunc<F> {
    /// `t_i = t_init / i`
    TemperatureFast,
    /// `t_i = t_init / ln(i)`
    Boltzmann,
    /// `t_i = t_init * x^i`
    Exponential(F),
    // /// User-provided temperature function. The first parameter must be the current temperature and
    // /// the second parameter must be the iteration number.
    // Custom(Box<Fn(f64, u64) -> f64>),
}

impl<F> std::default::Default for SATempFunc<F> {
    fn default() -> Self {
        SATempFunc::Boltzmann
    }
}

/// Simulated Annealing
///
/// [Example](https://github.com/argmin-rs/argmin/blob/master/examples/simulatedannealing.rs)
///
/// # References
///
/// \[0\] [Wikipedia](https://en.wikipedia.org/wiki/Simulated_annealing)
///
/// \[1\] S Kirkpatrick, CD Gelatt Jr, MP Vecchi. (1983). "Optimization by Simulated Annealing".
/// Science 13 May 1983, Vol. 220, Issue 4598, pp. 671-680
/// DOI: 10.1126/science.220.4598.671
#[derive(Clone, Serialize, Deserialize)]
pub struct SimulatedAnnealing<F, R> {
    /// Initial temperature
    init_temp: F,
    /// which temperature function?
    temp_func: SATempFunc<F>,
    /// Number of iterations used for the calculation of temperature. This is needed for
    /// reannealing!
    temp_iter: u64,
    /// Iterations since the last accepted solution
    stall_iter_accepted: u64,
    /// Stop if stall_iter_accepted exceeds this number
    stall_iter_accepted_limit: u64,
    /// Iterations since the last best solution was found
    stall_iter_best: u64,
    /// Stop if stall_iter_best exceeds this number
    stall_iter_best_limit: u64,
    /// Reanneal after this number of iterations is reached
    reanneal_fixed: u64,
    /// Similar to `iter`, but will be reset to 0 when reannealing is performed
    reanneal_iter_fixed: u64,
    /// Reanneal after no accepted solution has been found for `reanneal_accepted` iterations
    reanneal_accepted: u64,
    /// Similar to `stall_iter_accepted`, but will be reset to 0 when reannealing  is performed
    reanneal_iter_accepted: u64,
    /// Reanneal after no new best solution has been found for `reanneal_best` iterations
    reanneal_best: u64,
    /// Similar to `stall_iter_best`, but will be reset to 0 when reannealing is performed
    reanneal_iter_best: u64,
    /// current temperature
    cur_temp: F,
    /// random number generator
    rng: R,
}

impl<F, R> SimulatedAnnealing<F, R>
where
    F: ArgminFloat,
{
    /// Constructor
    ///
    /// Parameter:
    ///
    /// * `init_temp`: initial temperature
    /// * `rng`: an RNG (must implement Serialize)
    pub fn new(init_temp: F, rng: R) -> Result<Self, Error> {
        if init_temp <= F::from_f64(0.0).unwrap() {
            Err(ArgminError::InvalidParameter {
                text: "Initial temperature must be > 0.".to_string(),
            }
            .into())
        } else {
            Ok(SimulatedAnnealing {
                init_temp,
                temp_func: SATempFunc::TemperatureFast,
                temp_iter: 0,
                stall_iter_accepted: 0,
                stall_iter_accepted_limit: std::u64::MAX,
                stall_iter_best: 0,
                stall_iter_best_limit: std::u64::MAX,
                reanneal_fixed: std::u64::MAX,
                reanneal_iter_fixed: 0,
                reanneal_accepted: std::u64::MAX,
                reanneal_iter_accepted: 0,
                reanneal_best: std::u64::MAX,
                reanneal_iter_best: 0,
                cur_temp: init_temp,
                rng,
            })
        }
    }

    /// Set temperature function to one of the options in `SATempFunc`.
    pub fn temp_func(mut self, temperature_func: SATempFunc<F>) -> Self {
        self.temp_func = temperature_func;
        self
    }

    /// The optimization stops after there has been no accepted solution after `iter` iterations
    pub fn stall_accepted(mut self, iter: u64) -> Self {
        self.stall_iter_accepted_limit = iter;
        self
    }

    /// The optimization stops after there has been no new best solution after `iter` iterations
    pub fn stall_best(mut self, iter: u64) -> Self {
        self.stall_iter_best_limit = iter;
        self
    }

    /// Start reannealing after `iter` iterations
    pub fn reannealing_fixed(mut self, iter: u64) -> Self {
        self.reanneal_fixed = iter;
        self
    }

    /// Start reannealing after no accepted solution has been found for `iter` iterations
    pub fn reannealing_accepted(mut self, iter: u64) -> Self {
        self.reanneal_accepted = iter;
        self
    }

    /// Start reannealing after no new best solution has been found for `iter` iterations
    pub fn reannealing_best(mut self, iter: u64) -> Self {
        self.reanneal_best = iter;
        self
    }

    /// Update the temperature based on the current iteration number.
    ///
    /// Updates are performed based on specific update functions. See `SATempFunc` for details.
    fn update_temperature(&mut self) {
        self.cur_temp = match self.temp_func {
            SATempFunc::TemperatureFast => {
                self.init_temp / F::from_u64(self.temp_iter + 1).unwrap()
            }
            SATempFunc::Boltzmann => self.init_temp / F::from_u64(self.temp_iter + 1).unwrap().ln(),
            SATempFunc::Exponential(x) => {
                self.init_temp * x.powf(F::from_u64(self.temp_iter + 1).unwrap())
            }
        };
    }

    /// Perform reannealing
    fn reanneal(&mut self) -> (bool, bool, bool) {
        let out = (
            self.reanneal_iter_fixed >= self.reanneal_fixed,
            self.reanneal_iter_accepted >= self.reanneal_accepted,
            self.reanneal_iter_best >= self.reanneal_best,
        );
        if out.0 || out.1 || out.2 {
            self.reanneal_iter_fixed = 0;
            self.reanneal_iter_accepted = 0;
            self.reanneal_iter_best = 0;
            self.cur_temp = self.init_temp;
            self.temp_iter = 0;
        }
        out
    }

    /// Update the stall iter variables
    fn update_stall_and_reanneal_iter(&mut self, accepted: bool, new_best: bool) {
        self.stall_iter_accepted = if accepted {
            0
        } else {
            self.stall_iter_accepted + 1
        };

        self.reanneal_iter_accepted = if accepted {
            0
        } else {
            self.reanneal_iter_accepted + 1
        };

        self.stall_iter_best = if new_best {
            0
        } else {
            self.stall_iter_best + 1
        };

        self.reanneal_iter_best = if new_best {
            0
        } else {
            self.reanneal_iter_best + 1
        };
    }
}

impl<O, F, R> Solver<O> for SimulatedAnnealing<F, R>
where
    O: ArgminOp<Output = F, Float = F>,
    F: ArgminFloat,
    R: Rng + Serialize,
{
    const NAME: &'static str = "Simulated Annealing";
    fn init(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<Option<ArgminIterData<O>>, Error> {
        let cost = op.apply(&state.get_param())?;
        Ok(Some(
            ArgminIterData::new()
                .param(state.get_param())
                .cost(cost)
                .kv(make_kv!(
                    "initial_temperature" => self.init_temp;
                    "stall_iter_accepted_limit" => self.stall_iter_accepted_limit;
                    "stall_iter_best_limit" => self.stall_iter_best_limit;
                    "reanneal_fixed" => self.reanneal_fixed;
                    "reanneal_accepted" => self.reanneal_accepted;
                    "reanneal_best" => self.reanneal_best;
                )),
        ))
    }

    /// Perform one iteration of SA algorithm
    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<ArgminIterData<O>, Error> {
        // Careful: The order in here is *very* important, even if it may not seem so. Everything
        // is linked to the iteration number, and getting things mixed up will lead to strange
        // behaviour.

        let prev_param = state.get_param();
        let prev_cost = state.get_cost();

        // Make a move
        let new_param = op.modify(&prev_param, self.cur_temp)?;

        // Evaluate cost function with new parameter vector
        let new_cost = op.apply(&new_param)?;

        // Acceptance function
        //
        // Decide whether new parameter vector should be accepted.
        // If no, move on with old parameter vector.
        //
        // Any solution which satisfies `next_cost < prev_cost` will be accepted. Solutions worse
        // than the previous one are accepted with a probability given as:
        //
        // `1 / (1 + exp((next_cost - prev_cost) / current_temperature))`,
        //
        // which will always be between 0 and 0.5.
        let prob: f64 = self.rng.gen();
        let prob = F::from_f64(prob).unwrap();
        let accepted = (new_cost < prev_cost)
            || (F::from_f64(1.0).unwrap()
                / (F::from_f64(1.0).unwrap() + ((new_cost - prev_cost) / self.cur_temp).exp())
                > prob);

        let new_best_found = new_cost < state.get_best_cost();

        // Update stall iter variables
        self.update_stall_and_reanneal_iter(accepted, new_best_found);

        let (r_fixed, r_accepted, r_best) = self.reanneal();

        // Update temperature for next iteration.
        self.temp_iter += 1;
        // Todo: this variable may not be necessary (temp_iter does the same?)
        self.reanneal_iter_fixed += 1;

        self.update_temperature();

        Ok(if accepted {
            ArgminIterData::new().param(new_param).cost(new_cost)
        } else {
            ArgminIterData::new().param(prev_param).cost(prev_cost)
        }
        .kv(make_kv!(
            "t" => self.cur_temp;
            "new_be" => new_best_found;
            "acc" => accepted;
            "st_i_be" => self.stall_iter_best;
            "st_i_ac" => self.stall_iter_accepted;
            "ra_i_fi" => self.reanneal_iter_fixed;
            "ra_i_be" => self.reanneal_iter_best;
            "ra_i_ac" => self.reanneal_iter_accepted;
            "ra_fi" => r_fixed;
            "ra_be" => r_best;
            "ra_ac" => r_accepted;
        )))
    }

    fn terminate(&mut self, _state: &IterState<O>) -> TerminationReason {
        if self.stall_iter_accepted > self.stall_iter_accepted_limit {
            return TerminationReason::AcceptedStallIterExceeded;
        }
        if self.stall_iter_best > self.stall_iter_best_limit {
            return TerminationReason::BestStallIterExceeded;
        }
        TerminationReason::NotTerminated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_trait_impl;

    test_trait_impl!(sa, SimulatedAnnealing<f64, StdRng>);
}
