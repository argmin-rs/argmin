// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Simulated Annealing
//!
//! Simulated Annealing (SA) is a stochastic optimization method which imitates annealing in
//! metallurgy. For details see [`SimulatedAnnealing`].
//!
//! ## References
//!
//! [Wikipedia](https://en.wikipedia.org/wiki/Simulated_annealing)
//!
//! S Kirkpatrick, CD Gelatt Jr, MP Vecchi. (1983). "Optimization by Simulated Annealing".
//! Science 13 May 1983, Vol. 220, Issue 4598, pp. 671-680
//! DOI: 10.1126/science.220.4598.671

use crate::core::{
    ArgminFloat, CostFunction, Error, IterState, Problem, SerializeAlias, Solver,
    TerminationReason, TerminationStatus, KV,
};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// This trait handles the annealing of a parameter vector. Problems which are to be solved using
/// [`SimulatedAnnealing`] must implement this trait.
pub trait Anneal {
    /// Type of the parameter vector
    type Param;
    /// Return type of the anneal function
    type Output;
    /// Precision of floats
    type Float;

    /// Anneal a parameter vector
    fn anneal(&self, param: &Self::Param, extent: Self::Float) -> Result<Self::Output, Error>;
}

/// Wraps a call to `anneal` defined in the `Anneal` trait and as such allows to call `anneal` on
/// an instance of `Problem`. Internally, the number of evaluations of `anneal` is counted.
impl<O: Anneal> Problem<O> {
    /// Calls `anneal` defined in the `Anneal` trait and keeps track of the number of evaluations.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{Problem, Error};
    /// # use argmin::solver::simulatedannealing::Anneal;
    /// #
    /// # #[derive(Eq, PartialEq, Debug, Clone)]
    /// # struct UserDefinedProblem {};
    /// #
    /// # impl Anneal for UserDefinedProblem {
    /// #     type Param = Vec<f64>;
    /// #     type Output = Vec<f64>;
    /// #     type Float = f64;
    /// #
    /// #     fn anneal(&self, param: &Self::Param, extent: Self::Float) -> Result<Self::Output, Error> {
    /// #         Ok(vec![1.0f64, 1.0f64])
    /// #     }
    /// # }
    /// // `UserDefinedProblem` implements `Anneal`.
    /// let mut problem1 = Problem::new(UserDefinedProblem {});
    ///
    /// let param = vec![2.0f64, 1.0f64];
    ///
    /// let res = problem1.anneal(&param, 1.0);
    ///
    /// assert_eq!(problem1.counts["anneal_count"], 1);
    /// # assert_eq!(res.unwrap(), vec![1.0f64, 1.0f64]);
    /// ```
    pub fn anneal(&mut self, param: &O::Param, extent: O::Float) -> Result<O::Output, Error> {
        self.problem("anneal_count", |problem| problem.anneal(param, extent))
    }
}

/// Temperature functions for Simulated Annealing.
///
/// Given the initial temperature `t_init` and the iteration number `i`, the current temperature
/// `t_i` is given as follows:
///
/// * `SATempFunc::TemperatureFast`: `t_i = t_init / i`
/// * `SATempFunc::Boltzmann`: `t_i = t_init / ln(i)`
/// * `SATempFunc::Exponential`: `t_i = t_init * 0.95^i`
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub enum SATempFunc<F> {
    /// `t_i = t_init / i`
    TemperatureFast,
    /// `t_i = t_init / ln(i)`
    #[default]
    Boltzmann,
    /// `t_i = t_init * x^i`
    Exponential(F),
    // /// User-provided temperature function. The first parameter must be the current temperature and
    // /// the second parameter must be the iteration number.
    // Custom(Box<dyn Fn(f64, u64) -> f64 + 'static>),
}

/// # Simulated Annealing
///
/// Simulated Annealing (SA) is a stochastic optimization method which imitates annealing in
/// metallurgy. Parameter vectors are randomly modified in each iteration, where the degree of
/// modification depends on the current temperature. The algorithm starts with a high temperature
/// (a lot of modification and hence movement in parameter space) and continuously cools down as
/// the iterations progress, hence narrowing down in the search. Under certain conditions,
/// reannealing (increasing the temperature) can be performed. Solutions which are better than the
/// previous one are always accepted and solutions which are worse are accepted with a probability
/// proportional to the cost function value difference of previous to current parameter vector.
/// These measures allow the algorithm to explore the parameter space in a large and a small scale
/// and hence it is able to overcome local minima.
///
/// The initial temperature has to be provided by the user as well as the a initial parameter
/// vector (via [`configure`](`crate::core::Executor::configure`) of
/// [`Executor`](`crate::core::Executor`).
///
/// The cooling schedule can be set with [`SimulatedAnnealing::with_temp_func`]. For the available
/// choices please see [`SATempFunc`].
///
/// Reannealing can be performed if no new best solution was found for `N` iterations
/// ([`SimulatedAnnealing::with_reannealing_best`]), or if no new accepted solution was found for
/// `N` iterations ([`SimulatedAnnealing::with_reannealing_accepted`]) or every `N` iterations
/// without any other conditions ([`SimulatedAnnealing::with_reannealing_fixed`]).
///
/// The user-provided problem must implement [`Anneal`] which defines how parameter vectors are
/// modified. Please see the Simulated Annealing example for one approach to do so for floating
/// point parameters.
///
/// ## Requirements on the optimization problem
///
/// The optimization problem is required to implement [`CostFunction`].
///
/// ## References
///
/// [Wikipedia](https://en.wikipedia.org/wiki/Simulated_annealing)
///
/// S Kirkpatrick, CD Gelatt Jr, MP Vecchi. (1983). "Optimization by Simulated Annealing".
/// Science 13 May 1983, Vol. 220, Issue 4598, pp. 671-680
/// DOI: 10.1126/science.220.4598.671
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct SimulatedAnnealing<F, R> {
    /// Initial temperature
    init_temp: F,
    /// Temperature function used for decreasing the temperature
    temp_func: SATempFunc<F>,
    /// Number of iterations used for the calculation of temperature. Needed for reannealing
    temp_iter: u64,
    /// Number of iterations since the last accepted solution
    stall_iter_accepted: u64,
    /// Stop if `stall_iter_accepted` exceeds this number
    stall_iter_accepted_limit: u64,
    /// Number of iterations since the last best solution was found
    stall_iter_best: u64,
    /// Stop if `stall_iter_best` exceeds this number
    stall_iter_best_limit: u64,
    /// Reanneal after this number of iterations is reached
    reanneal_fixed: u64,
    /// Number of iterations since beginning or last reannealing
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

impl<F> SimulatedAnnealing<F, Xoshiro256PlusPlus>
where
    F: ArgminFloat,
{
    /// Construct a new instance of [`SimulatedAnnealing`]
    ///
    /// Takes the initial temperature as input, which must be >0.
    ///
    /// Uses the `Xoshiro256PlusPlus` RNG internally. For use of another RNG, consider using
    /// [`SimulatedAnnealing::new_with_rng`].
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::simulatedannealing::SimulatedAnnealing;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// let sa = SimulatedAnnealing::new(100.0f64)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(initial_temperature: F) -> Result<Self, Error> {
        SimulatedAnnealing::new_with_rng(initial_temperature, Xoshiro256PlusPlus::from_entropy())
    }
}

impl<F, R> SimulatedAnnealing<F, R>
where
    F: ArgminFloat,
{
    /// Construct a new instance of [`SimulatedAnnealing`]
    ///
    /// Takes the initial temperature as input, which must be >0.
    /// Requires a RNG which must implement `rand::Rng` (and `serde::Serialize` if the `serde1`
    /// feature is enabled).
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::simulatedannealing::SimulatedAnnealing;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// # let my_rng = ();
    /// let sa = SimulatedAnnealing::new_with_rng(100.0f64, my_rng)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_with_rng(init_temp: F, rng: R) -> Result<Self, Error> {
        if init_temp <= float!(0.0) {
            Err(argmin_error!(
                InvalidParameter,
                "`SimulatedAnnealing`: Initial temperature must be > 0."
            ))
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

    /// Set temperature function
    ///
    /// The temperature function defines how the temperature is decreased over the course of the
    /// iterations.
    /// See [`SATempFunc`] for the available options. Defaults to [`SATempFunc::TemperatureFast`].
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::simulatedannealing::{SimulatedAnnealing, SATempFunc};
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// let sa = SimulatedAnnealing::new(100.0f64)?.with_temp_func(SATempFunc::Boltzmann);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn with_temp_func(mut self, temperature_func: SATempFunc<F>) -> Self {
        self.temp_func = temperature_func;
        self
    }

    /// If there are no accepted solutions for `iter` iterations, the algorithm stops.
    ///
    /// Defaults to `std::u64::MAX`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::simulatedannealing::{SimulatedAnnealing, SATempFunc};
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// let sa = SimulatedAnnealing::new(100.0f64)?.with_stall_accepted(1000);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn with_stall_accepted(mut self, iter: u64) -> Self {
        self.stall_iter_accepted_limit = iter;
        self
    }

    /// If there are no new best solutions for `iter` iterations, the algorithm stops.
    ///
    /// Defaults to `std::u64::MAX`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::simulatedannealing::{SimulatedAnnealing, SATempFunc};
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// let sa = SimulatedAnnealing::new(100.0f64)?.with_stall_best(2000);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn with_stall_best(mut self, iter: u64) -> Self {
        self.stall_iter_best_limit = iter;
        self
    }

    /// Set number of iterations after which reannealing is performed
    ///
    /// Every `iter` iterations, reannealing (resetting temperature to its initial value) will be
    /// performed. This may help in overcoming local minima.
    ///
    /// Defaults to `std::u64::MAX`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::simulatedannealing::{SimulatedAnnealing, SATempFunc};
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// let sa = SimulatedAnnealing::new(100.0f64)?.with_reannealing_fixed(5000);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn with_reannealing_fixed(mut self, iter: u64) -> Self {
        self.reanneal_fixed = iter;
        self
    }

    /// Set the number of iterations that need to pass after the last accepted solution was found
    /// for reannealing to be performed.
    ///
    /// If no new accepted solution is found for `iter` iterations, reannealing (resetting
    /// temperature to its initial value) is performed. This may help in overcoming local minima.
    ///
    /// Defaults to `std::u64::MAX`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::simulatedannealing::{SimulatedAnnealing, SATempFunc};
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// let sa = SimulatedAnnealing::new(100.0f64)?.with_reannealing_accepted(5000);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn with_reannealing_accepted(mut self, iter: u64) -> Self {
        self.reanneal_accepted = iter;
        self
    }

    /// Set the number of iterations that need to pass after the last best solution was found
    /// for reannealing to be performed.
    ///
    /// If no new best solution is found for `iter` iterations, reannealing (resetting temperature
    /// to its initial value) is performed. This may help in overcoming local minima.
    ///
    /// Defaults to `std::u64::MAX`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::simulatedannealing::{SimulatedAnnealing, SATempFunc};
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// let sa = SimulatedAnnealing::new(100.0f64)?.with_reannealing_best(5000);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn with_reannealing_best(mut self, iter: u64) -> Self {
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
        (self.stall_iter_accepted, self.reanneal_iter_accepted) = if accepted {
            (0, 0)
        } else {
            (
                self.stall_iter_accepted + 1,
                self.reanneal_iter_accepted + 1,
            )
        };

        (self.stall_iter_best, self.reanneal_iter_best) = if new_best {
            (0, 0)
        } else {
            (self.stall_iter_best + 1, self.reanneal_iter_best + 1)
        };
    }
}

impl<O, P, F, R> Solver<O, IterState<P, (), (), (), F>> for SimulatedAnnealing<F, R>
where
    O: CostFunction<Param = P, Output = F> + Anneal<Param = P, Output = P, Float = F>,
    P: Clone,
    F: ArgminFloat,
    R: Rng + SerializeAlias,
{
    fn name(&self) -> &str {
        "Simulated Annealing"
    }
    fn init(
        &mut self,
        problem: &mut Problem<O>,
        mut state: IterState<P, (), (), (), F>,
    ) -> Result<(IterState<P, (), (), (), F>, Option<KV>), Error> {
        let param = state.take_param().ok_or_else(argmin_error_closure!(
            NotInitialized,
            concat!(
                "`SimulatedAnnealing` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method."
            )
        ))?;

        let cost = state.get_cost();
        let cost = if cost.is_infinite() {
            problem.cost(&param)?
        } else {
            cost
        };

        Ok((
            state.param(param).cost(cost),
            Some(kv!(
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
        problem: &mut Problem<O>,
        mut state: IterState<P, (), (), (), F>,
    ) -> Result<(IterState<P, (), (), (), F>, Option<KV>), Error> {
        // Careful: The order in here is *very* important, even if it may not seem so. Everything
        // is linked to the iteration number, and getting things mixed up may lead to unexpected
        // behavior.

        let prev_param = state.take_param().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`SimulatedAnnealing`: Parameter vector in state not set."
        ))?;
        let prev_cost = state.get_cost();

        // Make a move
        let new_param = problem.anneal(&prev_param, self.cur_temp)?;

        // Evaluate cost function with new parameter vector
        let new_cost = problem.cost(&new_param)?;

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
        let prob = float!(prob);
        let accepted = (new_cost < prev_cost)
            || (float!(1.0) / (float!(1.0) + ((new_cost - prev_cost) / self.cur_temp).exp())
                > prob);

        let new_best_found = new_cost < state.best_cost;

        // Update stall iter variables
        self.update_stall_and_reanneal_iter(accepted, new_best_found);

        let (r_fixed, r_accepted, r_best) = self.reanneal();

        // Update temperature for next iteration.
        self.temp_iter += 1;
        // Actually not necessary as it does the same as `temp_iter`, but I'll leave it here for
        // better readability.
        self.reanneal_iter_fixed += 1;

        self.update_temperature();

        Ok((
            if accepted {
                state.param(new_param).cost(new_cost)
            } else {
                state.param(prev_param).cost(prev_cost)
            },
            Some(kv!(
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
            )),
        ))
    }

    fn terminate(&mut self, _state: &IterState<P, (), (), (), F>) -> TerminationStatus {
        if self.stall_iter_accepted > self.stall_iter_accepted_limit {
            return TerminationStatus::Terminated(TerminationReason::SolverExit(
                "AcceptedStallIterExceeded".to_string(),
            ));
        }
        if self.stall_iter_best > self.stall_iter_best_limit {
            return TerminationStatus::Terminated(TerminationReason::SolverExit(
                "BestStallIterExceeded".to_string(),
            ));
        }
        TerminationStatus::NotTerminated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{test_utils::TestProblem, ArgminError, State};
    use crate::test_trait_impl;
    use approx::assert_relative_eq;

    test_trait_impl!(sa, SimulatedAnnealing<f64, StdRng>);

    #[test]
    fn test_new() {
        let sa: SimulatedAnnealing<f64, Xoshiro256PlusPlus> =
            SimulatedAnnealing::new(100.0).unwrap();
        let SimulatedAnnealing {
            init_temp,
            temp_func,
            temp_iter,
            stall_iter_accepted,
            stall_iter_accepted_limit,
            stall_iter_best,
            stall_iter_best_limit,
            reanneal_fixed,
            reanneal_iter_fixed,
            reanneal_accepted,
            reanneal_iter_accepted,
            reanneal_best,
            reanneal_iter_best,
            cur_temp,
            rng: _rng,
        } = sa;

        assert_eq!(init_temp.to_ne_bytes(), 100.0f64.to_ne_bytes());
        assert_eq!(temp_func, SATempFunc::TemperatureFast);
        assert_eq!(temp_iter, 0);
        assert_eq!(stall_iter_accepted, 0);
        assert_eq!(stall_iter_accepted_limit, u64::MAX);
        assert_eq!(stall_iter_best, 0);
        assert_eq!(stall_iter_best_limit, u64::MAX);
        assert_eq!(reanneal_fixed, u64::MAX);
        assert_eq!(reanneal_iter_fixed, 0);
        assert_eq!(reanneal_accepted, u64::MAX);
        assert_eq!(reanneal_iter_accepted, 0);
        assert_eq!(reanneal_best, u64::MAX);
        assert_eq!(reanneal_iter_best, 0);
        assert_eq!(cur_temp.to_ne_bytes(), 100.0f64.to_ne_bytes());

        for temp in [0.0, -1.0, -std::f64::EPSILON, -100.0] {
            let res = SimulatedAnnealing::new(temp);
            assert_error!(
                res,
                ArgminError,
                "Invalid parameter: \"`SimulatedAnnealing`: Initial temperature must be > 0.\""
            );
        }
    }

    #[test]
    fn test_new_with_rng() {
        #[derive(Eq, PartialEq, Debug)]
        struct MyRng {}

        let sa: SimulatedAnnealing<f64, MyRng> =
            SimulatedAnnealing::new_with_rng(100.0, MyRng {}).unwrap();
        let SimulatedAnnealing {
            init_temp,
            temp_func,
            temp_iter,
            stall_iter_accepted,
            stall_iter_accepted_limit,
            stall_iter_best,
            stall_iter_best_limit,
            reanneal_fixed,
            reanneal_iter_fixed,
            reanneal_accepted,
            reanneal_iter_accepted,
            reanneal_best,
            reanneal_iter_best,
            cur_temp,
            rng,
        } = sa;

        assert_eq!(init_temp.to_ne_bytes(), 100.0f64.to_ne_bytes());
        assert_eq!(temp_func, SATempFunc::TemperatureFast);
        assert_eq!(temp_iter, 0);
        assert_eq!(stall_iter_accepted, 0);
        assert_eq!(stall_iter_accepted_limit, u64::MAX);
        assert_eq!(stall_iter_best, 0);
        assert_eq!(stall_iter_best_limit, u64::MAX);
        assert_eq!(reanneal_fixed, u64::MAX);
        assert_eq!(reanneal_iter_fixed, 0);
        assert_eq!(reanneal_accepted, u64::MAX);
        assert_eq!(reanneal_iter_accepted, 0);
        assert_eq!(reanneal_best, u64::MAX);
        assert_eq!(reanneal_iter_best, 0);
        assert_eq!(cur_temp.to_ne_bytes(), 100.0f64.to_ne_bytes());
        // important part
        assert_eq!(rng, MyRng {});

        for temp in [0.0, -1.0, -std::f64::EPSILON, -100.0] {
            let res = SimulatedAnnealing::new_with_rng(temp, MyRng {});
            assert_error!(
                res,
                ArgminError,
                "Invalid parameter: \"`SimulatedAnnealing`: Initial temperature must be > 0.\""
            );
        }
    }

    #[test]
    fn test_with_temp_func() {
        for func in [
            SATempFunc::TemperatureFast,
            SATempFunc::Boltzmann,
            SATempFunc::Exponential(2.0),
        ] {
            let sa = SimulatedAnnealing::new(100.0f64).unwrap();
            let sa = sa.with_temp_func(func);

            assert_eq!(sa.temp_func, func);
        }
    }

    #[test]
    fn test_with_stall_accepted() {
        for iter in [0, 1, 5, 10, 100, 100000] {
            let sa = SimulatedAnnealing::new(100.0f64).unwrap();
            let sa = sa.with_stall_accepted(iter);

            assert_eq!(sa.stall_iter_accepted_limit, iter);
        }
    }

    #[test]
    fn test_with_stall_best() {
        for iter in [0, 1, 5, 10, 100, 100000] {
            let sa = SimulatedAnnealing::new(100.0f64).unwrap();
            let sa = sa.with_stall_best(iter);

            assert_eq!(sa.stall_iter_best_limit, iter);
        }
    }

    #[test]
    fn test_with_reannealing_fixed() {
        for iter in [0, 1, 5, 10, 100, 100000] {
            let sa = SimulatedAnnealing::new(100.0f64).unwrap();
            let sa = sa.with_reannealing_fixed(iter);

            assert_eq!(sa.reanneal_fixed, iter);
        }
    }

    #[test]
    fn test_with_reannealing_accepted() {
        for iter in [0, 1, 5, 10, 100, 100000] {
            let sa = SimulatedAnnealing::new(100.0f64).unwrap();
            let sa = sa.with_reannealing_accepted(iter);

            assert_eq!(sa.reanneal_accepted, iter);
        }
    }

    #[test]
    fn test_with_reannealing_best() {
        for iter in [0, 1, 5, 10, 100, 100000] {
            let sa = SimulatedAnnealing::new(100.0f64).unwrap();
            let sa = sa.with_reannealing_best(iter);

            assert_eq!(sa.reanneal_best, iter);
        }
    }

    #[test]
    fn test_update_temperature() {
        for (func, val) in [
            (SATempFunc::TemperatureFast, 100.0f64 / 2.0),
            (SATempFunc::Boltzmann, 100.0f64 / 2.0f64.ln()),
            (SATempFunc::Exponential(3.0), 100.0 * 3.0f64.powi(2)),
        ] {
            let mut sa = SimulatedAnnealing::new(100.0f64)
                .unwrap()
                .with_temp_func(func);
            sa.temp_iter = 1;

            sa.update_temperature();

            assert_relative_eq!(sa.cur_temp, val, epsilon = f64::EPSILON);
        }
    }

    #[test]
    fn test_reanneal() {
        let mut sa_t = SimulatedAnnealing::new(100.0f64).unwrap();

        sa_t.reanneal_fixed = 10;
        sa_t.reanneal_accepted = 20;
        sa_t.reanneal_best = 30;
        sa_t.temp_iter = 40;
        sa_t.cur_temp = 50.0;

        for ((f, a, b), expected) in [
            ((0, 0, 0), (false, false, false)),
            ((10, 0, 0), (true, false, false)),
            ((11, 0, 0), (true, false, false)),
            ((0, 20, 0), (false, true, false)),
            ((0, 21, 0), (false, true, false)),
            ((0, 0, 30), (false, false, true)),
            ((0, 0, 31), (false, false, true)),
            ((10, 20, 0), (true, true, false)),
            ((10, 0, 30), (true, false, true)),
            ((0, 20, 30), (false, true, true)),
            ((10, 20, 30), (true, true, true)),
        ] {
            let mut sa = sa_t.clone();

            sa.reanneal_iter_fixed = f;
            sa.reanneal_iter_accepted = a;
            sa.reanneal_iter_best = b;

            assert_eq!(sa.reanneal(), expected);

            if expected.0 || expected.1 || expected.2 {
                assert_eq!(sa.reanneal_iter_fixed, 0);
                assert_eq!(sa.reanneal_iter_accepted, 0);
                assert_eq!(sa.reanneal_iter_best, 0);
                assert_eq!(sa.temp_iter, 0);
                assert_eq!(sa.cur_temp.to_ne_bytes(), sa.init_temp.to_ne_bytes());
            }
        }
    }

    #[test]
    fn test_update_stall_and_reanneal_iter() {
        let mut sa_t = SimulatedAnnealing::new(100.0f64).unwrap();

        sa_t.stall_iter_accepted = 10;
        sa_t.reanneal_iter_accepted = 20;
        sa_t.stall_iter_best = 30;
        sa_t.reanneal_iter_best = 40;

        for ((a, b), (sia, ria, sib, rib)) in [
            ((false, false), (11, 21, 31, 41)),
            ((false, true), (11, 21, 0, 0)),
            ((true, false), (0, 0, 31, 41)),
            ((true, true), (0, 0, 0, 0)),
        ] {
            let mut sa = sa_t.clone();

            sa.update_stall_and_reanneal_iter(a, b);

            assert_eq!(sa.stall_iter_accepted, sia);
            assert_eq!(sa.reanneal_iter_accepted, ria);
            assert_eq!(sa.stall_iter_best, sib);
            assert_eq!(sa.reanneal_iter_best, rib);
        }
    }

    #[test]
    fn test_init() {
        let param: Vec<f64> = vec![-1.0, 1.0];

        let stall_iter_accepted_limit = 10;
        let stall_iter_best_limit = 20;
        let reanneal_fixed = 30;
        let reanneal_accepted = 40;
        let reanneal_best = 50;

        let mut sa = SimulatedAnnealing::new(100.0f64)
            .unwrap()
            .with_stall_accepted(stall_iter_accepted_limit)
            .with_stall_best(stall_iter_best_limit)
            .with_reannealing_fixed(reanneal_fixed)
            .with_reannealing_accepted(reanneal_accepted)
            .with_reannealing_best(reanneal_best);

        // Forgot to initialize the parameter vector
        let state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
        let problem = TestProblem::new();
        let res = sa.init(&mut Problem::new(problem), state);
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Not initialized: \"`SimulatedAnnealing` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method.\""
            )
        );

        // All good.
        let state: IterState<Vec<f64>, (), (), (), f64> = IterState::new().param(param.clone());
        let problem = TestProblem::new();
        let (mut state_out, kv) = sa.init(&mut Problem::new(problem), state).unwrap();

        let kv_expected = kv!(
            "initial_temperature" => 100.0f64;
            "stall_iter_accepted_limit" => stall_iter_accepted_limit;
            "stall_iter_best_limit" => stall_iter_best_limit;
            "reanneal_fixed" => reanneal_fixed;
            "reanneal_accepted" => reanneal_accepted;
            "reanneal_best" => reanneal_best;
        );

        assert_eq!(kv.unwrap(), kv_expected);

        let s_param = state_out.take_param().unwrap();

        for (s, p) in s_param.iter().zip(param.iter()) {
            assert_eq!(s.to_ne_bytes(), p.to_ne_bytes());
        }

        assert_eq!(state_out.get_cost().to_ne_bytes(), 1.0f64.to_ne_bytes())
    }
}
