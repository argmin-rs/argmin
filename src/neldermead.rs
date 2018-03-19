// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Nelder-Mead method
//!
//! The Nelder-Mead method a heuristic search method for nonlinear optimization problems which does
//! not require derivatives.
//!
//! The method is based on simplices which consist of n+1 vertices for an optimization problem with
//! n dimensions.
//! The function to be optimized is evaluated at all vertices. Based on these cost function values
//! the behaviour of the cost function is extrapolated in order to find the next point to be
//! evaluated.
//!
//! The following actions are possible:
//!
//! 1) Reflection: (Parameter `alpha`, default `1`)
//! 2) Expansion: (Parameter `gamma`, default `2`)
//! 3) Contraction: (Parameter `rho`, default `0.5`)
//! 4) Shrink: (Parameter `sigma`, default `0.5`)
//!
//! TODO: More information as soon as rustdoc allows math.
//!
//! The initial simplex needs to be chosen carefully.
//!
//! # Example
//! ```rust
//! extern crate argmin;
//! use argmin::prelude::*;
//! use argmin::{ArgminProblem, NelderMead};
//! use argmin::testfunctions::rosenbrock;
//!
//! // Define cost function
//! let cost = |x: &Vec<f64>| -> f64 { rosenbrock(x, 1_f64, 100_f64) };
//!
//! // Set up problem
//! let mut prob = ArgminProblem::new(&cost);
//! prob.target_cost(0.01);
//!
//! // Set up GradientDecent solver
//! let mut solver = NelderMead::new();
//! solver.max_iters(100);
//!
//! // Choose the starting points.
//! let init_params = vec![vec![0.0, 0.1], vec![2.0, 1.5], vec![2.0, -1.0]];
//!
//! let result = solver.run(&prob, &init_params).unwrap();
//!
//! println!("{:?}", result);
//! ```

use std;
use errors::*;
use prelude::*;
use problem::ArgminProblem;
use result::ArgminResult;
use termination::TerminationReason;

/// Nelder Mead method
pub struct NelderMead<'a> {
    /// Maximum number of iterations
    max_iters: u64,
    /// alpha
    alpha: f64,
    /// gamma
    gamma: f64,
    /// rho
    rho: f64,
    /// sigma
    sigma: f64,
    /// current state
    state: Option<NelderMeadState<'a>>,
}

#[derive(Clone, Debug)]
struct NelderMeadParam {
    param: Vec<f64>,
    cost: f64,
}

struct NelderMeadState<'a> {
    problem: &'a ArgminProblem<'a, Vec<f64>, f64, Vec<f64>>,
    param_vecs: Vec<NelderMeadParam>,
    iter: u64,
}

impl<'a> NelderMeadState<'a> {
    /// Constructor
    pub fn new(
        problem: &'a ArgminProblem<'a, Vec<f64>, f64, Vec<f64>>,
        param_vecs: Vec<NelderMeadParam>,
    ) -> Self {
        NelderMeadState {
            problem,
            param_vecs,
            iter: 0_u64,
        }
    }
}

impl<'a> NelderMead<'a> {
    /// Return a GradientDescent struct
    pub fn new() -> Self {
        NelderMead {
            max_iters: std::u64::MAX,
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
            state: None,
        }
    }

    /// Set maximum number of iterations
    pub fn max_iters(&mut self, max_iters: u64) -> &mut Self {
        self.max_iters = max_iters;
        self
    }

    /// alpha
    pub fn alpha(&mut self, alpha: f64) -> &mut Self {
        self.alpha = alpha;
        self
    }

    /// gamma
    pub fn gamma(&mut self, gamma: f64) -> &mut Self {
        self.gamma = gamma;
        self
    }

    /// rho
    pub fn rho(&mut self, rho: f64) -> &mut Self {
        self.rho = rho;
        self
    }

    /// sigma
    pub fn sigma(&mut self, sigma: f64) -> &mut Self {
        self.sigma = sigma;
        self
    }

    fn sort_param_vecs(&mut self, state: &mut NelderMeadState) {
        state.param_vecs.sort_by(|a, b| {
            a.cost
                .partial_cmp(&b.cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Calculate centroid of all but the worst vectors
    fn calculate_centroid(&self, state: &NelderMeadState) -> Vec<f64> {
        let num_param = state.param_vecs.len() - 1;
        let mut x0: Vec<f64> = state.param_vecs[0].clone().param;
        for idx in 1..num_param {
            x0 = x0.iter()
                .zip(state.param_vecs[idx].param.iter())
                .map(|(a, b)| a + b)
                .collect();
        }
        x0.iter().map(|a| a / (num_param as f64)).collect()
    }

    fn reflect(&self, x0: &[f64], x: &[f64]) -> Vec<f64> {
        x0.iter()
            .zip(x.iter())
            .map(|(a, b)| a + self.alpha * (a - b))
            .collect()
    }

    fn expand(&self, x0: &[f64], x: &[f64]) -> Vec<f64> {
        x0.iter()
            .zip(x.iter())
            .map(|(a, b)| a + self.gamma * (b - a))
            .collect()
    }

    fn contract(&self, x0: &[f64], x: &[f64]) -> Vec<f64> {
        x0.iter()
            .zip(x.iter())
            .map(|(a, b)| a + self.rho * (b - a))
            .collect()
    }

    fn shrink(&mut self, state: &mut NelderMeadState) {
        for idx in 1..state.param_vecs.len() {
            state.param_vecs[idx].param = state
                .param_vecs
                .first()
                .unwrap()
                .param
                .iter()
                .zip(state.param_vecs[idx].param.iter())
                .map(|(a, b)| a + self.sigma * (b - a))
                .collect();
            state.param_vecs[idx].cost =
                (state.problem.cost_function)(&state.param_vecs[idx].param);
        }
    }
}

impl<'a> ArgminSolver<'a> for NelderMead<'a> {
    type Parameter = Vec<f64>;
    type CostValue = f64;
    type Hessian = Vec<f64>;
    type StartingPoints = Vec<Self::Parameter>;
    type ProblemDefinition = &'a ArgminProblem<'a, Self::Parameter, Self::CostValue, Self::Hessian>;

    /// initialization with predefined parameter vectors
    fn init(
        &mut self,
        problem: Self::ProblemDefinition,
        param_vecs: &Self::StartingPoints,
    ) -> Result<()> {
        let mut params: Vec<NelderMeadParam> = vec![];
        for param in param_vecs.iter() {
            params.push(NelderMeadParam {
                param: param.to_vec(),
                cost: (problem.cost_function)(param),
            });
        }
        let mut state = NelderMeadState::new(problem, params);
        self.sort_param_vecs(&mut state);
        self.state = Some(state);
        Ok(())
    }

    /// Compute next iteration
    fn next_iter(&mut self) -> Result<ArgminResult<Self::Parameter, Self::CostValue>> {
        let mut state = self.state.take().unwrap();
        self.sort_param_vecs(&mut state);
        let num_param = state.param_vecs[0].param.len();
        let x0 = self.calculate_centroid(&state);
        let xr = self.reflect(&x0, &state.param_vecs.last().unwrap().param);
        let xr_cost = (state.problem.cost_function)(&xr);
        if xr_cost < state.param_vecs[num_param - 2].cost && xr_cost >= state.param_vecs[0].cost {
            // reflection
            state.param_vecs.last_mut().unwrap().param = xr;
            state.param_vecs.last_mut().unwrap().cost = xr_cost;
        } else if xr_cost < state.param_vecs[0].cost {
            // expansion
            let xe = self.expand(&x0, &xr);
            let xe_cost = (state.problem.cost_function)(&xe);
            if xe_cost < xr_cost {
                state.param_vecs.last_mut().unwrap().param = xe;
                state.param_vecs.last_mut().unwrap().cost = xe_cost;
            } else {
                state.param_vecs.last_mut().unwrap().param = xr;
                state.param_vecs.last_mut().unwrap().cost = xr_cost;
            }
        } else if xr_cost >= state.param_vecs[num_param - 2].cost {
            // contraction
            let xc = self.contract(&x0, &state.param_vecs.last().unwrap().param);
            let xc_cost = (state.problem.cost_function)(&xc);
            if xc_cost < state.param_vecs.last().unwrap().cost {
                state.param_vecs.last_mut().unwrap().param = xc;
                state.param_vecs.last_mut().unwrap().cost = xc_cost;
            }
        } else {
            self.shrink(&mut state)
        }

        state.iter += 1;

        self.sort_param_vecs(&mut state);
        let param = state.param_vecs[0].clone();
        let mut out = ArgminResult::new(param.param, param.cost, state.iter);
        self.state = Some(state);
        out.set_termination_reason(self.terminate());
        Ok(out)
    }

    /// Stopping criterions
    make_terminate!(self,
        self.state.as_ref().unwrap().iter >= self.max_iters, TerminationReason::MaxItersReached;
        self.state.as_ref().unwrap().param_vecs[0].cost <= self.state.as_ref().unwrap().problem.target_cost, TerminationReason::TargetCostReached;
    );

    /// Run Nelder Mead optimization
    make_run!(
        Self::ProblemDefinition,
        Self::StartingPoints,
        Self::Parameter,
        Self::CostValue
    );
}

impl<'a> Default for NelderMead<'a> {
    fn default() -> Self {
        Self::new()
    }
}
