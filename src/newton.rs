// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Newton method
//!
//! TODO

use std;
use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;
use errors::*;
use prelude::*;
use problem::Problem;
use result::ArgminResult;

/// Newton method struct (duh)
pub struct Newton<'a> {
    /// step size
    gamma: f64,
    /// Maximum number of iterations
    max_iters: u64,
    /// current state
    state: Option<NewtonState<'a>>,
}

/// Indicates the current state of the Newton method
struct NewtonState<'a> {
    /// Reference to the problem. This is an Option<_> because it is initialized as `None`
    problem: &'a Problem<'a, Array1<f64>, f64, Array2<f64>>,
    /// Current parameter vector
    param: Array1<f64>,
    /// Current number of iteration
    iter: u64,
}

impl<'a> NewtonState<'a> {
    /// Constructor for `NewtonState`
    pub fn new(
        problem: &'a Problem<'a, Array1<f64>, f64, Array2<f64>>,
        param: Array1<f64>,
    ) -> Self {
        NewtonState {
            problem: problem,
            param: param,
            iter: 0_u64,
        }
    }
}

impl<'a> Newton<'a> {
    /// Return a `Newton` struct
    pub fn new() -> Self {
        Newton {
            gamma: 1.0,
            max_iters: std::u64::MAX,
            state: None,
        }
    }

    /// Set maximum number of iterations
    pub fn max_iters(&mut self, max_iters: u64) -> &mut Self {
        self.max_iters = max_iters;
        self
    }
}

impl<'a> ArgminSolver<'a> for Newton<'a> {
    type Parameter = Array1<f64>;
    type CostValue = f64;
    type Hessian = Array2<f64>;
    type StartingPoints = Self::Parameter;
    type ProblemDefinition = Problem<'a, Self::Parameter, Self::CostValue, Self::Hessian>;

    /// Initialize with a given problem and a starting point
    fn init(
        &mut self,
        problem: &'a Self::ProblemDefinition,
        init_param: &Self::StartingPoints,
    ) -> Result<()> {
        self.state = Some(NewtonState::new(problem, init_param.clone()));
        Ok(())
    }

    /// Compute next point
    fn next_iter(&mut self) -> Result<ArgminResult<Self::Parameter, Self::CostValue>> {
        // TODO: Move to next point
        // x_{n+1} = x_n - \gamma [Hf(x_n)]^-1 \nabla f(x_n)
        let mut state = self.state.take().unwrap();
        let g = (state.problem.gradient.unwrap())(&state.param);
        let h_inv = (state.problem.hessian.unwrap())(&state.param).inv()?;
        state.param = state.param - self.gamma * h_inv.dot(&g);
        state.iter += 1;
        let out = ArgminResult::new(state.param.clone(), std::f64::NAN, state.iter);
        self.state = Some(state);
        Ok(out)
    }

    /// Indicates whether any of the stopping criteria are met
    fn terminate(&self) -> bool {
        self.state.as_ref().unwrap().iter >= self.max_iters
    }

    /// Run Newton method
    make_run!(
        Self::ProblemDefinition,
        Self::StartingPoints,
        Self::Parameter,
        Self::CostValue
    );
}

impl<'a> Default for Newton<'a> {
    fn default() -> Self {
        Self::new()
    }
}
