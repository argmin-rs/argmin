// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

/// Newton method
///
/// TODO
use std;
use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;
use errors::*;
use problem::Problem;
use result::ArgminResult;
use ArgminSolver;

/// Newton method struct (duh)
pub struct Newton<'a> {
    /// step size
    gamma: f64,
    /// Maximum number of iterations
    max_iters: u64,
    /// current state
    state: NewtonState<'a>,
}

/// Indicates the current state of the Newton method
struct NewtonState<'a> {
    /// Reference to the problem. This is an Option<_> because it is initialized as `None`
    problem: Option<&'a Problem<'a, Array1<f64>, f64, Array2<f64>>>,
    /// Current parameter vector
    param: Array1<f64>,
    /// Current number of iteration
    iter: u64,
}

impl<'a> NewtonState<'a> {
    /// Constructor for `NewtonState`
    pub fn new() -> Self {
        NewtonState {
            problem: None,
            param: Array1::default(1),
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
            state: NewtonState::new(),
        }
    }

    /// Set maximum number of iterations
    pub fn max_iters(&mut self, max_iters: u64) -> &mut Self {
        self.max_iters = max_iters;
        self
    }
}

impl<'a> ArgminSolver<'a> for Newton<'a> {
    type A = Array1<f64>;
    type B = f64;
    type C = Array2<f64>;
    type D = Array1<f64>;
    type E = Problem<'a, Array1<f64>, f64, Array2<f64>>;

    /// Initialize with a given problem and a starting point
    fn init(&mut self, problem: &'a Self::E, init_param: &Array1<f64>) -> Result<()> {
        self.state = NewtonState {
            problem: Some(problem),
            param: init_param.to_owned(),
            iter: 0_u64,
        };
        Ok(())
    }

    /// Compute next point
    fn next_iter(&mut self) -> Result<ArgminResult<Array1<f64>, f64>> {
        // TODO: Move to next point
        // x_{n+1} = x_n - \gamma [Hf(x_n)]^-1 \nabla f(x_n)
        let g = (self.state.problem.unwrap().gradient.unwrap())(&self.state.param);
        let h_inv = (self.state.problem.unwrap().hessian.unwrap())(&self.state.param).inv()?;
        self.state.param = self.state.param.clone() - self.gamma * h_inv.dot(&g);
        self.state.iter += 1;
        Ok(ArgminResult::new(
            self.state.param.clone(),
            -1.0,
            self.state.iter,
        ))
    }

    /// Indicates whether any of the stopping criteria are met
    fn terminate(&self) -> bool {
        false
    }

    /// Run Newton method
    fn run(
        &mut self,
        problem: &'a Self::E,
        init_param: &Array1<f64>,
    ) -> Result<ArgminResult<Array1<f64>, f64>> {
        // initialize
        self.init(problem, init_param)?;

        let mut out;
        loop {
            out = self.next_iter()?;
            if self.terminate() {
                break;
            }
        }
        let fin_cost = (problem.cost_function)(&self.state.param);
        out.cost = fin_cost;
        Ok(out)
    }
}

impl<'a> Default for Newton<'a> {
    fn default() -> Self {
        Self::new()
    }
}
