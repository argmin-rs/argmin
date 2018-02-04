// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Gradient Descent
//!
//! TODO

use std;
use ndarray::Array1;
use errors::*;
use prelude::*;
use problem::ArgminProblem;
use result::ArgminResult;
use backtracking::BacktrackingLineSearch;

/// Gradient Descent gamma update method
pub enum GDGammaUpdate<'a> {
    /// Constant gamma
    Constant(f64),
    /// Gamma updated according to TODO
    /// Apparently this only works if the cost function is convex and the derivative of the cost
    /// function is Lipschitz.
    /// TODO: More detailed description (formula)
    BarzilaiBorwein,
    /// Backtracking line search
    BacktrackingLineSearch(BacktrackingLineSearch<'a>),
}

/// Gradient Descent struct (duh)
pub struct GradientDescent<'a> {
    /// step size
    gamma: GDGammaUpdate<'a>,
    /// Maximum number of iterations
    max_iters: u64,
    /// Precision
    precision: f64,
    /// current state
    state: GradientDescentState<'a>,
}

/// Indicates the current state of the Gradient Descent method.
struct GradientDescentState<'a> {
    /// Reference to the problem. This is an Option<_> because it is initialized as `None`
    problem: Option<&'a ArgminProblem<'a, Array1<f64>, f64, Array1<f64>>>,
    /// Previous parameter vector
    prev_param: Array1<f64>,
    /// Current parameter vector
    param: Array1<f64>,
    /// Current number of iteration
    iter: u64,
    /// Previous gamma
    prev_gamma: f64,
    /// Current gamma
    gamma: f64,
    /// Previous gradient
    prev_grad: Array1<f64>,
    /// Current gradient
    cur_grad: Array1<f64>,
}

impl<'a> GradientDescentState<'a> {
    /// Constructor for `GradientDescentState`
    pub fn new() -> Self {
        GradientDescentState {
            problem: None,
            prev_param: Array1::from_vec(vec![0_f64; 1]),
            param: Array1::from_vec(vec![0_f64; 1]),
            iter: 0_u64,
            prev_gamma: 0_f64,
            gamma: 0_f64,
            prev_grad: Array1::from_vec(vec![0_f64; 1]),
            cur_grad: Array1::from_vec(vec![0_f64; 1]),
        }
    }
}

impl<'a> GradientDescent<'a> {
    /// Return a GradientDescent struct
    pub fn new() -> Self {
        GradientDescent {
            gamma: GDGammaUpdate::BarzilaiBorwein,
            max_iters: std::u64::MAX,
            precision: 0.00000001,
            state: GradientDescentState::new(),
        }
    }

    /// Set gradient descent gamma update method
    pub fn gamma_update(&mut self, gamma_update_method: GDGammaUpdate<'a>) -> &mut Self {
        self.gamma = gamma_update_method;
        self
    }

    /// Set maximum number of iterations
    pub fn max_iters(&mut self, max_iters: u64) -> &mut Self {
        self.max_iters = max_iters;
        self
    }

    /// Set precision
    pub fn precision(&mut self, precision: f64) -> &mut Self {
        self.precision = precision;
        self
    }

    /// Update gamma
    fn update_gamma(&mut self) {
        self.state.prev_gamma = self.state.gamma;
        self.state.gamma = match self.gamma {
            GDGammaUpdate::Constant(g) => g,
            GDGammaUpdate::BarzilaiBorwein => {
                let mut grad_diff: f64;
                let mut top: f64 = 0.0;
                let mut bottom: f64 = 0.0;
                for idx in 0..self.state.cur_grad.len() {
                    grad_diff = self.state.cur_grad[idx] - self.state.prev_grad[idx];
                    top += (self.state.param[idx] - self.state.prev_param[idx]) * grad_diff;
                    bottom += grad_diff.powf(2.0);
                }
                top / bottom
            }
            GDGammaUpdate::BacktrackingLineSearch(ref bls) => {
                let result = bls.run(&(-self.state.cur_grad.clone()), &self.state.param)
                    .unwrap();
                result.0
            }
        };
    }
}

impl<'a> ArgminSolver<'a> for GradientDescent<'a> {
    type Parameter = Array1<f64>;
    type CostValue = f64;
    type Hessian = Array1<f64>;
    type StartingPoints = Array1<f64>;
    type ProblemDefinition = ArgminProblem<'a, Self::Parameter, Self::CostValue, Self::Hessian>;

    /// Initialize with a given problem and a starting point
    fn init(
        &mut self,
        problem: &'a Self::ProblemDefinition,
        init_param: &Self::StartingPoints,
    ) -> Result<()> {
        self.state = GradientDescentState {
            problem: Some(problem),
            prev_param: Array1::from_vec(vec![0_f64; init_param.len()]),
            param: init_param.to_owned(),
            iter: 0_u64,
            prev_gamma: 0_f64,
            gamma: match self.gamma {
                GDGammaUpdate::Constant(g) => g,
                GDGammaUpdate::BarzilaiBorwein | GDGammaUpdate::BacktrackingLineSearch(_) => 0.0001,
            },
            prev_grad: Array1::from_vec(vec![0_f64; init_param.len()]),
            cur_grad: (problem.gradient.unwrap())(&init_param.to_owned()),
        };
        Ok(())
    }

    /// Compute next point
    fn next_iter(&mut self) -> Result<ArgminResult<Self::Parameter, Self::CostValue>> {
        let gradient = self.state.problem.unwrap().gradient.unwrap();
        // let state = &mut self.state;
        self.state.prev_param = self.state.param.clone();
        self.state.prev_grad = self.state.cur_grad.clone();

        // Move to next point
        for i in 0..self.state.param.len() {
            self.state.param[i] -= self.state.cur_grad[i] * self.state.gamma;
        }

        // Calculate next gradient
        self.state.cur_grad = (gradient)(&self.state.param);

        // Update gamma
        self.update_gamma();
        self.state.iter += 1;
        Ok(ArgminResult::new(
            self.state.param.clone(),
            std::f64::NAN,
            self.state.iter,
        ))
    }

    /// Indicates whether any of the stopping criteria are met
    fn terminate(&self) -> bool {
        // use zip here...
        let prev_step_size = ((self.state.param[0] - self.state.prev_param[0]).powf(2.0)
            + (self.state.param[1] - self.state.prev_param[1]).powf(2.0))
            .sqrt();

        if prev_step_size < self.precision {
            return true;
        }
        if self.state.iter >= self.max_iters {
            return true;
        }
        false
    }

    /// Run gradient descent method
    make_run!(
        Self::ProblemDefinition,
        Self::StartingPoints,
        Self::Parameter,
        Self::CostValue
    );
}

impl<'a> Default for GradientDescent<'a> {
    fn default() -> Self {
        Self::new()
    }
}
