// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Landweber algorithm
//!
//! TODO

use std;
use ndarray::{Array1, Array2};
use errors::*;
use prelude::*;
use operator::ArgminOperator;
use result::ArgminResult;
use termination::TerminationReason;

/// Landweber algorithm struct (duh)
pub struct Landweber<'a> {
    /// relaxation factor
    /// must satisfy 0 < omega < 2/sigma_1^2 where sigma_1 is the largest singular value of the
    /// matrix.
    omega: f64,
    /// Maximum number of iterations
    max_iters: u64,
    /// current state
    state: Option<LandweberState<'a>>,
}

/// Indicates the current state of the Landweber algorithm
struct LandweberState<'a> {
    /// Reference to the problem. This is an Option<_> because it is initialized as `None`
    operator: &'a ArgminOperator<'a>,
    /// Current parameter vector
    param: Array1<f64>,
    /// Current number of iteration
    iter: u64,
    /// Current l2 norm of difference
    norm: f64,
}

impl<'a> LandweberState<'a> {
    /// Constructor for `LandweberState`
    pub fn new(operator: &'a ArgminOperator<'a>, param: Array1<f64>) -> Self {
        LandweberState {
            operator: operator,
            param: param,
            iter: 0_u64,
            norm: std::f64::NAN,
        }
    }
}

impl<'a> Landweber<'a> {
    /// Return a `Landweber` struct
    pub fn new(omega: f64) -> Self {
        Landweber {
            omega: omega,
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

impl<'a> ArgminSolver<'a> for Landweber<'a> {
    type Parameter = Array1<f64>;
    type CostValue = f64;
    type Hessian = Array2<f64>;
    type StartingPoints = Self::Parameter;
    type ProblemDefinition = ArgminOperator<'a>;

    /// Initialize with a given problem and a starting point
    fn init(
        &mut self,
        operator: &'a Self::ProblemDefinition,
        init_param: &Self::StartingPoints,
    ) -> Result<()> {
        self.state = Some(LandweberState::new(operator, init_param.clone()));
        Ok(())
    }

    /// Compute next point
    fn next_iter(&mut self) -> Result<ArgminResult<Self::Parameter, Self::CostValue>> {
        let mut state = self.state.take().unwrap();
        let prev_param = state.param.clone();
        let diff = state.operator.apply(&prev_param) - state.operator.y;
        state.param = state.param - self.omega * state.operator.apply_transpose(&diff);
        state.iter += 1;
        state.norm = diff.iter().map(|a| a.powf(2.0)).sum::<f64>().sqrt();
        let mut out = ArgminResult::new(state.param.clone(), state.norm, state.iter);
        self.state = Some(state);
        out.set_termination_reason(self.terminate());
        Ok(out)
    }

    /// Indicates whether any of the stopping criteria are met
    fn terminate(&self) -> TerminationReason {
        if self.state.as_ref().unwrap().iter >= self.max_iters {
            return TerminationReason::MaxItersReached;
        }
        if self.state.as_ref().unwrap().norm < self.state.as_ref().unwrap().operator.target_cost {
            return TerminationReason::TargetCostReached;
        }
        TerminationReason::NotTerminated
    }

    /// Run Landweber method
    make_run!(
        Self::ProblemDefinition,
        Self::StartingPoints,
        Self::Parameter,
        Self::CostValue
    );
}

impl<'a> Default for Landweber<'a> {
    fn default() -> Self {
        Self::new(1.0)
    }
}
