// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

/// Conjugate Gradient method
///
/// TODO
use std;
use ndarray::{Array1, Array2};
use errors::*;
use operator::ArgminOperator;
use result::ArgminResult;
use ArgminSolver;

/// Conjugate Gradient method struct (duh)
pub struct ConjugateGradient<'a> {
    /// Maximum number of iterations
    max_iters: u64,
    /// current state
    state: Option<ConjugateGradientState<'a>>,
}

/// Indicates the current state of the Conjugate Gradient algorithm
struct ConjugateGradientState<'a> {
    /// Reference to the operator
    operator: &'a ArgminOperator<'a>,
    /// Current parameter vector
    param: Array1<f64>,
    /// Conjugate vector
    p: Array1<f64>,
    /// Residual
    r: Array1<f64>,
    /// Current number of iteration
    iter: u64,
}

impl<'a> ConjugateGradientState<'a> {
    /// Constructor for `ConjugateGradientState`
    pub fn new(
        operator: &'a ArgminOperator<'a>,
        param: Array1<f64>,
        p: Array1<f64>,
        r: Array1<f64>,
    ) -> Self {
        ConjugateGradientState {
            operator: operator,
            param: param,
            p: p,
            r: r,
            iter: 0_u64,
        }
    }
}

impl<'a> ConjugateGradient<'a> {
    /// Return a `ConjugateGradient` struct
    pub fn new() -> Self {
        ConjugateGradient {
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

impl<'a> ArgminSolver<'a> for ConjugateGradient<'a> {
    type A = Array1<f64>;
    type B = f64;
    type C = Array2<f64>;
    type D = Array1<f64>;
    type E = ArgminOperator<'a>;

    /// Initialize with a given problem and a starting point
    fn init(&mut self, operator: &'a Self::E, init_param: &Array1<f64>) -> Result<()> {
        let mut r = operator.y - &operator.apply(init_param);
        if !operator.operator.is_square() {
            r = operator.apply_transpose(&r);
        }
        let p = r.clone();
        self.state = Some(ConjugateGradientState::new(
            operator,
            init_param.clone(),
            r,
            p,
        ));
        Ok(())
    }

    /// Compute next point
    fn next_iter(&mut self) -> Result<ArgminResult<Array1<f64>, f64>> {
        let mut state = self.state.take().unwrap();
        let mut ap = state.operator.apply(&state.p);
        if !state.operator.operator.is_square() {
            ap = state.operator.apply_transpose(&ap);
        }
        let rtr = state.r.iter().map(|a| a.powf(2.0)).sum::<f64>();
        let alpha: f64 = rtr
            / state
                .p
                .iter()
                .zip(ap.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>();
        state.param = state.param + alpha * &state.p;
        state.r = state.r - alpha * &ap;
        let beta: f64 = state.r.iter().map(|a| a.powf(2.0)).sum::<f64>() / rtr;
        state.p = beta * &state.p + &state.r;
        state.iter += 1;
        let norm: f64 = state.r.iter().map(|a| a.powf(2.0)).sum::<f64>().sqrt();
        let out = ArgminResult::new(state.param.clone(), norm, state.iter);
        self.state = Some(state);
        Ok(out)
    }

    /// Indicates whether any of the stopping criteria are met
    fn terminate(&self) -> bool {
        self.state.as_ref().unwrap().iter >= self.max_iters
    }

    /// Run Conjugate Gradient method
    fn run(
        &mut self,
        operator: &'a Self::E,
        init_param: &Array1<f64>,
    ) -> Result<ArgminResult<Array1<f64>, f64>> {
        // initialize
        self.init(operator, init_param)?;

        let mut res;
        loop {
            res = self.next_iter()?;
            if self.terminate() {
                break;
            }
        }
        Ok(res)
    }
}

impl<'a> Default for ConjugateGradient<'a> {
    fn default() -> Self {
        Self::new()
    }
}
