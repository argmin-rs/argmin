// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

/// Backtracking Line Search
///
/// TODO
use errors::*;
use ndarray::Array1;

/// Reasons why it stopped -- shouldnt be here probably
#[derive(Debug)]
pub enum TerminationReason {
    /// Maximum number of iterations reached
    MaxNumberIterations,
    /// It converged before reaching the maximum number of iterations.
    Converged,
    /// dont know
    Unkown,
}

/// Backtracking Line Search
pub struct BacktrackingLineSearch<'a> {
    /// Reference to cost function.
    cost_function: &'a Fn(&Array1<f64>) -> f64,
    /// Gradient
    gradient: &'a Fn(&Array1<f64>) -> Array1<f64>,
    /// Starting distance to the current point:
    alpha: f64,
    /// Maximum number of iterations
    max_iters: u64,
    /// Parameter `tau`
    tau: f64,
    /// Parameter `c`
    c: f64,
}

impl<'a> BacktrackingLineSearch<'a> {
    /// Initialize Backtracking Line Search
    ///
    /// Requires the cost function and gradient to be passed as parameter. The parameters
    /// `max_iters`, `tau`, and `c` are set to 100, 0.5 and 0.5, respectively.
    pub fn new(
        cost_function: &'a Fn(&Array1<f64>) -> f64,
        gradient: &'a Fn(&Array1<f64>) -> Array1<f64>,
    ) -> Self {
        BacktrackingLineSearch {
            cost_function: cost_function,
            gradient: gradient,
            alpha: 1.0,
            max_iters: 100,
            tau: 0.5,
            c: 0.5,
        }
    }

    /// Set the maximum distance from the starting point
    pub fn alpha(&mut self, alpha: f64) -> &mut Self {
        self.alpha = alpha;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iters(&mut self, max_iters: u64) -> &mut Self {
        self.max_iters = max_iters;
        self
    }

    /// Set c to a desired value between 0 and 1
    pub fn c(&mut self, c: f64) -> Result<&mut Self> {
        if c >= 1.0 || c <= 0.0 {
            return Err(ErrorKind::InvalidParameter(
                "BacktrackingLineSearch: Parameter `c` must satisfy 0 < c < 1.".into(),
            ).into());
        }
        self.c = c;
        Ok(self)
    }

    /// Set tau to a desired value between 0 and 1
    pub fn tau(&mut self, tau: f64) -> Result<&mut Self> {
        if tau >= 1.0 || tau <= 0.0 {
            return Err(ErrorKind::InvalidParameter(
                "BacktrackingLineSearch: Parameter `tau` must satisfy 0 < tau < 1.".into(),
            ).into());
        }
        self.tau = tau;
        Ok(self)
    }

    /// Run backtracking line search
    ///
    /// `p` is the search direction. Take care about whether you need it to be a unit vector or
    /// not! `p` will not be normalized!
    pub fn run(&self, p: &Array1<f64>, x: &Array1<f64>) -> Result<(f64, u64, TerminationReason)> {
        // compute m
        let m: f64 = p.t().dot(&((self.gradient)(x)));

        let t = -self.c * m;
        let fx = (self.cost_function)(x);
        let termination_reason;
        let mut idx = 0;
        let mut alpha = self.alpha;
        loop {
            let param = x + &(alpha * p);
            if fx - (self.cost_function)(&param) >= alpha * t {
                termination_reason = TerminationReason::Converged;
                break;
            }
            if idx > self.max_iters {
                termination_reason = TerminationReason::MaxNumberIterations;
                break;
            }
            idx += 1;
            alpha *= self.tau;
        }
        Ok((alpha, idx, termination_reason))
    }
}
