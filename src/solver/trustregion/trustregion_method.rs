// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # References:
//!
//! [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
//! Springer. ISBN 0-387-30303-0.

use crate::prelude::*;
use crate::solver::trustregion::reduction_ratio;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// The trust region method approximates the cost function within a certain region around the
/// current point in parameter space. Depending on the quality of this approximation, the region is
/// either expanded or contracted.
///
/// The calculation of the actual step length and direction is done by one of the following
/// methods:
///
/// * [Cauchy point](../cauchypoint/struct.CauchyPoint.html)
/// * [Dogleg method](../dogleg/struct.Dogleg.html)
/// * [Steihaug method](../steihaug/struct.Steihaug.html)
///
/// This subproblem can be set via `set_subproblem(...)`. If this is not provided, it will default
/// to the Steihaug method.
///
/// # Example
///
/// ```
/// extern crate argmin;
/// extern crate ndarray;
/// use argmin::prelude::*;
/// use argmin::solver::trustregion::{CauchyPoint, Dogleg, Steihaug, TrustRegion};
/// use argmin::testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative, rosenbrock_2d_hessian};
/// use ndarray::{Array, Array1, Array2};
/// # use serde::{Deserialize, Serialize};
///
/// # #[derive(Clone, Default, Serialize, Deserialize)]
/// # struct MyProblem {}
/// #
/// # impl ArgminOp for MyProblem {
/// #     type Param = Array1<f64>;
/// #     type Output = f64;
/// #     type Hessian = Array2<f64>;
/// #
/// #     fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
/// #         Ok(rosenbrock_2d(&p.to_vec(), 1.0, 100.0))
/// #     }
/// #
/// #     fn gradient(&self, p: &Self::Param) -> Result<Self::Param, Error> {
/// #         Ok(Array1::from_vec(rosenbrock_2d_derivative(
/// #             &p.to_vec(),
/// #             1.0,
/// #             100.0,
/// #         )))
/// #     }
/// #
/// #     fn hessian(&self, p: &Self::Param) -> Result<Self::Hessian, Error> {
/// #         let h = rosenbrock_2d_hessian(&p.to_vec(), 1.0, 100.0);
/// #         Ok(Array::from_shape_vec((2, 2), h)?)
/// #     }
/// # }
/// #
/// # fn run() -> Result<(), Error> {
/// // Define cost function
/// let cost = MyProblem {};
///
/// // Define inital parameter vector
/// // easy case
/// // let init_param: Array1<f64> = Array1::from_vec(vec![1.2, 1.2]);
/// // tough case
/// let init_param: Array1<f64> = Array1::from_vec(vec![-1.2, 1.0]);
///
/// // Set up the subproblem
/// let mut subproblem = Steihaug::new(cost.clone());
/// // let mut subproblem = CauchyPoint::new(cost.clone());
/// // let mut subproblem = Dogleg::new(cost.clone());
/// subproblem.set_max_iters(2);
///
/// // Set up the subproblem
/// let mut subproblem = Steihaug::new(cost.clone());
/// // let mut subproblem = CauchyPoint::new(cost.clone());
/// // let mut subproblem = Dogleg::new(cost.clone());
/// subproblem.set_max_iters(2);
///
/// // Set up solver
/// let mut solver = TrustRegion::new(cost, init_param, subproblem);
///
/// // Set the maximum number of iterations
/// solver.set_max_iters(2_000);
///
/// // Attach a logger
/// solver.add_logger(ArgminSlogLogger::term());
///
/// // Run solver
/// solver.run()?;
///
/// // Wait a second (lets the logger flush everything before printing again)
/// std::thread::sleep(std::time::Duration::from_secs(1));
///
/// // Print result
/// println!("{:?}", solver.result());
/// #     Ok(())
/// # }
/// #
/// # fn main() {
/// #     if let Err(ref e) = run() {
/// #         println!("{} {}", e.as_fail(), e.backtrace());
/// #         std::process::exit(1);
/// #     }
/// # }
/// ```
///
/// # References:
///
/// [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Serialize, Deserialize)]
pub struct TrustRegion<R> {
    /// Radius
    radius: f64,
    /// Maximum Radius
    max_radius: f64,
    /// eta \in [0, 1/4)
    eta: f64,
    /// subproblem
    subproblem: R,
    /// f(xk)
    fxk: f64,
    /// mk(0)
    mk0: f64,
}

impl<R> TrustRegion<R> where {
    /// Constructor
    ///
    /// Parameters:
    ///
    /// `operator`: operator
    pub fn new(subproblem: R) -> Self {
        TrustRegion {
            radius: 1.0,
            max_radius: 100.0,
            eta: 0.125,
            subproblem: subproblem,
            fxk: std::f64::NAN,
            mk0: std::f64::NAN,
        }
    }

    /// set radius
    pub fn radius(mut self, radius: f64) -> Self {
        self.radius = radius;
        self
    }

    /// Set maximum radius
    pub fn max_radius(mut self, max_radius: f64) -> Self {
        self.max_radius = max_radius;
        self
    }

    /// Set eta
    pub fn eta(mut self, eta: f64) -> Result<Self, Error> {
        if eta >= 0.25 || eta < 0.0 {
            return Err(ArgminError::InvalidParameter {
                text: "TrustRegion: eta must be in [0, 1/4).".to_string(),
            }
            .into());
        }
        self.eta = eta;
        Ok(self)
    }
}

impl<O, R> Solver<O> for TrustRegion<R>
where
    O: ArgminOp<Output = f64>,
    O::Param: Default
        + Clone
        + Debug
        + Serialize
        + ArgminMul<f64, O::Param>
        + ArgminWeightedDot<O::Param, f64, O::Hessian>
        + ArgminNorm<f64>
        + ArgminDot<O::Param, f64>
        + ArgminAdd<O::Param, O::Param>
        + ArgminSub<O::Param, O::Param>
        + ArgminZero
        + ArgminMul<f64, O::Param>,
    O::Hessian: Default + Clone + Debug + Serialize + ArgminDot<O::Param, O::Param>,
    R: ArgminTrustRegion + Solver<OpWrapper<O>>,
{
    fn init(
        &mut self,
        op: &mut OpWrapper<O>,
        state: IterState<O::Param, O::Hessian>,
    ) -> Result<Option<ArgminIterData<O>>, Error> {
        let grad = op.gradient(&state.cur_param)?;
        let hessian = op.hessian(&state.cur_param)?;
        self.fxk = op.apply(&state.cur_param)?;
        self.mk0 = self.fxk;
        Ok(Some(
            ArgminIterData::new()
                .param(state.cur_param)
                .cost(self.fxk)
                .grad(grad)
                .hessian(hessian),
        ))
    }

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        state: IterState<O::Param, O::Hessian>,
    ) -> Result<ArgminIterData<O>, Error> {
        self.subproblem.set_radius(self.radius);

        let pk = Executor::new(op.clone(), self.subproblem.clone(), state.cur_param.clone())
            .grad(state.cur_grad.clone())
            .hessian(state.cur_hessian.clone())
            .run_fast()?
            .param;

        let new_param = pk.add(&state.cur_param);
        let fxkpk = op.apply(&new_param)?;
        let mkpk =
            self.fxk + pk.dot(&state.cur_grad) + 0.5 * pk.weighted_dot(&state.cur_hessian, &pk);

        let rho = reduction_ratio(self.fxk, fxkpk, self.mk0, mkpk);

        let pk_norm = pk.norm();

        let cur_radius = self.radius;
        self.radius = if rho < 0.25 {
            0.25 * pk_norm
        } else if rho > 0.75 && (pk_norm - self.radius).abs() <= 10.0 * std::f64::EPSILON {
            self.max_radius.min(2.0 * self.radius)
        } else {
            self.radius
        };

        let out = if rho > self.eta {
            self.fxk = fxkpk;
            self.mk0 = fxkpk;
            let grad = op.gradient(&new_param)?;
            let hessian = op.hessian(&new_param)?;
            ArgminIterData::new()
                .param(new_param)
                .cost(fxkpk)
                .grad(grad)
                .hessian(hessian)
        } else {
            ArgminIterData::new().param(state.cur_param).cost(self.fxk)
        }
        .kv(make_kv!("radius" => cur_radius;));

        Ok(out)
    }

    fn terminate(&mut self, _state: &IterState<O::Param, O::Hessian>) -> TerminationReason {
        // todo
        TerminationReason::NotTerminated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::send_sync_test;
    use crate::solver::trustregion::steihaug::Steihaug;

    type Operator = MinimalNoOperator;

    send_sync_test!(trustregion, TrustRegion<Operator, Steihaug<Operator>>);
}
