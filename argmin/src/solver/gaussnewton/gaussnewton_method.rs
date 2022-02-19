// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # References:
//!
//! \[0\] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
//! Springer. ISBN 0-387-30303-0.

use crate::core::{
    ArgminError, ArgminFloat, ArgminIterData, ArgminOp, Error, IterState, OpWrapper, Solver, State,
    TerminationReason,
};
use argmin_math::{ArgminDot, ArgminInv, ArgminMul, ArgminNorm, ArgminSub, ArgminTranspose};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// Gauss-Newton method
///
/// # References:
///
/// \[0\] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct GaussNewton<F> {
    /// gamma
    gamma: F,
    /// Tolerance for the stopping criterion based on cost difference
    tol: F,
}

impl<F: ArgminFloat> GaussNewton<F> {
    /// Constructor
    pub fn new() -> Self {
        GaussNewton {
            gamma: F::from_f64(1.0).unwrap(),
            tol: F::epsilon().sqrt(),
        }
    }

    /// set gamma
    pub fn with_gamma(mut self, gamma: F) -> Result<Self, Error> {
        if gamma <= F::from_f64(0.0).unwrap() || gamma > F::from_f64(1.0).unwrap() {
            return Err(ArgminError::InvalidParameter {
                text: "Gauss-Newton: gamma must be in  (0, 1].".to_string(),
            }
            .into());
        }
        self.gamma = gamma;
        Ok(self)
    }

    /// Set tolerance for the stopping criterion based on cost difference
    pub fn with_tol(mut self, tol: F) -> Result<Self, Error> {
        if tol <= F::from_f64(0.0).unwrap() {
            return Err(ArgminError::InvalidParameter {
                text: "Gauss-Newton: tol must be positive.".to_string(),
            }
            .into());
        }
        self.tol = tol;
        Ok(self)
    }
}

impl<F: ArgminFloat> Default for GaussNewton<F> {
    fn default() -> GaussNewton<F> {
        GaussNewton::new()
    }
}

impl<O, F> Solver<IterState<O>> for GaussNewton<F>
where
    O: ArgminOp<Float = F>,
    O::Param: ArgminSub<O::Param, O::Param> + ArgminMul<O::Float, O::Param>,
    O::Output: ArgminNorm<O::Float>,
    O::Jacobian: ArgminTranspose<O::Jacobian>
        + ArgminInv<O::Jacobian>
        + ArgminDot<O::Jacobian, O::Jacobian>
        + ArgminDot<O::Output, O::Param>
        + ArgminDot<O::Param, O::Param>,
    F: ArgminFloat,
{
    const NAME: &'static str = "Gauss-Newton method";

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &mut IterState<O>,
    ) -> Result<ArgminIterData<IterState<O>>, Error> {
        let param = state.take_param().unwrap();
        let residuals = op.apply(&param)?;
        let jacobian = op.jacobian(&param)?;

        let p = jacobian
            .clone()
            .t()
            .dot(&jacobian)
            .inv()?
            .dot(&jacobian.t().dot(&residuals));

        let new_param = param.sub(&p.mul(&self.gamma));

        Ok(ArgminIterData::new()
            .param(new_param)
            .cost(residuals.norm()))
    }

    fn terminate(&mut self, state: &IterState<O>) -> TerminationReason {
        if (state.get_prev_cost() - state.get_cost()).abs() < self.tol {
            return TerminationReason::NoChangeInCost;
        }
        TerminationReason::NotTerminated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "ndarrayl")]
    use crate::core::Executor;
    use crate::test_trait_impl;
    #[cfg(feature = "ndarrayl")]
    use approx::assert_relative_eq;

    test_trait_impl!(gauss_newton_method, GaussNewton<f64>);

    #[test]
    fn test_default() {
        let GaussNewton { tol: t, gamma: g } = GaussNewton::<f64>::new();

        assert_eq!(g.to_ne_bytes(), (1.0f64).to_ne_bytes());
        assert_eq!(t.to_ne_bytes(), f64::EPSILON.sqrt().to_ne_bytes());
    }

    #[test]
    fn test_tolerance() {
        let tol1: f64 = 1e-4;

        let GaussNewton { tol: t, .. } = GaussNewton::new().with_tol(tol1).unwrap();

        assert_eq!(t.to_ne_bytes(), tol1.to_ne_bytes());
    }

    #[test]
    fn test_tolerance_error() {
        let tol = -2.0;
        let error = GaussNewton::new().with_tol(tol).err().unwrap();
        assert_eq!(
            error.downcast_ref::<ArgminError>().unwrap().to_string(),
            "Invalid parameter: \"Gauss-Newton: tol must be positive.\""
        );
    }

    #[test]
    fn test_gamma() {
        let gamma: f64 = 0.5;

        let GaussNewton { gamma: g, .. } = GaussNewton::new().with_gamma(gamma).unwrap();

        assert_eq!(g.to_ne_bytes(), gamma.to_ne_bytes());
    }

    #[test]
    fn test_gamma_errors() {
        let gamma = -0.5;
        let error = GaussNewton::new().with_gamma(gamma).err().unwrap();
        assert_eq!(
            error.downcast_ref::<ArgminError>().unwrap().to_string(),
            "Invalid parameter: \"Gauss-Newton: gamma must be in  (0, 1].\""
        );

        let gamma = 0.0;
        let error = GaussNewton::new().with_gamma(gamma).err().unwrap();
        assert_eq!(
            error.downcast_ref::<ArgminError>().unwrap().to_string(),
            "Invalid parameter: \"Gauss-Newton: gamma must be in  (0, 1].\""
        );

        let gamma = 2.0;
        let error = GaussNewton::new().with_gamma(gamma).err().unwrap();
        assert_eq!(
            error.downcast_ref::<ArgminError>().unwrap().to_string(),
            "Invalid parameter: \"Gauss-Newton: gamma must be in  (0, 1].\""
        );
    }

    #[cfg(feature = "ndarrayl")]
    #[test]
    fn test_solver() {
        use ndarray::{Array, Array1, Array2};
        use std::cell::RefCell;

        struct Problem {
            counter: RefCell<usize>,
        }

        impl ArgminOp for Problem {
            type Param = Array1<f64>;
            type Output = Array1<f64>;
            type Hessian = ();
            type Jacobian = Array2<f64>;
            type Float = f64;

            fn apply(&self, _p: &Self::Param) -> Result<Self::Output, Error> {
                if *self.counter.borrow() == 0 {
                    let mut c = self.counter.borrow_mut();
                    *c += 1;
                    Ok(Array1::from_vec(vec![0.5, 2.0]))
                } else {
                    Ok(Array1::from_vec(vec![0.3, 1.0]))
                }
            }

            fn jacobian(&self, _p: &Self::Param) -> Result<Self::Jacobian, Error> {
                Ok(Array::from_shape_vec((2, 2), vec![1f64, 2.0, 3.0, 4.0])?)
            }
        }

        // Single iteration, starting from [0, 0], gamma = 1
        let problem = Problem {
            counter: RefCell::new(0),
        };
        let solver: GaussNewton<f64> = GaussNewton::new();
        let init_param = Array1::from_vec(vec![0.0, 0.0]);

        let param = Executor::new(problem, solver)
            .configure(|config| config.param(init_param))
            .max_iters(1)
            .run()
            .unwrap()
            .state
            .get_best_param()
            .unwrap();
        assert_relative_eq!(param[0], -1.0, epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(param[1], 0.25, epsilon = f64::EPSILON.sqrt());

        // Two iterations, starting from [0, 0], gamma = 1
        let problem = Problem {
            counter: RefCell::new(0),
        };
        let solver: GaussNewton<f64> = GaussNewton::new();
        let init_param = Array1::from_vec(vec![0.0, 0.0]);

        let param = Executor::new(problem, solver)
            .configure(|config| config.param(init_param))
            .max_iters(2)
            .run()
            .unwrap()
            .state
            .get_best_param()
            .unwrap();
        assert_relative_eq!(param[0], -1.4, epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(param[1], 0.3, epsilon = f64::EPSILON.sqrt());

        // Single iteration, starting from [0, 0], gamma = 0.5
        let problem = Problem {
            counter: RefCell::new(0),
        };
        let solver: GaussNewton<f64> = GaussNewton::new().with_gamma(0.5).unwrap();
        let init_param = Array1::from_vec(vec![0.0, 0.0]);

        let param = Executor::new(problem, solver)
            .configure(|config| config.param(init_param))
            .max_iters(1)
            .run()
            .unwrap()
            .state
            .get_best_param()
            .unwrap();
        assert_relative_eq!(param[0], -0.5, epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(param[1], 0.125, epsilon = f64::EPSILON.sqrt());

        // Two iterations, starting from [0, 0], gamma = 0.5
        let problem = Problem {
            counter: RefCell::new(0),
        };
        let solver: GaussNewton<f64> = GaussNewton::new().with_gamma(0.5).unwrap();
        let init_param = Array1::from_vec(vec![0.0, 0.0]);

        let param = Executor::new(problem, solver)
            .configure(|config| config.param(init_param))
            .max_iters(2)
            .run()
            .unwrap()
            .state
            .get_best_param()
            .unwrap();
        assert_relative_eq!(param[0], -0.7, epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(param[1], 0.15, epsilon = f64::EPSILON.sqrt());
    }
}
