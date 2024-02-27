// Copyright 2019-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#![allow(non_snake_case)]

use approx::assert_relative_eq;
use ndarray::prelude::*;
use ndarray::{Array1, Array2};

use crate::core::{CostFunction, Error, Executor, Gradient, Hessian, State};
use crate::solver::gradientdescent::SteepestDescent;
use crate::solver::linesearch::{HagerZhangLineSearch, MoreThuenteLineSearch};
use crate::solver::newton::NewtonCG;
use crate::solver::quasinewton::{BFGS, DFP, LBFGS};

#[derive(Clone, Default, Debug)]
struct MaxEntropy {
    F: Array2<f64>,
    K: Array1<f64>,
    param_opt: Array1<f64>,
    param_init: Array1<f64>,
}

/// Base test case for a simple constrained entropy maximization problem
/// (the machine translation example of Berger et al in
/// Computational Linguistics, vol 22, num 1, pp 39--72, 1996.)
///
/// Adapted from scipy.optimize.test.test_optimize
impl MaxEntropy {
    fn new() -> MaxEntropy {
        let F: Array2<f64> = arr2(&[
            [1., 1., 1.],
            [1., 1., 0.],
            [1., 0., 1.],
            [1., 0., 0.],
            [1., 0., 0.],
        ]);
        let K: Array1<f64> = arr1(&[1., 0.3, 0.5]);
        let param_opt: Array1<f64> = arr1(&[0., -0.524869316, 0.487525860]);
        let param_init: Array1<f64> = arr1(&[0.0, 0.0, 0.0]);
        MaxEntropy {
            F,
            K,
            param_opt,
            param_init,
        }
    }
}

impl CostFunction for MaxEntropy {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let log_pdot = self.F.dot(&p.t());
        let log_z = log_pdot.mapv(|x| x.exp()).sum().ln();
        let loss = log_z - self.K.dot(&p.t());
        Ok(loss)
    }
}

impl Gradient for MaxEntropy {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;

    fn gradient(&self, p: &Self::Param) -> Result<Self::Param, Error> {
        let log_pdot = self.F.dot(&p.t());
        let log_z = log_pdot.mapv(|x| x.exp()).sum().ln();
        let y = (log_pdot - log_z).mapv(|x| x.exp());
        let grad = self.F.clone().t().dot(&y) - self.K.clone();
        Ok(grad)
    }
}

impl Hessian for MaxEntropy {
    type Param = Array1<f64>;
    type Hessian = Array2<f64>;

    fn hessian(&self, p: &Self::Param) -> Result<Self::Hessian, Error> {
        let log_pdot = self.F.dot(&p.t());
        let log_z = log_pdot.mapv(|x| x.exp()).sum().ln();
        let y = (log_pdot - log_z).mapv(|x| x.exp());
        let y2_diag = Array2::from_diag(&y);
        let tmp = self.F.clone() - self.F.clone().t().dot(&y);
        let hess = self.F.clone().t().dot(&y2_diag.dot(&tmp));
        Ok(hess)
    }
}

// TODO fix this, there should be only one macro.
macro_rules! entropy_max_tests {
    ($($name:ident: $solver:expr,)*) => {
    $(
        #[test]
        fn $name() {
            let cost_func = MaxEntropy::new();
            let res = Executor::new(cost_func.clone(), $solver)
                .configure(|state| {
                    state
                        .param(cost_func.param_init.clone())
                        .max_iters(100)
                })
                .run()
                .unwrap();

            assert_relative_eq!(
                cost_func.cost(res.state.get_param().unwrap()).unwrap(),
                cost_func.cost(&cost_func.param_opt).unwrap(),
                epsilon = 1e-6
            );
        }
    )*
    }
}

macro_rules! entropy_max_tests_with_inv_hessian {
    ($($name:ident: $solver:expr,)*) => {
    $(
        #[test]
        fn $name() {
            let cost_func = MaxEntropy::new();
            let res = Executor::new(cost_func.clone(), $solver)
                .configure(|state| {
                    state
                        .param(cost_func.param_init.clone())
                        .inv_hessian(Array2::eye(3))
                        .max_iters(100)
                })
                .run()
                .unwrap();

            assert_relative_eq!(
                cost_func.cost(res.state.get_param().unwrap()).unwrap(),
                cost_func.cost(&cost_func.param_opt).unwrap(),
                epsilon = 1e-6
            );
        }
    )*
    }
}

entropy_max_tests! {
     test_max_entropy_lbfgs_morethuente: LBFGS::new(MoreThuenteLineSearch::new(), 10),
     test_max_entropy_lbfgs_hagerzhang: LBFGS::new(HagerZhangLineSearch::new(), 10),
     test_max_entropy_newton_cg: NewtonCG::new(MoreThuenteLineSearch::new()),
     test_max_entropy_steepest_descent: SteepestDescent::new(MoreThuenteLineSearch::new()),
}

entropy_max_tests_with_inv_hessian! {
     test_max_entropy_bfgs: BFGS::new(MoreThuenteLineSearch::new()),
     test_max_entropy_dfp: DFP::new(MoreThuenteLineSearch::new()),
}

#[test]
fn test_lbfgs_func_count() {
    let cost = MaxEntropy::new();

    let linesearch = MoreThuenteLineSearch::new();
    let solver = LBFGS::new(linesearch, 10);
    let res = Executor::new(cost.clone(), solver)
        .configure(|config| config.param(cost.param_init.clone()).max_iters(100))
        .run()
        .unwrap();

    assert_relative_eq!(
        cost.cost(res.state.get_param().unwrap()).unwrap(),
        cost.cost(&cost.param_opt).unwrap(),
        epsilon = 1e-6
    );

    // Check the number of cost function evaluation and gradient
    // evaluation with that in scipy
    let func_counts = res.state.get_func_counts();
    assert!(func_counts["cost_count"] <= 7);
    // The following value is 5 in scipy.optimize, but the convergence
    // criteria is different
    assert!(func_counts["gradient_count"] <= 11);
}
