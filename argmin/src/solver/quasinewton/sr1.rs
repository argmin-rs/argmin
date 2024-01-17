// Copyright 2019-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{
    ArgminFloat, CostFunction, DeserializeOwnedAlias, Error, Executor, Gradient, IterState,
    LineSearch, OptimizationResult, Problem, SerializeAlias, Solver, TerminationReason,
    TerminationStatus, KV,
};
use argmin_math::{ArgminAdd, ArgminDot, ArgminL2Norm, ArgminMul, ArgminSub};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// # Symmetric rank-one (SR1) method
///
/// This method currently has problems: <https://github.com/argmin-rs/argmin/issues/221>.
///
/// ## Requirements on the optimization problem
///
/// The optimization problem is required to implement [`CostFunction`] and [`Gradient`].
///
/// ## Reference
///
/// Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct SR1<L, F> {
    /// parameter for skipping rule
    denominator_factor: F,
    /// line search
    linesearch: L,
    /// Tolerance for the stopping criterion based on the change of the norm on the gradient
    tol_grad: F,
    /// Tolerance for the stopping criterion based on the change of the cost stopping criterion
    tol_cost: F,
}

impl<L, F> SR1<L, F>
where
    F: ArgminFloat,
{
    /// Construct a new instance of [`SR1`]
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::quasinewton::SR1;
    /// # let linesearch = ();
    /// let sr1: SR1<_, f64> = SR1::new(linesearch);
    /// ```
    pub fn new(linesearch: L) -> Self {
        SR1 {
            denominator_factor: float!(1e-8),
            linesearch,
            tol_grad: F::epsilon().sqrt(),
            tol_cost: F::epsilon(),
        }
    }

    /// Set denominator factor
    ///
    /// If the denominator of the update is below the `denominator_factor` (scaled with other
    /// factors derived from the parameter vectors and the gradients), then the update of the
    /// inverse Hessian will be skipped.
    ///
    /// Must be in `(0, 1)` and defaults to `1e-8`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::quasinewton::SR1;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// # let linesearch = ();
    /// let sr1: SR1<_, f64> = SR1::new(linesearch).with_denominator_factor(1e-7)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_denominator_factor(mut self, denominator_factor: F) -> Result<Self, Error> {
        if denominator_factor <= float!(0.0) || denominator_factor >= float!(1.0) {
            Err(argmin_error!(
                InvalidParameter,
                "`SR1`: denominator_factor must be in (0, 1)."
            ))
        } else {
            self.denominator_factor = denominator_factor;
            Ok(self)
        }
    }

    /// The algorithm stops if the norm of the gradient is below `tol_grad`.
    ///
    /// The provided value must be non-negative. Defaults to `sqrt(EPSILON)`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::quasinewton::SR1;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// # let linesearch = ();
    /// let sr1: SR1<_, f64> = SR1::new(linesearch).with_tolerance_grad(1e-6)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_tolerance_grad(mut self, tol_grad: F) -> Result<Self, Error> {
        if tol_grad < float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`SR1`: gradient tolerance must be >= 0."
            ));
        }
        self.tol_grad = tol_grad;
        Ok(self)
    }

    /// Sets tolerance for the stopping criterion based on the change of the cost stopping criterion
    ///
    /// The provided value must be non-negative. Defaults to `EPSILON`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::quasinewton::SR1;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// # let linesearch = ();
    /// let sr1: SR1<_, f64> = SR1::new(linesearch).with_tolerance_cost(1e-6)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_tolerance_cost(mut self, tol_cost: F) -> Result<Self, Error> {
        if tol_cost < float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`SR1`: cost tolerance must be >= 0."
            ));
        }
        self.tol_cost = tol_cost;
        Ok(self)
    }
}

impl<O, L, P, G, H, F> Solver<O, IterState<P, G, (), H, (), F>> for SR1<L, F>
where
    O: CostFunction<Param = P, Output = F> + Gradient<Param = P, Gradient = G>,
    P: Clone
        + SerializeAlias
        + DeserializeOwnedAlias
        + ArgminSub<P, P>
        + ArgminDot<G, F>
        + ArgminDot<P, F>
        + ArgminDot<P, H>
        + ArgminL2Norm<F>
        + ArgminMul<F, P>,
    G: Clone
        + SerializeAlias
        + DeserializeOwnedAlias
        + ArgminSub<P, P>
        + ArgminL2Norm<F>
        + ArgminSub<G, G>,
    H: SerializeAlias
        + DeserializeOwnedAlias
        + ArgminDot<G, P>
        + ArgminDot<P, P>
        + ArgminAdd<H, H>
        + ArgminMul<F, H>,
    L: Clone + LineSearch<P, F> + Solver<O, IterState<P, G, (), (), (), F>>,
    F: ArgminFloat,
{
    const NAME: &'static str = "SR1";

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        mut state: IterState<P, G, (), H, (), F>,
    ) -> Result<(IterState<P, G, (), H, (), F>, Option<KV>), Error> {
        let param = state.take_param().ok_or_else(argmin_error_closure!(
            NotInitialized,
            concat!(
                "`SR1` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method."
            )
        ))?;

        let inv_hessian = state.take_inv_hessian().ok_or_else(argmin_error_closure!(
            NotInitialized,
            concat!(
                "`SR1` requires an initial inverse Hessian. ",
                "Please provide an initial guess via `Executor`s `configure` method."
            )
        ))?;

        let cost = state.get_cost();
        let cost = if cost.is_infinite() {
            problem.cost(&param)?
        } else {
            cost
        };

        let grad = state
            .take_gradient()
            .map(Result::Ok)
            .unwrap_or_else(|| problem.gradient(&param))?;
        Ok((
            state
                .param(param)
                .cost(cost)
                .gradient(grad)
                .inv_hessian(inv_hessian),
            None,
        ))
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        mut state: IterState<P, G, (), H, (), F>,
    ) -> Result<(IterState<P, G, (), H, (), F>, Option<KV>), Error> {
        let param = state.take_param().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`SR1`: Parameter vector in state not set."
        ))?;
        let cost = state.get_cost();

        let prev_grad = state.take_gradient().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`SR1`: Gradient in state not set."
        ))?;

        let mut inv_hessian = state.take_inv_hessian().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`SR1`: Inverse Hessian in state not set."
        ))?;

        let p = inv_hessian.dot(&prev_grad).mul(&float!(-1.0));

        self.linesearch.search_direction(p);

        // Run solver
        let OptimizationResult {
            problem: line_problem,
            state: mut linesearch_state,
            ..
        } = Executor::new(problem.take_problem().unwrap(), self.linesearch.clone())
            .configure(|config| {
                config
                    .param(param.clone())
                    .gradient(prev_grad.clone())
                    .cost(cost)
            })
            .ctrlc(false)
            .run()?;

        let xk1 = linesearch_state.take_param().unwrap();
        let next_cost = linesearch_state.get_cost();

        // take care of function eval counts
        problem.consume_problem(line_problem);

        let grad = problem.gradient(&xk1)?;
        let yk = grad.sub(&prev_grad);

        let sk = xk1.sub(&param);

        // let skmhkyk: P = sk.sub(&inv_hessian.dot(&yk));
        // let a: H = skmhkyk.dot(&skmhkyk);
        // let b: F = skmhkyk.dot(&yk);
        let ykmbksk: P = yk.sub(&inv_hessian.dot(&sk));
        let a: H = ykmbksk.dot(&ykmbksk);
        let b: F = ykmbksk.dot(&sk);

        // let hessian_update = b.abs() >= self.r * yk.l2_norm() * skmhkyk.l2_norm();
        let hessian_update = b.abs() >= self.denominator_factor * sk.l2_norm() * ykmbksk.l2_norm();

        if hessian_update {
            inv_hessian = inv_hessian.add(&a.mul(&(float!(1.0) / b)));
        }

        Ok((
            state
                .param(xk1)
                .cost(next_cost)
                .gradient(grad)
                .inv_hessian(inv_hessian),
            Some(kv!["denominator" => b; "hessian_update" => hessian_update;]),
        ))
    }

    fn terminate(&mut self, state: &IterState<P, G, (), H, (), F>) -> TerminationStatus {
        if state.get_gradient().unwrap().l2_norm() < self.tol_grad {
            return TerminationStatus::Terminated(TerminationReason::SolverConverged);
        }
        if (state.get_prev_cost() - state.cost).abs() < self.tol_cost {
            return TerminationStatus::Terminated(TerminationReason::SolverConverged);
        }
        TerminationStatus::NotTerminated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{test_utils::TestProblem, ArgminError, IterState, State};
    use crate::solver::linesearch::MoreThuenteLineSearch;
    use crate::test_trait_impl;

    test_trait_impl!(
        sr1,
        SR1<MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64>, f64>
    );

    #[test]
    fn test_new() {
        #[derive(Eq, PartialEq, Debug)]
        struct MyFakeLineSearch {}

        let sr1: SR1<_, f64> = SR1::new(MyFakeLineSearch {});
        let SR1 {
            denominator_factor,
            linesearch,
            tol_grad,
            tol_cost,
        } = sr1;

        assert_eq!(linesearch, MyFakeLineSearch {});
        assert_eq!(tol_grad.to_ne_bytes(), f64::EPSILON.sqrt().to_ne_bytes());
        assert_eq!(tol_cost.to_ne_bytes(), f64::EPSILON.to_ne_bytes());
        assert_eq!(denominator_factor.to_ne_bytes(), 1e-8f64.to_ne_bytes());
    }

    #[test]
    fn test_with_denominator_factor() {
        #[derive(Eq, PartialEq, Debug, Clone, Copy)]
        struct MyFakeLineSearch {}

        // correct parameters
        for tol in [f64::EPSILON, 1e-8, 1e-6, 1e-2, 1.0 - f64::EPSILON] {
            let sr1: SR1<_, f64> = SR1::new(MyFakeLineSearch {});
            let res = sr1.with_denominator_factor(tol);
            assert!(res.is_ok());

            let nm = res.unwrap();
            assert_eq!(nm.denominator_factor.to_ne_bytes(), tol.to_ne_bytes());
        }

        // incorrect parameters
        for tol in [-f64::EPSILON, 0.0, -1.0, 1.0] {
            let sr1: SR1<_, f64> = SR1::new(MyFakeLineSearch {});
            let res = sr1.with_denominator_factor(tol);
            assert_error!(
                res,
                ArgminError,
                "Invalid parameter: \"`SR1`: denominator_factor must be in (0, 1).\""
            );
        }
    }

    #[test]
    fn test_with_tolerance_grad() {
        #[derive(Eq, PartialEq, Debug, Clone, Copy)]
        struct MyFakeLineSearch {}

        // correct parameters
        for tol in [1e-6, 0.0, 1e-2, 1.0, 2.0] {
            let sr1: SR1<_, f64> = SR1::new(MyFakeLineSearch {});
            let res = sr1.with_tolerance_grad(tol);
            assert!(res.is_ok());

            let nm = res.unwrap();
            assert_eq!(nm.tol_grad.to_ne_bytes(), tol.to_ne_bytes());
        }

        // incorrect parameters
        for tol in [-f64::EPSILON, -1.0, -100.0, -42.0] {
            let sr1: SR1<_, f64> = SR1::new(MyFakeLineSearch {});
            let res = sr1.with_tolerance_grad(tol);
            assert_error!(
                res,
                ArgminError,
                "Invalid parameter: \"`SR1`: gradient tolerance must be >= 0.\""
            );
        }
    }

    #[test]
    fn test_with_tolerance_cost() {
        #[derive(Eq, PartialEq, Debug, Clone, Copy)]
        struct MyFakeLineSearch {}

        // correct parameters
        for tol in [1e-6, 0.0, 1e-2, 1.0, 2.0] {
            let sr1: SR1<_, f64> = SR1::new(MyFakeLineSearch {});
            let res = sr1.with_tolerance_cost(tol);
            assert!(res.is_ok());

            let nm = res.unwrap();
            assert_eq!(nm.tol_cost.to_ne_bytes(), tol.to_ne_bytes());
        }

        // incorrect parameters
        for tol in [-f64::EPSILON, -1.0, -100.0, -42.0] {
            let sr1: SR1<_, f64> = SR1::new(MyFakeLineSearch {});
            let res = sr1.with_tolerance_cost(tol);
            assert_error!(
                res,
                ArgminError,
                "Invalid parameter: \"`SR1`: cost tolerance must be >= 0.\""
            );
        }
    }

    #[test]
    fn test_init() {
        let linesearch = MoreThuenteLineSearch::new().with_c(1e-4, 0.9).unwrap();

        let param: Vec<f64> = vec![-1.0, 1.0];
        let inv_hessian: Vec<Vec<f64>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let mut sr1: SR1<_, f64> = SR1::new(linesearch);

        // Forgot to initialize the parameter vector
        let state: IterState<Vec<f64>, Vec<f64>, (), Vec<Vec<f64>>, (), f64> = IterState::new();
        let problem = TestProblem::new();
        let res = sr1.init(&mut Problem::new(problem), state);
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Not initialized: \"`SR1` requires an initial parameter vector. Please ",
                "provide an initial guess via `Executor`s `configure` method.\""
            )
        );

        // Forgot initial inverse Hessian guess
        let state: IterState<Vec<f64>, Vec<f64>, (), Vec<Vec<f64>>, (), f64> =
            IterState::new().param(param.clone());
        let problem = TestProblem::new();
        let res = sr1.init(&mut Problem::new(problem), state);

        assert_error!(
            res,
            ArgminError,
            concat!(
                "Not initialized: \"`SR1` requires an initial inverse Hessian. Please ",
                "provide an initial guess via `Executor`s `configure` method.\""
            )
        );

        // All good.
        let state: IterState<Vec<f64>, Vec<f64>, (), Vec<Vec<f64>>, (), f64> = IterState::new()
            .param(param.clone())
            .inv_hessian(inv_hessian.clone());
        let problem = TestProblem::new();
        let (mut state_out, kv) = sr1.init(&mut Problem::new(problem), state).unwrap();

        assert!(kv.is_none());

        let s_param = state_out.take_param().unwrap();

        for (s, p) in s_param.iter().zip(param.iter()) {
            assert_eq!(s.to_ne_bytes(), p.to_ne_bytes());
        }

        let s_grad = state_out.take_gradient().unwrap();

        for (s, p) in s_grad.iter().zip(param.iter()) {
            assert_eq!(s.to_ne_bytes(), p.to_ne_bytes());
        }

        let s_inv_hessian = state_out.take_inv_hessian().unwrap();

        for (s, h) in s_inv_hessian
            .iter()
            .flatten()
            .zip(inv_hessian.iter().flatten())
        {
            assert_eq!(s.to_ne_bytes(), h.to_ne_bytes());
        }

        assert_eq!(state_out.get_cost().to_ne_bytes(), 1.0f64.to_ne_bytes())
    }

    #[test]
    fn test_init_provided_cost() {
        let linesearch = MoreThuenteLineSearch::new().with_c(1e-4, 0.9).unwrap();

        let param: Vec<f64> = vec![-1.0, 1.0];
        let inv_hessian: Vec<Vec<f64>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let mut sr1: SR1<_, f64> = SR1::new(linesearch);

        let state: IterState<Vec<f64>, Vec<f64>, (), Vec<Vec<f64>>, (), f64> = IterState::new()
            .param(param)
            .inv_hessian(inv_hessian)
            .cost(1234.0);

        let problem = TestProblem::new();
        let (state_out, kv) = sr1.init(&mut Problem::new(problem), state).unwrap();

        assert!(kv.is_none());

        assert_eq!(state_out.get_cost().to_ne_bytes(), 1234.0f64.to_ne_bytes())
    }

    #[test]
    fn test_init_provided_grad() {
        let linesearch = MoreThuenteLineSearch::new().with_c(1e-4, 0.9).unwrap();

        let param: Vec<f64> = vec![-1.0, 1.0];
        let gradient: Vec<f64> = vec![4.0, 9.0];
        let inv_hessian: Vec<Vec<f64>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let mut sr1: SR1<_, f64> = SR1::new(linesearch);

        let state: IterState<Vec<f64>, Vec<f64>, (), Vec<Vec<f64>>, (), f64> = IterState::new()
            .param(param)
            .inv_hessian(inv_hessian)
            .gradient(gradient.clone());

        let problem = TestProblem::new();
        let (mut state_out, kv) = sr1.init(&mut Problem::new(problem), state).unwrap();

        assert!(kv.is_none());

        let s_grad = state_out.take_gradient().unwrap();

        for (s, g) in s_grad.iter().zip(gradient.iter()) {
            assert_eq!(s.to_ne_bytes(), g.to_ne_bytes());
        }
    }
}
