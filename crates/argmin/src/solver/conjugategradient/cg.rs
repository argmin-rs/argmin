// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{ArgminFloat, Error, IterState, Operator, Problem, Solver, State, KV};
use argmin_math::{ArgminConj, ArgminDot, ArgminL2Norm, ArgminMul, ArgminScaledAdd, ArgminSub};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// # Conjugate Gradient method
///
/// A solver for systems of linear equations with a symmetric and positive-definite matrix.
///
/// Solves systems of the form `A * x = b` where `x` and `b` are vectors and `A` is a symmetric and
/// positive-definite matrix.
///
/// Requires an initial parameter vector.
///
/// ## Requirements on the optimization problem
///
/// The optimization problem is required to implement [`Operator`].
///
/// ## Reference
///
/// Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct ConjugateGradient<P, F> {
    /// b (right hand side of `A * x = b`)
    b: P,
    /// p
    p: Option<P>,
    /// previous p
    p_prev: Option<P>,
    /// r^T * r
    rtr: F,
}

impl<P, F> ConjugateGradient<P, F>
where
    F: ArgminFloat,
{
    /// Constructs an instance of [`ConjugateGradient`]
    ///
    /// Takes `b`, the right hand side of `A * x = b` as input.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::conjugategradient::ConjugateGradient;
    /// # let b = vec![1.0f64, 1.0];
    /// let cg: ConjugateGradient<_, f64> = ConjugateGradient::new(b);
    /// ```
    pub fn new(b: P) -> Self {
        ConjugateGradient {
            b,
            p: None,
            p_prev: None,
            rtr: F::nan(),
        }
    }

    /// Return the previous search direction (Needed by [`NewtonCG`](`crate::solver::newton::NewtonCG`))
    ///
    /// Returns an error if the field `p_prev` is not initialized.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::conjugategradient::ConjugateGradient;
    /// # use argmin::core::Error;
    /// # let cg: ConjugateGradient<_, f64> = ConjugateGradient::new(vec![1.0f64, 1.0]);
    /// let p_prev: Result<_, _> = cg.get_prev_p();
    /// ```
    pub fn get_prev_p(&self) -> Result<&P, Error> {
        self.p_prev.as_ref().ok_or_else(argmin_error_closure!(
            NotInitialized,
            "Field `p_prev` of `ConjugateGradient` not initialized."
        ))
    }
}

impl<P, O, R, F> Solver<O, IterState<P, (), (), (), R, F>> for ConjugateGradient<P, F>
where
    O: Operator<Param = P, Output = P>,
    P: Clone + ArgminDot<P, F> + ArgminSub<P, R> + ArgminScaledAdd<P, F, P> + ArgminConj,
    R: ArgminMul<F, R> + ArgminMul<F, P> + ArgminConj + ArgminDot<R, F> + ArgminScaledAdd<P, F, R>,
    F: ArgminFloat + ArgminL2Norm<F>,
{
    const NAME: &'static str = "Conjugate Gradient";

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        state: IterState<P, (), (), (), R, F>,
    ) -> Result<(IterState<P, (), (), (), R, F>, Option<KV>), Error> {
        let init_param = state.get_param().ok_or_else(argmin_error_closure!(
            NotInitialized,
            concat!(
                "`ConjugateGradient` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method."
            )
        ))?;
        let ap = problem.apply(init_param)?;
        let r0: R = self.b.sub(&ap).mul(&(float!(-1.0)));
        self.p = Some(r0.mul(&(float!(-1.0))));
        self.rtr = r0.dot(&r0.conj());
        Ok((state.residuals(r0), None))
    }

    /// Perform one iteration of CG algorithm
    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        mut state: IterState<P, (), (), (), R, F>,
    ) -> Result<(IterState<P, (), (), (), R, F>, Option<KV>), Error> {
        let p = self.p.take().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`ConjugateGradient`: Field `p` not set"
        ))?;
        let r = state.take_residuals().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`ConjugateGradient`: Residuals in `state` not set"
        ))?;

        let apk = problem.apply(&p)?;
        let alpha = self.rtr.div(p.dot(&apk.conj()));
        let state_param = state.get_param().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`ConjugateGradient`: Parameter vector in `state` not set"
        ))?;
        let new_param = state_param.scaled_add(&alpha, &p);
        let r = r.scaled_add(&alpha, &apk);
        let rtr_n = r.dot(&r.conj());
        let beta = rtr_n.div(self.rtr);
        self.rtr = rtr_n;
        let p_n = <R as ArgminMul<F, P>>::mul(&r, &(float!(-1.0))).scaled_add(&beta, &p);
        let norm = r.dot(&r.conj()).l2_norm();

        self.p = Some(p_n);
        self.p_prev = Some(p);

        Ok((
            state.param(new_param).residuals(r).cost(norm),
            Some(kv!("alpha" => alpha; "beta" => beta;)),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{test_utils::TestProblem, ArgminError, IterState, Problem};
    use crate::test_trait_impl;
    use approx::assert_relative_eq;

    test_trait_impl!(conjugate_gradient, ConjugateGradient<Vec<f64>, f64>);

    #[test]
    fn test_new() {
        let cg: ConjugateGradient<_, f64> = ConjugateGradient::new(vec![1.0f64, 2.0]);
        let ConjugateGradient { b, p, p_prev, rtr } = cg;
        assert_eq!(b[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
        assert_eq!(b[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
        assert!(p.is_none());
        assert!(p_prev.is_none());
        assert!(rtr.is_nan());
    }

    #[test]
    fn test_get_prev_p_not_initialized() {
        let cg: ConjugateGradient<_, f64> = ConjugateGradient::new(vec![1.0f64, 2.0]);
        let res: Result<_, _> = cg.get_prev_p();
        assert_error!(
            res,
            ArgminError,
            "Not initialized: \"Field `p_prev` of `ConjugateGradient` not initialized.\""
        );
    }

    #[test]
    fn test_get_prev_p() {
        let mut cg: ConjugateGradient<_, f64> = ConjugateGradient::new(vec![1.0f64, 2.0]);
        cg.p_prev = Some(vec![3.0f64, 4.0]);
        let res: Result<_, _> = cg.get_prev_p();
        assert!(res.is_ok());
        let p_prev = res.unwrap();
        assert_eq!(p_prev[0].to_ne_bytes(), 3.0f64.to_ne_bytes());
        assert_eq!(p_prev[1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    }

    #[test]
    fn test_init_param_not_initialized() {
        let mut cg: ConjugateGradient<_, f64> = ConjugateGradient::new(vec![1.0f64, 2.0]);
        let res = cg.init(&mut Problem::new(TestProblem::new()), IterState::new());
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Not initialized: \"`ConjugateGradient` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method.\""
            )
        );
    }

    #[test]
    fn test_init() {
        let mut cg: ConjugateGradient<_, f64> = ConjugateGradient::new(vec![1.0f64, 2.0]);
        let state: IterState<Vec<f64>, (), (), (), Vec<f64>, f64> =
            IterState::new().param(vec![3.0, 4.0]);
        let (state_out, kv) = cg
            .init(&mut Problem::new(TestProblem::new()), state.clone())
            .unwrap();
        assert!(kv.is_none());

        let ConjugateGradient { b, p, p_prev, rtr } = cg;

        assert_relative_eq!(b[0], 1.0, epsilon = f64::EPSILON);
        assert_relative_eq!(b[1], 2.0, epsilon = f64::EPSILON);
        let r0 = [2.0f64, 2.0];
        assert_relative_eq!(
            r0[0],
            state_out.get_residuals().as_ref().unwrap()[0],
            epsilon = f64::EPSILON
        );
        assert_relative_eq!(
            r0[1],
            state_out.get_residuals().as_ref().unwrap()[1],
            epsilon = f64::EPSILON
        );
        let pp = [-2.0f64, -2.0];
        assert_relative_eq!(pp[0], p.as_ref().unwrap()[0], epsilon = f64::EPSILON);
        assert_relative_eq!(pp[1], p.as_ref().unwrap()[1], epsilon = f64::EPSILON);
        assert_relative_eq!(rtr, 8.0, epsilon = f64::EPSILON);
        assert!(p_prev.is_none());
    }

    #[test]
    fn test_next_iter_p_not_set() {
        let mut cg: ConjugateGradient<_, f64> = ConjugateGradient::new(vec![1.0f64, 2.0]);
        let state = IterState::new().param(vec![1.0f64]);
        assert!(cg.p.is_none());
        let res = cg.next_iter(&mut Problem::new(TestProblem::new()), state);
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Potential bug: \"`ConjugateGradient`: ",
                "Field `p` not set\". This is potentially a bug. ",
                "Please file a report on https://github.com/argmin-rs/argmin/issues"
            )
        );
    }

    #[test]
    fn test_next_iter_r_not_set() {
        let mut cg: ConjugateGradient<_, f64> = ConjugateGradient::new(vec![1.0f64, 2.0]);
        let state = IterState::new().param(vec![1.0f64]);
        cg.p = Some(vec![]);
        let res = cg.next_iter(&mut Problem::new(TestProblem::new()), state);
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Potential bug: \"`ConjugateGradient`: ",
                "Residuals in `state` not set\". This is potentially a bug. ",
                "Please file a report on https://github.com/argmin-rs/argmin/issues"
            )
        );
    }

    #[test]
    fn test_next_iter_state_param_not_set() {
        let mut cg: ConjugateGradient<_, f64> = ConjugateGradient::new(vec![1.0f64, 2.0]);
        let state = IterState::new().residuals(vec![]);
        cg.p = Some(vec![]);
        assert!(state.param.is_none());
        let res = cg.next_iter(&mut Problem::new(TestProblem::new()), state);
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Potential bug: \"`ConjugateGradient`: ",
                "Parameter vector in `state` not set\". This is potentially a bug. ",
                "Please file a report on https://github.com/argmin-rs/argmin/issues"
            )
        );
    }

    #[test]
    fn test_next_iter() {
        let mut cg: ConjugateGradient<_, f64> = ConjugateGradient::new(vec![2.0f64]);
        let state = IterState::new().param(vec![1.0f64]);
        let mut problem = Problem::new(TestProblem::new());
        let (state, _) = cg.init(&mut problem, state).unwrap();
        let rtr = cg.rtr;
        let p = cg.p.clone().unwrap()[0];
        let r = state.get_residuals().unwrap()[0];

        let apk = p;
        let alpha = rtr / (p * apk);
        let new_param = 1.0 + alpha * p;
        let r = r + alpha * apk;
        let rtr_n = -r * r;
        let beta = rtr_n / rtr;
        let p_n = -r + beta * p;
        let norm = (r * r).l2_norm();

        let (state, kv) = cg.next_iter(&mut problem, state).unwrap();
        assert!(kv.is_some());

        assert_relative_eq!(r, state.get_residuals().unwrap()[0]);
        assert_relative_eq!(p_n, cg.p.as_ref().unwrap()[0]);
        assert_relative_eq!(p, cg.p_prev.as_ref().unwrap()[0]);
        assert_relative_eq!(rtr_n, cg.rtr);

        assert_relative_eq!(norm, state.get_cost());
        assert_relative_eq!(new_param, state.get_param().unwrap()[0]);
    }
}
