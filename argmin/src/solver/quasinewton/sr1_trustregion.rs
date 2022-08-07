// Copyright 2019-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{
    ArgminFloat, CostFunction, DeserializeOwnedAlias, Error, Executor, Gradient, Hessian,
    IterState, OptimizationResult, Problem, SerializeAlias, Solver, TerminationReason,
    TrustRegionRadius, KV,
};
use argmin_math::{
    ArgminAdd, ArgminDot, ArgminMul, ArgminNorm, ArgminSub, ArgminWeightedDot, ArgminZeroLike,
};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// # SR1 Trust region method
///
/// A Quasi-Newton method which uses symmetric rank 1 (SR1) updating of the Hessian in a trust
/// region framework. An initial parameter vector must be provided, initial cost, gradient and
/// Hessian are optional and will be computed if not provided.
/// Requires a [trust region sub problem](`crate::solver::trustregion`).
///
/// ## Requirements on the optimization problem
///
/// The optimization problem is required to implement [`CostFunction`], [`Gradient`] and
/// [`Hessian`].
///
/// ## Reference
///
/// Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct SR1TrustRegion<R, F> {
    /// parameter for skipping rule
    denominator_factor: F,
    /// subproblem
    subproblem: R,
    /// Radius
    radius: F,
    /// eta \in (0, 10^-3)
    eta: F,
    /// Tolerance for the stopping criterion based on the change of the norm on the gradient
    tol_grad: F,
}

impl<R, F> SR1TrustRegion<R, F>
where
    F: ArgminFloat,
{
    /// Construct a new instance of [`SR1TrustRegion`]
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::quasinewton::SR1TrustRegion;
    /// # let subproblem = ();
    /// let subproblem = argmin::solver::trustregion::Steihaug::new().with_max_iters(20);
    /// # // The next line defines the type of `subproblem`. This is done here hidden in order to
    /// # // not litter the docs. When all of this is fed into an Executor, the compiler will
    /// # // figure out the types.
    /// # let subproblem: argmin::solver::trustregion::Steihaug<Vec<f64>, f64> = subproblem;
    /// let sr1: SR1TrustRegion<_, f64> = SR1TrustRegion::new(subproblem);
    /// ```
    pub fn new(subproblem: R) -> Self {
        SR1TrustRegion {
            denominator_factor: float!(1e-8),
            subproblem,
            radius: float!(1.0),
            eta: float!(0.5 * 1e-3),
            tol_grad: float!(1e-3),
        }
    }

    /// Set denominator factor
    ///
    /// If the denominator of the update is below the `demoninator_factor` (scaled with other
    /// factors derived from the parameter vectors and the gradients), then the update of the
    /// inverse Hessian will be skipped.
    ///
    /// Must be in `(0, 1)` and defaults to `1e-8`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::quasinewton::SR1TrustRegion;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// # let subproblem = ();
    /// let sr1: SR1TrustRegion<_, f64> = SR1TrustRegion::new(subproblem).with_denominator_factor(1e-7)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_denominator_factor(mut self, denominator_factor: F) -> Result<Self, Error> {
        if denominator_factor <= float!(0.0) || denominator_factor >= float!(1.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`SR1TrustRegion`: denominator_factor must be in (0, 1)."
            ));
        }
        self.denominator_factor = denominator_factor;
        Ok(self)
    }

    /// Set initial radius
    ///
    /// Defaults to 1.0.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::quasinewton::SR1TrustRegion;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// # let subproblem = ();
    /// let sr1: SR1TrustRegion<_, f64> = SR1TrustRegion::new(subproblem).with_radius(2.0);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn with_radius(mut self, radius: F) -> Self {
        self.radius = radius.abs();
        self
    }

    /// Set eta
    ///
    /// A step is taken if the actual reducation over the predicted reduction exceeds eta.
    /// Must be in (0, 10^-3) and defaults to 0.5 * 10^-3.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::quasinewton::SR1TrustRegion;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// # let subproblem = ();
    /// let sr1: SR1TrustRegion<_, f64> = SR1TrustRegion::new(subproblem).with_eta(10e-4)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_eta(mut self, eta: F) -> Result<Self, Error> {
        if eta >= float!(10e-3) || eta <= float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "SR1TrustRegion: eta must be in (0, 10^-3)."
            ));
        }
        self.eta = eta;
        Ok(self)
    }

    /// The algorithm stops if the norm of the gradient is below `tol_grad`.
    ///
    /// The provided value must be non-negative. Defaults to `10^-3`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::quasinewton::SR1TrustRegion;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// # let subproblem = ();
    /// let sr1: SR1TrustRegion<_, f64> = SR1TrustRegion::new(subproblem).with_tolerance_grad(1e-6)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_tolerance_grad(mut self, tol_grad: F) -> Result<Self, Error> {
        if tol_grad < float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`SR1TrustRegion`: gradient tolerance must be >= 0."
            ));
        }
        self.tol_grad = tol_grad;
        Ok(self)
    }
}

impl<O, R, P, G, B, F> Solver<O, IterState<P, G, (), B, F>> for SR1TrustRegion<R, F>
where
    O: CostFunction<Param = P, Output = F>
        + Gradient<Param = P, Gradient = G>
        + Hessian<Param = P, Hessian = B>,
    P: Clone
        + SerializeAlias
        + DeserializeOwnedAlias
        + ArgminSub<P, P>
        + ArgminAdd<P, P>
        + ArgminDot<P, F>
        + ArgminDot<P, B>
        + ArgminNorm<F>
        + ArgminZeroLike,
    G: Clone
        + SerializeAlias
        + DeserializeOwnedAlias
        + ArgminNorm<F>
        + ArgminDot<P, F>
        + ArgminSub<G, P>,
    B: Clone
        + SerializeAlias
        + DeserializeOwnedAlias
        + ArgminDot<P, P>
        + ArgminAdd<B, B>
        + ArgminMul<F, B>,
    R: Clone + TrustRegionRadius<F> + Solver<O, IterState<P, G, (), B, F>>,
    F: ArgminFloat + ArgminNorm<F>,
{
    const NAME: &'static str = "SR1 trust region";

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        mut state: IterState<P, G, (), B, F>,
    ) -> Result<(IterState<P, G, (), B, F>, Option<KV>), Error> {
        let param = state.take_param().ok_or_else(argmin_error_closure!(
            NotInitialized,
            concat!(
                "`SR1TrustRegion` requires an initial parameter vector. ",
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

        let hessian = state
            .take_hessian()
            .map(Result::Ok)
            .unwrap_or_else(|| problem.hessian(&param))?;

        Ok((
            state
                .param(param)
                .cost(cost)
                .gradient(grad)
                .hessian(hessian),
            None,
        ))
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        mut state: IterState<P, G, (), B, F>,
    ) -> Result<(IterState<P, G, (), B, F>, Option<KV>), Error> {
        let xk = state.take_param().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`SR1TrustRegion`: Parameter vector in state not set."
        ))?;

        let cost = state.get_cost();

        let prev_grad = state.take_gradient().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`SR1TrustRegion`: Gradient in state not set."
        ))?;

        let hessian = state.take_hessian().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`SR1TrustRegion`: Hessian in state not set."
        ))?;

        self.subproblem.set_radius(self.radius);

        let OptimizationResult {
            problem: sub_problem,
            state: mut sub_state,
            ..
        } = Executor::new(problem.take_problem().unwrap(), self.subproblem.clone())
            .configure(|config| {
                config
                    .param(xk.zero_like())
                    .hessian(hessian.clone())
                    .gradient(prev_grad.clone())
                    .cost(cost)
            })
            .ctrlc(false)
            .run()?;

        let sk = sub_state.take_param().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`SR1TrustRegion`: No parameters returned by line search."
        ))?;

        problem.consume_problem(sub_problem);

        let xksk = xk.add(&sk);
        let dfk1 = problem.gradient(&xksk)?;
        let yk = dfk1.sub(&prev_grad);
        let fk1 = problem.cost(&xksk)?;

        let ared = cost - fk1;
        let tmp1: F = prev_grad.dot(&sk);
        let tmp2: F = sk.weighted_dot(&hessian, &sk);
        let tmp2: F = tmp2.mul(float!(0.5));
        let pred = -tmp1 - tmp2;
        let ap = ared / pred;

        let (xk1, fk1, dfk1) = if ap > self.eta {
            (xksk, fk1, dfk1)
        } else {
            (xk, cost, prev_grad)
        };

        self.radius = if ap > float!(0.75) {
            if sk.norm() <= float!(0.8) * self.radius {
                self.radius
            } else {
                float!(2.0) * self.radius
            }
        } else if ap <= float!(0.75) && ap >= float!(0.1) {
            self.radius
        } else {
            float!(0.5) * self.radius
        };

        let bksk = hessian.dot(&sk);
        let ykbksk = yk.sub(&bksk);
        let skykbksk: F = sk.dot(&ykbksk);

        let hessian_update =
            skykbksk.abs() >= self.denominator_factor * sk.norm() * skykbksk.norm();
        let hessian = if hessian_update {
            let a: B = ykbksk.dot(&ykbksk);
            let b: F = sk.dot(&ykbksk);
            hessian.add(&a.mul(&(float!(1.0) / b)))
        } else {
            hessian
        };

        Ok((
            state.param(xk1).cost(fk1).gradient(dfk1).hessian(hessian),
            Some(make_kv!["ared" => ared;
                         "pred" => pred;
                         "ap" => ap;
                         "radius" => self.radius;
                         "hessian_update" => hessian_update;]),
        ))
    }

    fn terminate(&mut self, state: &IterState<P, G, (), B, F>) -> TerminationReason {
        if state.get_gradient().unwrap().norm() < self.tol_grad {
            return TerminationReason::TargetPrecisionReached;
        }
        TerminationReason::NotTerminated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{test_utils::TestProblem, ArgminError, IterState, State};
    use crate::solver::trustregion::CauchyPoint;
    use crate::test_trait_impl;

    test_trait_impl!(sr1, SR1TrustRegion<CauchyPoint<f64>, f64>);

    #[test]
    fn test_new() {
        #[derive(Eq, PartialEq, Debug)]
        struct MyFakeSubProblem {}

        let sr1: SR1TrustRegion<_, f64> = SR1TrustRegion::new(MyFakeSubProblem {});
        let SR1TrustRegion {
            denominator_factor,
            subproblem,
            radius,
            eta,
            tol_grad,
        } = sr1;

        assert_eq!(denominator_factor.to_ne_bytes(), 1e-8f64.to_ne_bytes());
        assert_eq!(subproblem, MyFakeSubProblem {});
        assert_eq!(radius.to_ne_bytes(), 1.0f64.to_ne_bytes());
        assert_eq!(eta.to_ne_bytes(), (0.5f64 * 1e-3f64).to_ne_bytes());
        assert_eq!(tol_grad.to_ne_bytes(), 1e-3f64.to_ne_bytes());
    }

    #[test]
    fn test_with_denominator_factor() {
        #[derive(Eq, PartialEq, Debug, Clone, Copy)]
        struct MyFakeSubProblem {}

        // correct parameters
        for tol in [f64::EPSILON, 1e-8, 1e-6, 1e-2, 1.0 - f64::EPSILON] {
            let sr1: SR1TrustRegion<_, f64> = SR1TrustRegion::new(MyFakeSubProblem {});
            let res = sr1.with_denominator_factor(tol);
            assert!(res.is_ok());

            let nm = res.unwrap();
            assert_eq!(nm.denominator_factor.to_ne_bytes(), tol.to_ne_bytes());
        }

        // incorrect parameters
        for tol in [-f64::EPSILON, 0.0, -1.0, 1.0] {
            let sr1: SR1TrustRegion<_, f64> = SR1TrustRegion::new(MyFakeSubProblem {});
            let res = sr1.with_denominator_factor(tol);
            assert_error!(
                res,
                ArgminError,
                "Invalid parameter: \"`SR1TrustRegion`: denominator_factor must be in (0, 1).\""
            );
        }
    }

    #[test]
    fn test_with_tolerance_grad() {
        #[derive(Eq, PartialEq, Debug, Clone, Copy)]
        struct MyFakeSubProblem {}

        // correct parameters
        for tol in [1e-6, 0.0, 1e-2, 1.0, 2.0] {
            let sr1: SR1TrustRegion<_, f64> = SR1TrustRegion::new(MyFakeSubProblem {});
            let res = sr1.with_tolerance_grad(tol);
            assert!(res.is_ok());

            let nm = res.unwrap();
            assert_eq!(nm.tol_grad.to_ne_bytes(), tol.to_ne_bytes());
        }

        // incorrect parameters
        for tol in [-f64::EPSILON, -1.0, -100.0, -42.0] {
            let sr1: SR1TrustRegion<_, f64> = SR1TrustRegion::new(MyFakeSubProblem {});
            let res = sr1.with_tolerance_grad(tol);
            assert_error!(
                res,
                ArgminError,
                "Invalid parameter: \"`SR1TrustRegion`: gradient tolerance must be >= 0.\""
            );
        }
    }

    #[test]
    fn test_init() {
        let subproblem = CauchyPoint::new();

        let param: Vec<f64> = vec![-1.0, 1.0];

        let mut sr1: SR1TrustRegion<_, f64> = SR1TrustRegion::new(subproblem);

        // Forgot to initialize the parameter vector
        let state: IterState<Vec<f64>, Vec<f64>, (), Vec<Vec<f64>>, f64> = IterState::new();
        let problem = TestProblem::new();
        let res = sr1.init(&mut Problem::new(problem), state);
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Not initialized: \"`SR1TrustRegion` requires an initial parameter vector. Please ",
                "provide an initial guess via `Executor`s `configure` method.\""
            )
        );

        // All good.
        let state: IterState<Vec<f64>, Vec<f64>, (), Vec<Vec<f64>>, f64> =
            IterState::new().param(param.clone());
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

        assert_eq!(state_out.get_cost().to_ne_bytes(), 1.0f64.to_ne_bytes())
    }

    #[test]
    fn test_init_provided_cost() {
        let subproblem = CauchyPoint::new();

        let param: Vec<f64> = vec![-1.0, 1.0];

        let mut sr1: SR1TrustRegion<_, f64> = SR1TrustRegion::new(subproblem);

        let state: IterState<Vec<f64>, Vec<f64>, (), Vec<Vec<f64>>, f64> =
            IterState::new().param(param).cost(1234.0);

        let problem = TestProblem::new();
        let (state_out, kv) = sr1.init(&mut Problem::new(problem), state).unwrap();

        assert!(kv.is_none());

        assert_eq!(state_out.get_cost().to_ne_bytes(), 1234.0f64.to_ne_bytes())
    }

    #[test]
    fn test_init_provided_grad() {
        let subproblem = CauchyPoint::new();

        let param: Vec<f64> = vec![-1.0, 1.0];
        let gradient: Vec<f64> = vec![4.0, 9.0];

        let mut sr1: SR1TrustRegion<_, f64> = SR1TrustRegion::new(subproblem);

        let state: IterState<Vec<f64>, Vec<f64>, (), Vec<Vec<f64>>, f64> =
            IterState::new().param(param).gradient(gradient.clone());

        let problem = TestProblem::new();
        let (mut state_out, kv) = sr1.init(&mut Problem::new(problem), state).unwrap();

        assert!(kv.is_none());

        let s_grad = state_out.take_gradient().unwrap();

        for (s, g) in s_grad.iter().zip(gradient.iter()) {
            assert_eq!(s.to_ne_bytes(), g.to_ne_bytes());
        }
    }

    #[test]
    fn test_init_provided_hessian() {
        let subproblem = CauchyPoint::new();

        let param: Vec<f64> = vec![-1.0, 1.0];
        let gradient: Vec<f64> = vec![4.0, 9.0];
        let hessian: Vec<Vec<f64>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let mut sr1: SR1TrustRegion<_, f64> = SR1TrustRegion::new(subproblem);

        let state: IterState<Vec<f64>, Vec<f64>, (), Vec<Vec<f64>>, f64> = IterState::new()
            .param(param)
            .gradient(gradient)
            .hessian(hessian.clone());

        let problem = TestProblem::new();
        let (mut state_out, kv) = sr1.init(&mut Problem::new(problem), state).unwrap();

        assert!(kv.is_none());

        let s_hessian = state_out.take_hessian().unwrap();

        for (s1, g1) in s_hessian.iter().zip(hessian.iter()) {
            for (s, g) in s1.iter().zip(g1.iter()) {
                assert_eq!(s.to_ne_bytes(), g.to_ne_bytes());
            }
        }
    }
}
