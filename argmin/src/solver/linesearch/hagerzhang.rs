// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{
    ArgminFloat, CostFunction, Error, Gradient, IterState, LineSearch, Problem, SerializeAlias,
    Solver, TerminationReason, TerminationStatus, KV,
};
use argmin_math::{ArgminDot, ArgminScaledAdd};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

type Triplet<F> = (F, F, F);

/// # Hager-Zhang line search
///
/// The Hager-Zhang line search is a method to find a step length which obeys the strong Wolfe
/// conditions.
///
/// ## Requirements on the optimization problem
///
/// The optimization problem is required to implement [`CostFunction`] and [`Gradient`].
///
/// ## Reference
///
/// William W. Hager and Hongchao Zhang. "A new conjugate gradient method with guaranteed
/// descent and an efficient line search." SIAM J. Optim. 16(1), 2006, 170-192.
/// DOI: <https://doi.org/10.1137/030601880>
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct HagerZhangLineSearch<P, G, F> {
    /// delta: (0, 0.5), used in the Wolfe conditions
    delta: F,
    /// sigma: [delta, 1), used in the Wolfe conditions
    sigma: F,
    /// epsilon: [0, infinity), used in the approximate Wolfe termination
    epsilon: F,
    /// epsilon_k
    epsilon_k: F,
    /// theta: (0, 1), used in the update rules when the potential intervals [a, c] or [c, b]
    /// violate the opposite slope condition
    theta: F,
    /// gamma: (0, 1), determines when a bisection step is performed
    gamma: F,
    /// eta: (0, infinity), used in the lower bound for beta_k^N
    eta: F,
    /// initial a
    a_x_init: F,
    /// a
    a_x: F,
    /// phi(a)
    a_f: F,
    /// phi'(a)
    a_g: F,
    /// initial b
    b_x_init: F,
    /// b
    b_x: F,
    /// phi(b)
    b_f: F,
    /// phi'(b)
    b_g: F,
    /// initial c
    c_x_init: F,
    /// c
    c_x: F,
    /// phi(c)
    c_f: F,
    /// phi'(c)
    c_g: F,
    /// best x
    best_x: F,
    /// best function value
    best_f: F,
    /// best slope
    best_g: F,
    /// initial parameter vector
    init_param: Option<P>,
    /// initial cost
    finit: F,
    /// initial gradient (builder)
    init_grad: Option<G>,
    /// Search direction (builder)
    search_direction: Option<P>,
    /// Search direction in 1D
    dginit: F,
}

impl<P, G, F> HagerZhangLineSearch<P, G, F>
where
    P: ArgminScaledAdd<P, F, P> + ArgminDot<G, F>,
    F: ArgminFloat,
{
    /// Construct a new instance of [`HagerZhangLineSearch`]
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::linesearch::HagerZhangLineSearch;
    /// let hzls: HagerZhangLineSearch<Vec<f64>, Vec<f64>, f64> = HagerZhangLineSearch::new();
    /// ```
    pub fn new() -> Self {
        HagerZhangLineSearch {
            delta: float!(0.1),
            sigma: float!(0.9),
            epsilon: float!(1e-6),
            epsilon_k: F::nan(),
            theta: float!(0.5),
            gamma: float!(0.66),
            eta: float!(0.01),
            a_x_init: F::epsilon(),
            a_x: F::nan(),
            a_f: F::nan(),
            a_g: F::nan(),
            b_x_init: float!(1e5),
            b_x: F::nan(),
            b_f: F::nan(),
            b_g: F::nan(),
            c_x_init: float!(1.0),
            c_x: F::nan(),
            c_f: F::nan(),
            c_g: F::nan(),
            best_x: float!(0.0),
            best_f: F::infinity(),
            best_g: F::nan(),
            init_param: None,
            init_grad: None,
            search_direction: None,
            dginit: F::nan(),
            finit: F::infinity(),
        }
    }

    /// Set delta and sigma.
    ///
    /// Delta defaults to `0.1` and must be in `(0, 1)`.
    /// Sigma defaults to `0.9` and must be in `[delta, 1)`.
    ///
    /// Delta and Sigma correspond to the constants `c1` and `c2` of the strong Wolfe conditions,
    /// respectively.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::linesearch::HagerZhangLineSearch;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// let hzls: HagerZhangLineSearch<Vec<f64>, Vec<f64>, f64> =
    ///     HagerZhangLineSearch::new().with_delta_sigma(0.2, 0.8)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_delta_sigma(mut self, delta: F, sigma: F) -> Result<Self, Error> {
        if delta <= float!(0.0) || delta >= float!(1.0) || sigma < delta || sigma >= float!(1.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`HagerZhangLineSearch`: delta must be in (0, 1) and sigma must be in [delta, 1)."
            ));
        }
        self.delta = delta;
        self.sigma = sigma;
        Ok(self)
    }

    /// Set epsilon
    ///
    /// Used in the approximate strong Wolfe condition.
    ///
    /// Must be non-negative and defaults to `1e-6`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::linesearch::HagerZhangLineSearch;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// let hzls: HagerZhangLineSearch<Vec<f64>, Vec<f64>, f64> =
    ///     HagerZhangLineSearch::new().with_epsilon(1e-8)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_epsilon(mut self, epsilon: F) -> Result<Self, Error> {
        if epsilon < float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`HagerZhangLineSearch`: epsilon must be >= 0."
            ));
        }
        self.epsilon = epsilon;
        Ok(self)
    }

    /// Set theta
    ///
    /// Used in the update rules when the potential intervals [a, c] or [c, b] violate the opposite
    /// slope condition.
    ///
    /// Must be in `(0, 1)` and defaults to `0.5`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::linesearch::HagerZhangLineSearch;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// let hzls: HagerZhangLineSearch<Vec<f64>, Vec<f64>, f64> =
    ///     HagerZhangLineSearch::new().with_theta(0.4)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_theta(mut self, theta: F) -> Result<Self, Error> {
        if theta <= float!(0.0) || theta >= float!(1.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`HagerZhangLineSearch`: theta must be in (0, 1)."
            ));
        }
        self.theta = theta;
        Ok(self)
    }

    /// Set gamma
    ///
    /// Determines when a bisection step is performed.
    ///
    /// Must be in `(0, 1)` and defaults to `0.66`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::linesearch::HagerZhangLineSearch;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// let hzls: HagerZhangLineSearch<Vec<f64>, Vec<f64>, f64> =
    ///     HagerZhangLineSearch::new().with_gamma(0.7)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_gamma(mut self, gamma: F) -> Result<Self, Error> {
        if gamma <= float!(0.0) || gamma >= float!(1.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`HagerZhangLineSearch`: gamma must be in (0, 1)."
            ));
        }
        self.gamma = gamma;
        Ok(self)
    }

    /// Set eta
    ///
    /// Used in the lower bound for `beta_k^N`.
    ///
    /// Must be larger than zero and defaults to `0.01`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::linesearch::HagerZhangLineSearch;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// let hzls: HagerZhangLineSearch<Vec<f64>, Vec<f64>, f64> =
    ///     HagerZhangLineSearch::new().with_eta(0.02)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_eta(mut self, eta: F) -> Result<Self, Error> {
        if eta <= float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`HagerZhangLineSearch`: eta must be > 0."
            ));
        }
        self.eta = eta;
        Ok(self)
    }

    /// Set lower and upper bound of step
    ///
    /// Defaults to a minimum step length of `EPSILON` and a maximum step length of `1e5`.
    ///
    /// The chosen values must satisfy `0 <= step_min < step_max`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::linesearch::HagerZhangLineSearch;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// let hzls: HagerZhangLineSearch<Vec<f64>, Vec<f64>, f64> =
    ///     HagerZhangLineSearch::new().with_bounds(1e-3, 1.0)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_bounds(mut self, step_min: F, step_max: F) -> Result<Self, Error> {
        if step_min < float!(0.0) || step_max <= step_min {
            return Err(argmin_error!(
                InvalidParameter,
                concat!(
                    "`HagerZhangLineSearch`: minimum and maximum step length must be chosen ",
                    "such that 0 <= step_min < step_max."
                )
            ));
        }
        self.a_x_init = step_min;
        self.b_x_init = step_max;
        Ok(self)
    }

    fn update<O>(
        &mut self,
        problem: &mut Problem<O>,
        (a_x, a_f, a_g): Triplet<F>,
        (b_x, b_f, b_g): Triplet<F>,
        (c_x, c_f, c_g): Triplet<F>,
    ) -> Result<(Triplet<F>, Triplet<F>), Error>
    where
        O: CostFunction<Param = P, Output = F> + Gradient<Param = P, Gradient = G>,
    {
        // U0
        if c_x <= a_x || c_x >= b_x {
            // nothing changes.
            return Ok(((a_x, a_f, a_g), (b_x, b_f, b_g)));
        }

        // U1
        if c_g >= float!(0.0) {
            return Ok(((a_x, a_f, a_g), (c_x, c_f, c_g)));
        }

        // U2
        if c_g < float!(0.0) && c_f <= self.finit + self.epsilon_k {
            return Ok(((c_x, c_f, c_g), (b_x, b_f, b_g)));
        }

        // U3
        if c_g < float!(0.0) && c_f > self.finit + self.epsilon_k {
            let mut ah_x = a_x;
            let mut ah_f = a_f;
            let mut ah_g = a_g;
            let mut bh_x = c_x;
            loop {
                let d_x = (float!(1.0) - self.theta) * ah_x + self.theta * bh_x;
                let d_f = self.calc(problem, d_x)?;
                let d_g = self.calc_grad(problem, d_x)?;
                if d_g >= float!(0.0) {
                    return Ok(((ah_x, ah_f, ah_g), (d_x, d_f, d_g)));
                }
                if d_g < float!(0.0) && d_f <= self.finit + self.epsilon_k {
                    ah_x = d_x;
                    ah_f = d_f;
                    ah_g = d_g;
                }
                if d_g < float!(0.0) && d_f > self.finit + self.epsilon_k {
                    bh_x = d_x;
                }
            }
        }

        // return Ok(((a_x, a_f, a_g), (b_x, b_f, b_g)));
        Err(argmin_error!(
            PotentialBug,
            "`HagerZhangLineSearch`: Reached unreachable point in `update` method."
        ))
    }

    /// secant step
    fn secant(&self, a_x: F, a_g: F, b_x: F, b_g: F) -> F {
        (a_x * b_g - b_x * a_g) / (b_g - a_g)
    }

    /// double secant step
    fn secant2<O>(
        &mut self,
        problem: &mut Problem<O>,
        (a_x, a_f, a_g): Triplet<F>,
        (b_x, b_f, b_g): Triplet<F>,
    ) -> Result<(Triplet<F>, Triplet<F>), Error>
    where
        O: CostFunction<Param = P, Output = F> + Gradient<Param = P, Gradient = G>,
    {
        // S1
        let c_x = self.secant(a_x, a_g, b_x, b_g);
        let c_f = self.calc(problem, c_x)?;
        let c_g = self.calc_grad(problem, c_x)?;
        let mut c_bar_x: F = float!(0.0);

        let ((aa_x, aa_f, aa_g), (bb_x, bb_f, bb_g)) =
            self.update(problem, (a_x, a_f, a_g), (b_x, b_f, b_g), (c_x, c_f, c_g))?;

        // S2
        if (c_x - bb_x).abs() < F::epsilon() {
            c_bar_x = self.secant(b_x, b_g, bb_x, bb_g);
        }

        // S3
        if (c_x - aa_x).abs() < F::epsilon() {
            c_bar_x = self.secant(a_x, a_g, aa_x, aa_g);
        }

        // S4
        if (c_x - aa_x).abs() < F::epsilon() || (c_x - bb_x).abs() < F::epsilon() {
            let c_bar_f = self.calc(problem, c_bar_x)?;
            let c_bar_g = self.calc_grad(problem, c_bar_x)?;

            let (a_bar, b_bar) = self.update(
                problem,
                (aa_x, aa_f, aa_g),
                (bb_x, bb_f, bb_g),
                (c_bar_x, c_bar_f, c_bar_g),
            )?;
            Ok((a_bar, b_bar))
        } else {
            Ok(((aa_x, aa_f, aa_g), (bb_x, bb_f, bb_g)))
        }
    }

    fn calc<O>(&mut self, problem: &mut Problem<O>, alpha: F) -> Result<F, Error>
    where
        O: CostFunction<Param = P, Output = F>,
    {
        let tmp = self
            .init_param
            .as_ref()
            .ok_or_else(argmin_error_closure!(
                PotentialBug,
                "`HagerZhangLineSearch`: `init_param` is `None` in `calc`."
            ))?
            .scaled_add(&alpha, self.search_direction.as_ref().unwrap());
        problem.cost(&tmp)
    }

    fn calc_grad<O>(&mut self, problem: &mut Problem<O>, alpha: F) -> Result<F, Error>
    where
        O: Gradient<Param = P, Gradient = G>,
    {
        let tmp = self
            .init_param
            .as_ref()
            .ok_or_else(argmin_error_closure!(
                PotentialBug,
                "`HagerZhangLineSearch`: `init_param` is `None` in `calc_grad`."
            ))?
            .scaled_add(&alpha, self.search_direction.as_ref().unwrap());
        let grad = problem.gradient(&tmp)?;
        Ok(self.search_direction.as_ref().unwrap().dot(&grad))
    }

    fn set_best(&mut self) {
        if self.a_f <= self.b_f && self.a_f <= self.c_f {
            self.best_x = self.a_x;
            self.best_f = self.a_f;
            self.best_g = self.a_g;
        }

        if self.b_f <= self.a_f && self.b_f <= self.c_f {
            self.best_x = self.b_x;
            self.best_f = self.b_f;
            self.best_g = self.b_g;
        }

        if self.c_f <= self.a_f && self.c_f <= self.b_f {
            self.best_x = self.c_x;
            self.best_f = self.c_f;
            self.best_g = self.c_g;
        }
    }
}

impl<P, G, F> Default for HagerZhangLineSearch<P, G, F>
where
    P: ArgminScaledAdd<P, F, P> + ArgminDot<G, F>,
    F: ArgminFloat,
{
    fn default() -> Self {
        HagerZhangLineSearch::new()
    }
}

impl<P, G, F> LineSearch<P, F> for HagerZhangLineSearch<P, G, F> {
    /// Set search direction
    fn search_direction(&mut self, search_direction: P) {
        self.search_direction = Some(search_direction);
    }

    /// Set initial alpha value
    fn initial_step_length(&mut self, alpha: F) -> Result<(), Error> {
        self.c_x_init = alpha;
        Ok(())
    }
}

impl<P, G, O, F> Solver<O, IterState<P, G, (), (), F>> for HagerZhangLineSearch<P, G, F>
where
    O: CostFunction<Param = P, Output = F> + Gradient<Param = P, Gradient = G>,
    P: Clone + SerializeAlias + ArgminDot<G, F> + ArgminScaledAdd<P, F, P>,
    G: Clone + SerializeAlias + ArgminDot<P, F>,
    F: ArgminFloat,
{
    const NAME: &'static str = "Hager-Zhang line search";

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        mut state: IterState<P, G, (), (), F>,
    ) -> Result<(IterState<P, G, (), (), F>, Option<KV>), Error> {
        check_param!(
            self.search_direction,
            concat!(
                "`HagerZhangLineSearch`: Search direction not initialized. ",
                "Call `search_direction` before executing the solver."
            )
        );

        self.init_param = Some(state.take_param().ok_or_else(argmin_error_closure!(
            NotInitialized,
            concat!(
                "`HagerZhangLineSearch` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method."
            )
        ))?);

        let cost = state.get_cost();
        self.finit = if cost.is_infinite() {
            problem.cost(self.init_param.as_ref().unwrap())?
        } else {
            cost
        };

        self.init_grad = Some(
            state
                .take_gradient()
                .map(Result::Ok)
                .unwrap_or_else(|| problem.gradient(self.init_param.as_ref().unwrap()))?,
        );

        self.a_x = self.a_x_init;
        self.b_x = self.b_x_init;
        self.c_x = self.c_x_init;

        self.a_f = self.calc(problem, self.a_x)?;
        self.a_g = self.calc_grad(problem, self.a_x)?;
        self.b_f = self.calc(problem, self.b_x)?;
        self.b_g = self.calc_grad(problem, self.b_x)?;
        self.c_f = self.calc(problem, self.c_x)?;
        self.c_g = self.calc_grad(problem, self.c_x)?;

        self.epsilon_k = self.epsilon * self.finit.abs();

        self.dginit = self
            .init_grad
            .as_ref()
            .unwrap()
            .dot(self.search_direction.as_ref().unwrap());

        self.set_best();
        let new_param = self
            .init_param
            .as_ref()
            .unwrap()
            .scaled_add(&self.best_x, self.search_direction.as_ref().unwrap());
        let best_f = self.best_f;

        Ok((state.param(new_param).cost(best_f), None))
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        state: IterState<P, G, (), (), F>,
    ) -> Result<(IterState<P, G, (), (), F>, Option<KV>), Error> {
        // L1
        let aa = (self.a_x, self.a_f, self.a_g);
        let bb = (self.b_x, self.b_f, self.b_g);
        let ((mut at_x, mut at_f, mut at_g), (mut bt_x, mut bt_f, mut bt_g)) =
            self.secant2(problem, aa, bb)?;

        // L2
        if bt_x - at_x > self.gamma * (self.b_x - self.a_x) {
            let c_x = (at_x + bt_x) / float!(2.0);
            let tmp = self
                .init_param
                .as_ref()
                .unwrap()
                .scaled_add(&c_x, self.search_direction.as_ref().unwrap());
            let c_f = problem.cost(&tmp)?;
            let grad = problem.gradient(&tmp)?;
            let c_g = self.search_direction.as_ref().unwrap().dot(&grad);
            let ((an_x, an_f, an_g), (bn_x, bn_f, bn_g)) = self.update(
                problem,
                (at_x, at_f, at_g),
                (bt_x, bt_f, bt_g),
                (c_x, c_f, c_g),
            )?;
            at_x = an_x;
            at_f = an_f;
            at_g = an_g;
            bt_x = bn_x;
            bt_f = bn_f;
            bt_g = bn_g;
        }

        // L3
        self.a_x = at_x;
        self.a_f = at_f;
        self.a_g = at_g;
        self.b_x = bt_x;
        self.b_f = bt_f;
        self.b_g = bt_g;

        self.set_best();
        let new_param = self
            .init_param
            .as_ref()
            .unwrap()
            .scaled_add(&self.best_x, self.search_direction.as_ref().unwrap());
        Ok((state.param(new_param).cost(self.best_f), None))
    }

    fn terminate(&mut self, _state: &IterState<P, G, (), (), F>) -> TerminationStatus {
        if self.best_f - self.finit <= self.delta * self.best_x * self.dginit
            && self.best_g >= self.sigma * self.dginit
        {
            return TerminationStatus::Terminated(TerminationReason::SolverConverged);
        }
        if (float!(2.0) * self.delta - float!(1.0)) * self.dginit >= self.best_g
            && self.best_g >= self.sigma * self.dginit
            && self.best_f <= self.finit + self.epsilon_k
        {
            return TerminationStatus::Terminated(TerminationReason::SolverConverged);
        }
        TerminationStatus::NotTerminated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{test_utils::TestProblem, ArgminError, IterState, Problem, State};
    use crate::test_trait_impl;
    use approx::assert_relative_eq;

    test_trait_impl!(hagerzhang, HagerZhangLineSearch<Vec<f64>, Vec<f64>, f64>);

    #[test]
    fn test_new() {
        let hzls: HagerZhangLineSearch<Vec<f64>, Vec<f64>, f64> = HagerZhangLineSearch::new();
        let HagerZhangLineSearch {
            delta,
            sigma,
            epsilon,
            epsilon_k,
            theta,
            gamma,
            eta,
            a_x_init,
            a_x,
            a_f,
            a_g,
            b_x_init,
            b_x,
            b_f,
            b_g,
            c_x_init,
            c_x,
            c_f,
            c_g,
            best_x,
            best_f,
            best_g,
            init_param,
            init_grad,
            search_direction,
            dginit,
            finit,
        } = hzls;

        assert_relative_eq!(delta, 0.1f64, epsilon = f64::EPSILON);
        assert_relative_eq!(sigma, 0.9f64, epsilon = f64::EPSILON);
        assert_relative_eq!(epsilon, 1e-6f64, epsilon = f64::EPSILON);
        assert!(epsilon_k.is_nan());
        assert_relative_eq!(theta, 0.5f64, epsilon = f64::EPSILON);
        assert_relative_eq!(gamma, 0.66f64, epsilon = f64::EPSILON);
        assert_relative_eq!(eta, 0.01f64, epsilon = f64::EPSILON);
        assert_relative_eq!(a_x_init, f64::EPSILON, epsilon = f64::EPSILON);
        assert!(a_x.is_nan());
        assert!(a_f.is_nan());
        assert!(a_g.is_nan());
        assert_relative_eq!(b_x_init, 1e5f64, epsilon = f64::EPSILON);
        assert!(b_x.is_nan());
        assert!(b_f.is_nan());
        assert!(b_g.is_nan());
        assert_relative_eq!(c_x_init, 1.0f64, epsilon = f64::EPSILON);
        assert!(c_x.is_nan());
        assert!(c_f.is_nan());
        assert!(c_g.is_nan());
        assert_relative_eq!(best_x, 0.0f64, epsilon = f64::EPSILON);
        assert!(best_f.is_infinite());
        assert!(best_f.is_sign_positive());
        assert!(best_g.is_nan());
        assert!(init_param.is_none());
        assert!(init_grad.is_none());
        assert!(search_direction.is_none());
        assert!(dginit.is_nan());
        assert!(finit.is_infinite());
        assert!(finit.is_sign_positive());
    }

    #[test]
    fn test_with_delta_sigma() {
        // correct parameters
        for (delta, sigma) in [
            (0.2, 0.8),
            (0.5, 0.5),
            (0.0 + f64::EPSILON, 0.5),
            (0.2, 1.0 - f64::EPSILON),
            (0.5, 0.5),
        ] {
            let hzls: HagerZhangLineSearch<Vec<f64>, Vec<f64>, f64> = HagerZhangLineSearch::new();
            let res = hzls.with_delta_sigma(delta, sigma);
            assert!(res.is_ok());

            let hzls = res.unwrap();
            assert_relative_eq!(hzls.delta, delta, epsilon = f64::EPSILON);
            assert_relative_eq!(hzls.sigma, sigma, epsilon = f64::EPSILON);
        }

        // incorrect parameters
        for (delta, sigma) in [
            (-1.0, 0.5),
            (0.0, 0.5),
            (1.0, 0.5),
            (2.0, 0.5),
            (0.5, 0.5 - f64::EPSILON),
            (0.5, 0.0),
            (0.5, 1.0),
            (0.5, 2.0),
            (0.6, 0.2),
        ] {
            let hzls: HagerZhangLineSearch<Vec<f64>, Vec<f64>, f64> = HagerZhangLineSearch::new();
            let res = hzls.with_delta_sigma(delta, sigma);
            assert_error!(
                res,
                ArgminError,
                concat!(
                    "Invalid parameter: \"`HagerZhangLineSearch`: ",
                    "delta must be in (0, 1) and sigma must be in [delta, 1).\""
                )
            );
        }
    }

    #[test]
    fn test_with_epsilon() {
        // correct parameters
        for epsilon in [1e-6, 0.0, 1e-2, 1.0, 2.0] {
            let hzls: HagerZhangLineSearch<Vec<f64>, Vec<f64>, f64> = HagerZhangLineSearch::new();
            let res = hzls.with_epsilon(epsilon);
            assert!(res.is_ok());

            let hzls = res.unwrap();
            assert_relative_eq!(hzls.epsilon, epsilon, epsilon = f64::EPSILON);
        }

        // incorrect parameters
        for epsilon in [-f64::EPSILON, -1.0, -100.0, -42.0] {
            let hzls: HagerZhangLineSearch<Vec<f64>, Vec<f64>, f64> = HagerZhangLineSearch::new();
            let res = hzls.with_epsilon(epsilon);
            assert_error!(
                res,
                ArgminError,
                concat!(
                    "Invalid parameter: \"`HagerZhangLineSearch`: ",
                    "epsilon must be >= 0.\""
                )
            );
        }
    }

    #[test]
    fn test_with_theta() {
        // correct parameters
        for theta in [0.0 + f64::EPSILON, 1e-2, 0.5, 0.6, 1.0 - f64::EPSILON] {
            let hzls: HagerZhangLineSearch<Vec<f64>, Vec<f64>, f64> = HagerZhangLineSearch::new();
            let res = hzls.with_theta(theta);
            assert!(res.is_ok());

            let hzls = res.unwrap();
            assert_relative_eq!(hzls.theta, theta, epsilon = f64::EPSILON);
        }

        // incorrect parameters
        for theta in [0.0, 1.0, -100.0, 42.0] {
            let hzls: HagerZhangLineSearch<Vec<f64>, Vec<f64>, f64> = HagerZhangLineSearch::new();
            let res = hzls.with_theta(theta);
            assert_error!(
                res,
                ArgminError,
                concat!(
                    "Invalid parameter: \"`HagerZhangLineSearch`: ",
                    "theta must be in (0, 1).\""
                )
            );
        }
    }

    #[test]
    fn test_with_gamma() {
        // correct parameters
        for gamma in [0.0 + f64::EPSILON, 1e-2, 0.5, 0.6, 1.0 - f64::EPSILON] {
            let hzls: HagerZhangLineSearch<Vec<f64>, Vec<f64>, f64> = HagerZhangLineSearch::new();
            let res = hzls.with_gamma(gamma);
            assert!(res.is_ok());

            let hzls = res.unwrap();
            assert_relative_eq!(hzls.gamma, gamma, epsilon = f64::EPSILON);
        }

        // incorrect parameters
        for gamma in [0.0, 1.0, -100.0, 42.0] {
            let hzls: HagerZhangLineSearch<Vec<f64>, Vec<f64>, f64> = HagerZhangLineSearch::new();
            let res = hzls.with_gamma(gamma);
            assert_error!(
                res,
                ArgminError,
                concat!(
                    "Invalid parameter: \"`HagerZhangLineSearch`: ",
                    "gamma must be in (0, 1).\""
                )
            );
        }
    }

    #[test]
    fn test_with_eta() {
        // correct parameters
        for eta in [0.0 + f64::EPSILON, 1e-2, 0.5, 1.0, 10.0] {
            let hzls: HagerZhangLineSearch<Vec<f64>, Vec<f64>, f64> = HagerZhangLineSearch::new();
            let res = hzls.with_eta(eta);
            assert!(res.is_ok());

            let hzls = res.unwrap();
            assert_relative_eq!(hzls.eta, eta);
        }

        // incorrect parameters
        for eta in [0.0, -f64::EPSILON, -100.0, -42.0] {
            let hzls: HagerZhangLineSearch<Vec<f64>, Vec<f64>, f64> = HagerZhangLineSearch::new();
            let res = hzls.with_eta(eta);
            assert_error!(
                res,
                ArgminError,
                concat!(
                    "Invalid parameter: \"`HagerZhangLineSearch`: ",
                    "eta must be > 0.\""
                )
            );
        }
    }

    #[test]
    fn test_with_bounds() {
        // correct parameters
        for (min, max) in [
            (0.2, 0.8),
            (0.5 - f64::EPSILON, 0.5),
            (0.5, 0.5 + f64::EPSILON),
            (0.0, 0.5),
            (0.0 + f64::EPSILON, 0.5),
            (50.0, 100.0),
        ] {
            let hzls: HagerZhangLineSearch<Vec<f64>, Vec<f64>, f64> = HagerZhangLineSearch::new();
            let res = hzls.with_bounds(min, max);
            assert!(res.is_ok());

            let hzls = res.unwrap();
            assert_relative_eq!(hzls.a_x_init, min, epsilon = f64::EPSILON);
            assert_relative_eq!(hzls.b_x_init, max, epsilon = f64::EPSILON);
        }

        // incorrect parameters
        for (min, max) in [
            (-1.0, 0.5),
            (0.5, 0.5),
            (0.5 + f64::EPSILON, 0.5),
            (0.5, 0.0),
            (-1000.0, -100.0),
        ] {
            let hzls: HagerZhangLineSearch<Vec<f64>, Vec<f64>, f64> = HagerZhangLineSearch::new();
            let res = hzls.with_bounds(min, max);
            assert_error!(
                res,
                ArgminError,
                concat!(
                    "Invalid parameter: \"`HagerZhangLineSearch`: minimum and maximum step length ",
                    "must be chosen such that 0 <= step_min < step_max.\""
                )
            );
        }
    }

    #[test]
    fn test_init_search_direction_not_set() {
        let mut hzls: HagerZhangLineSearch<Vec<f64>, Vec<f64>, f64> = HagerZhangLineSearch::new();
        let res = hzls.init(&mut Problem::new(TestProblem::new()), IterState::new());
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Not initialized: \"`HagerZhangLineSearch`: Search direction not initialized. ",
                "Call `search_direction` before executing the solver.\""
            )
        );
    }

    #[test]
    fn test_init_param_not_set() {
        let mut hzls: HagerZhangLineSearch<Vec<f64>, Vec<f64>, f64> = HagerZhangLineSearch::new();
        hzls.search_direction(vec![1.0f64]);
        let res = hzls.init(&mut Problem::new(TestProblem::new()), IterState::new());
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Not initialized: \"`HagerZhangLineSearch` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method.\""
            )
        );
    }
}
