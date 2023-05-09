// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

// Deactivating this lint here because it would make the Boolean expressions more difficult to
// read.
#![allow(clippy::nonminimal_bool)]

use crate::core::{
    ArgminFloat, CostFunction, Error, Gradient, IterState, LineSearch, Problem, SerializeAlias,
    Solver, State, TerminationReason, KV,
};
use argmin_math::{ArgminDot, ArgminScaledAdd};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::{default::Default, fmt::Debug};

/// # More-Thuente line search
///
/// The More-Thuente line search is a method which finds an appropriate step length from a starting
/// point and a search direction. This point obeys the strong Wolfe conditions.
///
/// With the method [`with_c`](`MoreThuenteLineSearch::with_c`) the scaling factors for the
/// sufficient decrease condition and the curvature condition can be supplied. By default they are
/// set to `c1 = 1e-4` and `c2 = 0.9`.
///
/// Bounds on the range where step lengths are being searched for can be set with
/// [`with_bounds`](`MoreThuenteLineSearch::with_bounds`) which accepts a lower and an upper bound.
/// Both values need to be non-negative and `lower < upper`.
///
/// One of the reasons for the algorithm to terminate is when the the relative width of the
/// uncertainty interval is smaller than a given tolerance (default: `1e-10`). This tolerance can
/// be set via [`with_width_tolerance`](`MoreThuenteLineSearch::with_width_tolerance`) and must be
/// non-negative.
///
/// TODO: Add missing stopping criteria!
///
/// ## Requirements on the optimization problem
///
/// The optimization problem is required to implement [`CostFunction`] and [`Gradient`].
///
/// ## References
///
/// This implementation follows the excellent MATLAB implementation of Dianne P. O'Leary at
/// <http://www.cs.umd.edu/users/oleary/software/>
///
/// Jorge J. More and David J. Thuente. "Line search algorithms with guaranteed sufficient
/// decrease." ACM Trans. Math. Softw. 20, 3 (September 1994), 286-307.
/// DOI: <https://doi.org/10.1145/192115.192132>
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct MoreThuenteLineSearch<P, G, F> {
    /// Search direction
    search_direction: Option<P>,
    /// initial parameter vector
    init_param: Option<P>,
    /// initial cost
    finit: F,
    /// initial gradient
    init_grad: Option<G>,
    /// Search direction in 1D
    dginit: F,
    /// dgtest
    dgtest: F,
    /// c1
    ftol: F,
    /// c2
    gtol: F,
    /// xtrapf
    xtrapf: F,
    /// width of interval
    width: F,
    /// width of what?
    width1: F,
    /// xtol
    xtol: F,
    /// alpha
    alpha: F,
    /// stpmin
    stpmin: F,
    /// stpmax
    stpmax: F,
    /// current step
    stp: Step<F>,
    /// stx (one endpoint of uncertainty interval)
    stx: Step<F>,
    /// sty (another endpoint of uncertainty interval)
    sty: Step<F>,
    /// f
    f: F,
    /// bracketed
    brackt: bool,
    /// stage1
    stage1: bool,
    /// infoc
    infoc: usize,
}

#[derive(Clone, Eq, PartialEq, Debug)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
struct Step<F> {
    pub x: F,
    pub fx: F,
    pub gx: F,
}

impl<F> Step<F> {
    /// Create a new instance of `Step`
    pub fn new(x: F, fx: F, gx: F) -> Self {
        Step { x, fx, gx }
    }
}

impl<F> Default for Step<F>
where
    F: ArgminFloat,
{
    fn default() -> Self {
        Step {
            x: float!(0.0),
            fx: float!(0.0),
            gx: float!(0.0),
        }
    }
}

impl<P, G, F> MoreThuenteLineSearch<P, G, F>
where
    F: ArgminFloat,
{
    /// Construct a new instance of `MoreThuenteLineSearch`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::linesearch::MoreThuenteLineSearch;
    /// let mtls: MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64> = MoreThuenteLineSearch::new();
    /// ```
    pub fn new() -> Self {
        MoreThuenteLineSearch {
            search_direction: None,
            init_param: None,
            finit: F::infinity(),
            init_grad: None,
            dginit: float!(0.0),
            dgtest: float!(0.0),
            ftol: float!(1e-4),
            gtol: float!(0.9),
            xtrapf: float!(4.0),
            width: F::nan(),
            width1: F::nan(),
            xtol: float!(1e-10),
            alpha: float!(1.0),
            stpmin: F::epsilon().sqrt(),
            stpmax: F::infinity(),
            stp: Step::default(),
            stx: Step::default(),
            sty: Step::default(),
            f: F::nan(),
            brackt: false,
            stage1: true,
            infoc: 1,
        }
    }

    /// Set the constants c1 and c2 for the sufficient decrease and curvature conditions,
    /// respectively. `0 < c1 < c2 < 1` must hold.
    ///
    /// The default values are `c1 = 1e-4` and `c2 = 0.9`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::linesearch::MoreThuenteLineSearch;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// let mtls: MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64> =
    ///     MoreThuenteLineSearch::new().with_c(1e-3, 0.8)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_c(mut self, c1: F, c2: F) -> Result<Self, Error> {
        if c1 <= float!(0.0) || c1 >= c2 {
            return Err(argmin_error!(
                InvalidParameter,
                "`MoreThuenteLineSearch`: Parameter c1 must be in (0, c2)."
            ));
        }
        if c2 <= c1 || c2 >= float!(1.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`MoreThuenteLineSearch`: Parameter c2 must be in (c1, 1)."
            ));
        }
        self.ftol = c1;
        self.gtol = c2;
        Ok(self)
    }

    /// Set lower and upper bound of step
    ///
    /// Defaults are `step_min = sqrt(EPS)` and `step_max = INF`.
    ///
    /// `step_min` must be smaller than `step_max`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::linesearch::MoreThuenteLineSearch;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// let mtls: MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64> =
    ///     MoreThuenteLineSearch::new().with_bounds(1e-6, 10.0)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_bounds(mut self, step_min: F, step_max: F) -> Result<Self, Error> {
        if step_min < float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`MoreThuenteLineSearch`: step_min must be >= 0.0."
            ));
        }
        if step_max <= step_min {
            return Err(argmin_error!(
                InvalidParameter,
                "`MoreThuenteLineSearch`: step_min must be smaller than step_max."
            ));
        }
        self.stpmin = step_min;
        self.stpmax = step_max;
        Ok(self)
    }

    /// Set relative tolerance on width of uncertainty interval
    ///
    /// The algorithm terminates when the relative width of the uncertainty interval is below the
    /// supplied tolerance.
    ///
    /// Must be non-negative and defaults to `1e-10`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::linesearch::MoreThuenteLineSearch;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// let mtls: MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64> =
    ///     MoreThuenteLineSearch::new().with_width_tolerance(1e-9)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_width_tolerance(mut self, xtol: F) -> Result<Self, Error> {
        if xtol < float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`MoreThuenteLineSearch`: relative width tolerance must be >= 0.0."
            ));
        }
        self.xtol = xtol;
        Ok(self)
    }
}

impl<P, G, F> Default for MoreThuenteLineSearch<P, G, F>
where
    F: ArgminFloat,
{
    fn default() -> Self {
        MoreThuenteLineSearch::new()
    }
}

impl<P, G, F> LineSearch<P, F> for MoreThuenteLineSearch<P, G, F>
where
    F: ArgminFloat,
{
    /// Set search direction
    fn search_direction(&mut self, search_direction: P) {
        self.search_direction = Some(search_direction);
    }

    /// Set initial alpha value
    fn initial_step_length(&mut self, alpha: F) -> Result<(), Error> {
        if alpha <= float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "MoreThuenteLineSearch: Initial alpha must be > 0."
            ));
        }
        self.alpha = alpha;
        Ok(())
    }
}

impl<P, G, O, F> Solver<O, IterState<P, G, (), (), F>> for MoreThuenteLineSearch<P, G, F>
where
    O: CostFunction<Param = P, Output = F> + Gradient<Param = P, Gradient = G>,
    P: Clone + Debug + SerializeAlias + ArgminDot<G, F> + ArgminScaledAdd<P, F, P>,
    G: Clone + SerializeAlias + ArgminDot<P, F>,
    F: ArgminFloat,
{
    const NAME: &'static str = "More-Thuente Line search";

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        mut state: IterState<P, G, (), (), F>,
    ) -> Result<(IterState<P, G, (), (), F>, Option<KV>), Error> {
        check_param!(
            self.search_direction,
            concat!(
                "`MoreThuenteLineSearch`: Search direction not initialized. ",
                "Call `search_direction` before executing the solver."
            )
        );

        self.init_param = Some(state.take_param().ok_or_else(argmin_error_closure!(
            NotInitialized,
            concat!(
                "`MoreThuenteLineSearch` requires an initial parameter vector. ",
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

        self.dginit = self
            .init_grad
            .as_ref()
            .unwrap()
            .dot(self.search_direction.as_ref().unwrap());

        // compute search direction in 1D
        if self.dginit >= float!(0.0) {
            return Err(argmin_error!(
                ConditionViolated,
                "`MoreThuenteLineSearch`: Search direction must be a descent direction."
            ));
        }

        self.stage1 = true;
        self.brackt = false;

        self.dgtest = self.ftol * self.dginit;
        self.width = self.stpmax - self.stpmin;
        self.width1 = float!(2.0) * self.width;
        self.f = self.finit;

        self.stp = Step::new(self.alpha, F::nan(), F::nan());
        self.stx = Step::new(float!(0.0), self.finit, self.dginit);
        self.sty = Step::new(float!(0.0), self.finit, self.dginit);

        Ok((state, None))
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        state: IterState<P, G, (), (), F>,
    ) -> Result<(IterState<P, G, (), (), F>, Option<KV>), Error> {
        // set the minimum and maximum steps to correspond to the present interval of uncertainty
        let mut info = 0;
        let (stmin, stmax) = if self.brackt {
            (self.stx.x.min(self.sty.x), self.stx.x.max(self.sty.x))
        } else {
            (
                self.stx.x,
                self.stp.x + self.xtrapf * (self.stp.x - self.stx.x),
            )
        };

        // alpha needs to be within bounds
        self.stp.x = self.stp.x.max(self.stpmin);
        self.stp.x = self.stp.x.min(self.stpmax);

        // If an unusual termination is to occur then let alpha be the lowest point obtained so
        // far.
        if (self.brackt && (self.stp.x <= stmin || self.stp.x >= stmax))
            || (self.brackt && (stmax - stmin) <= self.xtol * stmax)
            || self.infoc == 0
        {
            self.stp.x = self.stx.x;
        }

        // Evaluate the function and gradient at new stp.x and compute the directional derivative
        let new_param = self
            .init_param
            .as_ref()
            .unwrap()
            .scaled_add(&self.stp.x, self.search_direction.as_ref().unwrap());
        self.f = problem.cost(&new_param)?;
        let new_grad = problem.gradient(&new_param)?;
        let cur_cost = self.f;
        let cur_param = new_param;
        let cur_grad = new_grad.clone();
        // self.stx.fx = new_cost;
        let dg = self.search_direction.as_ref().unwrap().dot(&new_grad);
        let ftest1 = self.finit + self.stp.x * self.dgtest;
        // self.stp.fx = new_cost;
        // self.stp.gx = dg;

        if (self.brackt && (self.stp.x <= stmin || self.stp.x >= stmax)) || self.infoc == 0 {
            info = 6;
        }

        if (self.stp.x - self.stpmax).abs() < F::epsilon() && self.f <= ftest1 && dg <= self.dgtest
        {
            info = 5;
        }

        if (self.stp.x - self.stpmin).abs() < F::epsilon() && (self.f > ftest1 || dg >= self.dgtest)
        {
            info = 4;
        }

        if self.brackt && stmax - stmin <= self.xtol * stmax {
            info = 2;
        }

        if self.f <= ftest1 && dg.abs() <= self.gtol * (-self.dginit) {
            info = 1;
        }

        if info != 0 {
            return Ok((
                state
                    .param(cur_param)
                    .cost(cur_cost)
                    .gradient(cur_grad)
                    .terminate_with(TerminationReason::SolverConverged),
                None,
            ));
        }

        if self.stage1 && self.f <= ftest1 && dg >= self.ftol.min(self.gtol) * self.dginit {
            self.stage1 = false;
        }

        if self.stage1 && self.f <= self.stp.fx && self.f > ftest1 {
            let fm = self.f - self.stp.x * self.dgtest;
            let fxm = self.stx.fx - self.stx.x * self.dgtest;
            let fym = self.sty.fx - self.sty.x * self.dgtest;
            let dgm = dg - self.dgtest;
            let dgxm = self.stx.gx - self.dgtest;
            let dgym = self.sty.gx - self.dgtest;

            let (stx1, sty1, stp1, brackt1, _stmin, _stmax, infoc) = cstep(
                Step::new(self.stx.x, fxm, dgxm),
                Step::new(self.sty.x, fym, dgym),
                Step::new(self.stp.x, fm, dgm),
                self.brackt,
                stmin,
                stmax,
            )?;

            self.stx.x = stx1.x;
            self.sty.x = sty1.x;
            self.stp.x = stp1.x;
            self.stx.fx = self.stx.fx + stx1.x * self.dgtest;
            self.sty.fx = self.sty.fx + sty1.x * self.dgtest;
            self.stx.gx = self.stx.gx + self.dgtest;
            self.sty.gx = self.sty.gx + self.dgtest;
            self.brackt = brackt1;
            self.stp = stp1;
            self.infoc = infoc;
        } else {
            let (stx1, sty1, stp1, brackt1, _stmin, _stmax, infoc) = cstep(
                self.stx.clone(),
                self.sty.clone(),
                Step::new(self.stp.x, self.f, dg),
                self.brackt,
                stmin,
                stmax,
            )?;
            self.stx = stx1;
            self.sty = sty1;
            self.stp = stp1;
            self.f = self.stp.fx;
            // dg = self.stp.gx;
            self.brackt = brackt1;
            self.infoc = infoc;
        }

        if self.brackt {
            if (self.sty.x - self.stx.x).abs() >= float!(0.66) * self.width1 {
                self.stp.x = self.stx.x + float!(0.5) * (self.sty.x - self.stx.x);
            }
            self.width1 = self.width;
            self.width = (self.sty.x - self.stx.x).abs();
        }

        Ok((state, None))
    }
}

type CstepReturnValue<F> = (Step<F>, Step<F>, Step<F>, bool, F, F, usize);

fn cstep<F: ArgminFloat>(
    stx: Step<F>,
    sty: Step<F>,
    stp: Step<F>,
    brackt: bool,
    stpmin: F,
    stpmax: F,
) -> Result<CstepReturnValue<F>, Error> {
    let mut info: usize = 0;
    let bound: bool;
    let mut stpf: F;
    let stpc: F;
    let stpq: F;
    let mut brackt = brackt;

    // check inputs
    if (brackt && (stp.x <= stx.x.min(sty.x) || stp.x >= stx.x.max(sty.x)))
        || stx.gx * (stp.x - stx.x) >= float!(0.0)
        || stpmax < stpmin
    {
        return Ok((stx, sty, stp, brackt, stpmin, stpmax, info));
    }

    // determine if the derivatives have opposite sign
    let sgnd = stp.gx * (stx.gx / stx.gx.abs());

    if stp.fx > stx.fx {
        // First case. A higher function value. The minimum is bracketed. If the cubic step is closer to
        // stx.x than the quadratic step, the cubic step is taken, else the average of the cubic and
        // the quadratic steps is taken.
        info = 1;
        bound = true;
        let theta = float!(3.0) * (stx.fx - stp.fx) / (stp.x - stx.x) + stx.gx + stp.gx;
        let tmp = vec![theta, stx.gx, stp.gx];
        // Check for a NaN or Inf in tmp before sorting
        if tmp.iter().any(|n| n.is_nan() || n.is_infinite()) {
            return Err(argmin_error!(
                ConditionViolated,
                "MoreThuenteLineSearch: NaN or Inf encountered during iteration"
            ));
        }
        let s = tmp.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let mut gamma = *s * ((theta / *s).powi(2) - (stx.gx / *s) * (stp.gx / *s)).sqrt();
        if stp.x < stx.x {
            gamma = -gamma;
        }

        let p = (gamma - stx.gx) + theta;
        let q = ((gamma - stx.gx) + gamma) + stp.gx;
        let r = p / q;
        stpc = stx.x + r * (stp.x - stx.x);
        stpq = stx.x
            + ((stx.gx / ((stx.fx - stp.fx) / (stp.x - stx.x) + stx.gx)) / float!(2.0))
                * (stp.x - stx.x);
        if (stpc - stx.x).abs() < (stpq - stx.x).abs() {
            stpf = stpc;
        } else {
            stpf = stpc + (stpq - stpc) / float!(2.0);
        }
        brackt = true;
    } else if sgnd < float!(0.0) {
        // Second case. A lower function value and derivatives of opposite sign. The minimum is
        // bracketed. If the cubic step is closer to stx.x than the quadratic (secant) step, the
        // cubic step is taken, else the quadratic step is taken.
        info = 2;
        bound = false;
        let theta = float!(3.0) * (stx.fx - stp.fx) / (stp.x - stx.x) + stx.gx + stp.gx;
        let tmp = vec![theta, stx.gx, stp.gx];
        // Check for a NaN or Inf in tmp before sorting
        if tmp.iter().any(|n| n.is_nan() || n.is_infinite()) {
            return Err(argmin_error!(
                ConditionViolated,
                "MoreThuenteLineSearch: NaN or Inf encountered during iteration"
            ));
        }
        let s = tmp.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let mut gamma = *s * ((theta / *s).powi(2) - (stx.gx / *s) * (stp.gx / *s)).sqrt();
        if stp.x > stx.x {
            gamma = -gamma;
        }
        let p = (gamma - stp.gx) + theta;
        let q = ((gamma - stp.gx) + gamma) + stx.gx;
        let r = p / q;
        stpc = stp.x + r * (stx.x - stp.x);
        stpq = stp.x + (stp.gx / (stp.gx - stx.gx)) * (stx.x - stp.x);
        if (stpc - stp.x).abs() > (stpq - stp.x).abs() {
            stpf = stpc;
        } else {
            stpf = stpq;
        }
        brackt = true;
    } else if stp.gx.abs() < stx.gx.abs() {
        // Third case. A lower function value, derivatives of the same sign, and the magnitude of
        // the derivative decreases. The cubic step is only used if the cubic tends to infinity in
        // the direction of the step or if the minimum of the cubic is beyond stp.x. Otherwise the
        // cubic step is defined to be either stpmin or stpmax. The quadratic (secant) step is
        // also computed and if the minimum is bracketed then the step closest to stx.x is taken,
        // else the step farthest away is taken.
        info = 3;
        bound = true;
        let theta = float!(3.0) * (stx.fx - stp.fx) / (stp.x - stx.x) + stx.gx + stp.gx;
        let tmp = vec![theta, stx.gx, stp.gx];
        // Check for a NaN or Inf in tmp before sorting
        if tmp.iter().any(|n| n.is_nan() || n.is_infinite()) {
            return Err(argmin_error!(
                ConditionViolated,
                "`MoreThuenteLineSearch`: NaN or Inf encountered during iteration"
            ));
        }
        let s = tmp.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        // the case gamma == 0 only arises if the cubic does not tend to infinity in the direction
        // of the step.

        let mut gamma = *s
            * float!(0.0)
                .max((theta / *s).powi(2) - (stx.gx / *s) * (stp.gx / *s))
                .sqrt();
        if stp.x > stx.x {
            gamma = -gamma;
        }

        let p = (gamma - stp.gx) + theta;
        let q = (gamma + (stx.gx - stp.gx)) + gamma;
        let r = p / q;
        if r < float!(0.0) && gamma != float!(0.0) {
            stpc = stp.x + r * (stx.x - stp.x);
        } else if stp.x > stx.x {
            stpc = stpmax;
        } else {
            stpc = stpmin;
        }
        stpq = stp.x + (stp.gx / (stp.gx - stx.gx)) * (stx.x - stp.x);
        if brackt {
            if (stp.x - stpc).abs() < (stp.x - stpq).abs() {
                stpf = stpc;
            } else {
                stpf = stpq;
            }
        } else if (stp.x - stpc).abs() > (stp.x - stpq).abs() {
            stpf = stpc;
        } else {
            stpf = stpq;
        }
    } else {
        // Fourth case. A lower function value, derivatives of the same sign, and the magnitude of
        // the derivative does not decrease. If the minimum is not bracketed, the step is either
        // stpmin or stpmax, else the cubic step is taken.
        info = 4;
        bound = false;
        if brackt {
            let theta = float!(3.0) * (stp.fx - sty.fx) / (sty.x - stp.x) + sty.gx + stp.gx;
            let tmp = vec![theta, sty.gx, stp.gx];
            // Check for a NaN or Inf in tmp before sorting
            if tmp.iter().any(|n| n.is_nan() || n.is_infinite()) {
                return Err(argmin_error!(
                    ConditionViolated,
                    "MoreThuenteLineSearch: NaN or Inf encountered during iteration"
                ));
            }
            let s = tmp.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            let mut gamma = *s * ((theta / *s).powi(2) - (sty.gx / *s) * (stp.gx / *s)).sqrt();
            if stp.x > sty.x {
                gamma = -gamma;
            }
            let p = (gamma - stp.gx) + theta;
            let q = ((gamma - stp.gx) + gamma) + sty.gx;
            let r = p / q;
            stpc = stp.x + r * (sty.x - stp.x);
            stpf = stpc;
        } else if stp.x > stx.x {
            stpf = stpmax;
        } else {
            stpf = stpmin;
        }
    }
    // Update the interval of uncertainty. This update does not depend on the new step or the case
    // analysis above.

    let mut stx_o = stx;
    let mut sty_o = sty;
    let mut stp_o = stp;
    if stp_o.fx > stx_o.fx {
        sty_o = Step::new(stp_o.x, stp_o.fx, stp_o.gx);
    } else {
        if sgnd < float!(0.0) {
            sty_o = Step::new(stx_o.x, stx_o.fx, stx_o.gx);
        }
        stx_o = Step::new(stp_o.x, stp_o.fx, stp_o.gx);
    }

    // compute the new step and safeguard it.

    stpf = stpmax.min(stpf);
    stpf = stpmin.max(stpf);

    stp_o.x = stpf;
    if brackt && bound {
        if sty_o.x > stx_o.x {
            stp_o.x = stp_o.x.min(stx_o.x + float!(0.66) * (sty_o.x - stx_o.x));
        } else {
            stp_o.x = stp_o.x.max(stx_o.x + float!(0.66) * (sty_o.x - stx_o.x));
        }
    }

    Ok((stx_o, sty_o, stp_o, brackt, stpmin, stpmax, info))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{test_utils::TestProblem, ArgminError, IterState, Problem};
    use crate::test_trait_impl;

    test_trait_impl!(morethuente, MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64>);

    #[test]
    fn test_new() {
        let mtls: MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64> = MoreThuenteLineSearch::new();
        let MoreThuenteLineSearch {
            search_direction,
            init_param,
            finit,
            init_grad,
            dginit,
            dgtest,
            ftol,
            gtol,
            xtrapf,
            width,
            width1,
            xtol,
            alpha,
            stpmin,
            stpmax,
            stp,
            stx,
            sty,
            f,
            brackt,
            stage1,
            infoc,
        } = mtls;

        assert!(search_direction.is_none());
        assert!(init_param.is_none());
        assert!(finit.is_infinite());
        assert!(finit.is_sign_positive());
        assert!(init_grad.is_none());
        assert_eq!(dginit.to_ne_bytes(), 0.0f64.to_ne_bytes());
        assert_eq!(dgtest.to_ne_bytes(), 0.0f64.to_ne_bytes());
        assert_eq!(ftol.to_ne_bytes(), 1e-4f64.to_ne_bytes());
        assert_eq!(gtol.to_ne_bytes(), 0.9f64.to_ne_bytes());
        assert_eq!(xtrapf.to_ne_bytes(), 4.0f64.to_ne_bytes());
        assert!(width.is_nan());
        assert!(width1.is_nan());
        assert_eq!(xtol.to_ne_bytes(), 1e-10f64.to_ne_bytes());
        assert_eq!(alpha.to_ne_bytes(), 1.0f64.to_ne_bytes());
        assert_eq!(stpmin.to_ne_bytes(), f64::EPSILON.sqrt().to_ne_bytes());
        assert!(stpmax.is_infinite());
        assert!(stpmax.is_sign_positive());
        assert_eq!(stp, Step::default());
        assert_eq!(stx, Step::default());
        assert_eq!(sty, Step::default());
        assert!(f.is_nan());
        assert!(!brackt);
        assert!(stage1);
        assert_eq!(infoc, 1);
    }

    #[test]
    fn test_with_c_correct() {
        let mtls: MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64> = MoreThuenteLineSearch::new();
        let res = mtls.with_c(0.1, 0.9);
        assert!(res.is_ok());

        let mtls = res.unwrap();
        assert_eq!(mtls.ftol.to_ne_bytes(), 0.1f64.to_ne_bytes());
        assert_eq!(mtls.gtol.to_ne_bytes(), 0.9f64.to_ne_bytes());
    }

    #[test]
    fn test_with_c_c1_larger_than_c2() {
        let mtls: MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64> = MoreThuenteLineSearch::new();
        let res = mtls.with_c(0.9, 0.1);
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Invalid parameter: \"`MoreThuenteLineSearch`: ",
                "Parameter c1 must be in (0, c2).\""
            )
        );
    }

    #[test]
    fn test_with_c_c1_smaller_than_0() {
        let mtls: MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64> = MoreThuenteLineSearch::new();
        let res = mtls.with_c(-0.9, 0.99);
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Invalid parameter: \"`MoreThuenteLineSearch`: ",
                "Parameter c1 must be in (0, c2).\""
            )
        );
    }

    #[test]
    fn test_with_c_c2_larger_than_1() {
        let mtls: MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64> = MoreThuenteLineSearch::new();
        let res = mtls.with_c(0.1, 1.01);
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Invalid parameter: \"`MoreThuenteLineSearch`: ",
                "Parameter c2 must be in (c1, 1).\""
            )
        );
    }

    #[test]
    fn test_with_bounds_correct() {
        let mtls: MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64> = MoreThuenteLineSearch::new();
        let res = mtls.with_bounds(0.1, 0.9);
        assert!(res.is_ok());

        let mtls = res.unwrap();
        assert_eq!(mtls.stpmin.to_ne_bytes(), 0.1f64.to_ne_bytes());
        assert_eq!(mtls.stpmax.to_ne_bytes(), 0.9f64.to_ne_bytes());
    }

    #[test]
    fn test_with_bounds_step_min_smaller_than_0() {
        let mtls: MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64> = MoreThuenteLineSearch::new();
        let res = mtls.with_bounds(-0.1, 0.99);
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Invalid parameter: \"`MoreThuenteLineSearch`: ",
                "step_min must be >= 0.0.\""
            )
        );
    }

    #[test]
    fn test_with_bounds_step_min_larger_than_step_max() {
        let mtls: MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64> = MoreThuenteLineSearch::new();
        let res = mtls.with_bounds(10.0, 0.99);
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Invalid parameter: \"`MoreThuenteLineSearch`: ",
                "step_min must be smaller than step_max.\""
            )
        );
    }

    #[test]
    fn test_with_width_tolerance_correct() {
        let mtls: MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64> = MoreThuenteLineSearch::new();
        let res = mtls.with_width_tolerance(1e-9);
        assert!(res.is_ok());

        let mtls = res.unwrap();
        assert_eq!(mtls.xtol.to_ne_bytes(), 1e-9f64.to_ne_bytes());
    }

    #[test]
    fn test_with_width_tolerance_negative_xtol() {
        let mtls: MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64> = MoreThuenteLineSearch::new();
        let res = mtls.with_width_tolerance(-1e-10);
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Invalid parameter: \"`MoreThuenteLineSearch`: ",
                "relative width tolerance must be >= 0.0.\""
            )
        );
    }

    #[test]
    fn test_init_search_direction_not_set() {
        let mut mtls: MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64> = MoreThuenteLineSearch::new();
        let res = mtls.init(&mut Problem::new(TestProblem::new()), IterState::new());
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Not initialized: \"`MoreThuenteLineSearch`: Search direction not initialized. ",
                "Call `search_direction` before executing the solver.\""
            )
        );
    }

    #[test]
    fn test_init_param_not_set() {
        let mut mtls: MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64> = MoreThuenteLineSearch::new();
        mtls.search_direction(vec![1.0f64]);
        let res = mtls.init(&mut Problem::new(TestProblem::new()), IterState::new());
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Not initialized: \"`MoreThuenteLineSearch` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method.\""
            )
        );
    }
}
