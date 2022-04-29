// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Nelder-Mead method
//!
//! The Nelder-Mead method a heuristic search method for nonlinear optimization problems which does
//! not require derivatives.
//!
//! See [`NelderMead`] for details.
//!
//! ## Reference
//!
//! <https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method>

use crate::core::{
    ArgminFloat, CostFunction, Error, IterState, Problem, SerializeAlias, Solver,
    TerminationReason, KV,
};
use argmin_math::{ArgminAdd, ArgminMul, ArgminSub};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::fmt;

/// # Nelder-Mead method
///
/// The Nelder-Mead method a heuristic search method for nonlinear optimization problems which does
/// not require derivatives.
///
/// The method is based on simplices which consist of n+1 vertices for an optimization problem with
/// n dimensions.
/// The function to be optimized is evaluated at all vertices. Based on these cost function values
/// the behaviour of the cost function is extrapolated in order to find the next point to be
/// evaluated.
///
/// The following actions are possible:
///
/// 1) Reflection (Parameter `alpha`, defaults to `1`)
/// 2) Expansion (Parameter `gamma`, defaults to `2`)
/// 3) Contraction (Parameter `rho`, defaults to `0.5`)
/// 4) Shrink (Parameter `sigma`, defaults to `0.5`)
///
/// ## Reference
///
/// [Wikipedia](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method)
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct NelderMead<P, F> {
    /// alpha
    alpha: F,
    /// gamma
    gamma: F,
    /// rho
    rho: F,
    /// sigma
    sigma: F,
    /// parameters
    params: Vec<(P, F)>,
    /// Sample standard deviation tolerance
    sd_tolerance: F,
}

impl<P, F> NelderMead<P, F>
where
    P: Clone + ArgminAdd<P, P> + ArgminSub<P, P> + ArgminMul<F, P>,
    F: ArgminFloat,
{
    /// Construct a new instance of `NelderMead`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::neldermead::NelderMead;
    /// # let vec_of_parameters = vec![vec![1.0], vec![2.0], vec![3.0]];
    /// let nm: NelderMead<Vec<f64>, f64> = NelderMead::new(vec_of_parameters);
    /// ```
    pub fn new(params: Vec<P>) -> Self {
        NelderMead {
            alpha: float!(1.0),
            gamma: float!(2.0),
            rho: float!(0.5),
            sigma: float!(0.5),
            params: params.into_iter().map(|p| (p, F::nan())).collect(),
            sd_tolerance: F::epsilon(),
        }
    }

    /// Set sample standard deviation tolerance
    ///
    /// Must be non-negative and defaults to `EPSILON`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::neldermead::NelderMead;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// # let vec_of_parameters = vec![vec![1.0], vec![2.0], vec![3.0]];
    /// let nm: NelderMead<Vec<f64>, f64> =
    ///     NelderMead::new(vec_of_parameters).with_sd_tolerance(1e-6)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_sd_tolerance(mut self, tol: F) -> Result<Self, Error> {
        if tol < float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`Nelder-Mead`: sd_tolerance must be >= 0."
            ));
        }
        self.sd_tolerance = tol;
        Ok(self)
    }

    /// Set alpha parameter for reflection
    ///
    /// Must be larger than 0 and defaults to 1.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::neldermead::NelderMead;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// # let vec_of_parameters = vec![vec![1.0], vec![2.0], vec![3.0]];
    /// let nm: NelderMead<Vec<f64>, f64> =
    ///     NelderMead::new(vec_of_parameters).with_alpha(0.9)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_alpha(mut self, alpha: F) -> Result<Self, Error> {
        if alpha <= float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`Nelder-Mead`: alpha must be > 0."
            ));
        }
        self.alpha = alpha;
        Ok(self)
    }

    /// Set gamma for expansion
    ///
    /// Must be larger than 1 and defaults to 2.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::neldermead::NelderMead;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// # let vec_of_parameters = vec![vec![1.0], vec![2.0], vec![3.0]];
    /// let nm: NelderMead<Vec<f64>, f64> =
    ///     NelderMead::new(vec_of_parameters).with_gamma(1.9)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_gamma(mut self, gamma: F) -> Result<Self, Error> {
        if gamma <= float!(1.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`Nelder-Mead`: gamma must be > 1."
            ));
        }
        self.gamma = gamma;
        Ok(self)
    }

    /// Set rho for contraction
    ///
    /// Must be in (0, 0.5] and defaults to 0.5.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::neldermead::NelderMead;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// # let vec_of_parameters = vec![vec![1.0], vec![2.0], vec![3.0]];
    /// let nm: NelderMead<Vec<f64>, f64> =
    ///     NelderMead::new(vec_of_parameters).with_rho(0.4)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_rho(mut self, rho: F) -> Result<Self, Error> {
        if rho <= float!(0.0) || rho > float!(0.5) {
            return Err(argmin_error!(
                InvalidParameter,
                "`Nelder-Mead`: rho must be in (0, 0.5]."
            ));
        }
        self.rho = rho;
        Ok(self)
    }

    /// Set sigma for shrinking
    ///
    /// Must be in (0, 1] and defaults to 0.5.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::neldermead::NelderMead;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// # let vec_of_parameters = vec![vec![1.0], vec![2.0], vec![3.0]];
    /// let nm: NelderMead<Vec<f64>, f64> =
    ///     NelderMead::new(vec_of_parameters).with_sigma(0.4)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_sigma(mut self, sigma: F) -> Result<Self, Error> {
        if sigma <= float!(0.0) || sigma > float!(1.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`Nelder-Mead`: sigma must be in (0, 1]."
            ));
        }
        self.sigma = sigma;
        Ok(self)
    }

    /// Sort parameters vectors based on their cost function values
    fn sort_param_vecs(&mut self) {
        self.params
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// Calculate centroid of all vectors but the worst
    fn calculate_centroid(&self) -> P {
        // Number of parameters is number of parameter vectors minus 1
        let num_param = self.params.len() - 1;
        self.params
            .iter()
            // Avoid the worst vector
            .take(num_param)
            // First one is used as the accumulator, therefore exclude it from the iterator
            .skip(1)
            // Add all vectors to the first
            .fold(self.params[0].0.clone(), |acc, p| acc.add(&p.0))
            // Scale
            .mul(&(float!(1.0) / (float!(num_param as f64))))
    }

    /// Reflect
    fn reflect(&self, x0: &P, x: &P) -> P {
        x0.add(&x0.sub(x).mul(&self.alpha))
    }

    /// Expand
    fn expand(&self, x0: &P, x: &P) -> P {
        x0.add(&x.sub(x0).mul(&self.gamma))
    }

    /// Contract
    fn contract(&self, x0: &P, x: &P) -> P {
        x0.add(&x.sub(x0).mul(&self.rho))
    }

    /// Shrink
    fn shrink<S>(&mut self, mut cost: S) -> Result<(), Error>
    where
        S: FnMut(&P) -> Result<F, Error>,
    {
        // The best parameter vector unfortunately has to be cloned once.
        let x0 = self.params[0].0.clone();
        self.params
            .iter_mut()
            // Best one is not modified
            .skip(1)
            .try_for_each(|(p, c)| -> Result<(), Error> {
                *p = x0.add(&p.sub(&x0).mul(&self.sigma));
                *c = (cost)(p)?;
                Ok(())
            })?;
        Ok(())
    }
}

#[derive(Debug)]
enum Action {
    Reflection,
    Expansion,
    Contraction,
    Shrink,
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Action::Reflection => write!(f, "Reflection"),
            Action::Expansion => write!(f, "Expansion"),
            Action::Contraction => write!(f, "Contraction"),
            Action::Shrink => write!(f, "Shrink"),
        }
    }
}

impl<O, P, F> Solver<O, IterState<P, (), (), (), F>> for NelderMead<P, F>
where
    O: CostFunction<Param = P, Output = F>,
    P: Clone + SerializeAlias + ArgminSub<P, P> + ArgminAdd<P, P> + ArgminMul<F, P>,
    F: ArgminFloat + std::iter::Sum<F>,
{
    const NAME: &'static str = "Nelder-Mead method";

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        state: IterState<P, (), (), (), F>,
    ) -> Result<(IterState<P, (), (), (), F>, Option<KV>), Error> {
        self.params
            .iter_mut()
            .for_each(|(p, c)| *c = problem.cost(p).unwrap());

        self.sort_param_vecs();

        Ok((
            state.param(self.params[0].0.clone()).cost(self.params[0].1),
            None,
        ))
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        state: IterState<P, (), (), (), F>,
    ) -> Result<(IterState<P, (), (), (), F>, Option<KV>), Error> {
        let num_param = self.params.len();

        let x0 = self.calculate_centroid();

        let xr = self.reflect(&x0, &self.params[num_param - 1].0);
        let xr_cost = problem.cost(&xr)?;

        let action = if xr_cost < self.params[num_param - 2].1 && xr_cost >= self.params[0].1 {
            // reflection
            self.params.last_mut().unwrap().0 = xr;
            self.params.last_mut().unwrap().1 = xr_cost;
            Action::Reflection
        } else if xr_cost < self.params[0].1 {
            // expansion
            let xe = self.expand(&x0, &xr);
            let xe_cost = problem.cost(&xe)?;
            if xe_cost < xr_cost {
                self.params.last_mut().unwrap().0 = xe;
                self.params.last_mut().unwrap().1 = xe_cost;
            } else {
                self.params.last_mut().unwrap().0 = xr;
                self.params.last_mut().unwrap().1 = xr_cost;
            }
            Action::Expansion
        } else if xr_cost >= self.params[num_param - 2].1 {
            // contraction
            let xc = self.contract(&x0, &self.params[num_param - 1].0);
            let xc_cost = problem.cost(&xc)?;
            if xc_cost < self.params[num_param - 1].1 {
                self.params.last_mut().unwrap().0 = xc;
                self.params.last_mut().unwrap().1 = xc_cost;
            }
            Action::Contraction
        } else {
            // shrink
            self.shrink(|x| problem.cost(x))?;
            Action::Shrink
        };

        self.sort_param_vecs();

        Ok((
            state.param(self.params[0].0.clone()).cost(self.params[0].1),
            Some(make_kv!("action" => action;)),
        ))
    }

    fn terminate(&mut self, _state: &IterState<P, (), (), (), F>) -> TerminationReason {
        let n = float!(self.params.len() as f64);
        let c0: F = self.params.iter().map(|(_, c)| *c).sum::<F>() / n;
        let s: F = (float!(1.0) / (n - float!(1.0))
            * self
                .params
                .iter()
                .map(|(_, c)| (*c - c0).powi(2))
                .sum::<F>())
        .sqrt();
        if s < self.sd_tolerance {
            return TerminationReason::TargetToleranceReached;
        }
        TerminationReason::NotTerminated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{test_utils::TestProblem, ArgminError};
    use crate::test_trait_impl;
    use approx::assert_relative_eq;

    test_trait_impl!(nelder_mead, NelderMead<TestProblem, f64>);

    #[test]
    fn test_new() {
        let params = vec![vec![1.0], vec![2.0]];
        let nm: NelderMead<Vec<f64>, f64> = NelderMead::new(params);

        let NelderMead {
            alpha,
            gamma,
            rho,
            sigma,
            params,
            sd_tolerance,
        } = nm;

        assert_eq!(alpha.to_ne_bytes(), 1.0f64.to_ne_bytes());
        assert_eq!(gamma.to_ne_bytes(), 2.0f64.to_ne_bytes());
        assert_eq!(rho.to_ne_bytes(), 0.5f64.to_ne_bytes());
        assert_eq!(sigma.to_ne_bytes(), 0.5f64.to_ne_bytes());
        assert_eq!(params[0].0[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
        assert_eq!(params[1].0[0].to_ne_bytes(), 2.0f64.to_ne_bytes());
        assert_eq!(params[0].1.to_ne_bytes(), f64::NAN.to_ne_bytes());
        assert_eq!(params[1].1.to_ne_bytes(), f64::NAN.to_ne_bytes());
        assert_eq!(sd_tolerance.to_ne_bytes(), f64::EPSILON.to_ne_bytes());
    }

    #[test]
    fn test_with_sd_tolerance() {
        // correct parameters
        for tol in [1e-6, 0.0, 1e-2, 1.0, 2.0] {
            let params = vec![vec![1.0], vec![2.0]];
            let nm: NelderMead<Vec<f64>, f64> = NelderMead::new(params);
            let res = nm.with_sd_tolerance(tol);
            assert!(res.is_ok());

            let nm = res.unwrap();
            assert_eq!(nm.sd_tolerance.to_ne_bytes(), tol.to_ne_bytes());
        }

        // incorrect parameters
        for tol in [-f64::EPSILON, -1.0, -100.0, -42.0] {
            let params = vec![vec![1.0], vec![2.0]];
            let nm: NelderMead<Vec<f64>, f64> = NelderMead::new(params);
            let res = nm.with_sd_tolerance(tol);
            assert_error!(
                res,
                ArgminError,
                concat!(
                    "Invalid parameter: \"`Nelder-Mead`: ",
                    "sd_tolerance must be >= 0.\""
                )
            );
        }
    }

    #[test]
    fn test_with_alpha() {
        // correct parameters
        for alpha in [f64::EPSILON, 1e-6, 1e-2, 1.0, 2.0] {
            let params = vec![vec![1.0], vec![2.0]];
            let nm: NelderMead<Vec<f64>, f64> = NelderMead::new(params);
            let res = nm.with_alpha(alpha);
            assert!(res.is_ok());

            let nm = res.unwrap();
            assert_eq!(nm.alpha.to_ne_bytes(), alpha.to_ne_bytes());
        }

        // incorrect parameters
        for alpha in [-f64::EPSILON, -1.0, -100.0, -42.0] {
            let params = vec![vec![1.0], vec![2.0]];
            let nm: NelderMead<Vec<f64>, f64> = NelderMead::new(params);
            let res = nm.with_alpha(alpha);
            assert_error!(
                res,
                ArgminError,
                concat!(
                    "Invalid parameter: \"`Nelder-Mead`: ",
                    "alpha must be > 0.\""
                )
            );
        }
    }

    #[test]
    fn test_with_rho() {
        // correct parameters
        for rho in [f64::EPSILON, 0.1, 0.3, 0.5] {
            let params = vec![vec![1.0], vec![2.0]];
            let nm: NelderMead<Vec<f64>, f64> = NelderMead::new(params);
            let res = nm.with_rho(rho);
            assert!(res.is_ok());

            let nm = res.unwrap();
            assert_eq!(nm.rho.to_ne_bytes(), rho.to_ne_bytes());
        }

        // incorrect parameters
        for rho in [-1.0, 0.0, 0.5 + f64::EPSILON, 1.0] {
            let params = vec![vec![1.0], vec![2.0]];
            let nm: NelderMead<Vec<f64>, f64> = NelderMead::new(params);
            let res = nm.with_rho(rho);
            assert_error!(
                res,
                ArgminError,
                concat!(
                    "Invalid parameter: \"`Nelder-Mead`: ",
                    "rho must be in (0, 0.5].\""
                )
            );
        }
    }

    #[test]
    fn test_with_sigma() {
        // correct parameters
        for sigma in [f64::EPSILON, 0.3, 0.5, 0.9, 1.0 - f64::EPSILON] {
            let params = vec![vec![1.0], vec![2.0]];
            let nm: NelderMead<Vec<f64>, f64> = NelderMead::new(params);
            let res = nm.with_sigma(sigma);
            assert!(res.is_ok());

            let nm = res.unwrap();
            assert_eq!(nm.sigma.to_ne_bytes(), sigma.to_ne_bytes());
        }

        // incorrect parameters
        for sigma in [-1.0, 0.0, 1.0 + f64::EPSILON, 10.0] {
            let params = vec![vec![1.0], vec![2.0]];
            let nm: NelderMead<Vec<f64>, f64> = NelderMead::new(params);
            let res = nm.with_sigma(sigma);
            assert_error!(
                res,
                ArgminError,
                concat!(
                    "Invalid parameter: \"`Nelder-Mead`: ",
                    "sigma must be in (0, 1].\""
                )
            );
        }
    }

    #[test]
    fn test_sort_param_vecs() {
        let params: Vec<Vec<f64>> = vec![vec![2.0], vec![1.0], vec![3.0]];
        let params_sorted: Vec<Vec<f64>> = vec![vec![1.0], vec![2.0], vec![3.0]];
        let mut nm: NelderMead<_, f64> = NelderMead::new(params);
        nm.params.iter_mut().for_each(|(p, c)| *c = p[0]);
        nm.sort_param_vecs();
        for ((p, c), ps) in nm.params.iter().zip(params_sorted.iter()) {
            assert_eq!(p[0].to_ne_bytes(), ps[0].to_ne_bytes());
            assert_eq!(c.to_ne_bytes(), ps[0].to_ne_bytes());
        }
    }

    #[test]
    fn test_calculate_centroid() {
        let params: Vec<Vec<f64>> = vec![vec![0.2, 0.0], vec![0.4, 1.0], vec![1.0, 0.0]];
        let mut nm: NelderMead<_, f64> = NelderMead::new(params);
        nm.params
            .iter_mut()
            .enumerate()
            .for_each(|(i, (_, c))| *c = i as f64);
        nm.sort_param_vecs();
        let centroid = nm.calculate_centroid();
        assert_relative_eq!(centroid[0], 0.3f64, epsilon = f64::EPSILON);
        assert_relative_eq!(centroid[1], 0.5f64, epsilon = f64::EPSILON);
    }

    #[test]
    fn test_reflect() {
        let params: Vec<Vec<f64>> = vec![vec![0.0, 1.0], vec![1.0, 0.0], vec![0.0, 0.0]];
        let mut nm: NelderMead<_, f64> = NelderMead::new(params);
        nm.params
            .iter_mut()
            .enumerate()
            .for_each(|(i, (_, c))| *c = i as f64);
        nm.sort_param_vecs();
        let centroid = nm.calculate_centroid();
        let reflected = nm.reflect(&centroid, &vec![0.0, 0.0]);
        assert_relative_eq!(reflected[0], 1.0f64, epsilon = f64::EPSILON);
        assert_relative_eq!(reflected[1], 1.0f64, epsilon = f64::EPSILON);
    }

    #[test]
    fn test_expand() {
        let params: Vec<Vec<f64>> = vec![vec![0.0, 1.0], vec![1.0, 0.0], vec![0.0, 0.0]];
        let mut nm: NelderMead<_, f64> = NelderMead::new(params);
        nm.params
            .iter_mut()
            .enumerate()
            .for_each(|(i, (_, c))| *c = i as f64);
        nm.sort_param_vecs();
        let centroid = nm.calculate_centroid();
        let expanded = nm.expand(&centroid, &vec![1.0, 1.0]);
        assert_relative_eq!(expanded[0], 1.5f64, epsilon = f64::EPSILON);
        assert_relative_eq!(expanded[1], 1.5f64, epsilon = f64::EPSILON);
    }

    #[test]
    fn test_contract() {
        let params: Vec<Vec<f64>> = vec![vec![0.0, 1.0], vec![1.0, 0.0], vec![0.0, 0.0]];
        let mut nm: NelderMead<_, f64> = NelderMead::new(params);
        nm.params
            .iter_mut()
            .enumerate()
            .for_each(|(i, (_, c))| *c = i as f64);
        nm.sort_param_vecs();
        let centroid = nm.calculate_centroid();
        let contracted = nm.contract(&centroid, &vec![1.0, 1.0]);
        assert_relative_eq!(contracted[0], 0.75f64, epsilon = f64::EPSILON);
        assert_relative_eq!(contracted[1], 0.75f64, epsilon = f64::EPSILON);
    }

    #[test]
    fn test_shrink() {
        let params: Vec<Vec<f64>> = vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0]];
        let params_shrunk: Vec<Vec<f64>> = vec![vec![0.0, 0.0], vec![0.0, 0.5], vec![0.5, 0.0]];
        let mut nm: NelderMead<_, f64> = NelderMead::new(params);
        nm.params
            .iter_mut()
            .enumerate()
            .for_each(|(i, (_, c))| *c = i as f64);
        nm.sort_param_vecs();
        nm.shrink(|_| Ok(1.0f64)).unwrap();

        for ((p, _), ps) in nm.params.iter().zip(params_shrunk.iter()) {
            assert_eq!(p[0].to_ne_bytes(), ps[0].to_ne_bytes());
            assert_eq!(p[1].to_ne_bytes(), ps[1].to_ne_bytes());
        }
    }
}
