// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Covariance matrix adaptation evolution strategy (CMA-ES)
//!
//! For details see [`CMAES`].

use crate::core::{
    ArgminFloat, CostFunction, Error, PopulationState, Problem, SerializeAlias, Solver, SyncAlias,
    KV,
};
use argmin_math::{
    ArgminAdd, ArgminArgsort, ArgminAxisIter, ArgminBroadcast, ArgminDiv, ArgminDot,
    ArgminEigenSym, ArgminEye, ArgminFrom, ArgminIter, ArgminMul, ArgminMutIter, ArgminNorm,
    ArgminOuter, ArgminRandomMatrix, ArgminSize, ArgminSub, ArgminTDot, ArgminTake,
    ArgminTransition, ArgminTranspose, ArgminZeroLike,
};
use num_traits::NumCast;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::ops::{AddAssign, MulAssign};

/// # CMA-ES
///
/// CMA-ES is a stochastic, derivative-free, population-based method that uses generate-update paradigm
/// to find the optimal value of a function. The method generates a population from a multivariate normal
/// distribution and then updates the parameters of the distribution from a set of best individuals in
/// this population. This method is well suited for continuous problem optimization.
///
/// Example:
///
/// ```
/// use argmin::core::{CostFunction, Error, Executor};
/// use argmin::solver::cma_es::CMAES;
/// use argmin_testfunctions::rosenbrock_2d;
///
/// let cost = Rosenbrock { a: 1.0, b: 100.0 };
///
/// struct Rosenbrock {
///     a: f32,
///     b: f32,
/// }
///
/// impl CostFunction for Rosenbrock {
///     type Param = Vec<f32>;
///     type Output = f32;
///
///     fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
///         Ok(rosenbrock_2d(p, self.a, self.b))
///     }
/// }
///
/// let solver = CMAES::new(vec![5.; 2], 5., 40);
///
/// let res = Executor::new(cost, solver).configure(|state| state.max_iters(100)).run();
/// ```
///
/// ## Reference
///
/// <https://arxiv.org/pdf/1604.00772.pdf>

#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct CMAES<P: ArgminTransition, F> {
    centroid: P,
    weights: P,
    sigma: F,
    mueff: F,
    lambda: usize,
    dim: usize,
    mu: usize,
    b: P::Array2D,
    bd: P::Array2D,
    c: P::Array2D,
    diag_d: P,
    ps: P,
    pc: P,
    cs: F,
    cc: F,
    ccov1: F,
    ccovmu: F,
    chi_n: F,
    damps: F,
}

impl<P, F> CMAES<P, F>
where
    F: ArgminFloat + MulAssign + AddAssign + NumCast,
    P: ArgminTransition
        + ArgminSize<usize>
        + ArgminZeroLike
        + ArgminIter<F>
        + ArgminMutIter<F>
        + ArgminArgsort
        + ArgminTake<usize>
        + ArgminFrom<F, usize>
        + ArgminDiv<F, P>,
    P::Array2D: Clone
        + ArgminRandomMatrix
        + ArgminEye
        + ArgminDot<P::Array2D, P::Array2D>
        + ArgminEigenSym<P>
        + ArgminTake<usize>
        + ArgminBroadcast<P, P::Array2D>
        + ArgminMul<F, P::Array2D>
        + ArgminTranspose<P::Array2D>,
{
    /// Construct a new instance of [`CMAES`]
    ///
    /// * `centroid` - initial value of the parameters
    /// * `sigma` - initial standard deviation of the distribution
    /// * `lambda` - number of children in each generation
    ///
    /// # Example
    ///
    /// ```
    /// use argmin::solver::cma_es::CMAES;
    ///
    /// let solver = CMAES::new(vec![5.; 2], 5., 40);
    /// ```
    pub fn new(centroid: P, sigma: F, lambda: usize) -> Self {
        let two_f = F::from(2).unwrap();
        let half_f = F::from(0.5).unwrap();
        let dim = centroid.shape();
        let dim_f = F::from(dim).unwrap();
        let pc = centroid.zero_like();
        let ps = centroid.zero_like();
        let chi_n = dim_f.sqrt()
            * (F::one() - F::one() / (F::from(4).unwrap() * dim_f)
                + F::one() / (F::from(21).unwrap() * dim_f.powf(two_f)));

        let c = P::Array2D::eye(dim);

        let (mut diag_d, mut b) = c.eig_sym();

        let indx = diag_d.argsort();

        diag_d = diag_d.take(&indx, 0);

        diag_d.iterator_mut().for_each(|v| *v = v.powf(half_f));

        b = b.take(&indx, 0);

        let bd = b.broadcast_mul(&diag_d);

        let mu = lambda / 2;
        let w_c = F::from(0.5 + mu as f64).unwrap().ln();
        let mut weights = P::from_iterator(mu, (1..mu + 1).map(|i| w_c - F::from(i).unwrap().ln()));

        weights = weights.div(&weights.iterator().fold(F::zero(), |sum, &val| sum + val));
        let mueff = F::one()
            / weights
                .iterator()
                .fold(F::zero(), |sum, &val| sum + val.powf(two_f));

        let cc = F::from(4. / (dim as f64 + 4.)).unwrap();
        let cs = (mueff + two_f) / (F::from(dim).unwrap() + mueff + F::from(3.).unwrap());
        let ccov1 = two_f / (F::from(dim as f64 + 1.3).unwrap().powf(two_f) + mueff);
        let mut ccovmu = two_f * (mueff - two_f + F::one() / mueff)
            / (F::from(dim as f64 + 2.).unwrap().powf(two_f) + mueff);
        ccovmu = ccovmu.min(F::one() - ccov1);
        let damps = F::one()
            + two_f * F::zero().max(((mueff - F::one()) / (dim_f + F::one())).sqrt() - F::one())
            + cs;

        CMAES {
            centroid,
            weights,
            sigma,
            mueff,
            lambda,
            dim,
            mu,
            b,
            bd,
            c,
            diag_d,
            cs,
            cc,
            ccov1,
            ccovmu,
            ps,
            pc,
            chi_n,
            damps,
        }
    }

    fn generate(&self) -> P::Array2D {
        let population = P::Array2D::standard_normal(self.lambda, self.dim);

        let bd_t = self.bd.clone().t();

        population
            .dot(&bd_t)
            .mul(&self.sigma)
            .broadcast_add(&self.centroid)
    }
}

impl<O, P, F> Solver<O, PopulationState<P, F, P::Array2D>> for CMAES<P, F>
where
    O: CostFunction<Param = P, Output = F> + SyncAlias,
    Vec<F>: ArgminArgsort,
    F: ArgminFloat + MulAssign + AddAssign + NumCast + ArgminDiv<P, P>,
    P: SerializeAlias
        + Clone
        + SyncAlias
        + ArgminTransition
        + ArgminSize<usize>
        + ArgminZeroLike
        + ArgminIter<F>
        + ArgminMutIter<F>
        + ArgminArgsort
        + ArgminTake<usize>
        + ArgminFrom<F, usize>
        + ArgminDiv<F, P>
        + ArgminTDot<P::Array2D, P>
        + ArgminSub<P, P>
        + ArgminMul<P, P>
        + ArgminMul<F, P>
        + ArgminAdd<P, P>
        + ArgminNorm<F>
        + ArgminOuter<P, P::Array2D>,
    P::Array2D: SerializeAlias
        + Clone
        + ArgminRandomMatrix
        + ArgminEye
        + ArgminDot<P::Array2D, P::Array2D>
        + ArgminDot<P, P>
        + ArgminEigenSym<P>
        + ArgminTake<usize>
        + ArgminMul<F, P::Array2D>
        + ArgminBroadcast<P, P::Array2D>
        + ArgminAxisIter<P>
        + ArgminTranspose<P::Array2D>
        + ArgminDiv<F, P::Array2D>
        + ArgminAdd<P::Array2D, P::Array2D>,
{
    const NAME: &'static str = "CMA-ES method";

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        mut state: PopulationState<P, F, P::Array2D>,
    ) -> Result<(PopulationState<P, F, P::Array2D>, Option<KV>), Error> {
        let two_f = F::from(2).unwrap();
        let half_f = F::from(0.5).unwrap();

        state.population = Some(self.generate());

        let fitness: Vec<F> = problem
            .bulk_cost(&state.get_population().unwrap().row_iterator().collect())
            .unwrap();

        let fitness_indices = fitness.argsort();

        let best_ind = state
            .get_population()
            .unwrap()
            .take(&fitness_indices[..self.mu], 0);

        let new_centroid = self.weights.tdot(&best_ind);

        let c_diff = new_centroid.sub(&self.centroid);

        let b_t = self.b.clone().t();

        self.ps = self.ps.mul(&(F::one() - self.cs)).add(
            &(self
                .b
                .dot(&ArgminDiv::div(&F::one(), &self.diag_d).mul(&b_t.dot(&c_diff)))
                .mul(&((self.cs * (two_f - self.cs) * self.mueff).sqrt() / self.sigma))),
        );

        let hsig = if self.ps.norm()
            / (F::one()
                - (F::one() - self.cs).powf(two_f * (F::from(state.iter + 1).unwrap() + F::one())))
            .sqrt()
            / self.chi_n
            < F::from(1.4 + 2. / (self.dim as f64 + 1.)).unwrap()
        {
            F::one()
        } else {
            F::zero()
        };

        self.pc = self.pc.mul(&(F::one() - self.cc)).add(
            &c_diff.mul(&(hsig * (self.cc * (two_f - self.cc) * self.mueff).sqrt() / self.sigma)),
        );

        let artmp = best_ind.broadcast_sub(&self.centroid);

        let artmp_t = artmp.clone().t();

        self.c = self
            .c
            .mul(
                &(F::one() - self.ccov1 - self.ccovmu
                    + (F::one() - hsig) * self.ccov1 * self.cc * (two_f - self.cc)),
            )
            .add(&self.pc.outer(&self.pc).mul(&self.ccov1))
            .add(
                &artmp_t
                    .broadcast_mul(&self.weights)
                    .dot(&artmp)
                    .mul(&self.ccovmu)
                    .div(&self.sigma.powf(two_f)),
            );

        self.sigma *= ((self.ps.norm() / self.chi_n - F::one()) * self.cs / self.damps).exp();
        if self.sigma.is_infinite() {
            self.sigma = F::max_value();
        }

        (self.diag_d, self.b) = self.c.eig_sym();

        self.diag_d.iterator_mut().for_each(|v| *v += F::epsilon());

        let indx = self.diag_d.argsort();

        self.diag_d = self.diag_d.take(&indx, 0);

        self.diag_d.iterator_mut().for_each(|v| *v = v.powf(half_f));

        self.b = self.b.take(&indx, 0);

        self.bd = self.b.broadcast_mul(&self.diag_d);

        self.centroid = new_centroid;

        state.best_cost = fitness.iter().fold(F::infinity(), |a, &b| a.min(b));

        state.cost =
            fitness.iter().fold(F::zero(), |a, &b| a + b) / F::from(fitness.len()).unwrap();

        state.best_individual = Some(self.centroid.clone());

        Ok((state, None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{CostFunction, Error, Executor};
    use argmin_testfunctions::{rosenbrock, rosenbrock_2d};
    #[cfg(feature = "nalgebral")]
    use nalgebra::{dvector, DVector};
    #[cfg(feature = "ndarrayl")]
    use ndarray::{array, Array1};

    test_trait_impl!(cmaes_method, CMAES<Vec<f64>, f64>);

    #[test]
    fn test_new() {
        let precision = 1e-4;
        let solver = CMAES::<Vec<f64>, f64>::new(vec![1., 2.], 4., 5);
        let bd_sum: f64 = solver.bd.iter().flat_map(|v| v.iter()).sum();
        let ps_sum: f64 = solver.ps.iter().sum();
        let pc_sum: f64 = solver.pc.iter().sum();
        assert_eq!(solver.dim, 2);
        assert_eq!(solver.mu, 2);
        assert!((solver.weights[0] - 0.8042).abs() <= precision);
        assert!((solver.weights[1] - 0.1958).abs() <= precision);
        assert!((solver.mueff - 1.4598).abs() <= precision);
        assert!((bd_sum - 2.0).abs() <= precision);
        assert!((ps_sum - 0.0).abs() <= precision);
        assert!((pc_sum - 0.0).abs() <= precision);
        assert!((solver.damps - 1.5356).abs() <= precision);
        assert!((solver.cs - 0.5356).abs() <= precision);
        assert!((solver.cc - 0.6667).abs() <= precision);
        assert!((solver.ccov1 - 0.1619).abs() <= precision);
        assert!((solver.ccovmu - 0.0166).abs() <= precision);
        assert!((solver.chi_n - 1.2543).abs() <= precision);
    }

    #[test]
    fn test_solver() {
        let precision = 1e-4;

        let cost = Rosenbrock { a: 1.0, b: 100.0 };

        struct Rosenbrock {
            a: f64,
            b: f64,
        }
        impl CostFunction for Rosenbrock {
            type Param = Vec<f64>;
            type Output = f64;

            fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
                Ok(rosenbrock_2d(p, self.a, self.b))
            }
        }

        let solver = CMAES::new(vec![5f64; 2], 5., 100);

        let res = Executor::new(cost, solver)
            .configure(|state| state.max_iters(300))
            .run();

        assert!(res.is_ok());

        let state = res.unwrap().state;
        assert!(state.best_individual.is_some());

        let solution = state.best_individual.unwrap();
        assert!((solution[0] - 1.0).abs() <= precision);
        assert!((solution[1] - 1.0).abs() <= precision);
    }

    #[test]
    fn test_solver_multidimensional() {
        let precision = 1e-2;

        let cost = Rosenbrock { a: 1.0, b: 100.0 };

        struct Rosenbrock {
            a: f64,
            b: f64,
        }
        impl CostFunction for Rosenbrock {
            type Param = Vec<f64>;
            type Output = f64;

            fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
                Ok(rosenbrock(p, self.a, self.b))
            }
        }

        let solver = CMAES::new(vec![5f64; 5], 5., 100);

        let res = Executor::new(cost, solver)
            .configure(|state| state.max_iters(2000))
            .run();

        assert!(res.is_ok());

        let state = res.unwrap().state;
        assert!(state.best_individual.is_some());

        let solution = state.best_individual.unwrap();
        assert!((solution[0] - 1.0).abs() <= precision);
        assert!((solution[1] - 1.0).abs() <= precision);
    }

    #[cfg(feature = "ndarrayl")]
    #[test]
    fn test_ndarray_new() {
        let precision = 1e-4;
        let solver = CMAES::<Array1<f64>, f64>::new(array![1., 2.], 4., 5);
        let bd_sum: f64 = solver.bd.sum();
        let ps_sum: f64 = solver.ps.sum();
        let pc_sum: f64 = solver.pc.sum();
        assert_eq!(solver.dim, 2);
        assert_eq!(solver.mu, 2);
        assert!((solver.weights[0] - 0.8042).abs() <= precision);
        assert!((solver.weights[1] - 0.1958).abs() <= precision);
        assert!((solver.mueff - 1.4598).abs() <= precision);
        assert!((bd_sum - 2.0).abs() <= precision);
        assert!((ps_sum - 0.0).abs() <= precision);
        assert!((pc_sum - 0.0).abs() <= precision);
        assert!((solver.damps - 1.5356).abs() <= precision);
        assert!((solver.cs - 0.5356).abs() <= precision);
        assert!((solver.cc - 0.6667).abs() <= precision);
        assert!((solver.ccov1 - 0.1619).abs() <= precision);
        assert!((solver.ccovmu - 0.0166).abs() <= precision);
        assert!((solver.chi_n - 1.2543).abs() <= precision);
    }

    #[cfg(feature = "ndarrayl")]
    #[test]
    fn test_ndarray_solver() {
        let precision = 1e-6;

        let cost = Rosenbrock { a: 1.0, b: 100.0 };

        struct Rosenbrock {
            a: f64,
            b: f64,
        }
        impl CostFunction for Rosenbrock {
            type Param = Array1<f64>;
            type Output = f64;

            fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
                Ok(rosenbrock_2d(&[p[0], p[1]], self.a, self.b))
            }
        }

        let solver = CMAES::new(array![5., 5.], 5., 100);

        let res = Executor::new(cost, solver)
            .configure(|state| state.max_iters(100))
            .run();

        assert!(res.is_ok());

        let state = res.unwrap().state;
        assert!(state.best_individual.is_some());

        let solution = state.best_individual.unwrap();
        assert!((solution[0] - 1.0).abs() <= precision);
        assert!((solution[1] - 1.0).abs() <= precision);
    }

    #[cfg(feature = "ndarrayl")]
    #[test]
    fn test_ndarray_solver_multidimensional() {
        let precision = 1e-6;

        let cost = Rosenbrock { a: 1.0, b: 100.0 };

        struct Rosenbrock {
            a: f64,
            b: f64,
        }
        impl CostFunction for Rosenbrock {
            type Param = Array1<f64>;
            type Output = f64;

            fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
                Ok(rosenbrock(&[p[0], p[1], p[2], p[3], p[4]], self.a, self.b))
            }
        }

        let solver = CMAES::new(array![5., 5., 5., 5., 5.], 5., 100);

        let res = Executor::new(cost, solver)
            .configure(|state| state.max_iters(1000))
            .run();

        assert!(res.is_ok());

        let state = res.unwrap().state;
        assert!(state.best_individual.is_some());

        let solution = state.best_individual.unwrap();
        assert!((solution[0] - 1.0).abs() <= precision);
        assert!((solution[1] - 1.0).abs() <= precision);
        assert!((solution[2] - 1.0).abs() <= precision);
        assert!((solution[3] - 1.0).abs() <= precision);
        assert!((solution[4] - 1.0).abs() <= precision);
    }

    #[cfg(feature = "nalgebral")]
    #[test]
    fn test_nalgebra_new() {
        let precision = 1e-4;
        let solver = CMAES::<DVector<f64>, f64>::new(dvector![1., 2.], 4., 5);
        let bd_sum: f64 = solver.bd.sum();
        let ps_sum: f64 = solver.ps.sum();
        let pc_sum: f64 = solver.pc.sum();
        assert_eq!(solver.dim, 2);
        assert_eq!(solver.mu, 2);
        assert!((solver.weights[0] - 0.8042).abs() <= precision);
        assert!((solver.weights[1] - 0.1958).abs() <= precision);
        assert!((solver.mueff - 1.4598).abs() <= precision);
        assert!((bd_sum - 2.0).abs() <= precision);
        assert!((ps_sum - 0.0).abs() <= precision);
        assert!((pc_sum - 0.0).abs() <= precision);
        assert!((solver.damps - 1.5356).abs() <= precision);
        assert!((solver.cs - 0.5356).abs() <= precision);
        assert!((solver.cc - 0.6667).abs() <= precision);
        assert!((solver.ccov1 - 0.1619).abs() <= precision);
        assert!((solver.ccovmu - 0.0166).abs() <= precision);
        assert!((solver.chi_n - 1.2543).abs() <= precision);
    }

    #[cfg(feature = "nalgebral")]
    #[test]
    fn test_nalgebra_solver() {
        let precision = 1e-6;

        let cost = Rosenbrock { a: 1.0, b: 100.0 };

        struct Rosenbrock {
            a: f64,
            b: f64,
        }
        impl CostFunction for Rosenbrock {
            type Param = DVector<f64>;
            type Output = f64;

            fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
                Ok(rosenbrock_2d(&[p[0], p[1]], self.a, self.b))
            }
        }

        let solver = CMAES::new(dvector![5., 5.], 5., 40);

        let res = Executor::new(cost, solver)
            .configure(|state| state.max_iters(100))
            .run();

        assert!(res.is_ok());

        let state = res.unwrap().state;
        assert!(state.best_individual.is_some());

        let solution = state.best_individual.unwrap();
        assert!((solution[0] - 1.0).abs() <= precision);
        assert!((solution[1] - 1.0).abs() <= precision);
    }

    #[cfg(feature = "nalgebral")]
    #[test]
    fn test_nalgebra_solver_multidimensional() {
        let precision = 1e-2;

        let cost = Rosenbrock { a: 1.0, b: 100.0 };

        struct Rosenbrock {
            a: f64,
            b: f64,
        }
        impl CostFunction for Rosenbrock {
            type Param = DVector<f64>;
            type Output = f64;

            fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
                Ok(rosenbrock(&[p[0], p[1], p[2], p[3], p[4]], self.a, self.b))
            }
        }

        let solver = CMAES::new(dvector![5., 5., 5., 5., 5.], 5., 100);

        let res = Executor::new(cost, solver)
            .configure(|state| state.max_iters(2000))
            .run();

        assert!(res.is_ok());

        let state = res.unwrap().state;
        assert!(state.best_individual.is_some());

        let solution = state.best_individual.unwrap();
        assert!((solution[0] - 1.0).abs() <= precision);
        assert!((solution[1] - 1.0).abs() <= precision);
        assert!((solution[2] - 1.0).abs() <= precision);
        assert!((solution[3] - 1.0).abs() <= precision);
        assert!((solution[4] - 1.0).abs() <= precision);
    }
}
