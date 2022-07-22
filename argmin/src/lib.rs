// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! argmin is a numerical optimization library written entirely in Rust.
//!
//! Its goal is to offer a wide range of optimization algorithms with a consistent interface.
//! It is type-agnostic by the design, meaning that any type and/or math backend, such as
//! `nalgebra` or `ndarray` can be used -- even your own.
//!
//! Observers allow one to track the progress of iterations, either by using one of the provided
//! ones for logging to screen or disk or by implementing your own.
//!
//! An optional checkpointing mechanism helps to mitigate the negative effects of crashes in
//! unstable computing environments.
//!
//! Due to Rusts powerful generics and traits, most features can be exchanged by your own tailored
//! implementations.
//!
//! argmin is designed to simplify the implementation of optimization algorithms and as such can
//! also be used as a toolbox for the development of new algorithms. One can focus on the algorithm
//! itself, while the handling of termination, parameter vectors, populations, gradients, Jacobians
//! and Hessians is taken care of by the library.
//!
//! # Highlights
//!
//! * [Checkpointing](`crate::core::checkpointing`)
//! * [Observers](`crate::core::observers`)
//!
//!
//! # Algorithms
//!
//! - [Line searches](`crate::solver::linesearch`)
//!   - [Backtracking line search](`crate::solver::linesearch::BacktrackingLineSearch`)
//!   - [More-Thuente line search](`crate::solver::linesearch::MoreThuenteLineSearch`)
//!   - [Hager-Zhang line search](`crate::solver::linesearch::HagerZhangLineSearch`)
//!
//! - [Trust region method](`crate::solver::trustregion::TrustRegion`)
//!   - [Cauchy point method](`crate::solver::trustregion::CauchyPoint`)
//!   - [Dogleg method](`crate::solver::trustregion::Dogleg`)
//!   - [Steihaug method](`crate::solver::trustregion::Steihaug`)
//!   
//! - [Steepest descent](`crate::solver::gradientdescent::SteepestDescent`)
//!
//! - [Conjugate gradient methods](`crate::solver::conjugategradient`)
//!   - [Conjugate gradient method](`crate::solver::conjugategradient::ConjugateGradient`)
//!   - [Nonlinear conjugate gradient method](`crate::solver::conjugategradient::NonlinearConjugateGradient`)
//!
//! - [Newton methods](`crate::solver::newton`)
//!   - [Newton's method](`crate::solver::newton::Newton`)
//!   - [Newton-CG](solver/newton/newton_cg/struct.NewtonCG.html)
//!
//! - [Quasi-Newton methods](`crate::solver::quasinewton`)
//!   - [BFGS](`crate::solver::quasinewton::BFGS`)
//!   - [L-BFGS](`crate::solver::quasinewton::LBFGS`)
//!   - [DFP](`crate::solver::quasinewton::DFP`)
//!   - [SR1](`crate::solver::quasinewton::SR1`)
//!   - [SR1-TrustRegion](`crate::solver::quasinewton::SR1TrustRegion`)
//!
//! - [Gauss-Newton methods](`crate::solver::gaussnewton`)
//!   - [Gauss-Newton method](`crate::solver::gaussnewton::GaussNewton`)
//!   - [Gauss-Newton method with linesearch](`crate::solver::gaussnewton::GaussNewtonLS`)
//!
//! - [Golden-section search](`crate::solver::goldensectionsearch::GoldenSectionSearch`)
//!
//! - [Landweber iteration](`crate::solver::landweber::Landweber`)
//!
//! - [Brent's methods](`crate::solver::brent`)
//!   - [Brent's minimization method](`crate::solver::brent::BrentOpt`)
//!   - [Brent's root finding method](`crate::solver::brent::BrentRoot`)
//!
//! - [Nelder-Mead method](`crate::solver::neldermead::NelderMead`)
//!
//! - [Simulated Annealing](`crate::solver::simulatedannealing::SimulatedAnnealing`)
//!
//! - [Particle Swarm Optimization](`crate::solver::particleswarm::ParticleSwarm`)
//!
//! # License
//!
//! Licensed under either of
//!
//!   * Apache License, Version 2.0,
//!     ([LICENSE-APACHE](https://github.com/argmin-rs/argmin/blob/main/LICENSE-APACHE) or
//!     <http://www.apache.org/licenses/LICENSE-2.0>)
//!   * MIT License ([LICENSE-MIT](https://github.com/argmin-rs/argmin/blob/main/LICENSE-MIT) or
//!     <http://opensource.org/licenses/MIT>)
//!
//! at your option.
//!
//! ## Contribution
//!
//! Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion
//! in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above,
//! without any additional terms or conditions.

#![warn(missing_docs)]
#![allow(unused_attributes)]
// Explicitly disallow EQ comparison of floats. (This clippy lint is denied by default; however,
// this is just to make sure that it will always stay this way.)
#![deny(clippy::float_cmp)]

#[macro_use]
pub mod core;

/// Solvers
pub mod solver;

#[cfg(test)]
#[cfg(feature = "_ndarrayl")]
mod tests;
