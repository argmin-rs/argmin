// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! argmin is a numerical optimization library written entirely in Rust.
//!
//! [Documentation of most recent release](https://docs.rs/argmin/latest/argmin/)
//!
//! [Documentation of main branch](https://argmin-rs.github.io/argmin/argmin/)
//!
//! # Design goals
//!
//! argmin aims at offering a wide range of optimization algorithms with a consistent interface,
//! written purely in Rust. It comes with additional features such as checkpointing and observers
//! which for instance make it possible to log the progress of an optimization to screen or file.
//!
//! It further provides a framework for implementing iterative optimization algorithms in a
//! convenient manner. Essentially, a single iteration of the algorithm needs to be implemented and
//! everything else, such as handling termination, parameter vectors, gradients and Hessians, is
//! taken care of by the library.
//!
//! This library uses generics to be as type-agnostic as possible. Abstractions over common math
//! functions enable the use of common backends such as `ndarray` and `nalgebra` via the
//! `argmin-math` crate. All operations can be performed with 32 and 64 bit floats. Custom types are
//! of course also supported.
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
