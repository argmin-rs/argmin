// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! This crate contains a wide range of methods for the calculation of gradients, Jacobians and
//! Hessians using forward and central differences.
//! The methods have been implemented for input vectors of the type `Vec<f64>` and
//! `ndarray::Array1<f64>`.
//! Central differences are more accurate but require more evaluations of the cost function and are
//! therefore computationally more expensive.
//!
//! # References
//!
//! Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
//! Springer. ISBN 0-387-30303-0.
//!
//! # Usage
//!
//! Add this to your `dependencies` section of `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! finitediff = "0.1.4"
//! ```
//!
//! To use the `FiniteDiff` trait implementations on the `ndarray` types, please activate the
//! `ndarray` feature:
//!
//! ```toml
//! [dependencies]
//! finitediff = { version = "0.1.4", features = ["ndarray"] }
//! ```
//!
//! # Examples
//!
//! * [Calculation of the gradient](#calculation-of-the-gradient)
//!   * [For `Vec<f64>`](#for-vecf64)
//!   * [For `ndarray::Array1<f64>`](#for-ndarrayarray1f64)
//! * [Calculation of the Jacobian](#calculation-of-the-jacobian)
//!   * [Full Jacobian](#full-jacobian)
//!   * [Product of the Jacobian `J(x)` with a vector `p`](#product-of-the-jacobian-jx-with-a-vector-p)
//!   * [Sparse Jacobian](#sparse-jacobian)
//! * [Calculation of the Hessian](#calculation-of-the-hessian)
//!   * [Full Hessian](#full-hessian)
//!   * [Product of the Hessian `H(x)` with a vector `p`](#product-of-the-hessian-hx-with-a-vector-p)
//!   * [Calculation of the Hessian without knowledge of the gradient](#calculation-of-the-hessian-without-knowledge-of-the-gradient)
//!   * [Calculation of the sparse Hessian without knowledge of the gradient](#calculation-of-the-sparse-hessian-without-knowledge-of-the-gradient)
//!
//!
//! ## Calculation of the gradient
//!
//! This section illustrates the computation of a gradient of a function `f` at a point `x` of type
//! `Vec<f64>`. Using forward differences requires `n+1` evaluations of the function `f` while
//! central differences requires `2*n` evaluations.
//!
//! ### For `Vec<f64>`
//!
//! ```rust
//! use finitediff::FiniteDiff;
//!
//! // Define cost function `f(x)`
//! let f = |x: &Vec<f64>| -> f64 {
//!     // ...
//! #     x[0] + x[1].powi(2)
//! };
//!
//! // Point at which gradient should be calculated
//! let x = vec![1.0f64, 1.0];
//!
//! // Calculate gradient of `f` at `x` using forward differences
//! let grad_forward = x.forward_diff(&f);
//!
//! // Calculate gradient of `f` at `x` using central differences
//! let grad_central = x.central_diff(&f);
//! #
//! #  // Desired solution
//! #  let res = vec![1.0f64, 2.0];
//! #
//! #  // Check result
//! #  for i in 0..2 {
//! #      assert!((res[i] - grad_forward[i]).abs() < 1e-6);
//! #      assert!((res[i] - grad_central[i]).abs() < 1e-6);
//! #  }
//! ```
//!
//! ### For `ndarray::Array1<f64>`
//!
//! ```rust
//! # #[cfg(feature = "ndarray")]
//! # {
//! use ndarray::{array, Array1};
//! use finitediff::FiniteDiff;
//!
//! // Define cost function `f(x)`
//! let f = |x: &Array1<f64>| -> f64 {
//!     // ...
//! #     x[0] + x[1].powi(2)
//! };
//!
//! // Point at which gradient should be calculated
//! let x = array![1.0f64, 1.0];
//!
//! // Calculate gradient of `f` at `x` using forward differences
//! let grad_forward = x.forward_diff(&f);
//!
//! // Calculate gradient of `f` at `x` using central differences
//! let grad_central = x.central_diff(&f);
//! #
//! #  // Desired solution
//! #  let res = vec![1.0f64, 2.0];
//! #
//! #  // Check result
//! #  for i in 0..2 {
//! #      assert!((res[i] - grad_forward[i]).abs() < 1e-6);
//! #      assert!((res[i] - grad_central[i]).abs() < 1e-6);
//! #  }
//! # }
//! ```
//!
//! ## Calculation of the Jacobian
//!
//! Note that the same interface is also implemented for `ndarray::Array1<f64>` (not shown).
//!
//! ### Full Jacobian
//!
//! The calculation of the full Jacobian requires `n+1` evaluations of the function `f`.
//!
//! ```rust
//! use finitediff::FiniteDiff;
//!
//! let f = |x: &Vec<f64>| -> Vec<f64> {
//!     // ...
//! #      vec![
//! #          2.0 * (x[1].powi(3) - x[0].powi(2)),
//! #          3.0 * (x[1].powi(3) - x[0].powi(2)) + 2.0 * (x[2].powi(3) - x[1].powi(2)),
//! #          3.0 * (x[2].powi(3) - x[1].powi(2)) + 2.0 * (x[3].powi(3) - x[2].powi(2)),
//! #          3.0 * (x[3].powi(3) - x[2].powi(2)) + 2.0 * (x[4].powi(3) - x[3].powi(2)),
//! #          3.0 * (x[4].powi(3) - x[3].powi(2)) + 2.0 * (x[5].powi(3) - x[4].powi(2)),
//! #          3.0 * (x[5].powi(3) - x[4].powi(2)),
//! #      ]
//! };
//!
//! let x = vec![1.0f64, 1.0, 1.0, 1.0, 1.0, 1.0];
//!
//! // Using forward differences
//! let jacobian_forward = x.forward_jacobian(&f);
//!
//! // Using central differences
//! let jacobian_central = x.central_jacobian(&f);
//!
//! #  let res = vec![
//! #      vec![-4.0, -6.0, 0.0, 0.0, 0.0, 0.0],
//! #      vec![6.0, 5.0, -6.0, 0.0, 0.0, 0.0],
//! #      vec![0.0, 6.0, 5.0, -6.0, 0.0, 0.0],
//! #      vec![0.0, 0.0, 6.0, 5.0, -6.0, 0.0],
//! #      vec![0.0, 0.0, 0.0, 6.0, 5.0, -6.0],
//! #      vec![0.0, 0.0, 0.0, 0.0, 6.0, 9.0],
//! #  ];
//! #
//! #  // Check result
//! #  for i in 0..6 {
//! #      for j in 0..6 {
//! #          assert!((res[i][j] - jacobian_forward[i][j]).abs() < 1e-6);
//! #          assert!((res[i][j] - jacobian_central[i][j]).abs() < 1e-6);
//! #      }
//! #  }
//! ```
//!
//! ### Product of the Jacobian `J(x)` with a vector `p`
//!
//! Directly computing `J(x)*p` can be much more efficient than computing `J(x)` first and then
//! multiplying it with `p`. While computing the full Jacobian `J(x)` requires `n+1` evaluations of
//! `f`, `J(x)*p` only requires `2`.
//!
//! ```rust
//! use finitediff::FiniteDiff;
//!
//! let f = |x: &Vec<f64>| -> Vec<f64> {
//!     // ...
//! #      vec![
//! #          2.0 * (x[1].powi(3) - x[0].powi(2)),
//! #          3.0 * (x[1].powi(3) - x[0].powi(2)) + 2.0 * (x[2].powi(3) - x[1].powi(2)),
//! #          3.0 * (x[2].powi(3) - x[1].powi(2)) + 2.0 * (x[3].powi(3) - x[2].powi(2)),
//! #          3.0 * (x[3].powi(3) - x[2].powi(2)) + 2.0 * (x[4].powi(3) - x[3].powi(2)),
//! #          3.0 * (x[4].powi(3) - x[3].powi(2)) + 2.0 * (x[5].powi(3) - x[4].powi(2)),
//! #          3.0 * (x[5].powi(3) - x[4].powi(2)),
//! #      ]
//! };
//!
//! let x = vec![1.0f64, 1.0, 1.0, 1.0, 1.0, 1.0];
//! let p = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
//!
//! // using forward differences
//! let jacobian_forward = x.forward_jacobian_vec_prod(&f, &p);
//!
//! // using central differences
//! let jacobian_central = x.central_jacobian_vec_prod(&f, &p);
//! #
//! #  let res = vec![8.0, 22.0, 27.0, 32.0, 37.0, 24.0];
//! #
//! #  // Check result
//! #  for i in 0..6 {
//! #      assert!((res[i] - jacobian_forward[i]).abs() < 11.0*1e-6);
//! #      assert!((res[i] - jacobian_central[i]).abs() < 11.0*1e-6);
//! #  }
//! ```
//!
//! ### Sparse Jacobian
//!
//! If the Jacobian is sparse its structure can be exploited using perturbation vectors. See
//! Nocedal & Wright for details.
//!
//! ```rust
//! use finitediff::{FiniteDiff, PerturbationVector};
//!
//! let f = |x: &Vec<f64>| -> Vec<f64> {
//!     // ...
//! #      vec![
//! #          2.0 * (x[1].powi(3) - x[0].powi(2)),
//! #          3.0 * (x[1].powi(3) - x[0].powi(2)) + 2.0 * (x[2].powi(3) - x[1].powi(2)),
//! #          3.0 * (x[2].powi(3) - x[1].powi(2)) + 2.0 * (x[3].powi(3) - x[2].powi(2)),
//! #          3.0 * (x[3].powi(3) - x[2].powi(2)) + 2.0 * (x[4].powi(3) - x[3].powi(2)),
//! #          3.0 * (x[4].powi(3) - x[3].powi(2)) + 2.0 * (x[5].powi(3) - x[4].powi(2)),
//! #          3.0 * (x[5].powi(3) - x[4].powi(2)),
//! #      ]
//! };
//!
//! let x = vec![1.0f64, 1.0, 1.0, 1.0, 1.0, 1.0];
//!
//! let pert = vec![
//!     PerturbationVector::new()
//!         .add(0, vec![0, 1])
//!         .add(3, vec![2, 3, 4]),
//!     PerturbationVector::new()
//!         .add(1, vec![0, 1, 2])
//!         .add(4, vec![3, 4, 5]),
//!     PerturbationVector::new()
//!         .add(2, vec![1, 2, 3])
//!         .add(5, vec![4, 5]),
//! ];
//!
//! // using forward differences
//! let jacobian_forward = x.forward_jacobian_pert(&f, &pert);
//!
//! // using central differences
//! let jacobian_central = x.central_jacobian_pert(&f, &pert);
//! #
//! #  let res = vec![
//! #      vec![-4.0, -6.0, 0.0, 0.0, 0.0, 0.0],
//! #      vec![6.0, 5.0, -6.0, 0.0, 0.0, 0.0],
//! #      vec![0.0, 6.0, 5.0, -6.0, 0.0, 0.0],
//! #      vec![0.0, 0.0, 6.0, 5.0, -6.0, 0.0],
//! #      vec![0.0, 0.0, 0.0, 6.0, 5.0, -6.0],
//! #      vec![0.0, 0.0, 0.0, 0.0, 6.0, 9.0],
//! #  ];
//! #
//! #  // Check result
//! #  for i in 0..6 {
//! #      for j in 0..6 {
//! #          assert!((res[i][j] - jacobian_forward[i][j]).abs() < 1e-6);
//! #          assert!((res[i][j] - jacobian_central[i][j]).abs() < 1e-6);
//! #      }
//! #  }
//! ```
//!
//! ## Calculation of the Hessian
//!
//! Note that the same interface is also implemented for `ndarray::Array1<f64>` (not shown).
//!
//! ### Full Hessian
//!
//! ```rust
//! use finitediff::FiniteDiff;
//!
//! let g = |x: &Vec<f64>| -> Vec<f64> {
//!     // ...
//! #     vec![1.0, 2.0 * x[1], x[3].powi(2), 2.0 * x[3] * x[2]]
//! };
//!
//! let x = vec![1.0f64, 1.0, 1.0, 1.0];
//!
//! // using forward differences
//! let hessian_forward = x.forward_hessian(&g);
//!
//! // using central differences
//! let hessian_central = x.central_hessian(&g);
//! #
//! #  let res = vec![
//! #      vec![0.0, 0.0, 0.0, 0.0],
//! #      vec![0.0, 2.0, 0.0, 0.0],
//! #      vec![0.0, 0.0, 0.0, 2.0],
//! #      vec![0.0, 0.0, 2.0, 2.0],
//! #  ];
//! #
//! #  // Check result
//! #  for i in 0..4 {
//! #      for j in 0..4 {
//! #          assert!((res[i][j] - hessian_forward[i][j]).abs() < 1e-6);
//! #          assert!((res[i][j] - hessian_central[i][j]).abs() < 1e-6);
//! #      }
//! #  }
//! ```
//!
//! ### Product of the Hessian `H(x)` with a vector `p`
//!
//! ```rust
//! use finitediff::FiniteDiff;
//!
//! let g = |x: &Vec<f64>| -> Vec<f64> {
//!     // ...
//! #     vec![1.0, 2.0 * x[1], x[3].powi(2), 2.0 * x[3] * x[2]]
//! };
//!
//! let x = vec![1.0f64, 1.0, 1.0, 1.0];
//! let p = vec![2.0, 3.0, 4.0, 5.0];
//!
//! // using forward differences
//! let hessian_forward = x.forward_hessian_vec_prod(&g, &p);
//!
//! // using forward differences
//! let hessian_central = x.central_hessian_vec_prod(&g, &p);
//! #
//! #  let res = vec![0.0, 6.0, 10.0, 18.0];
//! #
//! #  for i in 0..4 {
//! #      assert!((res[i] - hessian_forward[i]).abs() < 1e-6);
//! #      assert!((res[i] - hessian_central[i]).abs() < 1e-6);
//! #  }
//! ```
//!
//! ### Calculation of the Hessian without knowledge of the gradient
//!
//! ```rust
//! use finitediff::FiniteDiff;
//!
//! let f = |x: &Vec<f64>| -> f64 {
//!     // ...
//! #     x[0] + x[1].powi(2) + x[2] * x[3].powi(2)
//! };
//!
//! let x = vec![1.0f64, 1.0, 1.0, 1.0];
//!
//! let hessian = x.forward_hessian_nograd(&f);
//! #
//! #  let res = vec![
//! #      vec![0.0, 0.0, 0.0, 0.0],
//! #      vec![0.0, 2.0, 0.0, 0.0],
//! #      vec![0.0, 0.0, 0.0, 2.0],
//! #      vec![0.0, 0.0, 2.0, 2.0],
//! #  ];
//! #
//! #  // Check result
//! #  for i in 0..4 {
//! #      for j in 0..4 {
//! #          assert!((res[i][j] - hessian[i][j]).abs() < 1e-6)
//! #      }
//! #  }
//! ```
//!
//! ### Calculation of the sparse Hessian without knowledge of the gradient
//!
//! ```rust
//! use finitediff::FiniteDiff;
//!
//! let f = |x: &Vec<f64>| -> f64 {
//!     // ...
//! #     x[0] + x[1].powi(2) + x[2] * x[3].powi(2)
//! };
//!
//! let x = vec![1.0f64, 1.0, 1.0, 1.0];
//!
//! // Indices at which the Hessian should be evaluated. All other
//! // elements of the Hessian will be zero
//! let indices = vec![[1, 1], [2, 3], [3, 3]];
//!
//! let hessian = x.forward_hessian_nograd_sparse(&f, indices);
//! #
//! #  let res = vec![
//! #      vec![0.0, 0.0, 0.0, 0.0],
//! #      vec![0.0, 2.0, 0.0, 0.0],
//! #      vec![0.0, 0.0, 0.0, 2.0],
//! #      vec![0.0, 0.0, 2.0, 2.0],
//! #  ];
//! #
//! #  // Check result
//! #  for i in 0..4 {
//! #      for j in 0..4 {
//! #          assert!((res[i][j] - hessian[i][j]).abs() < 1e-6)
//! #      }
//! #  }
//! ```

mod array;
#[cfg(feature = "ndarray")]
mod ndarray_m;
mod pert;
mod utils;
mod vec;

pub use pert::{PerturbationVector, PerturbationVectors};

pub trait FiniteDiff
where
    Self: Sized,
{
    type Scalar;
    type Jacobian;
    type Hessian;
    type OperatorOutput;

    /// Forward difference calculated as
    ///
    /// `df/dx_i (x) \approx (f(x + sqrt(EPS_F64) * e_i) - f(x))/sqrt(EPS_F64)  \forall i`
    ///
    /// where `f` is the cost function and `e_i` is the `i`th unit vector.
    /// For a parameter vector of length `n`, this requires `n+1` evaluations of `f`.
    fn forward_diff(&self, f: &dyn Fn(&Self) -> Self::Scalar) -> Self;

    /// Central difference calculated as
    ///
    /// `df/dx_i (x) \approx (f(x + sqrt(EPS_F64) * e_i) - f(x - sqrt(EPS_F64) * e_i))/(2.0 * sqrt(EPS_F64))  \forall i`
    ///
    /// where `f` is the cost function and `e_i` is the `i`th unit vector.
    /// For a parameter vector of length `n`, this requires `2*n` evaluations of `f`.
    fn central_diff(&self, f: &dyn Fn(&Self) -> Self::Scalar) -> Self;

    /// Calculation of the Jacobian J(x) of a vector function `fs` using forward differences:
    ///
    /// `dfs/dx_i (x) \approx (fs(x + sqrt(EPS_F64) * e_i) - fs(x))/sqrt(EPS_F64)  \forall i`
    ///
    /// where `e_i` is the `i`th unit vector.
    /// For a parameter vector of length `n`, this requires `n+1` evaluations of `fs`.
    fn forward_jacobian(&self, fs: &dyn Fn(&Self) -> Self::OperatorOutput) -> Self::Jacobian;

    /// Calculation of the Jacobian J(x) of a vector function `fs` using central differences:
    ///
    /// `dfs/dx_i (x) \approx (fs(x + sqrt(EPS_F64) * e_i) - fs(x - sqrt(EPS_F64) * e_i))/(2.0 * sqrt(EPS_F64))  \forall i`
    ///
    /// where `e_i` is the `i`th unit vector.
    /// For a parameter vector of length `n`, this requires `2*n` evaluations of `fs`.
    fn central_jacobian(&self, fs: &dyn Fn(&Self) -> Self::OperatorOutput) -> Self::Jacobian;

    /// Calculation of the product of the Jacobian J(x) of a vector function `fs` with a vector `p`
    /// using forward differences:
    ///
    /// `J(x)*p \approx (fs(x + sqrt(EPS_F64) * p) - fs(x))/sqrt(EPS_F64)  \forall i`
    ///
    /// where `e_i` is the `i`th unit vector.
    /// This requires 2 evaluations of `fs`.
    fn forward_jacobian_vec_prod(
        &self,
        fs: &dyn Fn(&Self) -> Self::OperatorOutput,
        p: &Self,
    ) -> Self;

    /// Calculation of the product of the Jacobian J(x) of a vector function `fs` with a vector `p`
    /// using central differences:
    ///
    /// `J(x)*p \approx (fs(x + sqrt(EPS_F64) * p) - fs(x - sqrt(EPS_F64) * p))/(2.0 * sqrt(EPS_F64))  \forall i`
    ///
    /// where `e_i` is the `i`th unit vector.
    /// This requires 2 evaluations of `fs`.
    fn central_jacobian_vec_prod(
        &self,
        fs: &dyn Fn(&Self) -> Self::OperatorOutput,
        p: &Self,
    ) -> Self;

    fn forward_jacobian_pert(
        &self,
        fs: &dyn Fn(&Self) -> Self::OperatorOutput,
        pert: &PerturbationVectors,
    ) -> Self::Jacobian;

    fn central_jacobian_pert(
        &self,
        fs: &dyn Fn(&Self) -> Self::OperatorOutput,
        pert: &PerturbationVectors,
    ) -> Self::Jacobian;

    /// Calculation of the Hessian using forward differences
    ///
    /// `dg/dx_i (x) \approx (g(x + sqrt(EPS_F64) * e_i) - g(x))/sqrt(EPS_F64)  \forall i`
    ///
    /// where `g` is a function which computes the gradient of some other function f and `e_i` is
    /// the `i`th unit vector.
    /// For a parameter vector of length `n`, this requires `n+1` evaluations of `g`.
    fn forward_hessian(&self, g: &dyn Fn(&Self) -> Self::OperatorOutput) -> Self::Hessian;

    /// Calculation of the Hessian using central differences
    ///
    /// `dg/dx_i (x) \approx (g(x + sqrt(EPS_F64) * e_i) - g(x - sqrt(EPS_F64) * e_i))/(2.0 * sqrt(EPS_F64))  \forall i`
    ///
    /// where `g` is a function which computes the gradient of some other function f and `e_i` is
    /// the `i`th unit vector.
    /// For a parameter vector of length `n`, this requires `2*n` evaluations of `g`.
    fn central_hessian(&self, g: &dyn Fn(&Self) -> Self::OperatorOutput) -> Self::Hessian;

    /// Calculation of the product of the Hessian H(x) of a function `g` with a vector `p`
    /// using forward differences:
    ///
    /// `H(x)*p \approx (g(x + sqrt(EPS_F64) * p) - g(x))/sqrt(EPS_F64)  \forall i`
    ///
    /// where `g` is a function which computes the gradient of some other function f and `e_i` is
    /// the `i`th unit vector.
    /// This requires 2 evaluations of `g`.
    fn forward_hessian_vec_prod(&self, g: &dyn Fn(&Self) -> Self::OperatorOutput, p: &Self)
        -> Self;

    /// Calculation of the product of the Hessian H(x) of a function `g` with a vector `p`
    /// using central differences:
    ///
    /// `H(x)*p \approx (g(x + sqrt(EPS_F64) * p) - g(x - sqrt(EPS_F64) * p))/(2.0 * sqrt(EPS_F64))  \forall i`
    ///
    /// where `g` is a function which computes the gradient of some other function f and `e_i` is
    /// the `i`th unit vector.
    /// This requires 2 evaluations of `g`.
    fn central_hessian_vec_prod(&self, g: &dyn Fn(&Self) -> Self::OperatorOutput, p: &Self)
        -> Self;

    /// Calculation of the Hessian using forward differences without knowledge of the gradient:
    ///
    /// `df/(dx_i dx_j) (x) \approx (f(x + sqrt(EPS_F64) * e_i + sqrt(EPS_F64) * e_j) - f(x + sqrt(EPS_F64) + e_i) - f(x + sqrt(EPS_F64) * e_j) + f(x))/EPS_F64  \forall i`
    ///
    /// where `e_i` and `e_j` are the `i`th and `j`th unit vector, respectively.
    // /// For a parameter vector of length `n`, this requires `n*(n+1)/2` evaluations of `g`.
    fn forward_hessian_nograd(&self, f: &dyn Fn(&Self) -> Self::Scalar) -> Self::Hessian;

    /// Calculation of a sparse Hessian using forward differences without knowledge of the gradient:
    ///
    /// `df/(dx_i dx_j) (x) \approx (f(x + sqrt(EPS_F64) * e_i + sqrt(EPS_F64) * e_j) - f(x + sqrt(EPS_F64) + e_i) - f(x + sqrt(EPS_F64) * e_j) + f(x))/EPS_F64  \forall i`
    ///
    /// where `e_i` and `e_j` are the `i`th and `j`th unit vector, respectively.
    /// The indices which are to be evaluated need to be provided via `indices`. Note that due to
    /// the symmetry of the Hessian, an index `(a, b)` will also compute the value of the Hessian at
    /// `(b, a)`.
    // /// For a parameter vector of length `n`, this requires `n*(n+1)/2` evaluations of `g`.
    fn forward_hessian_nograd_sparse(
        &self,
        f: &dyn Fn(&Self) -> Self::Scalar,
        indices: Vec<[usize; 2]>,
    ) -> Self::Hessian;
}
