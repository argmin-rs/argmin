// Copyright 2018-2024 argmin developers
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
//! # fn main() -> Result<(), anyhow::Error> {
//! use finitediff::vec;
//!
//! // Define function `f(x)`
//! let f = |x: &Vec<f64>| -> Result<f64, anyhow::Error> {
//!     // ...
//! #     Ok(x[0] + x[1].powi(2))
//! };
//!
//! // Point at which gradient should be calculated
//! let x = vec![1.0f64, 1.0];
//!
//! // Calculate gradient of `f` at `x` using forward differences
//! let g_forward = vec::forward_diff(&f);
//! let grad_forward = g_forward(&x)?;
//!
//! // Calculate gradient of `f` at `x` using central differences
//! let g_central = vec::central_diff(&f);
//! let grad_central = g_central(&x)?;
//! #
//! #  // Desired solution
//! #  let res = vec![1.0f64, 2.0];
//! #
//! #  // Check result
//! #  for i in 0..2 {
//! #      assert!((res[i] - grad_forward[i]).abs() < 1e-6);
//! #      assert!((res[i] - grad_central[i]).abs() < 1e-6);
//! #  }
//! # Ok(())
//! # }
//! ```
//!
//! ### For `ndarray::Array1<f64>`
//!
//! ```rust
//! # fn main() -> Result<(), anyhow::Error> {
//! # #[cfg(feature = "ndarray")]
//! # {
//! use ndarray::{array, Array1};
//! use finitediff::ndarr;
//!
//! // Define cost function `f(x)`
//! let f = |x: &Array1<f64>| -> Result<f64, anyhow::Error> {
//!     // ...
//! #     Ok(x[0] + x[1].powi(2))
//! };
//!
//! // Point at which gradient should be calculated
//! let x = array![1.0f64, 1.0];
//!
//! // Calculate gradient of `f` at `x` using forward differences
//! let g_forward = ndarr::forward_diff(&f);
//! let grad_forward = g_forward(&x)?;
//!
//! // Calculate gradient of `f` at `x` using central differences
//! let g_central = ndarr::central_diff(&f);
//! let grad_central = g_central(&x)?;
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
//! # Ok(())
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
//! # fn main() -> Result<(), anyhow::Error> {
//! use finitediff::vec;
//!
//! let f = |x: &Vec<f64>| -> Result<Vec<f64>, anyhow::Error> {
//!     // ...
//! #      Ok(vec![
//! #          2.0 * (x[1].powi(3) - x[0].powi(2)),
//! #          3.0 * (x[1].powi(3) - x[0].powi(2)) + 2.0 * (x[2].powi(3) - x[1].powi(2)),
//! #          3.0 * (x[2].powi(3) - x[1].powi(2)) + 2.0 * (x[3].powi(3) - x[2].powi(2)),
//! #          3.0 * (x[3].powi(3) - x[2].powi(2)) + 2.0 * (x[4].powi(3) - x[3].powi(2)),
//! #          3.0 * (x[4].powi(3) - x[3].powi(2)) + 2.0 * (x[5].powi(3) - x[4].powi(2)),
//! #          3.0 * (x[5].powi(3) - x[4].powi(2)),
//! #      ])
//! };
//!
//! let x = vec![1.0f64, 1.0, 1.0, 1.0, 1.0, 1.0];
//!
//! // Using forward differences
//! let j_forward = vec::forward_jacobian(&f);
//! let jacobian_forward = j_forward(&x)?;
//!
//! // Using central differences
//! let j_central = vec::central_jacobian(&f);
//! let jacobian_central = j_central(&x)?;
//!
//! #  let res = vec![
//! #      vec![-4.0, 6.0, 0.0, 0.0, 0.0, 0.0],
//! #      vec![-6.0, 5.0, 6.0, 0.0, 0.0, 0.0],
//! #      vec![0.0, -6.0, 5.0, 6.0, 0.0, 0.0],
//! #      vec![0.0, 0.0, -6.0, 5.0, 6.0, 0.0],
//! #      vec![0.0, 0.0, 0.0, -6.0, 5.0, 6.0],
//! #      vec![0.0, 0.0, 0.0, 0.0, -6.0, 9.0],
//! #  ];
//! #
//! #  // Check result
//! #  for i in 0..6 {
//! #      for j in 0..6 {
//! #          assert!((res[i][j] - jacobian_forward[i][j]).abs() < 1e-6);
//! #          assert!((res[i][j] - jacobian_central[i][j]).abs() < 1e-6);
//! #      }
//! #  }
//! # Ok(())
//! # }
//! ```
//!
//! ### Product of the Jacobian `J(x)` with a vector `p`
//!
//! Directly computing `J(x)*p` can be much more efficient than computing `J(x)` first and then
//! multiplying it with `p`. While computing the full Jacobian `J(x)` requires `n+1` evaluations of
//! `f`, `J(x)*p` only requires `2`.
//!
//! ```rust
//! # fn main() -> Result<(), anyhow::Error> {
//! use finitediff::vec;
//!
//! let f = |x: &Vec<f64>| -> Result<Vec<f64>, anyhow::Error> {
//!     // ...
//! #      Ok(vec![
//! #          2.0 * (x[1].powi(3) - x[0].powi(2)),
//! #          3.0 * (x[1].powi(3) - x[0].powi(2)) + 2.0 * (x[2].powi(3) - x[1].powi(2)),
//! #          3.0 * (x[2].powi(3) - x[1].powi(2)) + 2.0 * (x[3].powi(3) - x[2].powi(2)),
//! #          3.0 * (x[3].powi(3) - x[2].powi(2)) + 2.0 * (x[4].powi(3) - x[3].powi(2)),
//! #          3.0 * (x[4].powi(3) - x[3].powi(2)) + 2.0 * (x[5].powi(3) - x[4].powi(2)),
//! #          3.0 * (x[5].powi(3) - x[4].powi(2)),
//! #      ])
//! };
//!
//! let x = vec![1.0f64, 1.0, 1.0, 1.0, 1.0, 1.0];
//! let p = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
//!
//! // using forward differences
//! let j_forward = vec::forward_jacobian_vec_prod(&f);
//! let jacobian_forward = j_forward(&x, &p)?;
//!
//! // using central differences
//! let j_central = vec::central_jacobian_vec_prod(&f);
//! let jacobian_central = j_central(&x, &p)?;
//! #
//! #  let res = vec![8.0, 22.0, 27.0, 32.0, 37.0, 24.0];
//! #
//! #  // Check result
//! #  for i in 0..6 {
//! #      assert!((res[i] - jacobian_forward[i]).abs() < 11.0*1e-6);
//! #      assert!((res[i] - jacobian_central[i]).abs() < 11.0*1e-6);
//! #  }
//! # Ok(())
//! # }
//! ```
//!
//! ### Sparse Jacobian
//!
//! If the Jacobian is sparse its structure can be exploited using perturbation vectors. See
//! Nocedal & Wright for details.
//!
//! ```rust
//! # fn main() -> Result<(), anyhow::Error> {
//! use finitediff::{vec, PerturbationVector};
//!
//! let f = |x: &Vec<f64>| -> Result<Vec<f64>, anyhow::Error> {
//!     // ...
//! #      Ok(vec![
//! #          2.0 * (x[1].powi(3) - x[0].powi(2)),
//! #          3.0 * (x[1].powi(3) - x[0].powi(2)) + 2.0 * (x[2].powi(3) - x[1].powi(2)),
//! #          3.0 * (x[2].powi(3) - x[1].powi(2)) + 2.0 * (x[3].powi(3) - x[2].powi(2)),
//! #          3.0 * (x[3].powi(3) - x[2].powi(2)) + 2.0 * (x[4].powi(3) - x[3].powi(2)),
//! #          3.0 * (x[4].powi(3) - x[3].powi(2)) + 2.0 * (x[5].powi(3) - x[4].powi(2)),
//! #          3.0 * (x[5].powi(3) - x[4].powi(2)),
//! #      ])
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
//! let j_forward = vec::forward_jacobian_pert(&f);
//! let jacobian_forward = j_forward(&x, &pert)?;
//!
//! // using central differences
//! let j_central = vec::central_jacobian_pert(&f);
//! let jacobian_central = j_central(&x, &pert)?;
//! #
//! #  let res = vec![
//! #      vec![-4.0, 6.0, 0.0, 0.0, 0.0, 0.0],
//! #      vec![-6.0, 5.0, 6.0, 0.0, 0.0, 0.0],
//! #      vec![0.0, -6.0, 5.0, 6.0, 0.0, 0.0],
//! #      vec![0.0, 0.0, -6.0, 5.0, 6.0, 0.0],
//! #      vec![0.0, 0.0, 0.0, -6.0, 5.0, 6.0],
//! #      vec![0.0, 0.0, 0.0, 0.0, -6.0, 9.0],
//! #  ];
//! #
//! #  // Check result
//! #  for i in 0..6 {
//! #      for j in 0..6 {
//! #          assert!((res[i][j] - jacobian_forward[i][j]).abs() < 1e-6);
//! #          assert!((res[i][j] - jacobian_central[i][j]).abs() < 1e-6);
//! #      }
//! #  }
//! # Ok(())
//! # }
//! ```
//!
//! ## Calculation of the Hessian
//!
//! Note that the same interface is also implemented for `ndarray::Array1<f64>` (not shown).
//!
//! ### Full Hessian
//!
//! ```rust
//! # fn main() -> Result<(), anyhow::Error> {
//! use finitediff::vec;
//!
//! let g = |x: &Vec<f64>| -> Result<Vec<f64>, anyhow::Error> {
//!     // ...
//! #     Ok(vec![1.0, 2.0 * x[1], x[3].powi(2), 2.0 * x[3] * x[2]])
//! };
//!
//! let x = vec![1.0f64, 1.0, 1.0, 1.0];
//!
//! // using forward differences
//! let h_forward = vec::forward_hessian(&g);
//! let hessian_forward = h_forward(&x)?;
//!
//! // using central differences
//! let h_central = vec::central_hessian(&g);
//! let hessian_central = h_central(&x)?;
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
//! # Ok(())
//! # }
//! ```
//!
//! ### Product of the Hessian `H(x)` with a vector `p`
//!
//! ```rust
//! # fn main() -> Result<(), anyhow::Error> {
//! use finitediff::vec;
//!
//! let g = |x: &Vec<f64>| -> Result<Vec<f64>, anyhow::Error> {
//!     // ...
//! #     Ok(vec![1.0, 2.0 * x[1], x[3].powi(2), 2.0 * x[3] * x[2]])
//! };
//!
//! let x = vec![1.0f64, 1.0, 1.0, 1.0];
//! let p = vec![2.0, 3.0, 4.0, 5.0];
//!
//! // using forward differences
//! let h_forward = vec::forward_hessian_vec_prod(&g);
//! let hessian_forward = h_forward(&x, &p)?;
//!
//! // using forward differences
//! let h_central = vec::central_hessian_vec_prod(&g);
//! let hessian_central = h_central(&x, &p)?;
//! #
//! #  let res = vec![0.0, 6.0, 10.0, 18.0];
//! #
//! #  for i in 0..4 {
//! #      assert!((res[i] - hessian_forward[i]).abs() < 1e-6);
//! #      assert!((res[i] - hessian_central[i]).abs() < 1e-6);
//! #  }
//! # Ok(())
//! # }
//! ```
//!
//! ### Calculation of the Hessian without knowledge of the gradient
//!
//! ```rust
//! # fn main() -> Result<(), anyhow::Error> {
//! use finitediff::vec;
//!
//! let f = |x: &Vec<f64>| -> Result<f64, anyhow::Error> {
//!     // ...
//! #     Ok(x[0] + x[1].powi(2) + x[2] * x[3].powi(2))
//! };
//!
//! let x = vec![1.0f64, 1.0, 1.0, 1.0];
//!
//! let h = vec::forward_hessian_nograd(&f);
//! let hessian = h(&x)?;
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
//! # Ok(())
//! # }
//! ```
//!
//! ### Calculation of the sparse Hessian without knowledge of the gradient
//!
//! ```rust
//! # fn main() -> Result<(), anyhow::Error> {
//! use finitediff::vec;
//!
//! let f = |x: &Vec<f64>| -> Result<f64, anyhow::Error> {
//!     // ...
//! #     Ok(x[0] + x[1].powi(2) + x[2] * x[3].powi(2))
//! };
//!
//! let x = vec![1.0f64, 1.0, 1.0, 1.0];
//!
//! // Indices at which the Hessian should be evaluated. All other
//! // elements of the Hessian will be zero
//! let indices = vec![[1, 1], [2, 3], [3, 3]];
//!
//! let h = vec::forward_hessian_nograd_sparse(&f);
//! let hessian = h(&x, indices)?;
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
//! # Ok(())
//! # }
//! ```

pub mod array;
#[cfg(feature = "ndarray")]
pub mod ndarr;
mod pert;
mod utils;
pub mod vec;

pub use pert::{PerturbationVector, PerturbationVectors};
