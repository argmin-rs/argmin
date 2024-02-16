// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! A collection of two- and multidimensional test functions (and their derivatives and Hessians)
//! for optimization algorithms. For two-dimensional test functions, the derivate and Hessian
//! calculation does not allocate. For multi-dimensional tes functions, the derivative and Hessian
//! calculation comes in two variants. One variant returns `Vec`s and hence does allocate. This is
//! needed for cases, where the number of parameters is only known at run time. In case the number
//! of parameters are known at compile-time, the `_const` variants can be used, which return fixed
//! size arrays and hence do not allocate.
//!
//! The derivative and Hessian calculation is always named `<test function name>_derivative` and
//! `<test function name>_hessian`, respectively. The const generics variants are defined as
//! `<test function name>_derivative_const` and `<test function name>_hessian_const`.
//!
//! Some functions, such as `ackley`, `rosenbrock` and `rastrigin` come with additional optional
//! parameters which change the shape of the functions. These additional parameters are exposed in
//! `ackley_abc`, `rosenbrock_ab` and `rastrigin_a`.
//!
//! All functions are generic over their inputs and work with `[f64]` and `[f32]`.
//!
//! ## Python wrapper
//!
//! Thanks to the python module
//! [`argmin-testfunctions-py`](https://pypi.org/project/argmin-testfunctions-py/), it is possible
//! to use the functions in Python as well. Note that the derivative and Hessian calculation used
//! in the wrapper will always allocate.
//!
//! ## Running the tests and benchmarks
//!
//! The tests can be run with
//!
//! ```bash
//! cargo test
//! ```
//!
//! The test functions derivatives and Hessians are tested against
//! [finitediff](https://crates.io/crates/finitediff) using
//! [proptest](https://crates.io/crates/proptest) to sample the functions at various points.
//!
//! All functions are benchmarked using [criterion.rs](https://crates.io/crates/criterion).
//! Run the benchmarks with
//!
//! ```bash
//! cargo bench
//! ```
//!
//! The report is available in `target/criterion/report/index.html`.
//!
//! ## Contributing
//!
//! This library is the most useful the more test functions it contains, therefore any contributions
//! are highly welcome. For inspiration on what to implement and how to proceed, feel free to have
//! a look at this [issue](https://github.com/argmin-rs/argmin/issues/450).
//!
//! While most of the implemented functions are probably already quite efficient, there are probably
//! a few which may profit from performance improvements.
//!
//! ## License
//!
//! Licensed under either of
//!
//!   * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
//!   * MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)
//!
//! at your option.
//!
//! ### Contribution
//!
//! Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you,
//! as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

mod ackley;
mod beale;
mod booth;
mod bukin;
mod crossintray;
mod easom;
mod eggholder;
mod goldsteinprice;
mod himmelblau;
mod holdertable;
mod levy;
mod matyas;
mod mccorminck;
mod picheny;
mod rastrigin;
mod rosenbrock;
mod schaffer;
mod sphere;
mod styblinskitang;
mod threehumpcamel;
mod zero;

pub use ackley::*;
pub use beale::*;
pub use booth::*;
pub use bukin::*;
pub use crossintray::*;
pub use easom::*;
pub use eggholder::*;
pub use goldsteinprice::*;
pub use himmelblau::*;
pub use holdertable::*;
pub use levy::*;
pub use matyas::*;
pub use mccorminck::*;
pub use picheny::*;
pub use rastrigin::*;
pub use rosenbrock::*;
pub use schaffer::*;
pub use sphere::*;
pub use styblinskitang::*;
pub use threehumpcamel::*;
pub use zero::*;
