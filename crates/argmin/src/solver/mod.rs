// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

pub mod brent;
pub mod conjugategradient;
pub mod gaussnewton;
pub mod goldensectionsearch;
pub mod gradientdescent;
pub mod landweber;
pub mod linesearch;
pub mod neldermead;
pub mod newton;
#[cfg(feature = "rand")]
pub mod particleswarm;
pub mod quasinewton;
#[cfg(feature = "rand")]
pub mod simulatedannealing;
pub mod trustregion;
