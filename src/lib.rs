// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Optimizaton toolbox
//!
//! TODOs

#![warn(missing_docs)]

extern crate argmin_conjugategradient;
extern crate argmin_core;
extern crate argmin_gradientdescent;
extern crate argmin_linesearch;
extern crate argmin_simulatedannealing;

/// Definition of all relevant traits
pub mod prelude;

// #[cfg(test)]
// mod tests {
//     #[test]
//     fn it_works() {
//         assert_eq!(2 + 2, 4);
//     }
// }
