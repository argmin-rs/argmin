// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Cauchy point
//!
//!
//! ## Reference
//!
//! TODO
//!
// //!
// //! # Example
// //!
// //! ```rust
// //! todo
// //! ```

use prelude::*;
use std;

/// Cauchy Point
#[derive(ArgminSolver)]
pub struct CauchyPoint<T>
where
    T: Clone + std::default::Default + std::fmt::Debug,
{
    /// base
    base: ArgminBase<T, f64>,
}

impl<T> CauchyPoint<T>
where
    T: Clone + std::default::Default + std::fmt::Debug,
    // CauchyPoint<T>: ArgminSolver<Parameters = T, OperatorOutput = f64>,
{
    /// Constructor
    ///
    /// Parameters:
    ///
    /// `operator`: operator
    pub fn new(operator: Box<ArgminOperator<Parameters = T, OperatorOutput = f64>>) -> Self {
        CauchyPoint {
            base: ArgminBase::new(operator, T::default()),
        }
    }
}

impl<T> ArgminNextIter for CauchyPoint<T>
where
    T: Clone + std::default::Default + std::fmt::Debug,
{
    type Parameters = T;
    type OperatorOutput = f64;

    fn init(&mut self) -> Result<(), Error> {
        // This is not an interative method.
        self.set_max_iters(1);
        Ok(())
    }

    fn next_iter(&mut self) -> Result<ArgminIterationData<Self::Parameters>, Error> {
        // let out = ArgminIterationData::new(new_param, self.best_f);
        // Ok(out)
        unimplemented!()
    }
}
