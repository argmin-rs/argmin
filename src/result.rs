// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

/// `ArgminResult`
///
/// TODO
use parameter::ArgminParameter;
use ArgminCostValue;

/// Return struct for all solvers.
#[derive(Debug)]
pub struct ArgminResult<T: ArgminParameter, U: ArgminCostValue> {
    /// Final parameter vector
    pub param: T,
    /// Final cost value
    pub cost: U,
    /// Number of iterations
    pub iters: u64,
}

impl<T: ArgminParameter, U: ArgminCostValue> ArgminResult<T, U> {
    /// Constructor
    ///
    /// `param`: Final (best) parameter vector
    /// `cost`: Final (best) cost function value
    /// `iters`: Number of iterations
    pub fn new(param: T, cost: U, iters: u64) -> Self {
        ArgminResult { param, cost, iters }
    }
}
