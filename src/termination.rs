// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! TODO Documentation

/// Indicates why the optimization algorithm stopped
#[derive(Debug)]
pub enum TerminationReason {
    /// Maximum number of iterations reached
    MaxItersReached,
    /// Target cost function value reached
    TargetCostReached,
    /// Target precision reached
    TargetPrecisionReached,
    /// Not covered by the above
    Other,
    /// In case it has not terminated yet
    NotTerminated,
}

impl TerminationReason {
    /// Returns `true` if a solver terminated and `false` otherwise
    pub fn terminated(&self) -> bool {
        match *self {
            TerminationReason::NotTerminated => false,
            _ => true,
        }
    }
}
