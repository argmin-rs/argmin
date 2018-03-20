// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Macros
//!
//! Macros to generate the `run` and `terminate` methods of the solvers.

/// This macro generates the `run` function for every solver which implements `ArgminSolver`.
#[macro_export]
macro_rules! make_run {
    ( $ProblemDefinition:ty, $StartingPoints:ty, $Parameter:ty, $CostValue:ty ) => {
        fn run(
            &mut self,
            operator: $ProblemDefinition,
            init_param: &$StartingPoints,
        ) -> Result<ArgminResult<$Parameter, $CostValue>> {
            self.init(operator, init_param)?;

            let mut res;
            loop {
                res = self.next_iter()?;
                if res.terminated {
                    break;
                }
            }
            Ok(res)
        }
    }
}

/// This macro generates the `terminate` function for every solver which implements `ArgminSolver`.
#[macro_export]
macro_rules! make_terminate {
    ($condition:expr, $reason:path;) => {
        if $condition {
            return $reason;
        }
    };
    ($condition:expr, $reason:path ; $($x: expr, $y:path;)*) => {
            make_terminate!( $condition, $reason; );
            make_terminate!( $($x, $y;)* );
    };
    ($self:ident, $($x: expr, $y:path;)*) => {
        fn terminate(&$self) -> TerminationReason {
            make_terminate!( $($x, $y;)* );
            TerminationReason::NotTerminated
        }
    };
}
