// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

/// This macro generates the `run` function for every solver which implements `ArgminSolver`.
#[macro_export]
macro_rules! make_run {
    ( $E:ty, $D:ty, $A:ty, $B:ty ) => {
        fn run(
            &mut self,
            operator: &'a $E,
            init_param: &$D,
        ) -> Result<ArgminResult<$A, $B>> {
            self.init(operator, init_param)?;

            let mut res;
            loop {
                res = self.next_iter()?;
                if self.terminate() {
                    break;
                }
            }
            Ok(res)
        }
    }
}
