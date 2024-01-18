// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

// All code that does not depend on a linked ndarray-linalg backend can still be tested as normal.
// To avoid dublicating tests and to allow convenient testing of functionality that does not need ndarray-linalg the tests are still included here.
// The tests expect the name for the crate containing the tested functions to be argmin_math
#[cfg(test)]
use crate as argmin_math;
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/ndarray-tests-src/scaledsub.rs"
));
