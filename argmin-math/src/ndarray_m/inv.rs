// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminInv;
use crate::Error;
use ndarray::Array2;
use ndarray_linalg::Inverse;
use num_complex::Complex;

macro_rules! make_inv {
    ($t:ty) => {
        impl ArgminInv<Array2<$t>> for Array2<$t>
        where
            Array2<$t>: Inverse,
        {
            #[inline]
            fn inv(&self) -> Result<Array2<$t>, Error> {
                Ok(<Self as Inverse>::inv(&self)?)
            }
        }

        // inverse for scalars (1d solvers)
        impl ArgminInv<$t> for $t {
            #[inline]
            fn inv(&self) -> Result<$t, Error> {
                Ok(1.0 / self)
            }
        }
    };
}

make_inv!(f32);
make_inv!(f64);
make_inv!(Complex<f32>);
make_inv!(Complex<f64>);

// The tests expect the name for the crate containing the tested functions to be argmin_math
#[cfg(test)]
use crate as argmin_math;
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../argmin-math-ndarray-linalg-tests/src/inv.rs"
));
