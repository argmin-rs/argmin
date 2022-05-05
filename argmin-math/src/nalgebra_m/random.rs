// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminRandomMatrix;

use num_traits::{FromPrimitive, One, Zero};
use rand::Rng;
use rand_distr::StandardNormal;

use nalgebra::{
    base::{allocator::Allocator, dimension::Dim},
    DefaultAllocator, OMatrix, Scalar,
};

impl<N, R, C> ArgminRandomMatrix for OMatrix<N, R, C>
where
    N: Scalar + Zero + One + FromPrimitive,
    R: Dim,
    C: Dim,
    DefaultAllocator: Allocator<N, R, C>,
{
    #[inline]
    fn standard_normal(nrows: usize, ncols: usize) -> OMatrix<N, R, C> {
        let mut rng = rand::thread_rng();
        OMatrix::from_fn_generic(R::from_usize(nrows), C::from_usize(ncols), |_, _| {
            N::from_f64(rng.sample(StandardNormal)).unwrap()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_standard_normal_ $t>]() {
                    let e: DMatrix<$t> = <DMatrix<$t> as ArgminRandomMatrix>::standard_normal(100, 100);
                    let z_score = 1.96;
                    let sd = 1.0;

                    let n_outlier: usize = e.iter().map(|x: &$t| if *x > z_score * sd || *x < -(z_score * sd) {1} else {0}).sum();
                    assert!(n_outlier > 400 && n_outlier < 600); // Should be somewhere around 0.05 * 10000
                    assert_eq!(e.nrows(), 100);
                    assert_eq!(e.ncols(), 100);
                }
            }
        };
    }

    make_test!(f32);
    make_test!(f64);
}
