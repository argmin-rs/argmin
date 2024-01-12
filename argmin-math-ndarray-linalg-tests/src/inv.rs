// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    use argmin_math::ArgminInv;
    
    use ndarray::array;
    use ndarray::Array2;
    
    
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_inv_ $t>]() {
                    let a = array![
                        [2 as $t, 5 as $t],
                        [1 as $t, 3 as $t],
                    ];
                    let target = array![
                        [3 as $t, -5 as $t],
                        [-1 as $t, 2 as $t],
                    ];
                    let res = <Array2<$t> as ArgminInv<Array2<$t>>>::inv(&a).unwrap();
                    for i in 0..2 {
                        for j in 0..2 {
                            // TODO: before ndarray 0.14 / ndarray-linalg 0.13, comparison with
                            // EPSILON worked, now errors are larger (and dependent on the BLAS
                            // backend)
                            assert!((((res[(i, j)] - target[(i, j)]) as f64).abs()) < 0.000001);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_inv_scalar_ $t>]() {
                    let a = 2.0;
                    let target = 0.5;
                    let res = <$t as ArgminInv<$t>>::inv(&a).unwrap();
                    assert!(((res - target) as f64).abs() < 0.000001);
                }
            }
        };
    }

    make_test!(f32);
    make_test!(f64);
}
