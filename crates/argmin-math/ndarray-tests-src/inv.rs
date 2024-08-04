// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    use approx::assert_relative_eq;
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
                            assert_relative_eq!(res[(i, j)], target[(i, j)], epsilon = $t::EPSILON.sqrt());
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
                    assert_relative_eq!(res as f64, target as f64, epsilon = f64::EPSILON);
                }
            }
        };
    }

    make_test!(f32);
    make_test!(f64);
}
