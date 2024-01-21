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
    use argmin_math::ArgminEye;
    use ndarray::array;
    use ndarray::Array2;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_eye_ $t>]() {
                    let e: Array2<$t> = <Array2<$t> as ArgminEye>::eye(3);
                    let res = array![
                        [1 as $t, 0 as $t, 0 as $t],
                        [0 as $t, 1 as $t, 0 as $t],
                        [0 as $t, 0 as $t, 1 as $t]
                    ];
                    for i in 0..3 {
                        for j in 0..3 {
                            assert_relative_eq!(res[(i, j)] as f64, e[(i, j)] as f64, epsilon = std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_eye_like_ $t>]() {
                    let a = array![
                        [0 as $t, 2 as $t, 6 as $t],
                        [3 as $t, 2 as $t, 7 as $t],
                        [9 as $t, 8 as $t, 1 as $t]
                    ];
                    let e: Array2<$t> = a.eye_like();
                    let res = array![
                        [1 as $t, 0 as $t, 0 as $t],
                        [0 as $t, 1 as $t, 0 as $t],
                        [0 as $t, 0 as $t, 1 as $t]
                    ];
                    for i in 0..3 {
                        for j in 0..3 {
                            assert_relative_eq!(res[(i, j)] as f64, e[(i, j)] as f64, epsilon = std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                #[should_panic]
                #[allow(unused)]
                fn [<test_eye_like_panic_ $t>]() {
                    let a = array![
                        [0 as $t, 2 as $t, 6 as $t],
                        [3 as $t, 2 as $t, 7 as $t],
                    ];
                    let e: Array2<$t> = a.eye_like();
                }
            }
        };
    }

    make_test!(isize);
    make_test!(usize);
    make_test!(i8);
    make_test!(u8);
    make_test!(i16);
    make_test!(u16);
    make_test!(i32);
    make_test!(u32);
    make_test!(i64);
    make_test!(u64);
    make_test!(f32);
    make_test!(f64);
}
