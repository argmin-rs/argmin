// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminEigenSym;
use ndarray::{Array1, Array2};
use ndarray_linalg::{Eigh, UPLO};

macro_rules! make_eigen {
    ($t:ty) => {
        impl ArgminEigenSym<Array1<$t>> for Array2<$t> {
            #[inline]
            fn eig_sym(&self) -> (Array1<$t>, Self) {
                self.eigh(UPLO::Lower).unwrap()
            }
        }
    };
}

make_eigen!(f32);
make_eigen!(f64);

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {

                #[test]
                fn [<test_eigen_simple $t>]() {
                    let m = array![
                        [5. as $t, 3. as $t],
                        [3. as $t, 5. as $t]
                    ];

                    let (eigenvalues, eigenvectors) = m.eig_sym();
                    assert_eq!(eigenvalues.len(), 2);
                    let mut diag = Array2::zeros((2, 2));
                    for i in 0..2{
                        diag[[i, i]] = eigenvalues[i];
                    }

                    let reconstructed_m = &eigenvectors.dot(&diag).dot(&eigenvectors.clone().t());

                    for i in 0..m.nrows(){
                        for j in 0..m.ncols(){
                            assert!((m[[i, j]] - reconstructed_m[[i, j]]).abs() < 1e-8);
                        }
                    }
                }

                #[test]
                fn [<test_eye_eigen_ $t>]() {

                    let eye = array![
                        [1. as $t, 0. as $t, 0. as $t],
                        [0. as $t, 1. as $t, 0. as $t],
                        [0. as $t, 0. as $t, 1. as $t],
                    ];

                    let (eigenvalues, eigenvectors) = ArgminEigenSym::<Array1<$t>>::eig_sym(&eye);

                    let expected_eigenvalues: Array1<$t> = Array1::ones(3);

                    for i in 0..eye.nrows(){
                        for j in 0..eye.ncols(){
                            assert!((eye[[i, j]] - eigenvectors[[i, j]]).abs() < <$t>::EPSILON);
                        }
                    }

                    for i in 0..eye.nrows(){
                        assert!((expected_eigenvalues[i] - eigenvalues[i]).abs() < <$t>::EPSILON);
                    }

                }
            }
        };
    }

    make_test!(f32);
    make_test!(f64);
}
