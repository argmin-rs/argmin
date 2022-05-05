// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminEigenSym;

use nalgebra::linalg::SymmetricEigen;
use nalgebra::{
    base::{
        allocator::Allocator,
        dimension::{Dim, DimDiff, DimSub, U1},
    },
    ComplexField, DefaultAllocator, OMatrix, OVector,
};

impl<T, D> ArgminEigenSym<OVector<T::RealField, D>> for OMatrix<T, D, D>
where
    T: ComplexField,
    D: Dim + DimSub<U1>,
    DefaultAllocator: Allocator<T, D, D>
        + Allocator<T::RealField, D>
        + Allocator<T, DimDiff<D, U1>>
        + Allocator<T::RealField, DimDiff<D, U1>>
        + Allocator<T::RealField, U1, D>,
{
    #[inline]
    fn eig_sym(&self) -> (OVector<T::RealField, D>, Self) {
        let sym_eigen = SymmetricEigen::new(self.clone());
        (sym_eigen.eigenvalues, sym_eigen.eigenvectors.transpose())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{dmatrix, matrix, Matrix2, Vector3};
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {

                #[test]
                fn [<test_eigen_simple_ $t>]() {
                    let m = dmatrix![5. as $t, 3.; 3., 5.];

                    let (eigenvalues, eigenvectors) = m.eig_sym();

                    assert_eq!(eigenvalues.len(), 2);

                    let mut diag = Matrix2::<$t>::zeros();
                    for i in 0..2{
                        diag[(i, i)] = eigenvalues[i];
                    }

                    let reconstructed_m = (eigenvectors.clone().transpose() * diag) * eigenvectors;

                    for i in 0..m.nrows(){
                        for j in 0..m.ncols(){
                            assert!((m[(i, j)] - reconstructed_m[(i, j)]).abs() < 1e-6);
                        }
                    }
                }

                #[test]
                fn [<test_eye_eigen_ $t>]() {

                    let eye = matrix![1 as $t, 0., 0.; 0., 1., 0.; 0., 0., 1.];

                    let (eigenvalues, eigenvectors) = eye.eig_sym();

                    let expected_eigenvalues = Vector3::from_element(1.);

                    for i in 0..eye.nrows(){
                        for j in 0..eye.ncols(){
                            assert!((eye[(i, j)] - eigenvectors[(i, j)]).abs() < <$t>::EPSILON);
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
