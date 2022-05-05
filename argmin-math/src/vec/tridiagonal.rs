// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#![allow(non_snake_case)]

pub(crate) struct TriDiagonalTransformer<T> {
    pub householder_vectors: Vec<Vec<T>>,
    pub main: Vec<T>,
    pub secondary: Vec<T>,
}

macro_rules! make_tridiagonal_transformer {
    ($t:ty) => {
        impl TriDiagonalTransformer<$t> {
            pub fn new(matrix: &Vec<Vec<$t>>) -> Self {
                let m = matrix.len();

                let mut householder_vectors = matrix.clone();
                let mut main = vec![0 as $t; m];
                let mut secondary = vec![0 as $t; m - 1];

                let mut z = vec![0 as $t; m];

                for k in 0..(m - 1) {
                    //zero-out a row and a column simultaneously
                    main[k] = householder_vectors[k][k];
                    let mut x_norm_sqr = 0.;
                    for j in k + 1..m {
                        let c = householder_vectors[k][j];
                        x_norm_sqr += c * c;
                    }
                    let a = if householder_vectors[k][k + 1] > 0. {
                        -x_norm_sqr.sqrt()
                    } else {
                        x_norm_sqr.sqrt()
                    };
                    secondary[k] = a;
                    if a != 0. {
                        // apply Householder transform from left and right simultaneously
                        householder_vectors[k][k + 1] -= a;
                        let beta = -1. / (a * householder_vectors[k][k + 1]);

                        // compute a = beta A v, where v is the Householder vector
                        // this loop is written in such a way
                        //   1) only the upper triangular part of the matrix is accessed
                        //   2) access is cache-friendly for a matrix stored in rows
                        z[k + 1..m].fill(0.);
                        for i in k + 1..m {
                            let h_i = &householder_vectors[i];
                            let h_k_i = householder_vectors[k][i];
                            let mut z_i = h_i[i] * h_k_i;
                            for j in i + 1..m {
                                let h_i_j = h_i[j];
                                z_i += h_i_j * householder_vectors[k][j];
                                z[j] += h_i_j * h_k_i;
                            }
                            z[i] = beta * (z[i] + z_i);
                        }

                        // compute gamma = beta vT z / 2
                        let mut gamma = 0.;
                        for i in k + 1..m {
                            gamma += z[i] * householder_vectors[k][i];
                        }
                        gamma *= beta / 2.;

                        // compute z = z - gamma v
                        for i in k + 1..m {
                            z[i] -= gamma * householder_vectors[k][i];
                        }

                        // update matrix: A = A - v zT - z vT
                        // only the upper triangular part of the matrix is updated
                        for i in k + 1..m {
                            for j in i..m {
                                householder_vectors[i][j] -= householder_vectors[k][i] * z[j]
                                    + z[i] * householder_vectors[k][j];
                            }
                        }
                    }
                }
                main[m - 1] = householder_vectors[m - 1][m - 1];

                TriDiagonalTransformer {
                    householder_vectors,
                    main,
                    secondary,
                }
            }

            pub fn get_QT(&self) -> Vec<Vec<$t>> {
                let m = self.householder_vectors.len();
                let mut qta = vec![vec![0.; m]; m];

                // build up first part of the matrix by applying Householder transforms
                for k in (1..m).rev() {
                    let h_k = &self.householder_vectors[k - 1];
                    qta[k][k] = 1.;
                    if h_k[k] != 0. {
                        let inv = 1. / (self.secondary[k - 1] * h_k[k]);
                        let mut beta = 1. / self.secondary[k - 1];
                        qta[k][k] = 1. + beta * h_k[k];
                        for i in k + 1..m {
                            qta[k][i] = beta * h_k[i];
                        }
                        for j in k + 1..m {
                            beta = 0.;
                            for i in k + 1..m {
                                beta += qta[j][i] * h_k[i];
                            }
                            beta *= inv;
                            qta[j][k] = beta * h_k[k];
                            for i in k + 1..m {
                                qta[j][i] += beta * h_k[i];
                            }
                        }
                    }
                }
                qta[0][0] = 1.;
                qta
            }

            #[allow(dead_code)]
            pub fn get_T(&self) -> Vec<Vec<$t>> {
                let m = self.main.len();
                let mut ta = vec![vec![0.; m]; m];
                for i in 0..m {
                    ta[i][i] = self.main[i];
                    if i > 0 {
                        ta[i][i - 1] = self.secondary[i - 1];
                    }
                    if i < self.main.len() - 1 {
                        ta[i][i + 1] = self.secondary[i];
                    }
                }
                ta
            }
        }
    };
}

make_tridiagonal_transformer!(f64);
make_tridiagonal_transformer!(f32);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArgminDot, ArgminTranspose};
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {

                #[test]
                fn [<test_tridiagonal_3 $t>]() {
                    let m = vec![
                        vec![1., 3., 4.],
                        vec![3., 2., 2.],
                        vec![4., 2., 0.],
                    ];

                    let transformer = TriDiagonalTransformer::<$t>::new(&m);

                    let m_restored = transformer.get_QT().t().dot(&transformer.get_T()).dot(&transformer.get_QT());

                    for i in 0..m.len() {
                        for j in 0..m.len() {
                            assert!((m[i][j] - m_restored[i][j]).abs() <= 1e-4);
                        }
                    }

                }

                #[test]
                fn [<test_tridiagonal_5 $t>]() {
                    let m = vec![
                        vec![1., 2., 3., 1., 1.],
                        vec![2., 1., 1., 3., 1.],
                        vec![3., 1., 1., 1., 2.],
                        vec![1., 3., 1., 2., 1.],
                        vec![1., 1., 2., 1., 3.],
                    ];

                    let transformer = TriDiagonalTransformer::<$t>::new(&m);

                    let m_restored = transformer.get_QT().t().dot(&transformer.get_T()).dot(&transformer.get_QT());

                    for i in 0..m.len() {
                        for j in 0..m.len() {
                            assert!((m[i][j] - m_restored[i][j]).abs() <= 1e-4);
                        }
                    }

                }

            }
        }
    }

    make_test!(f64);
    make_test!(f32);
}
