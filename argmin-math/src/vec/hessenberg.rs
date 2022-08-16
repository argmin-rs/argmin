// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

/// Transforms a matrix into a Hessenberg form.
/// This transformation is an intermediate step to Eigendecomposition
pub(crate) struct HessenbergTransformer<T> {
    pub p: Vec<Vec<T>>,
    pub h: Vec<Vec<T>>,
}

macro_rules! make_hessenberg_transformer {
    ($t:ty) => {
        impl HessenbergTransformer<$t> {
            pub fn new(matrix: &Vec<Vec<$t>>) -> Self {
                let m = matrix.len();
                let mut householder_vectors = matrix.clone();

                let mut ort = vec![0.; m];

                HessenbergTransformer::<$t>::transform(&mut householder_vectors, &mut ort);

                HessenbergTransformer {
                    p: HessenbergTransformer::<$t>::get_p(&householder_vectors, &mut ort),
                    h: HessenbergTransformer::<$t>::get_h(&householder_vectors),
                }
            }

            fn transform(householder_vectors: &mut Vec<Vec<$t>>, ort: &mut [$t]) {
                let n = householder_vectors.len();
                let high = n - 1;

                for m in 1..=(high - 1) {
                    // Scale column.
                    let mut scale = 0.;
                    for i in m..=high {
                        scale += householder_vectors[i][m - 1].abs();
                    }

                    if (scale - 0.).abs() > <$t>::EPSILON {
                        let mut h = 0.;

                        for i in (m..=high).rev() {
                            ort[i] = householder_vectors[i][m - 1] / scale;
                            h += ort[i] * ort[i];
                        }

                        let g = if ort[m] > 0. { -h.sqrt() } else { h.sqrt() };

                        h -= ort[m] * g;
                        ort[m] -= g;

                        for j in m..n {
                            let mut f = 0.;
                            for i in (m..=high).rev() {
                                f += ort[i] * householder_vectors[i][j];
                            }
                            f /= h;
                            for i in m..=high {
                                householder_vectors[i][j] -= f * ort[i];
                            }
                        }

                        for i in 0..=high {
                            let mut f = 0.;
                            for j in (m..=high).rev() {
                                f += ort[j] * householder_vectors[i][j];
                            }
                            f /= h;
                            for j in m..=high {
                                householder_vectors[i][j] -= f * ort[j];
                            }
                        }

                        ort[m] *= scale;
                        householder_vectors[m][m - 1] = scale * g;
                    }
                }
            }

            fn get_p(householder_vectors: &Vec<Vec<$t>>, ort: &mut [$t]) -> Vec<Vec<$t>> {
                let n = householder_vectors.len();
                let high = n - 1;

                let mut p = vec![vec![0.; n]; n];

                for i in 0..n {
                    for j in 0..n {
                        p[i][j] = if i == j { 1. } else { 0. };
                    }
                }

                for m in (1..=(high - 1)).rev() {
                    if householder_vectors[m][m - 1] != 0.0 {
                        for i in (m + 1)..=high {
                            ort[i] = householder_vectors[i][m - 1];
                        }

                        for j in m..=high {
                            let mut g = 0.0;

                            for i in m..=high {
                                g += ort[i] * p[i][j];
                            }

                            // Double division avoids possible underflow
                            g = (g / ort[m]) / householder_vectors[m][m - 1];

                            for i in m..=high {
                                p[i][j] += g * ort[i];
                            }
                        }
                    }
                }

                p
            }

            fn get_h(householder_vectors: &Vec<Vec<$t>>) -> Vec<Vec<$t>> {
                let m = householder_vectors.len();
                let mut h = vec![vec![0.; m]; m];
                for i in 0..m {
                    if i > 0 {
                        // copy the entry of the lower sub-diagonal
                        h[i][i - 1] = householder_vectors[i][i - 1];
                    }

                    // copy upper triangular part of the matrix
                    for j in i..m {
                        h[i][j] = householder_vectors[i][j];
                    }
                }

                h
            }
        }
    };
}

make_hessenberg_transformer!(f32);
make_hessenberg_transformer!(f64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArgminDot, ArgminEye, ArgminTranspose};
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {

                #[test]
                fn [<test_hessenberg_5 $t>]() {
                    let m = vec![
                        vec![5., 4., 3., 2., 1.],
                        vec![1., 4., 0., 3., 3.],
                        vec![2., 0., 3., 0., 0.],
                        vec![3., 2., 1., 2., 5.],
                        vec![4., 2., 1., 4., 1.],
                    ];

                    let transformer = HessenbergTransformer::<$t>::new(&m);

                    [<check_orthogonal $t>](&transformer.p);
                    [<check_hessenberg_form $t>](&transformer.h)
                }

                #[test]
                fn [<test_hessenberg_3 $t>]() {
                    let m = vec![
                        vec![2., -1., 1.],
                        vec![-1., 2., 1.],
                        vec![1., -1., 2.],
                    ];

                    let transformer = HessenbergTransformer::<$t>::new(&m);

                    [<check_orthogonal $t>](&transformer.p);
                    [<check_hessenberg_form $t>](&transformer.h)
                }

                fn [<check_orthogonal $t>](m: &Vec<Vec<$t>>) {
                    let eye = m.eye_like();
                    let m_t_m = m.clone().t().dot(m);
                    m_t_m.iter().flatten().zip(eye.iter().flatten()).for_each(|(&a, &b)| assert!((a - b).abs() < 1e-4));
                }

                fn [<check_hessenberg_form $t>](m: &Vec<Vec<$t>>) {
                    let n_rows = m.len();
                    assert!(n_rows > 0);
                    let n_cols = m[0].len();

                    for i in 0..n_rows {
                        for j in 0..n_cols {
                            if i > j + 1 {
                                assert!((0. - m[i][j]).abs() <= <$t>::EPSILON);
                            }
                        }
                    }
                }
            }
        }
    }

    make_test!(f64);
    make_test!(f32);
}
