// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#![allow(non_snake_case)]
use crate::vec::HessenbergTransformer;

const SCHUR_MAX_ITERATIONS: usize = 1000;

/// Transforms a matrix into a Schur form.
/// This transformation is an intermediate step to Eigendecomposition
pub(crate) struct SchurTransformer<T> {
    pub p: Vec<Vec<T>>,
    pub t: Vec<Vec<T>>,
}

struct ShiftInfo<T> {
    x: T,
    y: T,
    w: T,
    ex_shift: T,
}

macro_rules! make_schur_transformer {

    ($t:ty) => {

        impl ShiftInfo<$t> {
            fn new() -> Self {
                ShiftInfo {x: 0., y: 0., w: 0., ex_shift: 0.}
            }
        }

        impl SchurTransformer<$t> {

            pub fn new(matrix: &Vec<Vec<$t>>) -> Self {

                let mut h_transformer = HessenbergTransformer::<$t>::new(&matrix);

                SchurTransformer::<$t>::transform(&mut h_transformer);

                SchurTransformer{p: h_transformer.p, t: h_transformer.h}

            }

            fn transform(h_transformer: &mut HessenbergTransformer<$t>) {
                let n = h_transformer.h.len();

                let mut norm = 0.;
                for i in 0..n {
                    // as matrix T is (quasi-)triangular, also take the sub-diagonal element into account
                    for j in i.checked_sub(1).unwrap_or(0)..n {
                        norm += (h_transformer.h[i][j]).abs();
                    }
                }

                let mut shift = ShiftInfo::<$t>::new();

                let mut iteration = 0;
                let mut iu_cond = Some(n - 1);

                while let Some(iu) = iu_cond {

                    // Look for single small sub-diagonal element
                    let il = SchurTransformer::<$t>::find_small_sub_diagonal_element(iu, norm, h_transformer);

                    if il == iu {
                        // One root found
                        h_transformer.h[iu][iu] += shift.ex_shift;
                        iu_cond = iu.checked_sub(1);
                        iteration = 0;
                    } else if il == iu - 1 {

                        let mut p = (h_transformer.h[iu - 1][iu - 1] - h_transformer.h[iu][iu]) / 2.0;
                        let mut q = p * p + h_transformer.h[iu][iu - 1] * h_transformer.h[iu - 1][iu];
                        h_transformer.h[iu][iu] += shift.ex_shift;
                        h_transformer.h[iu - 1][iu - 1] += shift.ex_shift;

                        if q >= 0. {
                            let mut z = q.abs().sqrt();
                            if p >= 0. {
                                z += p;
                            } else {
                                z = p - z;
                            }
                            let x = h_transformer.h[iu][iu - 1];
                            let s = x.abs() + z.abs();
                            p = x / s;
                            q = z / s;
                            let r = (p * p + q * q).sqrt();
                            p /= r;
                            q /= r;

                            // Row modification
                            for j in (iu - 1)..n {
                                z = h_transformer.h[iu - 1][j];
                                h_transformer.h[iu - 1][j] = q * z + p * h_transformer.h[iu][j];
                                h_transformer.h[iu][j] = q * h_transformer.h[iu][j] - p * z;
                            }

                            // Column modification
                            for i in 0..=iu {
                                z = h_transformer.h[i][iu - 1];
                                h_transformer.h[i][iu - 1] = q * z + p * h_transformer.h[i][iu];
                                h_transformer.h[i][iu] = q * h_transformer.h[i][iu] - p * z;
                            }

                            // Accumulate transformations
                            for i in 0..=(n-1) {
                                z = h_transformer.p[i][iu - 1];
                                h_transformer.p[i][iu - 1] = q * z + p * h_transformer.p[i][iu];
                                h_transformer.p[i][iu] = q * h_transformer.p[i][iu] - p * z;
                            }
                        }

                        iu_cond = iu.checked_sub(2);
                        iteration = 0;

                    } else {
                        // No convergence yet
                        SchurTransformer::<$t>::compute_shift(il, iu, iteration, &mut shift, h_transformer);
                        iteration += 1;

                        if iteration > SCHUR_MAX_ITERATIONS {
                            panic!("Exceeded maximum number of iterations when calculating Schur transformation");
                        }

                        let mut h_vec = vec![0.; 3];

                        let im = SchurTransformer::<$t>::init_QR_step(il, iu, &shift, h_transformer, &mut h_vec);
                        SchurTransformer::<$t>::perform_double_QR_step(il, im, iu, &mut shift, h_transformer, &mut h_vec);

                    }

                }
            }

            fn find_small_sub_diagonal_element(start_idx: usize, norm: $t, h_transformer: &HessenbergTransformer<$t>) -> usize {
                let mut l = start_idx;
                while l > 0 {
                    let mut s = (h_transformer.h[l - 1][l - 1]).abs() + (h_transformer.h[l][l]).abs();
                    if s == 0. {
                        s = norm;
                    }
                    if (h_transformer.h[l][l - 1]).abs() < <$t>::EPSILON * s {
                        break;
                    }
                    l -= 1;
                }
                return l;
            }

            fn compute_shift(l: usize, idx: usize, iteration: usize, shift: &mut ShiftInfo<$t>, h_transformer: &mut HessenbergTransformer<$t>) {
                // Form shift
                shift.x = h_transformer.h[idx][idx];
                shift.y = 0.;
                shift.w = 0.;
                if l < idx {
                    shift.y = h_transformer.h[idx - 1][idx - 1];
                    shift.w = h_transformer.h[idx][idx - 1] * h_transformer.h[idx - 1][idx];
                }

                // Wilkinson's original ad hoc shift
                if iteration == 10 {
                    shift.ex_shift += shift.x;
                    for i in 0..=idx {
                        h_transformer.h[i][i] -= shift.x;
                    }
                    let s = (h_transformer.h[idx][idx - 1]).abs() + (h_transformer.h[idx - 1][idx - 2]).abs();
                    shift.x = 0.75 * s;
                    shift.y = 0.75 * s;
                    shift.w = -0.4375 * s * s;
                }

                // MATLAB's new ad hoc shift
                if iteration == 30 {
                    let mut s = (shift.y - shift.x) / 2.0;
                    s = s * s + shift.w;
                    if s > 0. {
                        s = s.sqrt();
                        if (shift.y < shift.x) {
                            s = -s;
                        }
                        s = shift.x - shift.w / ((shift.y - shift.x) / 2.0 + s);
                        for i in 0..=idx {
                            h_transformer.h[i][i] -= s;
                        }
                        shift.ex_shift += s;
                        (shift.x, shift.y, shift.w) = (0.964, 0.964, 0.964);
                    }
                }

            }

            fn init_QR_step(il: usize, iu: usize, shift: &ShiftInfo<$t>, h_transformer: &HessenbergTransformer<$t>, h_vec: &mut [$t]) -> usize{

                let mut im = iu - 2;
                while im >= il {
                    let z = h_transformer.h[im][im];
                    let r = shift.x - z;
                    let s = shift.y - z;
                    h_vec[0] = (r * s - shift.w) / h_transformer.h[im + 1][im] + h_transformer.h[im][im + 1];
                    h_vec[1] = h_transformer.h[im + 1][im + 1] - z - r - s;
                    h_vec[2] = h_transformer.h[im + 2][im + 1];

                    if im == il {
                        break;
                    }

                    let lhs = (h_transformer.h[im][im - 1]).abs() * (h_vec[1].abs() + h_vec[2].abs());
                    let rhs = h_vec[0].abs() * ((h_transformer.h[im - 1][im - 1]).abs() + z.abs() + (h_transformer.h[im + 1][im + 1]).abs());

                    if lhs < <$t>::EPSILON * rhs {
                        break;
                    }

                    im -= 1;
                }

                im

            }

            fn perform_double_QR_step(il: usize, im: usize, iu: usize, shift: &mut ShiftInfo<$t>, h_transformer: &mut HessenbergTransformer<$t>, h_vec: &mut [$t]) {
                let n = h_transformer.h.len();
                let mut p = h_vec[0];
                let mut q = h_vec[1];
                let mut r = h_vec[2];

                for k in im..=(iu-1) {
                    let notlast = k != (iu - 1);
                    if k != im {
                        p = h_transformer.h[k][k - 1];
                        q = h_transformer.h[k + 1][k - 1];
                        r = if notlast {h_transformer.h[k + 2][k - 1]} else {0.};
                        shift.x = p.abs() + q.abs() + r.abs();
                        if (shift.x - 0.0).abs() <= <$t>::EPSILON {
                            continue;
                        }
                        p /= shift.x;
                        q /= shift.x;
                        r /= shift.x;
                    }
                    let mut s = (p * p + q * q + r * r).sqrt();
                    if p < 0. {
                        s = -s;
                    }
                    if s != 0. {
                        if k != im {
                            h_transformer.h[k][k - 1] = -s * shift.x;
                        } else if il != im {
                            h_transformer.h[k][k - 1] = -h_transformer.h[k][k - 1];
                        }
                        p += s;
                        shift.x = p / s;
                        shift.y = q / s;
                        let z = r / s;
                        q /= p;
                        r /= p;

                        // Row modification
                        for j in k..n {
                            p = h_transformer.h[k][j] + q * h_transformer.h[k + 1][j];
                            if notlast {
                                p += r * h_transformer.h[k + 2][j];
                                h_transformer.h[k + 2][j] -= p * z;
                            }
                            h_transformer.h[k][j] -= p * shift.x;
                            h_transformer.h[k + 1][j] -= p * shift.y;
                        }

                        // Column modification
                        for i in 0..=iu.min(k + 3) {
                            p = shift.x * h_transformer.h[i][k] + shift.y * h_transformer.h[i][k + 1];
                            if notlast {
                                p += z * h_transformer.h[i][k + 2];
                                h_transformer.h[i][k + 2] -= p * r;
                            }
                            h_transformer.h[i][k] -= p;
                            h_transformer.h[i][k + 1] -= p * q;
                        }

                        // Accumulate transformations
                        let high = h_transformer.h.len() - 1;
                        for i in 0..=high {
                            p = shift.x * h_transformer.p[i][k] + shift.y * h_transformer.p[i][k + 1];
                            if notlast {
                                p += z * h_transformer.p[i][k + 2];
                                h_transformer.p[i][k + 2] -= p * r;
                            }
                            h_transformer.p[i][k] -= p;
                            h_transformer.p[i][k + 1] -= p * q;
                        }
                    }
                }

                for i in im+2..=iu {
                    h_transformer.h[i][i-2] = 0.;
                    if i > im + 2 {
                        h_transformer.h[i][i-3] = 0.;
                    }
                }

            }

        }

    }
}

make_schur_transformer!(f64);
make_schur_transformer!(f32);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArgminDot, ArgminEye, ArgminTranspose};
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {

                #[test]
                fn [<test_schur_5 $t>]() {
                    let m = vec![
                        vec![5., 4., 3., 2., 1.],
                        vec![1., 4., 0., 3., 3.],
                        vec![2., 0., 3., 0., 0.],
                        vec![3., 2., 1., 2., 5.],
                        vec![4., 2., 1., 4., 1.],
                    ];

                    let transformer = SchurTransformer::<$t>::new(&m);

                    [<check_orthogonal $t>](&transformer.p);
                    [<check_schur_form $t>](&transformer.t)
                }

                #[test]
                fn [<test_schur_3 $t>]() {
                    let m = vec![
                        vec![2., -1., 1.],
                        vec![-1., 2., 1.],
                        vec![1., -1., 2.],
                    ];

                    let transformer = SchurTransformer::<$t>::new(&m);

                    [<check_orthogonal $t>](&transformer.p);
                    [<check_schur_form $t>](&transformer.t)
                }

                fn [<check_orthogonal $t>](m: &Vec<Vec<$t>>) {
                    let eye = m.eye_like();
                    let m_t_m = m.clone().t().dot(m);
                    m_t_m.iter().flatten().zip(eye.iter().flatten()).for_each(|(&a, &b)| assert!((a - b).abs() < 1e-4));
                }

                fn [<check_schur_form $t>](m: &Vec<Vec<$t>>) {
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
