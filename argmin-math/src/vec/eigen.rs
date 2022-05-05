// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::vec::{SchurTransformer, TriDiagonalTransformer};
use crate::{ArgminEigen, ArgminEigenSym};
use num_complex::Complex;

const EIGEN_MAX_ITERATIONS: usize = 30;

macro_rules! make_eigen {
    ($t:ty) => {
        impl ArgminEigen<Vec<$t>> for Vec<Vec<$t>> {
            #[inline]
            fn eig(&self) -> (Vec<$t>, Vec<$t>, Self) {
                fn transform_to_schur(
                    matrix: &Vec<Vec<$t>>,
                ) -> (SchurTransformer<$t>, Vec<$t>, Vec<$t>) {
                    let schur_transform = SchurTransformer::<$t>::new(matrix);
                    let mut real_eigenvalues = vec![0.; schur_transform.t.len()];
                    let mut imag_eigenvalues = vec![0.; schur_transform.t.len()];

                    let mut i = 0;
                    while i < real_eigenvalues.len() {
                        if i == (real_eigenvalues.len() - 1)
                            || (schur_transform.t[i + 1][i] - 0.).abs() <= <$t>::EPSILON
                        {
                            real_eigenvalues[i] = schur_transform.t[i][i];
                        } else {
                            let x = schur_transform.t[i + 1][i + 1];
                            let p = 0.5 * (schur_transform.t[i][i] - x);
                            let z = (p * p
                                + schur_transform.t[i + 1][i] * schur_transform.t[i][i + 1])
                                .abs()
                                .sqrt();
                            real_eigenvalues[i] = x + p;
                            imag_eigenvalues[i] = z;
                            real_eigenvalues[i + 1] = x + p;
                            imag_eigenvalues[i + 1] = -z;
                            i += 1;
                        }
                        i += 1;
                    }

                    (schur_transform, real_eigenvalues, imag_eigenvalues)
                }

                fn find_eigenvectors_from_schur(
                    schur_transform: SchurTransformer<$t>,
                    real_eigenvalues: Vec<$t>,
                    imag_eigenvalues: Vec<$t>,
                ) -> (Vec<Vec<$t>>, Vec<$t>, Vec<$t>) {
                    let mut matrix_t = schur_transform.t;
                    let mut matrix_p = schur_transform.p;

                    let n = matrix_t.len();

                    // compute matrix norm
                    let mut norm = 0.;
                    for i in 0..n {
                        for j in i.saturating_sub(1)..n {
                            norm += matrix_t[i][j].abs();
                        }
                    }

                    let mut r = 0.;
                    let mut s = 0.;
                    let mut z = 0.;

                    for idx in ((n - 1)..=0).rev() {
                        let p = real_eigenvalues[idx];
                        let mut q = imag_eigenvalues[idx];

                        if (q - 0.).abs() <= <$t>::EPSILON {
                            let mut l = idx;
                            matrix_t[idx][idx] = 1.;

                            for i in ((idx - 1)..=0).rev() {
                                let w = matrix_t[i][i] - p;
                                r = 0.;
                                for j in l..=idx {
                                    r += matrix_t[i][j] * matrix_t[j][idx];
                                }
                                if imag_eigenvalues[i] < (0. - <$t>::EPSILON) {
                                    z = w;
                                    s = r;
                                } else {
                                    l = i;
                                    if (imag_eigenvalues[i] - 0.).abs() <= <$t>::EPSILON {
                                        if w != 0. {
                                            matrix_t[i][idx] = -r / w;
                                        } else {
                                            matrix_t[i][idx] = -r / (<$t>::EPSILON * norm);
                                        }
                                    } else {
                                        // Solve real equations
                                        let x = matrix_t[i][i + 1];
                                        let y = matrix_t[i + 1][i];
                                        q = (real_eigenvalues[i] - p) * (real_eigenvalues[i] - p)
                                            + imag_eigenvalues[i] * imag_eigenvalues[i];
                                        let t = (x * s - z * r) / q;
                                        matrix_t[i][idx] = t;
                                        if x.abs() > z.abs() {
                                            matrix_t[i + 1][idx] = (-r - w * t) / x;
                                        } else {
                                            matrix_t[i + 1][idx] = (-s - y * t) / z;
                                        }
                                    }

                                    // Overflow control
                                    let t = matrix_t[i][idx].abs();
                                    if (<$t>::EPSILON * t) * t > 1. {
                                        for j in i..=idx {
                                            matrix_t[j][idx] /= t;
                                        }
                                    }
                                }
                            }
                        } else if q < 0. {
                            let mut l = idx - 1;

                            // Last vector component imaginary so matrix is triangular
                            if (matrix_t[idx][idx - 1]).abs() > (matrix_t[idx - 1][idx]).abs() {
                                matrix_t[idx - 1][idx - 1] = q / matrix_t[idx][idx - 1];
                                matrix_t[idx - 1][idx] =
                                    -(matrix_t[idx][idx] - p) / matrix_t[idx][idx - 1];
                            } else {
                                let result = Complex::new(0.0, -matrix_t[idx - 1][idx])
                                    / Complex::new(matrix_t[idx - 1][idx - 1] - p, q);
                                matrix_t[idx - 1][idx - 1] = result.re;
                                matrix_t[idx - 1][idx] = result.im;
                            }

                            matrix_t[idx][idx - 1] = 0.;
                            matrix_t[idx][idx] = 1.;

                            for i in ((idx - 2)..=0).rev() {
                                let mut ra = 0.;
                                let mut sa = 0.;
                                for j in l..=idx {
                                    ra += matrix_t[i][j] * matrix_t[j][idx - 1];
                                    sa += matrix_t[i][j] * matrix_t[j][idx];
                                }
                                let w = matrix_t[i][i] - p;

                                if imag_eigenvalues[i] < 0.0 - <$t>::EPSILON {
                                    z = w;
                                    r = ra;
                                    s = sa;
                                } else {
                                    l = i;
                                    if (imag_eigenvalues[i] - 0.0).abs() <= <$t>::EPSILON {
                                        let c = Complex::new(-ra, -sa) / Complex::new(w, q);
                                        matrix_t[i][idx - 1] = c.re;
                                        matrix_t[i][idx] = c.im;
                                    } else {
                                        // Solve complex equations
                                        let x = matrix_t[i][i + 1];
                                        let y = matrix_t[i + 1][i];
                                        let mut vr = (real_eigenvalues[i] - p)
                                            * (real_eigenvalues[i] - p)
                                            + imag_eigenvalues[i] * imag_eigenvalues[i]
                                            - q * q;
                                        let vi = (real_eigenvalues[i] - p) * 2.0 * q;
                                        if (vr - 0.).abs() <= <$t>::EPSILON
                                            && (vi - 0.).abs() <= <$t>::EPSILON
                                        {
                                            vr = <$t>::EPSILON
                                                * norm
                                                * (w.abs() + q.abs() + x.abs() + y.abs() + z.abs());
                                        }
                                        let c = Complex::new(
                                            x * r - z * ra + q * sa,
                                            x * s - z * sa - q * ra,
                                        ) / Complex::new(vr, vi);
                                        matrix_t[i][idx - 1] = c.re;
                                        matrix_t[i][idx] = c.im;

                                        if x.abs() > (z.abs() + q.abs()) {
                                            matrix_t[i + 1][idx - 1] = (-ra
                                                - w * matrix_t[i][idx - 1]
                                                + q * matrix_t[i][idx])
                                                / x;
                                            matrix_t[i + 1][idx] = (-sa
                                                - w * matrix_t[i][idx]
                                                - q * matrix_t[i][idx - 1])
                                                / x;
                                        } else {
                                            let c2 = Complex::new(
                                                -r - y * matrix_t[i][idx - 1],
                                                -s - y * matrix_t[i][idx],
                                            ) / Complex::new(z, q);
                                            matrix_t[i + 1][idx - 1] = c2.re;
                                            matrix_t[i + 1][idx] = c2.im;
                                        }
                                    }

                                    // Overflow control
                                    let t = matrix_t[i][idx - 1].abs().max(matrix_t[i][idx].abs());
                                    if (<$t>::EPSILON * t) * t > 1. {
                                        for j in i..=idx {
                                            matrix_t[j][idx - 1] /= t;
                                            matrix_t[j][idx] /= t;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Back transformation to get eigenvectors of original matrix
                    for j in ((n - 1)..=0).rev() {
                        for i in 0..=(n - 1) {
                            z = 0.;
                            for k in 0..=j.min(n - 1) {
                                z += matrix_p[i][k] * matrix_t[k][j];
                            }
                            matrix_p[i][j] = z;
                        }
                    }

                    let mut eigenvectors: Vec<Vec<$t>> = Vec::with_capacity(n);
                    let mut tmp = vec![0.; n];
                    for i in 0..n {
                        for j in 0..n {
                            tmp[j] = matrix_p[i][j];
                        }
                        eigenvectors.push(tmp.clone());
                    }

                    (eigenvectors, real_eigenvalues, imag_eigenvalues)
                }

                let (schur_transform, real_eigenvalues, imag_eigenvalues) =
                    transform_to_schur(self);

                let (eigenvectors, real_eigenvalues, imag_eigenvalues) =
                    find_eigenvectors_from_schur(
                        schur_transform,
                        real_eigenvalues,
                        imag_eigenvalues,
                    );

                (real_eigenvalues, imag_eigenvalues, eigenvectors)
            }
        }
    };
}

macro_rules! make_eigen_sym {
    ($t:ty) => {
        impl ArgminEigenSym<Vec<$t>> for Vec<Vec<$t>> {
            #[inline]
            fn eig_sym(&self) -> (Vec<$t>, Self) {
                let transformer = TriDiagonalTransformer::<$t>::new(self);

                let mut z = transformer.get_QT().clone();
                let n = transformer.main.len();
                let mut eigenvalues = vec![0.; n];
                let mut e = vec![0.; n];
                for i in 0..n-1 {
                    eigenvalues[i] = transformer.main[i];
                    e[i] = transformer.secondary[i];
                }
                eigenvalues[n - 1] = transformer.main[n - 1];
                e[n - 1] = 0.;

                let mut max_absolute_value = 0.;
                for i in 0..n {
                    if eigenvalues[i].abs() > max_absolute_value {
                        max_absolute_value = eigenvalues[i].abs();
                    }
                    if e[i].abs() > max_absolute_value {
                        max_absolute_value = e[i].abs();
                    }
                }

                if max_absolute_value != 0. {
                    for i in 0..n {
                        if eigenvalues[i].abs() <= <$t>::EPSILON * max_absolute_value {
                            eigenvalues[i] = 0.;
                        }
                        if e[i].abs() <= <$t>::EPSILON * max_absolute_value {
                            e[i] = 0.;
                        }
                    }
                }

                for j in 0..n {
                    let mut its = 0;
                    let mut m;
                    loop {
                        m = j;
                        while m < n-1 {
                            let delta = eigenvalues[m].abs() + eigenvalues[m + 1].abs();
                            if ((e[m].abs() + delta) - delta).abs() <= <$t>::EPSILON {
                                break;
                            }
                            m += 1;
                        }
                        if m != j {
                            if its == EIGEN_MAX_ITERATIONS {
                                panic!("Exceeded maximum number of iterations when calculating QL transformation");
                            }
                            its += 1;
                            let mut q = (eigenvalues[j + 1] - eigenvalues[j]) / (2. * e[j]);
                            let mut t = (1. + q * q).sqrt();
                            if q < 0. {
                                q = eigenvalues[m] - eigenvalues[j] + e[j] / (q - t);
                            } else {
                                q = eigenvalues[m] - eigenvalues[j] + e[j] / (q + t);
                            }
                            let mut u = 0.;
                            let mut s = 1.;
                            let mut c = 1.;
                            let mut i = m - 1;
                            while i >= j {
                                let mut p = s * e[i];
                                let h = c * e[i];
                                if (p.abs() >= q.abs()) {
                                    c = q / p;
                                    t = (c * c + 1.).sqrt();
                                    e[i + 1] = p * t;
                                    s = 1. / t;
                                    c *= s;
                                } else {
                                    s = p / q;
                                    t = (s * s + 1.0).sqrt();
                                    e[i + 1] = q * t;
                                    c = 1.0 / t;
                                    s *= c;
                                }
                                if (e[i + 1] == 0.0) {
                                    eigenvalues[i + 1] -= u;
                                    e[m] = 0.0;
                                    break;
                                }
                                q = eigenvalues[i + 1] - u;
                                t = (eigenvalues[i] - q) * s + 2. * c * h;
                                u = s * t;
                                eigenvalues[i + 1] = q + u;
                                q = c * t - h;
                                for ia in 0..n {
                                    p = z[i + 1][ia];
                                    z[i + 1][ia] = s * z[i][ia] + c * p;
                                    z[i][ia] = c * z[i][ia] - s * p;
                                }
                                if i == 0 {
                                    break;
                                } else {
                                    i -= 1;
                                }
                            }
                            if t == 0. && i >= j {
                                continue;
                            }
                            eigenvalues[j] -= u;
                            e[j] = q;
                            e[m] = 0.;
                        }

                        if m == j {
                            break
                        }
                    }
                }

                for i in 0..n {
                    let mut k = i;
                    let mut p = eigenvalues[i];
                    for j in i+1..n {
                        if eigenvalues[j] > p {
                            k = j;
                            p = eigenvalues[j];
                        }
                    }
                    if k != i {
                        eigenvalues[k] = eigenvalues[i];
                        eigenvalues[i] = p;
                        for j in 0..n {
                            p = z[i][j];
                            z[i][j] = z[k][j];
                            z[k][j] = p;
                        }
                    }
                }

                max_absolute_value = 0.;
                for i in 0..n {
                    if eigenvalues[i].abs() > max_absolute_value {
                        max_absolute_value=eigenvalues[i].abs();
                    }
                }

                if max_absolute_value != 0. {
                    for i in 0..n {
                        if eigenvalues[i].abs() < <$t>::EPSILON * max_absolute_value {
                            eigenvalues[i] = 0.;
                        }
                    }
                }

                let mut eigenvectors = vec![vec![0.; n]; n];
                for i in 0..n {
                    for j in 0..n {
                        eigenvectors[i][j] = z[i][j];
                    }
                }

                (eigenvalues, eigenvectors)

            }
        }
    };
}

make_eigen!(f32);
make_eigen!(f64);
make_eigen_sym!(f32);
make_eigen_sym!(f64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArgminDot, ArgminTranspose};
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {

                #[test]
                fn [<test_eigen_simple $t>]() {
                    let m = vec![
                        vec![5. as $t, 3. as $t],
                        vec![3. as $t, 5. as $t]
                    ];

                    let (eigenvalues, _, eigenvectors) = m.eig();
                    assert_eq!(eigenvalues.len(), 2);
                    let mut diag = vec![vec![0.; 2]; 2];
                    for i in 0..2{
                        diag[i][i] = eigenvalues[i];
                    }

                    let reconstructed_m = &eigenvectors.dot(&diag).dot(&eigenvectors.clone().t());

                    reconstructed_m.iter().flatten().zip(m.iter().flatten()).for_each(|(a, b)| assert!((a-b).abs() <= 1e-4));
                }

                #[test]
                fn [<test_eigen_sym_simple $t>]() {
                    let m = vec![
                        vec![5. as $t, 10. as $t, 15. as $t],
                        vec![10. as $t, 20. as $t, 30. as $t],
                        vec![15. as $t, 30. as $t, 45. as $t]
                    ];

                    let (eigenvalues, _) = m.eig_sym();

                    assert_eq!(eigenvalues.len(), 3);
                    assert!((eigenvalues[0] - 70.).abs() <= 1e-6);
                    assert!((eigenvalues[1] - 0.).abs() <= 1e-6);
                    assert!((eigenvalues[2] - 0.).abs() <= 1e-6);
                }

                #[test]
                fn [<test_eye_eigen_ $t>]() {

                    let eye = vec![
                        vec![1. as $t, 0. as $t, 0. as $t],
                        vec![0. as $t, 1. as $t, 0. as $t],
                        vec![0. as $t, 0. as $t, 1. as $t],
                    ];

                    let (eigenvalues, _, eigenvectors) = <Vec<Vec<$t>> as ArgminEigen<Vec<$t>>>::eig(&eye);

                    let expected_eigenvalues = vec![1. as $t; 3];

                    for i in 0..eye.len(){
                        for j in 0..eye.len(){
                            assert!((eye[i][j] - eigenvectors[i][j]).abs() < <$t>::EPSILON);
                        }
                    }

                    for i in 0..eye.len(){
                        assert!((expected_eigenvalues[i] - eigenvalues[i]).abs() < <$t>::EPSILON);
                    }

                }
            }

        };
    }

    make_test!(f32);
    make_test!(f64);
}
