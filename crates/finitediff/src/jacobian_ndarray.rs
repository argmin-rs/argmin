// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use ndarray::Array2;

use crate::pert::*;
use crate::utils::*;
use crate::EPS_F64;

pub fn forward_jacobian_ndarray_f64(
    x: &ndarray::Array1<f64>,
    fs: &dyn Fn(&ndarray::Array1<f64>) -> ndarray::Array1<f64>,
) -> ndarray::Array2<f64> {
    let fx = (fs)(&x);
    let mut xt = x.clone();
    let rn = fx.len();
    let n = x.len();
    let mut out = Array2::zeros((n, rn));
    for i in 0..n {
        let fx1 = mod_and_calc_ndarray_f64(&mut xt, fs, i, EPS_F64.sqrt());
        for j in 0..rn {
            out[(i, j)] = (fx1[j] - fx[j]) / EPS_F64.sqrt();
        }
    }
    out
}

pub fn central_jacobian_ndarray_f64(
    x: &ndarray::Array1<f64>,
    fs: &dyn Fn(&ndarray::Array1<f64>) -> ndarray::Array1<f64>,
) -> ndarray::Array2<f64> {
    let mut xt = x.clone();

    // TODO: get rid of this! fx is only needed to calculate rn in order to be able to allocate the
    // array for the jacobian.
    let fx = (fs)(&x);

    let rn = fx.len();
    let n = x.len();

    let mut out = Array2::zeros((n, rn));
    for i in 0..n {
        let fx1 = mod_and_calc_ndarray_f64(&mut xt, fs, i, EPS_F64.sqrt());
        let fx2 = mod_and_calc_ndarray_f64(&mut xt, fs, i, -EPS_F64.sqrt());
        for j in 0..rn {
            out[(i, j)] = (fx1[j] - fx2[j]) / (2.0 * EPS_F64.sqrt());
        }
    }
    out
}

pub fn forward_jacobian_vec_prod_ndarray_f64(
    x: &ndarray::Array1<f64>,
    fs: &dyn Fn(&ndarray::Array1<f64>) -> ndarray::Array1<f64>,
    p: &ndarray::Array1<f64>,
) -> ndarray::Array1<f64> {
    let fx = (fs)(&x);
    let x1 = x + &p.mapv(|pi| EPS_F64.sqrt() * pi);
    let fx1 = (fs)(&x1);
    (fx1 - fx) / EPS_F64.sqrt()
}

pub fn central_jacobian_vec_prod_ndarray_f64(
    x: &ndarray::Array1<f64>,
    fs: &dyn Fn(&ndarray::Array1<f64>) -> ndarray::Array1<f64>,
    p: &ndarray::Array1<f64>,
) -> ndarray::Array1<f64> {
    let x1 = x + &p.mapv(|pi| EPS_F64.sqrt() * pi);
    let x2 = x + &p.mapv(|pi| -EPS_F64.sqrt() * pi);
    let fx1 = (fs)(&x1);
    let fx2 = (fs)(&x2);
    (fx1 - fx2) / (2.0 * EPS_F64.sqrt())
}

pub fn forward_jacobian_pert_ndarray_f64(
    x: &ndarray::Array1<f64>,
    fs: &dyn Fn(&ndarray::Array1<f64>) -> ndarray::Array1<f64>,
    pert: &PerturbationVectors,
) -> ndarray::Array2<f64> {
    let fx = (fs)(&x);
    let mut xt = x.clone();
    let mut out = ndarray::Array2::zeros((fx.len(), x.len()));
    for pert_item in pert.iter() {
        for j in pert_item.x_idx.iter() {
            xt[*j] += EPS_F64.sqrt();
        }

        let fx1 = (fs)(&xt);

        for j in pert_item.x_idx.iter() {
            xt[*j] = x[*j];
        }

        for (k, x_idx) in pert_item.x_idx.iter().enumerate() {
            for j in pert_item.r_idx[k].iter() {
                out[(*x_idx, *j)] = (fx1[*j] - fx[*j]) / EPS_F64.sqrt();
            }
        }
    }
    out
}

pub fn central_jacobian_pert_ndarray_f64(
    x: &ndarray::Array1<f64>,
    fs: &dyn Fn(&ndarray::Array1<f64>) -> ndarray::Array1<f64>,
    pert: &PerturbationVectors,
) -> ndarray::Array2<f64> {
    let mut out = ndarray::Array2::zeros((1, 1));
    let mut xt = x.clone();
    for (i, pert_item) in pert.iter().enumerate() {
        for j in pert_item.x_idx.iter() {
            xt[*j] += EPS_F64.sqrt();
        }

        let fx1 = (fs)(&xt);

        for j in pert_item.x_idx.iter() {
            xt[*j] = x[*j] - EPS_F64.sqrt();
        }

        let fx2 = (fs)(&xt);

        for j in pert_item.x_idx.iter() {
            xt[*j] = x[*j];
        }

        if i == 0 {
            out = ndarray::Array2::zeros((fx1.len(), x.len()));
        }

        for (k, x_idx) in pert_item.x_idx.iter().enumerate() {
            for j in pert_item.r_idx[k].iter() {
                out[(*x_idx, *j)] = (fx1[*j] - fx2[*j]) / (2.0 * EPS_F64.sqrt());
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray;
    use ndarray::{array, Array1};

    const COMP_ACC: f64 = 1e-6;

    fn f(x: &Array1<f64>) -> Array1<f64> {
        array![
            2.0 * (x[1].powi(3) - x[0].powi(2)),
            3.0 * (x[1].powi(3) - x[0].powi(2)) + 2.0 * (x[2].powi(3) - x[1].powi(2)),
            3.0 * (x[2].powi(3) - x[1].powi(2)) + 2.0 * (x[3].powi(3) - x[2].powi(2)),
            3.0 * (x[3].powi(3) - x[2].powi(2)) + 2.0 * (x[4].powi(3) - x[3].powi(2)),
            3.0 * (x[4].powi(3) - x[3].powi(2)) + 2.0 * (x[5].powi(3) - x[4].powi(2)),
            3.0 * (x[5].powi(3) - x[4].powi(2)),
        ]
    }

    fn res1() -> Vec<Vec<f64>> {
        vec![
            vec![-4.0, -6.0, 0.0, 0.0, 0.0, 0.0],
            vec![6.0, 5.0, -6.0, 0.0, 0.0, 0.0],
            vec![0.0, 6.0, 5.0, -6.0, 0.0, 0.0],
            vec![0.0, 0.0, 6.0, 5.0, -6.0, 0.0],
            vec![0.0, 0.0, 0.0, 6.0, 5.0, -6.0],
            vec![0.0, 0.0, 0.0, 0.0, 6.0, 9.0],
        ]
    }

    fn res2() -> Vec<f64> {
        vec![8.0, 22.0, 27.0, 32.0, 37.0, 24.0]
    }

    fn x() -> Array1<f64> {
        array![1.0f64, 1.0, 1.0, 1.0, 1.0, 1.0]
    }

    fn p() -> Array1<f64> {
        array![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]
    }

    fn pert() -> PerturbationVectors {
        vec![
            PerturbationVector::new()
                .add(0, vec![0, 1])
                .add(3, vec![2, 3, 4]),
            PerturbationVector::new()
                .add(1, vec![0, 1, 2])
                .add(4, vec![3, 4, 5]),
            PerturbationVector::new()
                .add(2, vec![1, 2, 3])
                .add(5, vec![4, 5]),
        ]
    }

    #[test]
    fn test_forward_jacobian_ndarray_f64() {
        let jacobian = forward_jacobian_ndarray_f64(&x(), &f);
        let res = res1();
        // println!("{:?}", jacobian);
        for i in 0..6 {
            for j in 0..6 {
                assert!((res[i][j] - jacobian[(i, j)]).abs() < COMP_ACC)
            }
        }
    }

    #[test]
    fn test_central_jacobian_ndarray_f64() {
        let jacobian = central_jacobian_ndarray_f64(&x(), &f);
        let res = res1();
        // println!("{:?}", jacobian);
        for i in 0..6 {
            for j in 0..6 {
                assert!((res[i][j] - jacobian[(i, j)]).abs() < COMP_ACC);
            }
        }
    }

    #[test]
    fn test_forward_jacobian_vec_prod_ndarray_f64() {
        let jacobian = forward_jacobian_vec_prod_ndarray_f64(&x(), &f, &p());
        let res = res2();
        // println!("{:?}", jacobian);
        // the accuracy for this is pretty bad!!
        for i in 0..6 {
            assert!((res[i] - jacobian[i]).abs() < 11.0 * COMP_ACC)
        }
    }

    #[test]
    fn test_central_jacobian_vec_prod_ndarray_f64() {
        let jacobian = central_jacobian_vec_prod_ndarray_f64(&x(), &f, &p());
        let res = res2();
        // println!("{:?}", jacobian);
        for i in 0..6 {
            assert!((res[i] - jacobian[i]).abs() < COMP_ACC)
        }
    }

    #[test]
    fn test_forward_jacobian_pert_ndarray_f64() {
        let jacobian = forward_jacobian_pert_ndarray_f64(&x(), &f, &pert());
        let res = res1();
        // println!("jacobian:\n{:?}", jacobian);
        // println!("res:\n{:?}", res);
        for i in 0..6 {
            for j in 0..6 {
                assert!((res[i][j] - jacobian[(i, j)]).abs() < COMP_ACC)
            }
        }
    }

    #[test]
    fn test_central_jacobian_pert_ndarray_f64() {
        let jacobian = central_jacobian_pert_ndarray_f64(&x(), &f, &pert());
        let res = res1();
        // println!("jacobian:\n{:?}", jacobian);
        // println!("res:\n{:?}", res);
        for i in 0..6 {
            for j in 0..6 {
                assert!((res[i][j] - jacobian[(i, j)]).abs() < COMP_ACC)
            }
        }
    }
}
