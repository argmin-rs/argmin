// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use std::ops::AddAssign;

use num::{Float, FromPrimitive};

use crate::pert::PerturbationVectors;
use crate::utils::mod_and_calc;

pub fn forward_jacobian_vec<F>(x: &Vec<F>, fs: &dyn Fn(&Vec<F>) -> Vec<F>) -> Vec<Vec<F>>
where
    F: Float,
{
    let fx = (fs)(x);
    let mut xt = x.clone();
    let eps_sqrt = F::epsilon().sqrt();
    (0..x.len())
        .map(|i| {
            let fx1 = mod_and_calc(&mut xt, fs, i, eps_sqrt);
            fx1.iter()
                .zip(fx.iter())
                .map(|(&a, &b)| (a - b) / eps_sqrt)
                .collect::<Vec<F>>()
        })
        .collect()
}

pub fn central_jacobian_vec<F>(x: &[F], fs: &dyn Fn(&Vec<F>) -> Vec<F>) -> Vec<Vec<F>>
where
    F: Float + FromPrimitive,
{
    let mut xt = x.to_owned();
    let eps_cbrt = F::epsilon().cbrt();
    (0..x.len())
        .map(|i| {
            let fx1 = mod_and_calc(&mut xt, fs, i, eps_cbrt);
            let fx2 = mod_and_calc(&mut xt, fs, i, -eps_cbrt);
            fx1.iter()
                .zip(fx2.iter())
                .map(|(&a, &b)| (a - b) / (F::from_f64(2.0).unwrap() * eps_cbrt))
                .collect::<Vec<F>>()
        })
        .collect()
}

pub fn forward_jacobian_vec_prod_vec<F>(
    x: &Vec<F>,
    fs: &dyn Fn(&Vec<F>) -> Vec<F>,
    p: &[F],
) -> Vec<F>
where
    F: Float,
{
    let fx = (fs)(x);
    let eps_sqrt = F::epsilon().sqrt();
    let x1 = x
        .iter()
        .zip(p.iter())
        .map(|(&xi, &pi)| xi + eps_sqrt * pi)
        .collect();
    let fx1 = (fs)(&x1);
    fx1.iter()
        .zip(fx.iter())
        .map(|(&a, &b)| (a - b) / eps_sqrt)
        .collect::<Vec<F>>()
}

pub fn central_jacobian_vec_prod_vec<F>(x: &[F], fs: &dyn Fn(&Vec<F>) -> Vec<F>, p: &[F]) -> Vec<F>
where
    F: Float + FromPrimitive,
{
    let eps_cbrt = F::epsilon().cbrt();
    let x1 = x
        .iter()
        .zip(p.iter())
        .map(|(&xi, &pi)| xi + eps_cbrt * pi)
        .collect();
    let x2 = x
        .iter()
        .zip(p.iter())
        .map(|(&xi, &pi)| xi - eps_cbrt * pi)
        .collect();
    let fx1 = (fs)(&x1);
    let fx2 = (fs)(&x2);
    fx1.iter()
        .zip(fx2.iter())
        .map(|(&a, &b)| (a - b) / (F::from_f64(2.0).unwrap() * eps_cbrt))
        .collect::<Vec<F>>()
}

pub fn forward_jacobian_pert_vec<F>(
    x: &Vec<F>,
    fs: &dyn Fn(&Vec<F>) -> Vec<F>,
    pert: &PerturbationVectors,
) -> Vec<Vec<F>>
where
    F: Float + FromPrimitive + AddAssign,
{
    let fx = (fs)(x);
    let eps_sqrt = F::epsilon().sqrt();
    let mut xt = x.clone();
    let mut out = vec![vec![F::from_f64(0.0).unwrap(); x.len()]; fx.len()];
    for pert_item in pert.iter() {
        for j in pert_item.x_idx.iter() {
            xt[*j] += eps_sqrt;
        }

        let fx1 = (fs)(&xt);

        for j in pert_item.x_idx.iter() {
            xt[*j] = x[*j];
        }

        for (k, x_idx) in pert_item.x_idx.iter().enumerate() {
            for j in pert_item.r_idx[k].iter() {
                out[*x_idx][*j] = (fx1[*j] - fx[*j]) / eps_sqrt;
            }
        }
    }
    out
}

pub fn central_jacobian_pert_vec<F>(
    x: &[F],
    fs: &dyn Fn(&Vec<F>) -> Vec<F>,
    pert: &PerturbationVectors,
) -> Vec<Vec<F>>
where
    F: Float + FromPrimitive + AddAssign,
{
    let mut out = vec![];
    let eps_cbrt = F::epsilon().cbrt();
    let mut xt = x.to_owned();
    for (i, pert_item) in pert.iter().enumerate() {
        for j in pert_item.x_idx.iter() {
            xt[*j] += eps_cbrt;
        }

        let fx1 = (fs)(&xt);

        for j in pert_item.x_idx.iter() {
            xt[*j] = x[*j] - eps_cbrt;
        }

        let fx2 = (fs)(&xt);

        for j in pert_item.x_idx.iter() {
            xt[*j] = x[*j];
        }

        if i == 0 {
            out = vec![vec![F::from_f64(0.0).unwrap(); x.len()]; fx1.len()];
        }

        for (k, x_idx) in pert_item.x_idx.iter().enumerate() {
            for j in pert_item.r_idx[k].iter() {
                out[*x_idx][*j] = (fx1[*j] - fx2[*j]) / (F::from_f64(2.0).unwrap() * eps_cbrt);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use crate::PerturbationVector;

    use super::*;

    const COMP_ACC: f64 = 1e-6;

    fn f(x: &Vec<f64>) -> Vec<f64> {
        vec![
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

    fn x() -> Vec<f64> {
        vec![1.0f64, 1.0, 1.0, 1.0, 1.0, 1.0]
    }

    fn p() -> Vec<f64> {
        vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]
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
    fn test_forward_jacobian_vec_f64() {
        let jacobian = forward_jacobian_vec(&x(), &f);
        let res = res1();
        // println!("{:?}", jacobian);
        for i in 0..6 {
            for j in 0..6 {
                assert!((res[i][j] - jacobian[i][j]).abs() < COMP_ACC)
            }
        }
    }

    #[test]
    fn test_central_jacobian_vec_f64() {
        let jacobian = central_jacobian_vec(&x(), &f);
        let res = res1();
        // println!("{:?}", jacobian);
        for i in 0..6 {
            for j in 0..6 {
                assert!((res[i][j] - jacobian[i][j]).abs() < COMP_ACC);
            }
        }
    }

    #[test]
    fn test_forward_jacobian_vec_prod_vec_f64() {
        let jacobian = forward_jacobian_vec_prod_vec(&x(), &f, &p());
        let res = res2();
        // println!("{:?}", jacobian);
        // the accuracy for this is pretty bad!!
        for i in 0..6 {
            assert!((res[i] - jacobian[i]).abs() < 11.0 * COMP_ACC)
        }
    }

    #[test]
    fn test_central_jacobian_vec_prod_vec_f64() {
        let jacobian = central_jacobian_vec_prod_vec(&x(), &f, &p());
        let res = res2();
        // println!("{:?}", jacobian);
        for i in 0..6 {
            assert!((res[i] - jacobian[i]).abs() < COMP_ACC)
        }
    }

    #[test]
    fn test_forward_jacobian_pert_vec_f64() {
        let jacobian = forward_jacobian_pert_vec(&x(), &f, &pert());
        let res = res1();
        // println!("jacobian:\n{:?}", jacobian);
        // println!("res:\n{:?}", res);
        for i in 0..6 {
            for j in 0..6 {
                assert!((res[i][j] - jacobian[i][j]).abs() < COMP_ACC)
            }
        }
    }

    #[test]
    fn test_central_jacobian_pert_vec_f64() {
        let jacobian = central_jacobian_pert_vec(&x(), &f, &pert());
        let res = res1();
        // println!("jacobian:\n{:?}", jacobian);
        // println!("res:\n{:?}", res);
        for i in 0..6 {
            for j in 0..6 {
                assert!((res[i][j] - jacobian[i][j]).abs() < COMP_ACC)
            }
        }
    }
}
