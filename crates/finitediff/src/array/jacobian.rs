// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use std::ops::AddAssign;

use anyhow::Error;
use num::{Float, FromPrimitive};

use crate::pert::PerturbationVectors;
use crate::utils::{mod_and_calc, mod_and_calc_const};

use super::OpFn;

pub fn forward_jacobian_const<const N: usize, const M: usize, F>(
    x: &[F; N],
    fs: OpFn<'_, N, M, F>,
) -> Result<[[F; N]; M], Error>
where
    F: Float + FromPrimitive,
{
    let fx = (fs)(x)?;
    let mut xt = *x;
    let eps_sqrt = F::epsilon().sqrt();
    let mut out = [[F::from_f64(0.0).unwrap(); N]; M];

    for i in 0..N {
        let fx1 = mod_and_calc_const(&mut xt, fs, i, eps_sqrt)?;

        for j in 0..M {
            out[j][i] = (fx1[j] - fx[j]) / eps_sqrt;
        }
    }
    Ok(out)
}

pub fn central_jacobian_const<const N: usize, const M: usize, F>(
    x: &[F; N],
    fs: OpFn<'_, N, M, F>,
) -> Result<[[F; N]; M], Error>
where
    F: Float + FromPrimitive,
{
    let mut xt = *x;
    let eps_cbrt = F::epsilon().cbrt();
    let mut out = [[F::from_f64(0.0).unwrap(); N]; M];
    for i in 0..M {
        let fx1 = mod_and_calc(&mut xt, fs, i, eps_cbrt)?;
        let fx2 = mod_and_calc(&mut xt, fs, i, -eps_cbrt)?;

        for j in 0..M {
            out[j][i] = (fx1[j] - fx2[j]) / (F::from_f64(2.0).unwrap() * eps_cbrt);
        }
    }
    Ok(out)
}

pub fn forward_jacobian_vec_prod_const<const N: usize, const M: usize, F>(
    x: &[F; N],
    fs: OpFn<'_, N, M, F>,
    p: &[F; N],
) -> Result<[F; M], Error>
where
    F: Float + FromPrimitive,
{
    let fx = (fs)(x)?;
    let eps_sqrt = F::epsilon().sqrt();
    let mut x1 = [F::from_f64(0.0).unwrap(); N];
    x1.iter_mut()
        .enumerate()
        .map(|(i, o)| *o = x[i] + eps_sqrt * p[i])
        .count();

    let fx1 = (fs)(&x1)?;
    let mut out = [F::from_f64(0.0).unwrap(); M];
    out.iter_mut()
        .enumerate()
        .map(|(i, o)| {
            *o = (fx1[i] - fx[i]) / eps_sqrt;
        })
        .count();
    Ok(out)
}

pub fn central_jacobian_vec_prod_const<const N: usize, const M: usize, F>(
    x: &[F; N],
    fs: OpFn<'_, N, M, F>,
    p: &[F; N],
) -> Result<[F; M], Error>
where
    F: Float + FromPrimitive,
{
    let eps_cbrt = F::epsilon().cbrt();
    let mut x1 = [F::from_f64(0.0).unwrap(); N];
    let mut x2 = [F::from_f64(0.0).unwrap(); N];
    x1.iter_mut()
        .zip(x2.iter_mut())
        .enumerate()
        .map(|(i, (x1, x2))| {
            let tmp = eps_cbrt * p[i];
            *x1 = x[i] + tmp;
            *x2 = x[i] - tmp;
        })
        .count();
    let fx1 = (fs)(&x1)?;
    let fx2 = (fs)(&x2)?;
    let mut out = [F::from_f64(0.0).unwrap(); M];
    out.iter_mut()
        .enumerate()
        .map(|(i, o)| {
            *o = (fx1[i] - fx2[i]) / (F::from_f64(2.0).unwrap() * eps_cbrt);
        })
        .count();
    Ok(out)
}

pub fn forward_jacobian_pert_const<const N: usize, const M: usize, F>(
    x: &[F; N],
    fs: OpFn<'_, N, M, F>,
    pert: &PerturbationVectors,
) -> Result<[[F; N]; M], Error>
where
    F: Float + FromPrimitive + AddAssign,
{
    let fx = (fs)(x)?;
    let eps_sqrt = F::epsilon().sqrt();
    let mut xt = *x;
    let mut out = [[F::from_f64(0.0).unwrap(); N]; M];
    for pert_item in pert.iter() {
        for j in pert_item.x_idx.iter() {
            xt[*j] += eps_sqrt;
        }

        let fx1 = (fs)(&xt)?;

        for j in pert_item.x_idx.iter() {
            xt[*j] = x[*j];
        }

        for (k, x_idx) in pert_item.x_idx.iter().enumerate() {
            for j in pert_item.r_idx[k].iter() {
                out[*j][*x_idx] = (fx1[*j] - fx[*j]) / eps_sqrt;
            }
        }
    }
    Ok(out)
}

pub fn central_jacobian_pert_const<const N: usize, const M: usize, F>(
    x: &[F; N],
    fs: OpFn<'_, N, M, F>,
    pert: &PerturbationVectors,
) -> Result<[[F; N]; M], Error>
where
    F: Float + FromPrimitive + AddAssign,
{
    let eps_cbrt = F::epsilon().cbrt();
    let mut xt = *x;
    let mut out = [[F::from_f64(0.0).unwrap(); N]; M];
    for pert_item in pert.iter() {
        for j in pert_item.x_idx.iter() {
            xt[*j] += eps_cbrt;
        }

        let fx1 = (fs)(&xt)?;

        for j in pert_item.x_idx.iter() {
            xt[*j] = x[*j] - eps_cbrt;
        }

        let fx2 = (fs)(&xt)?;

        for j in pert_item.x_idx.iter() {
            xt[*j] = x[*j];
        }

        for (k, x_idx) in pert_item.x_idx.iter().enumerate() {
            for j in pert_item.r_idx[k].iter() {
                out[*j][*x_idx] = (fx1[*j] - fx2[*j]) / (F::from_f64(2.0).unwrap() * eps_cbrt);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use crate::PerturbationVector;

    use super::*;

    const COMP_ACC: f64 = 1e-6;

    fn f(x: &[f64; 6]) -> Result<[f64; 6], Error> {
        Ok([
            2.0 * (x[1].powi(3) - x[0].powi(2)),
            3.0 * (x[1].powi(3) - x[0].powi(2)) + 2.0 * (x[2].powi(3) - x[1].powi(2)),
            3.0 * (x[2].powi(3) - x[1].powi(2)) + 2.0 * (x[3].powi(3) - x[2].powi(2)),
            3.0 * (x[3].powi(3) - x[2].powi(2)) + 2.0 * (x[4].powi(3) - x[3].powi(2)),
            3.0 * (x[4].powi(3) - x[3].powi(2)) + 2.0 * (x[5].powi(3) - x[4].powi(2)),
            3.0 * (x[5].powi(3) - x[4].powi(2)),
        ])
    }

    fn res1() -> [[f64; 6]; 6] {
        [
            [-4.0, 6.0, 0.0, 0.0, 0.0, 0.0],
            [-6.0, 5.0, 6.0, 0.0, 0.0, 0.0],
            [0.0, -6.0, 5.0, 6.0, 0.0, 0.0],
            [0.0, 0.0, -6.0, 5.0, 6.0, 0.0],
            [0.0, 0.0, 0.0, -6.0, 5.0, 6.0],
            [0.0, 0.0, 0.0, 0.0, -6.0, 9.0],
        ]
    }

    fn res2() -> [f64; 6] {
        [8.0, 22.0, 27.0, 32.0, 37.0, 24.0]
    }

    fn x() -> [f64; 6] {
        [1.0f64, 1.0, 1.0, 1.0, 1.0, 1.0]
    }

    fn p() -> [f64; 6] {
        [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]
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
    fn test_forward_jacobian_const_f64() {
        let jacobian = forward_jacobian_const(&x(), &f).unwrap();
        let res = res1();
        // println!("{:?}", jacobian);
        for i in 0..6 {
            for j in 0..6 {
                assert!((res[i][j] - jacobian[i][j]).abs() < COMP_ACC)
            }
        }
    }

    #[test]
    fn test_central_jacobian_const_f64() {
        let jacobian = central_jacobian_const(&x(), &f).unwrap();
        let res = res1();
        println!("{:?}", jacobian);
        for i in 0..6 {
            for j in 0..6 {
                assert!((res[i][j] - jacobian[i][j]).abs() < COMP_ACC);
            }
        }
    }

    #[test]
    fn test_forward_jacobian_vec_prod_const_f64() {
        let jacobian = forward_jacobian_vec_prod_const(&x(), &f, &p()).unwrap();
        let res = res2();
        // println!("{:?}", jacobian);
        // the accuracy for this is pretty bad!!
        for i in 0..6 {
            assert!((res[i] - jacobian[i]).abs() < 11.0 * COMP_ACC)
        }
    }

    #[test]
    fn test_central_jacobian_vec_prod_const_f64() {
        let jacobian = central_jacobian_vec_prod_const(&x(), &f, &p()).unwrap();
        let res = res2();
        // println!("{:?}", jacobian);
        for i in 0..6 {
            assert!((res[i] - jacobian[i]).abs() < COMP_ACC)
        }
    }

    #[test]
    fn test_forward_jacobian_pert_const_f64() {
        let jacobian = forward_jacobian_pert_const(&x(), &f, &pert()).unwrap();
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
    fn test_central_jacobian_pert_const_f64() {
        let jacobian = central_jacobian_pert_const(&x(), &f, &pert()).unwrap();
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
