// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use std::ops::AddAssign;

use anyhow::Error;
use num::{Float, FromPrimitive};

use crate::utils::{mod_and_calc, restore_symmetry_const, KV};

use super::{CostFn, GradientFn};

pub fn forward_hessian_const<const N: usize, F>(
    x: &[F; N],
    grad: GradientFn<'_, N, F>,
) -> Result<[[F; N]; N], Error>
where
    F: Float + FromPrimitive,
{
    let eps_sqrt = F::epsilon().sqrt();
    let fx = (grad)(x)?;
    let mut xt = *x;
    let mut out = [[F::from_f64(0.0).unwrap(); N]; N];
    for (i, o_item) in out.iter_mut().enumerate().take(N) {
        let fx1 = mod_and_calc(&mut xt, grad, i, eps_sqrt)?;
        for j in 0..N {
            o_item[j] = (fx1[j] - fx[j]) / eps_sqrt;
        }
    }

    // restore symmetry
    Ok(restore_symmetry_const(out))
}

pub fn central_hessian_const<const N: usize, F>(
    x: &[F; N],
    grad: GradientFn<'_, N, F>,
) -> Result<[[F; N]; N], Error>
where
    F: Float + FromPrimitive,
{
    let eps_cbrt = F::epsilon().cbrt();
    let mut xt = x.to_owned();
    let mut out = [[F::from_f64(0.0).unwrap(); N]; N];

    for (i, o_item) in out.iter_mut().enumerate().take(N) {
        let fx1 = mod_and_calc(&mut xt, grad, i, eps_cbrt)?;
        let fx2 = mod_and_calc(&mut xt, grad, i, -eps_cbrt)?;
        for j in 0..N {
            o_item[j] = (fx1[j] - fx2[j]) / (F::from_f64(2.0).unwrap() * eps_cbrt);
        }
    }

    // restore symmetry
    Ok(restore_symmetry_const(out))
}

pub fn forward_hessian_vec_prod_const<const N: usize, F>(
    x: &[F; N],
    grad: GradientFn<'_, N, F>,
    p: &[F; N],
) -> Result<[F; N], Error>
where
    F: Float + FromPrimitive,
{
    let eps_sqrt = F::epsilon().sqrt();
    let fx = (grad)(x)?;
    let mut out = [F::from_f64(0.0).unwrap(); N];

    let mut x1 = *x;
    for i in 1..N {
        x1[i] = x[i] + p[i] * eps_sqrt;
    }
    let fx1 = (grad)(&x1)?;

    for i in 0..N {
        out[i] = (fx1[i] - fx[i]) / eps_sqrt;
    }
    Ok(out)
}

pub fn central_hessian_vec_prod_const<const N: usize, F>(
    x: &[F; N],
    grad: GradientFn<'_, N, F>,
    p: &[F; N],
) -> Result<[F; N], Error>
where
    F: Float + FromPrimitive,
{
    let eps_cbrt = F::epsilon().cbrt();
    let mut x1 = *x;
    let mut x2 = *x;
    for i in 1..N {
        x1[i] = x[i] + p[i] * eps_cbrt;
        x2[i] = x[i] - p[i] * eps_cbrt;
    }
    let fx1 = (grad)(&x1)?;
    let fx2 = (grad)(&x2)?;

    let mut out = [F::from_f64(0.0).unwrap(); N];
    for i in 0..N {
        out[i] = (fx1[i] - fx2[i]) / (F::from_f64(2.0).unwrap() * eps_cbrt);
    }
    Ok(out)
}

pub fn forward_hessian_nograd_const<const N: usize, F>(
    x: &[F; N],
    f: CostFn<'_, N, F>,
) -> Result<[[F; N]; N], Error>
where
    F: Float + FromPrimitive + AddAssign,
{
    // TODO: Check why this is necessary
    let eps_nograd = F::from_f64(2.0).unwrap() * F::epsilon();
    let eps_sqrt_nograd = eps_nograd.sqrt();

    let fx = (f)(x)?;
    let mut xt = *x;

    // Precompute f(x + sqrt(EPS) * e_i) for all i
    let mut fxei = [F::from_f64(0.0).unwrap(); N];
    for (i, item) in fxei.iter_mut().enumerate().take(N) {
        *item = mod_and_calc(&mut xt, f, i, eps_sqrt_nograd)?;
    }

    let mut out = [[F::from_f64(0.0).unwrap(); N]; N];

    for i in 0..N {
        for j in 0..=i {
            let t = {
                let xti = xt[i];
                let xtj = xt[j];
                xt[i] += eps_sqrt_nograd;
                xt[j] += eps_sqrt_nograd;
                let fxij = (f)(&xt)?;
                xt[i] = xti;
                xt[j] = xtj;
                (fxij - fxei[i] - fxei[j] + fx) / eps_nograd
            };
            out[i][j] = t;
            out[j][i] = t;
        }
    }
    Ok(out)
}

pub fn forward_hessian_nograd_sparse_const<const N: usize, F>(
    x: &[F; N],
    f: CostFn<'_, N, F>,
    indices: Vec<[usize; 2]>,
) -> Result<[[F; N]; N], Error>
where
    F: Float + FromPrimitive + AddAssign,
{
    // TODO: Check why this is necessary
    let eps_nograd = F::from_f64(2.0).unwrap() * F::epsilon();
    let eps_sqrt_nograd = eps_nograd.sqrt();

    let fx = (f)(x)?;
    let mut xt = *x;

    let mut idxs: Vec<usize> = indices
        .iter()
        .flat_map(|i| i.iter())
        .cloned()
        .collect::<Vec<usize>>();
    idxs.sort();
    idxs.dedup();

    let mut fxei = KV::new(idxs.len());

    for idx in idxs.iter() {
        fxei.set(*idx, mod_and_calc(&mut xt, f, *idx, eps_sqrt_nograd)?);
    }

    let mut out = [[F::from_f64(0.0).unwrap(); N]; N];
    for [i, j] in indices {
        let t = {
            let xti = xt[i];
            let xtj = xt[j];
            xt[i] += eps_sqrt_nograd;
            xt[j] += eps_sqrt_nograd;
            let fxij = (f)(&xt)?;
            xt[i] = xti;
            xt[j] = xtj;

            let fxi = fxei.get(i).ok_or(anyhow::anyhow!("Bug?"))?;
            let fxj = fxei.get(j).ok_or(anyhow::anyhow!("Bug?"))?;
            (fxij - fxi - fxj + fx) / eps_nograd
        };
        out[i][j] = t;
        out[j][i] = t;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMP_ACC: f64 = 1e-6;

    fn f(x: &[f64; 4]) -> Result<f64, Error> {
        Ok(x[0] + x[1].powi(2) + x[2] * x[3].powi(2))
    }

    fn g(x: &[f64; 4]) -> Result<[f64; 4], Error> {
        Ok([1.0, 2.0 * x[1], x[3].powi(2), 2.0 * x[3] * x[2]])
    }

    fn x() -> [f64; 4] {
        [1.0f64, 1.0, 1.0, 1.0]
    }

    fn p() -> [f64; 4] {
        [2.0, 3.0, 4.0, 5.0]
    }

    fn res1() -> [[f64; 4]; 4] {
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 2.0, 2.0],
        ]
    }

    fn res2() -> [f64; 4] {
        [0.0, 6.0, 10.0, 18.0]
    }

    #[test]
    fn test_forward_hessian_vec_f64() {
        let hessian = forward_hessian_const(&x(), &g).unwrap();
        let res = res1();
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            for j in 0..4 {
                assert!((res[i][j] - hessian[i][j]).abs() < COMP_ACC)
            }
        }
    }

    #[test]
    fn test_central_hessian_vec_f64() {
        let hessian = central_hessian_const(&x(), &g).unwrap();
        let res = res1();
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            for j in 0..4 {
                assert!((res[i][j] - hessian[i][j]).abs() < COMP_ACC)
            }
        }
    }

    #[test]
    fn test_forward_hessian_vec_prod_vec_f64() {
        let hessian = forward_hessian_vec_prod_const(&x(), &g, &p()).unwrap();
        let res = res2();
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            assert!((res[i] - hessian[i]).abs() < COMP_ACC)
        }
    }

    #[test]
    fn test_central_hessian_vec_prod_vec_f64() {
        let hessian = central_hessian_vec_prod_const(&x(), &g, &p()).unwrap();
        let res = res2();
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            assert!((res[i] - hessian[i]).abs() < COMP_ACC)
        }
    }

    #[test]
    fn test_forward_hessian_nograd_vec_f64() {
        let hessian = forward_hessian_nograd_const(&x(), &f).unwrap();
        let res = res1();
        // println!("hessian:\n{:#?}", hessian);
        for i in 0..4 {
            for j in 0..4 {
                assert!((res[i][j] - hessian[i][j]).abs() < COMP_ACC)
            }
        }
    }

    #[test]
    fn test_forward_hessian_nograd_sparse_vec_f64() {
        let indices = vec![[1, 1], [2, 3], [3, 3]];
        let hessian = forward_hessian_nograd_sparse_const(&x(), &f, indices).unwrap();
        let res = res1();
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            for j in 0..4 {
                assert!((res[i][j] - hessian[i][j]).abs() < COMP_ACC)
            }
        }
    }
}
