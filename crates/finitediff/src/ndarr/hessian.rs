// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use std::ops::AddAssign;

use anyhow::Error;
use ndarray::Array2;
use ndarray::ScalarOperand;
use num::{Float, FromPrimitive};

use crate::utils::{mod_and_calc, restore_symmetry_ndarray, KV};

use super::CostFn;
use super::GradientFn;

pub fn forward_hessian_ndarray<F>(
    x: &ndarray::Array1<F>,
    grad: GradientFn<'_, F>,
) -> Result<ndarray::Array2<F>, Error>
where
    F: Float + FromPrimitive,
{
    let eps_sqrt = F::epsilon().sqrt();

    let mut xt = x.clone();
    let fx = (grad)(x)?;
    let rn = fx.len();
    let n = x.len();
    let mut out = Array2::zeros((n, rn));
    for i in 0..n {
        let fx1 = mod_and_calc(&mut xt, grad, i, eps_sqrt)?;
        for j in 0..rn {
            out[(i, j)] = (fx1[j] - fx[j]) / eps_sqrt;
        }
    }
    // restore symmetry
    Ok(restore_symmetry_ndarray(out))
}

pub fn central_hessian_ndarray<F>(
    x: &ndarray::Array1<F>,
    grad: GradientFn<'_, F>,
) -> Result<ndarray::Array2<F>, Error>
where
    F: Float + FromPrimitive,
{
    let eps_cbrt = F::epsilon().cbrt();

    let mut xt = x.clone();
    // TODO: get rid of this!
    let fx = (grad)(x)?;
    let rn = fx.len();
    let n = x.len();
    let mut out = ndarray::Array2::zeros((n, rn));
    for i in 0..n {
        let fx1 = mod_and_calc(&mut xt, grad, i, eps_cbrt)?;
        let fx2 = mod_and_calc(&mut xt, grad, i, -eps_cbrt)?;
        for j in 0..rn {
            out[(i, j)] = (fx1[j] - fx2[j]) / (F::from_f64(2.0).unwrap() * eps_cbrt);
        }
    }
    // restore symmetry
    Ok(restore_symmetry_ndarray(out))
}

pub fn forward_hessian_vec_prod_ndarray<F>(
    x: &ndarray::Array1<F>,
    grad: GradientFn<'_, F>,
    p: &ndarray::Array1<F>,
) -> Result<ndarray::Array1<F>, Error>
where
    F: Float + ScalarOperand,
{
    let eps_sqrt = F::epsilon().sqrt();

    let fx = (grad)(x)?;
    let x1 = x + &(p.mapv(|pi| pi * eps_sqrt));
    let fx1 = (grad)(&x1)?;
    Ok((fx1 - fx) / eps_sqrt)
}

pub fn central_hessian_vec_prod_ndarray<F>(
    x: &ndarray::Array1<F>,
    grad: GradientFn<'_, F>,
    p: &ndarray::Array1<F>,
) -> Result<ndarray::Array1<F>, Error>
where
    F: Float + FromPrimitive + ScalarOperand,
{
    let eps_cbrt = F::epsilon().cbrt();

    let x1 = x + &(p.mapv(|pi| pi * eps_cbrt));
    let x2 = x - &(p.mapv(|pi| pi * eps_cbrt));
    let fx1 = (grad)(&x1)?;
    let fx2 = (grad)(&x2)?;
    Ok((fx1 - fx2) / (F::from_f64(2.0).unwrap() * eps_cbrt))
}

pub fn forward_hessian_nograd_ndarray<F>(
    x: &ndarray::Array1<F>,
    f: CostFn<'_, F>,
) -> Result<ndarray::Array2<F>, Error>
where
    F: Float + FromPrimitive + AddAssign,
{
    // TODO: Check why this is necessary
    let eps_nograd = F::from_f64(2.0).unwrap() * F::epsilon();
    let eps_sqrt_nograd = eps_nograd.sqrt();

    let fx = (f)(x)?;
    let n = x.len();
    let mut xt = x.clone();

    // Precompute f(x + sqrt(EPS) * e_i) for all i
    let fxei: Vec<F> = (0..n)
        .map(|i| mod_and_calc(&mut xt, f, i, eps_sqrt_nograd))
        .collect::<Result<_, Error>>()?;

    let mut out = ndarray::Array2::zeros((n, n));
    for i in 0..n {
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
            out[(i, j)] = t;
            out[(j, i)] = t;
        }
    }
    Ok(out)
}

pub fn forward_hessian_nograd_sparse_ndarray<F>(
    x: &ndarray::Array1<F>,
    f: CostFn<'_, F>,
    indices: Vec<[usize; 2]>,
) -> Result<ndarray::Array2<F>, Error>
where
    F: Float + FromPrimitive + AddAssign,
{
    // TODO: Check why this is necessary
    let eps_nograd = F::from_f64(2.0).unwrap() * F::epsilon();
    let eps_sqrt_nograd = eps_nograd.sqrt();

    let fx = (f)(x)?;
    let n = x.len();
    let mut xt = x.clone();

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

    let mut out = ndarray::Array2::zeros((n, n));
    for [i, j] in indices {
        let t = {
            let xti = xt[i];
            let xtj = xt[j];
            xt[i] += eps_sqrt_nograd;
            xt[j] += eps_sqrt_nograd;
            let fxij = (f)(&xt)?;
            xt[i] = xti;
            xt[j] = xtj;

            let fxi = fxei.get(i).unwrap();
            let fxj = fxei.get(j).unwrap();

            (fxij - fxi - fxj + fx) / eps_nograd
        };
        out[(i, j)] = t;
        out[(j, i)] = t;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1};

    const COMP_ACC: f64 = 1e-6;

    fn f(x: &Array1<f64>) -> Result<f64, Error> {
        Ok(x[0] + x[1].powi(2) + x[2] * x[3].powi(2))
    }

    fn g(x: &Array1<f64>) -> Result<Array1<f64>, Error> {
        Ok(array![1.0, 2.0 * x[1], x[3].powi(2), 2.0 * x[3] * x[2]])
    }

    fn x() -> Array1<f64> {
        array![1.0f64, 1.0, 1.0, 1.0]
    }

    fn p() -> Array1<f64> {
        array![2.0, 3.0, 4.0, 5.0]
    }

    fn res1() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 2.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 2.0],
            vec![0.0, 0.0, 2.0, 2.0],
        ]
    }

    fn res2() -> Vec<f64> {
        vec![0.0, 6.0, 10.0, 18.0]
    }

    #[test]
    fn test_forward_hessian_ndarray_f64() {
        let hessian = forward_hessian_ndarray(&x(), &g).unwrap();
        let res = res1();
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            for j in 0..4 {
                assert!((res[i][j] - hessian[(i, j)]).abs() < COMP_ACC)
            }
        }
    }

    #[test]
    fn test_central_hessian_ndarray_f64() {
        let hessian = central_hessian_ndarray(&x(), &g).unwrap();
        let res = res1();
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            for j in 0..4 {
                assert!((res[i][j] - hessian[(i, j)]).abs() < COMP_ACC)
            }
        }
    }

    #[test]
    fn test_forward_hessian_vec_prod_ndarray_f64() {
        let hessian = forward_hessian_vec_prod_ndarray(&x(), &g, &p()).unwrap();
        let res = res2();
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            assert!((res[i] - hessian[i]).abs() < COMP_ACC)
        }
    }

    #[test]
    fn test_central_hessian_vec_prod_ndarray_f64() {
        let hessian = central_hessian_vec_prod_ndarray(&x(), &g, &p()).unwrap();
        let res = res2();
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            assert!((res[i] - hessian[i]).abs() < COMP_ACC)
        }
    }

    #[test]
    fn test_forward_hessian_nograd_ndarray_f64() {
        let hessian = forward_hessian_nograd_ndarray(&x(), &f).unwrap();
        let res = res1();
        // println!("hessian:\n{:#?}", hessian);
        for i in 0..4 {
            for j in 0..4 {
                assert!((res[i][j] - hessian[(i, j)]).abs() < COMP_ACC)
            }
        }
    }

    #[test]
    fn test_forward_hessian_nograd_sparse_ndarray_f64() {
        let indices = vec![[1, 1], [2, 3], [3, 3]];
        let hessian = forward_hessian_nograd_sparse_ndarray(&x(), &f, indices).unwrap();
        let res = res1();
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            for j in 0..4 {
                assert!((res[i][j] - hessian[(i, j)]).abs() < COMP_ACC)
            }
        }
    }
}
