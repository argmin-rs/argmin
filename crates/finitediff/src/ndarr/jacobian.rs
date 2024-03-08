// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use std::ops::AddAssign;

use anyhow::Error;
use ndarray::{Array1, Array2, ScalarOperand};
use num::{Float, FromPrimitive};

use crate::{pert::PerturbationVectors, utils::mod_and_calc};

use super::OpFn;

pub fn forward_jacobian_ndarray<F>(
    x: &ndarray::Array1<F>,
    fs: OpFn<'_, F>,
) -> Result<ndarray::Array2<F>, Error>
where
    F: Float,
{
    let eps_sqrt = F::epsilon().sqrt();

    let fx = (fs)(x)?;
    let mut xt = x.clone();
    let rn = fx.len();
    let n = x.len();
    let mut out = Array2::zeros((rn, n));
    for j in 0..n {
        let fx1 = mod_and_calc(&mut xt, fs, j, eps_sqrt)?;
        for i in 0..rn {
            out[(i, j)] = (fx1[i] - fx[i]) / eps_sqrt;
        }
    }
    Ok(out)
}

pub fn central_jacobian_ndarray<F>(
    x: &ndarray::Array1<F>,
    fs: OpFn<'_, F>,
) -> Result<ndarray::Array2<F>, Error>
where
    F: Float + FromPrimitive,
{
    let eps_cbrt = F::epsilon().cbrt();

    let mut xt = x.clone();

    let comp = |(a, b): (&F, &F)| (*a - *b) / (F::from_f64(2.0).unwrap() * eps_cbrt);
    let fx1 = mod_and_calc(&mut xt, fs, 0, eps_cbrt)?;
    let fx2 = mod_and_calc(&mut xt, fs, 0, -eps_cbrt)?;
    let tmp = Array1::from_iter(fx1.iter().zip(fx2.iter()).map(comp));

    let rn = tmp.len();
    let n = x.len();

    let mut out = Array2::zeros((rn, n));

    for i in 0..rn {
        out[(i, 0)] = tmp[i];
    }

    for j in 1..n {
        let fx1 = mod_and_calc(&mut xt, fs, j, eps_cbrt)?;
        let fx2 = mod_and_calc(&mut xt, fs, j, -eps_cbrt)?;
        for i in 0..rn {
            out[(i, j)] = comp((&fx1[i], &fx2[i]));
        }
    }
    Ok(out)
}

pub fn forward_jacobian_vec_prod_ndarray<F>(
    x: &ndarray::Array1<F>,
    fs: OpFn<'_, F>,
    p: &ndarray::Array1<F>,
) -> Result<ndarray::Array1<F>, Error>
where
    F: Float + ScalarOperand,
{
    let eps_sqrt = F::epsilon().sqrt();
    let fx = (fs)(x)?;
    let x1 = x + &p.mapv(|pi| eps_sqrt * pi);
    let fx1 = (fs)(&x1)?;
    Ok((fx1 - fx) / eps_sqrt)
}

pub fn central_jacobian_vec_prod_ndarray<F>(
    x: &ndarray::Array1<F>,
    fs: OpFn<'_, F>,
    p: &ndarray::Array1<F>,
) -> Result<ndarray::Array1<F>, Error>
where
    F: Float + FromPrimitive + ScalarOperand,
{
    let eps_cbrt = F::epsilon().sqrt();
    // TODO: Do this in a single array!
    let x1 = x + &p.mapv(|pi| eps_cbrt * pi);
    let x2 = x + &p.mapv(|pi| -eps_cbrt * pi);
    let fx1 = (fs)(&x1)?;
    let fx2 = (fs)(&x2)?;
    Ok((fx1 - fx2) / (F::from_f64(2.0).unwrap() * eps_cbrt))
}

pub fn forward_jacobian_pert_ndarray<F>(
    x: &ndarray::Array1<F>,
    fs: OpFn<'_, F>,
    pert: &PerturbationVectors,
) -> Result<ndarray::Array2<F>, Error>
where
    F: Float + AddAssign,
{
    let eps_sqrt = F::epsilon().sqrt();

    let fx = (fs)(x)?;
    let mut xt = x.clone();
    let mut out = ndarray::Array2::zeros((fx.len(), x.len()));
    for pert_item in pert.iter() {
        for i in pert_item.x_idx.iter() {
            xt[*i] += eps_sqrt;
        }

        let fx1 = (fs)(&xt)?;

        for i in pert_item.x_idx.iter() {
            xt[*i] = x[*i];
        }

        for (k, x_idx) in pert_item.x_idx.iter().enumerate() {
            for i in pert_item.r_idx[k].iter() {
                out[(*i, *x_idx)] = (fx1[*i] - fx[*i]) / eps_sqrt;
            }
        }
    }
    Ok(out)
}

pub fn central_jacobian_pert_ndarray<F>(
    x: &ndarray::Array1<F>,
    fs: OpFn<'_, F>,
    pert: &PerturbationVectors,
) -> Result<ndarray::Array2<F>, Error>
where
    F: Float + FromPrimitive + AddAssign,
{
    let eps_cbrt = F::epsilon().cbrt();

    let mut out = ndarray::Array2::zeros((1, 1));
    let mut xt = x.clone();
    for (j, pert_item) in pert.iter().enumerate() {
        for i in pert_item.x_idx.iter() {
            xt[*i] += eps_cbrt;
        }

        let fx1 = (fs)(&xt)?;

        for i in pert_item.x_idx.iter() {
            xt[*i] = x[*i] - eps_cbrt;
        }

        let fx2 = (fs)(&xt)?;

        for i in pert_item.x_idx.iter() {
            xt[*i] = x[*i];
        }

        // TODO: Move this out of loop (probably compute iteration 0 prior to rest of loop)
        if j == 0 {
            out = ndarray::Array2::zeros((fx1.len(), x.len()));
        }

        for (k, x_idx) in pert_item.x_idx.iter().enumerate() {
            for i in pert_item.r_idx[k].iter() {
                out[(*i, *x_idx)] = (fx1[*i] - fx2[*i]) / (F::from_f64(2.0).unwrap() * eps_cbrt);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use crate::PerturbationVector;

    use super::*;
    use ndarray::{array, Array1};

    const COMP_ACC: f64 = 1e-6;

    fn f(x: &Array1<f64>) -> Result<Array1<f64>, Error> {
        Ok(array![
            2.0 * (x[1].powi(3) - x[0].powi(2)),
            3.0 * (x[1].powi(3) - x[0].powi(2)) + 2.0 * (x[2].powi(3) - x[1].powi(2)),
            3.0 * (x[2].powi(3) - x[1].powi(2)) + 2.0 * (x[3].powi(3) - x[2].powi(2)),
            3.0 * (x[3].powi(3) - x[2].powi(2)) + 2.0 * (x[4].powi(3) - x[3].powi(2)),
            3.0 * (x[4].powi(3) - x[3].powi(2)) + 2.0 * (x[5].powi(3) - x[4].powi(2)),
            3.0 * (x[5].powi(3) - x[4].powi(2)),
        ])
    }

    fn res1() -> Vec<Vec<f64>> {
        vec![
            vec![-4.0, 6.0, 0.0, 0.0, 0.0, 0.0],
            vec![-6.0, 5.0, 6.0, 0.0, 0.0, 0.0],
            vec![0.0, -6.0, 5.0, 6.0, 0.0, 0.0],
            vec![0.0, 0.0, -6.0, 5.0, 6.0, 0.0],
            vec![0.0, 0.0, 0.0, -6.0, 5.0, 6.0],
            vec![0.0, 0.0, 0.0, 0.0, -6.0, 9.0],
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
        let jacobian = forward_jacobian_ndarray(&x(), &f).unwrap();
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
        let jacobian = central_jacobian_ndarray(&x(), &f).unwrap();
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
        let jacobian = forward_jacobian_vec_prod_ndarray(&x(), &f, &p()).unwrap();
        let res = res2();
        // println!("{:?}", jacobian);
        // the accuracy for this is pretty bad!!
        for i in 0..6 {
            assert!((res[i] - jacobian[i]).abs() < 11.0 * COMP_ACC)
        }
    }

    #[test]
    fn test_central_jacobian_vec_prod_ndarray_f64() {
        let jacobian = central_jacobian_vec_prod_ndarray(&x(), &f, &p()).unwrap();
        let res = res2();
        // println!("{:?}", jacobian);
        for i in 0..6 {
            assert!((res[i] - jacobian[i]).abs() < COMP_ACC)
        }
    }

    #[test]
    fn test_forward_jacobian_pert_ndarray_f64() {
        let jacobian = forward_jacobian_pert_ndarray(&x(), &f, &pert()).unwrap();
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
        let jacobian = central_jacobian_pert_ndarray(&x(), &f, &pert()).unwrap();
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
