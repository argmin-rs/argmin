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
use crate::utils::mod_and_calc;

use super::OpFn;

pub fn forward_jacobian_vec<F>(x: &Vec<F>, fs: OpFn<'_, F>) -> Result<Vec<Vec<F>>, Error>
where
    F: Float + FromPrimitive,
{
    let fx = (fs)(x)?;
    let mut xt = x.clone();
    let eps_sqrt = F::epsilon().sqrt();
    let mut out: Vec<Vec<F>> = vec![vec![F::from_f64(0.0).unwrap(); x.len()]; fx.len()];
    for j in 0..x.len() {
        let fx1 = mod_and_calc(&mut xt, fs, j, eps_sqrt)?;
        for i in 0..fx.len() {
            out[i][j] = (fx1[i] - fx[i]) / eps_sqrt;
        }
    }
    Ok(out)
}

pub fn central_jacobian_vec<F>(x: &[F], fs: OpFn<'_, F>) -> Result<Vec<Vec<F>>, Error>
where
    F: Float + FromPrimitive,
{
    let mut xt = x.to_owned();
    let eps_cbrt = F::epsilon().cbrt();

    let comp = |(a, b): (&F, &F)| (*a - *b) / (F::from_f64(2.0).unwrap() * eps_cbrt);

    // We need to compute first iteration here, in order to know which dimension the output
    // of `fs` has.
    let fx1 = mod_and_calc(&mut xt, fs, 0, eps_cbrt)?;
    let fx2 = mod_and_calc(&mut xt, fs, 0, -eps_cbrt)?;
    let t0 = fx1.iter().zip(fx2.iter()).map(comp).collect::<Vec<F>>();

    // Now we can create the actual Jacobian
    let mut out: Vec<Vec<F>> = vec![vec![F::from_f64(0.0).unwrap(); x.len()]; fx1.len()];

    // Fill in first column
    for i in 0..t0.len() {
        out[i][0] = t0[i];
    }

    // Fill in all the other columns
    for j in 1..x.len() {
        let fx1 = mod_and_calc(&mut xt, fs, j, eps_cbrt)?;
        let fx2 = mod_and_calc(&mut xt, fs, j, -eps_cbrt)?;
        for i in 0..fx1.len() {
            out[i][j] = comp((&fx1[i], &fx2[i]));
        }
    }
    Ok(out)
}

pub fn forward_jacobian_vec_prod_vec<F>(
    x: &Vec<F>,
    fs: OpFn<'_, F>,
    p: &[F],
) -> Result<Vec<F>, Error>
where
    F: Float,
{
    let fx = (fs)(x)?;
    let eps_sqrt = F::epsilon().sqrt();
    let x1 = x
        .iter()
        .zip(p.iter())
        .map(|(&xi, &pi)| xi + eps_sqrt * pi)
        .collect();
    let fx1 = (fs)(&x1)?;
    fx1.iter()
        .zip(fx.iter())
        .map(|(&a, &b)| Ok((a - b) / eps_sqrt))
        .collect::<Result<Vec<F>, Error>>()
}

pub fn central_jacobian_vec_prod_vec<F>(x: &[F], fs: OpFn<'_, F>, p: &[F]) -> Result<Vec<F>, Error>
where
    F: Float + FromPrimitive,
{
    let eps_cbrt = F::epsilon().cbrt();
    // TODO: Do this in a single vec
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
    let fx1 = (fs)(&x1)?;
    let fx2 = (fs)(&x2)?;
    fx1.iter()
        .zip(fx2.iter())
        .map(|(&a, &b)| Ok((a - b) / (F::from_f64(2.0).unwrap() * eps_cbrt)))
        .collect::<Result<Vec<F>, Error>>()
}

pub fn forward_jacobian_pert_vec<F>(
    x: &Vec<F>,
    fs: OpFn<'_, F>,
    pert: &PerturbationVectors,
) -> Result<Vec<Vec<F>>, Error>
where
    F: Float + FromPrimitive + AddAssign,
{
    let fx = (fs)(x)?;
    let eps_sqrt = F::epsilon().sqrt();
    let mut xt = x.clone();
    let mut out = vec![vec![F::from_f64(0.0).unwrap(); x.len()]; fx.len()];
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
                out[*i][*x_idx] = (fx1[*i] - fx[*i]) / eps_sqrt;
            }
        }
    }
    Ok(out)
}

pub fn central_jacobian_pert_vec<F>(
    x: &[F],
    fs: OpFn<'_, F>,
    pert: &PerturbationVectors,
) -> Result<Vec<Vec<F>>, Error>
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

        let fx1 = (fs)(&xt)?;

        for j in pert_item.x_idx.iter() {
            xt[*j] = x[*j] - eps_cbrt;
        }

        let fx2 = (fs)(&xt)?;

        for j in pert_item.x_idx.iter() {
            xt[*j] = x[*j];
        }

        // TODO: Move this out of loop (probably compute iteration 0 prior to rest of loop)
        if i == 0 {
            out = vec![vec![F::from_f64(0.0).unwrap(); x.len()]; fx1.len()];
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

    fn f(x: &Vec<f64>) -> Result<Vec<f64>, Error> {
        Ok(vec![
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
        let jacobian = forward_jacobian_vec(&x(), &f).unwrap();
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
        let jacobian = central_jacobian_vec(&x(), &f).unwrap();
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
        let jacobian = forward_jacobian_vec_prod_vec(&x(), &f, &p()).unwrap();
        let res = res2();
        // println!("{:?}", jacobian);
        // the accuracy for this is pretty bad!!
        for i in 0..6 {
            assert!((res[i] - jacobian[i]).abs() < 11.0 * COMP_ACC)
        }
    }

    #[test]
    fn test_central_jacobian_vec_prod_vec_f64() {
        let jacobian = central_jacobian_vec_prod_vec(&x(), &f, &p()).unwrap();
        let res = res2();
        // println!("{:?}", jacobian);
        for i in 0..6 {
            assert!((res[i] - jacobian[i]).abs() < COMP_ACC)
        }
    }

    #[test]
    fn test_forward_jacobian_pert_vec_f64() {
        let jacobian = forward_jacobian_pert_vec(&x(), &f, &pert()).unwrap();
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
        let jacobian = central_jacobian_pert_vec(&x(), &f, &pert()).unwrap();
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
