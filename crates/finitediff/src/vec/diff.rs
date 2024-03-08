// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use anyhow::Error;
use num::Float;
use num::FromPrimitive;

use crate::utils::mod_and_calc;

use super::CostFn;

pub fn forward_diff_vec<F>(x: &Vec<F>, f: CostFn<'_, F>) -> Result<Vec<F>, Error>
where
    F: Float,
{
    let fx = (f)(x)?;
    let mut xt = x.clone();
    let eps_sqrt = F::epsilon().sqrt();
    (0..x.len())
        .map(|i| -> Result<F, Error> {
            let fx1 = mod_and_calc(&mut xt, f, i, eps_sqrt)?;
            Ok((fx1 - fx) / eps_sqrt)
        })
        .collect()
}

pub fn central_diff_vec<F>(x: &[F], f: CostFn<'_, F>) -> Result<Vec<F>, Error>
where
    F: Float + FromPrimitive,
{
    let mut xt = x.to_owned();
    let eps_cbrt = F::epsilon().cbrt();
    (0..x.len())
        .map(|i| -> Result<_, Error> {
            let fx1 = mod_and_calc(&mut xt, f, i, eps_cbrt)?;
            let fx2 = mod_and_calc(&mut xt, f, i, -eps_cbrt)?;
            Ok((fx1 - fx2) / (F::from_f64(2.0).unwrap() * eps_cbrt))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMP_ACC: f64 = 1e-6;

    fn f(x: &Vec<f64>) -> Result<f64, Error> {
        Ok(x[0] + x[1].powi(2))
    }

    #[test]
    fn test_forward_diff_vec_f64() {
        let p = vec![1.0f64, 1.0f64];
        let grad = forward_diff_vec(&p, &f).unwrap();
        let res = [1.0f64, 2.0];

        (0..2)
            .map(|i| assert!((res[i] - grad[i]).abs() < COMP_ACC))
            .count();

        let p = vec![1.0f64, 2.0f64];
        let grad = forward_diff_vec(&p, &f).unwrap();
        let res = [1.0f64, 4.0];

        (0..2)
            .map(|i| assert!((res[i] - grad[i]).abs() < COMP_ACC))
            .count();
    }

    #[test]
    fn test_central_diff_vec_f64() {
        let p = vec![1.0f64, 1.0f64];
        let grad = central_diff_vec(&p, &f).unwrap();
        let res = [1.0f64, 2.0];

        (0..2)
            .map(|i| assert!((res[i] - grad[i]).abs() < COMP_ACC))
            .count();

        let p = vec![1.0f64, 2.0f64];
        let grad = central_diff_vec(&p, &f).unwrap();
        let res = [1.0f64, 4.0];

        (0..2)
            .map(|i| assert!((res[i] - grad[i]).abs() < COMP_ACC))
            .count();
    }
}
