// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use anyhow::Error;
use num::Float;
use num::FromPrimitive;

use crate::utils::mod_and_calc_const;

use super::CostFn;

pub fn forward_diff_const<const N: usize, F>(
    x: &[F; N],
    f: CostFn<'_, N, F>,
) -> Result<[F; N], Error>
where
    F: Float + FromPrimitive,
{
    let fx = (f)(x)?;
    let mut xt = *x;
    let eps_sqrt = F::epsilon().sqrt();
    let mut out = [F::from_f64(0.0).unwrap(); N];
    out.iter_mut()
        .enumerate()
        .map(|(i, o)| -> Result<_, Error> {
            let fx1 = mod_and_calc_const(&mut xt, f, i, eps_sqrt)?;
            *o = (fx1 - fx) / eps_sqrt;
            Ok(())
        })
        .count();
    Ok(out)
}

pub fn central_diff_const<const N: usize, F>(
    x: &[F; N],
    f: CostFn<'_, N, F>,
) -> Result<[F; N], Error>
where
    F: Float + FromPrimitive,
{
    let mut xt = *x;
    let eps_cbrt = F::epsilon().cbrt();
    let mut out = [F::from_f64(0.0).unwrap(); N];
    out.iter_mut()
        .enumerate()
        .map(|(i, o)| -> Result<_, Error> {
            let fx1 = mod_and_calc_const(&mut xt, f, i, eps_cbrt)?;
            let fx2 = mod_and_calc_const(&mut xt, f, i, -eps_cbrt)?;
            *o = (fx1 - fx2) / (F::from_f64(2.0).unwrap() * eps_cbrt);
            Ok(())
        })
        .count();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMP_ACC: f64 = 1e-6;

    fn f(x: &[f64; 2]) -> Result<f64, Error> {
        Ok(x[0] + x[1].powi(2))
    }

    fn f2(x: &[f64; 2]) -> Result<f64, Error> {
        Ok(x[0] + x[1].powi(2))
    }

    #[test]
    fn test_forward_diff_const_f64() {
        let p = [1.0f64, 1.0f64];
        let grad = forward_diff_const(&p, &f2).unwrap();
        let res = [1.0f64, 2.0];

        (0..2)
            .map(|i| assert!((res[i] - grad[i]).abs() < COMP_ACC))
            .count();

        let p = [1.0f64, 2.0f64];
        let grad = forward_diff_const(&p, &f2).unwrap();
        let res = [1.0f64, 4.0];

        (0..2)
            .map(|i| assert!((res[i] - grad[i]).abs() < COMP_ACC))
            .count();
    }

    #[test]
    fn test_central_diff_vec_f64() {
        let p = [1.0f64, 1.0f64];
        let grad = central_diff_const(&p, &f).unwrap();
        let res = [1.0f64, 2.0];

        (0..2)
            .map(|i| assert!((res[i] - grad[i]).abs() < COMP_ACC))
            .count();

        let p = [1.0f64, 2.0f64];
        let grad = central_diff_const(&p, &f).unwrap();
        let res = [1.0f64, 4.0];

        (0..2)
            .map(|i| assert!((res[i] - grad[i]).abs() < COMP_ACC))
            .count();
    }
}
