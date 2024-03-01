// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::utils::*;
use crate::EPS_F64;

/// I wish this wasn't necessary!
const EPS_F64_NOGRAD: f64 = EPS_F64 * 2.0;

pub fn forward_hessian_vec_f64(
    x: &Vec<f64>,
    grad: &dyn Fn(&Vec<f64>) -> Vec<f64>,
) -> Vec<Vec<f64>> {
    let fx = (grad)(x);
    let mut xt = x.clone();
    let out: Vec<Vec<f64>> = (0..x.len())
        .map(|i| {
            let fx1 = mod_and_calc_vec_f64(&mut xt, grad, i, EPS_F64.sqrt());
            fx1.iter()
                .zip(fx.iter())
                .map(|(a, b)| (a - b) / (EPS_F64.sqrt()))
                .collect::<Vec<f64>>()
        })
        .collect();

    // restore symmetry
    restore_symmetry_vec_f64(out)
}

pub fn central_hessian_vec_f64(
    x: &Vec<f64>,
    grad: &dyn Fn(&Vec<f64>) -> Vec<f64>,
) -> Vec<Vec<f64>> {
    let mut xt = x.clone();
    let out: Vec<Vec<f64>> = (0..x.len())
        .map(|i| {
            let fx1 = mod_and_calc_vec_f64(&mut xt, grad, i, EPS_F64.sqrt());
            let fx2 = mod_and_calc_vec_f64(&mut xt, grad, i, -EPS_F64.sqrt());
            fx1.iter()
                .zip(fx2.iter())
                .map(|(a, b)| (a - b) / (2.0 * EPS_F64.sqrt()))
                .collect::<Vec<f64>>()
        })
        .collect();

    // restore symmetry
    restore_symmetry_vec_f64(out)
}

pub fn forward_hessian_vec_prod_vec_f64(
    x: &Vec<f64>,
    grad: &dyn Fn(&Vec<f64>) -> Vec<f64>,
    p: &Vec<f64>,
) -> Vec<f64> {
    let fx = (grad)(x);
    let out: Vec<f64> = {
        let x1 = x
            .iter()
            .zip(p.iter())
            .map(|(xi, pi)| xi + pi * EPS_F64.sqrt())
            .collect();
        let fx1 = (grad)(&x1);
        fx1.iter()
            .zip(fx.iter())
            .map(|(a, b)| (a - b) / (EPS_F64.sqrt()))
            .collect::<Vec<f64>>()
    };
    out
}

pub fn central_hessian_vec_prod_vec_f64(
    x: &Vec<f64>,
    grad: &dyn Fn(&Vec<f64>) -> Vec<f64>,
    p: &Vec<f64>,
) -> Vec<f64> {
    let out: Vec<f64> = {
        let x1 = x
            .iter()
            .zip(p.iter())
            .map(|(xi, pi)| xi + pi * EPS_F64.sqrt())
            .collect();
        let x2 = x
            .iter()
            .zip(p.iter())
            .map(|(xi, pi)| xi - pi * EPS_F64.sqrt())
            .collect();
        let fx1 = (grad)(&x1);
        let fx2 = (grad)(&x2);
        fx1.iter()
            .zip(fx2.iter())
            .map(|(a, b)| (a - b) / (2.0 * EPS_F64.sqrt()))
            .collect::<Vec<f64>>()
    };
    out
}

pub fn forward_hessian_nograd_vec_f64(x: &Vec<f64>, f: &dyn Fn(&Vec<f64>) -> f64) -> Vec<Vec<f64>> {
    let fx = (f)(x);
    let n = x.len();
    let mut xt = x.clone();

    // Precompute f(x + sqrt(EPS) * e_i) for all i
    let fxei: Vec<f64> = (0..n)
        .map(|i| mod_and_calc_vec_f64(&mut xt, f, i, EPS_F64_NOGRAD.sqrt()))
        .collect();

    let mut out: Vec<Vec<f64>> = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..=i {
            let t = {
                let xti = xt[i];
                let xtj = xt[j];
                xt[i] += EPS_F64_NOGRAD.sqrt();
                xt[j] += EPS_F64_NOGRAD.sqrt();
                let fxij = (f)(&xt);
                xt[i] = xti;
                xt[j] = xtj;
                (fxij - fxei[i] - fxei[j] + fx) / EPS_F64_NOGRAD
            };
            out[i][j] = t;
            out[j][i] = t;
        }
    }
    out
}

pub fn forward_hessian_nograd_sparse_vec_f64(
    x: &Vec<f64>,
    f: &dyn Fn(&Vec<f64>) -> f64,
    indices: Vec<[usize; 2]>,
) -> Vec<Vec<f64>> {
    let fx = (f)(x);
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
        fxei.set(
            *idx,
            mod_and_calc_vec_f64(&mut xt, f, *idx, EPS_F64_NOGRAD.sqrt()),
        );
    }

    let mut out: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    for [i, j] in indices {
        let t = {
            let xti = xt[i];
            let xtj = xt[j];
            xt[i] += EPS_F64_NOGRAD.sqrt();
            xt[j] += EPS_F64_NOGRAD.sqrt();
            let fxij = (f)(&xt);
            xt[i] = xti;
            xt[j] = xtj;

            let fxi = fxei.get(i).unwrap();
            let fxj = fxei.get(j).unwrap();
            (fxij - fxi - fxj + fx) / EPS_F64_NOGRAD
        };
        out[i][j] = t;
        out[j][i] = t;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMP_ACC: f64 = 1e-6;

    fn f(x: &Vec<f64>) -> f64 {
        x[0] + x[1].powi(2) + x[2] * x[3].powi(2)
    }

    fn g(x: &Vec<f64>) -> Vec<f64> {
        vec![1.0, 2.0 * x[1], x[3].powi(2), 2.0 * x[3] * x[2]]
    }

    fn x() -> Vec<f64> {
        vec![1.0f64, 1.0, 1.0, 1.0]
    }

    fn p() -> Vec<f64> {
        vec![2.0, 3.0, 4.0, 5.0]
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
    fn test_forward_hessian_vec_f64() {
        let hessian = forward_hessian_vec_f64(&x(), &g);
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
        let hessian = central_hessian_vec_f64(&x(), &g);
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
        let hessian = forward_hessian_vec_prod_vec_f64(&x(), &g, &p());
        let res = res2();
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            assert!((res[i] - hessian[i]).abs() < COMP_ACC)
        }
    }

    #[test]
    fn test_central_hessian_vec_prod_vec_f64() {
        let hessian = central_hessian_vec_prod_vec_f64(&x(), &g, &p());
        let res = res2();
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            assert!((res[i] - hessian[i]).abs() < COMP_ACC)
        }
    }

    #[test]
    fn test_forward_hessian_nograd_vec_f64() {
        let hessian = forward_hessian_nograd_vec_f64(&x(), &f);
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
        let hessian = forward_hessian_nograd_sparse_vec_f64(&x(), &f, indices);
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
