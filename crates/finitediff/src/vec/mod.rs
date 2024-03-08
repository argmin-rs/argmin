// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

pub mod diff;
pub mod hessian;
pub mod jacobian;

use std::ops::AddAssign;

use anyhow::Error;
use num::{Float, FromPrimitive};

use crate::PerturbationVectors;
use diff::{central_diff_vec, forward_diff_vec};
use hessian::{
    central_hessian_vec, central_hessian_vec_prod_vec, forward_hessian_nograd_sparse_vec,
    forward_hessian_nograd_vec, forward_hessian_vec, forward_hessian_vec_prod_vec,
};
use jacobian::{
    central_jacobian_pert_vec, central_jacobian_vec, central_jacobian_vec_prod_vec,
    forward_jacobian_pert_vec, forward_jacobian_vec, forward_jacobian_vec_prod_vec,
};

pub(crate) type CostFn<'a, F> = &'a dyn Fn(&Vec<F>) -> Result<F, Error>;
pub(crate) type GradientFn<'a, F> = &'a dyn Fn(&Vec<F>) -> Result<Vec<F>, Error>;
pub(crate) type OpFn<'a, F> = &'a dyn Fn(&Vec<F>) -> Result<Vec<F>, Error>;

// pub trait GradientImpl<'a, F>: Fn(&Vec<F>) -> Result<Vec<F>, Error> + 'a {}
// impl<'a, F, T: Fn(&Vec<F>) -> Result<Vec<F>, Error> + 'a> GradientImpl<'a, F> for T {}
// pub fn forward_diff<F>(f: CostFn<'_, F>) -> impl GradientImpl<'_, F> { .. }

#[inline(always)]
pub fn forward_diff<F>(f: CostFn<'_, F>) -> impl Fn(&Vec<F>) -> Result<Vec<F>, Error> + '_
where
    F: Float + FromPrimitive,
{
    move |p: &Vec<F>| forward_diff_vec(p, f)
}

#[inline(always)]
pub fn central_diff<F>(f: CostFn<'_, F>) -> impl Fn(&Vec<F>) -> Result<Vec<F>, Error> + '_
where
    F: Float + FromPrimitive,
{
    move |p: &Vec<F>| central_diff_vec(p, f)
}

#[inline(always)]
pub fn forward_jacobian<F>(f: OpFn<'_, F>) -> impl Fn(&Vec<F>) -> Result<Vec<Vec<F>>, Error> + '_
where
    F: Float + FromPrimitive,
{
    move |p: &Vec<F>| forward_jacobian_vec(p, f)
}

#[inline(always)]
pub fn central_jacobian<F>(f: OpFn<'_, F>) -> impl Fn(&Vec<F>) -> Result<Vec<Vec<F>>, Error> + '_
where
    F: Float + FromPrimitive,
{
    move |p: &Vec<F>| central_jacobian_vec(p, f)
}

#[inline(always)]
pub fn forward_jacobian_vec_prod<F>(
    f: OpFn<'_, F>,
) -> impl Fn(&Vec<F>, &Vec<F>) -> Result<Vec<F>, Error> + '_
where
    F: Float + FromPrimitive,
{
    move |p: &Vec<F>, v: &Vec<F>| forward_jacobian_vec_prod_vec(p, f, v)
}

#[inline(always)]
pub fn central_jacobian_vec_prod<F>(
    f: OpFn<'_, F>,
) -> impl Fn(&Vec<F>, &Vec<F>) -> Result<Vec<F>, Error> + '_
where
    F: Float + FromPrimitive,
{
    move |p: &Vec<F>, v: &Vec<F>| central_jacobian_vec_prod_vec(p, f, v)
}

#[inline(always)]
pub fn forward_jacobian_pert<F>(
    f: OpFn<'_, F>,
) -> impl Fn(&Vec<F>, &PerturbationVectors) -> Result<Vec<Vec<F>>, Error> + '_
where
    F: Float + FromPrimitive + AddAssign,
{
    move |p: &Vec<F>, pert: &PerturbationVectors| forward_jacobian_pert_vec(p, f, pert)
}

#[inline(always)]
pub fn central_jacobian_pert<F>(
    f: OpFn<'_, F>,
) -> impl Fn(&Vec<F>, &PerturbationVectors) -> Result<Vec<Vec<F>>, Error> + '_
where
    F: Float + FromPrimitive + AddAssign,
{
    move |p: &Vec<F>, pert: &PerturbationVectors| central_jacobian_pert_vec(p, f, pert)
}

#[inline(always)]
pub fn forward_hessian<F>(
    f: GradientFn<'_, F>,
) -> impl Fn(&Vec<F>) -> Result<Vec<Vec<F>>, Error> + '_
where
    F: Float + FromPrimitive,
{
    move |p: &Vec<F>| forward_hessian_vec(p, f)
}

#[inline(always)]
pub fn central_hessian<F>(
    f: GradientFn<'_, F>,
) -> impl Fn(&Vec<F>) -> Result<Vec<Vec<F>>, Error> + '_
where
    F: Float + FromPrimitive,
{
    move |p: &Vec<F>| central_hessian_vec(p, f)
}

#[inline(always)]
pub fn forward_hessian_vec_prod<F>(
    f: GradientFn<'_, F>,
) -> impl Fn(&Vec<F>, &Vec<F>) -> Result<Vec<F>, Error> + '_
where
    F: Float + FromPrimitive,
{
    move |p: &Vec<F>, v: &Vec<F>| forward_hessian_vec_prod_vec(p, f, v)
}

#[inline(always)]
pub fn central_hessian_vec_prod<F>(
    f: GradientFn<'_, F>,
) -> impl Fn(&Vec<F>, &Vec<F>) -> Result<Vec<F>, Error> + '_
where
    F: Float + FromPrimitive,
{
    move |p: &Vec<F>, v: &Vec<F>| central_hessian_vec_prod_vec(p, f, v)
}

#[inline(always)]
pub fn forward_hessian_nograd<F>(
    f: CostFn<'_, F>,
) -> impl Fn(&Vec<F>) -> Result<Vec<Vec<F>>, Error> + '_
where
    F: Float + FromPrimitive + AddAssign,
{
    move |p: &Vec<F>| forward_hessian_nograd_vec(p, f)
}

#[inline(always)]
pub fn forward_hessian_nograd_sparse<F>(
    f: CostFn<'_, F>,
) -> impl Fn(&Vec<F>, Vec<[usize; 2]>) -> Result<Vec<Vec<F>>, Error> + '_
where
    F: Float + FromPrimitive + AddAssign,
{
    move |p: &Vec<F>, indices: Vec<[usize; 2]>| forward_hessian_nograd_sparse_vec(p, f, indices)
}

#[cfg(test)]
mod tests {
    use crate::{PerturbationVector, PerturbationVectors};

    use super::*;

    const COMP_ACC: f64 = 1e-6;

    fn f1(x: &Vec<f64>) -> Result<f64, Error> {
        Ok(x[0] + x[1].powi(2))
    }

    fn f2(x: &Vec<f64>) -> Result<Vec<f64>, Error> {
        Ok(vec![
            2.0 * (x[1].powi(3) - x[0].powi(2)),
            3.0 * (x[1].powi(3) - x[0].powi(2)) + 2.0 * (x[2].powi(3) - x[1].powi(2)),
            3.0 * (x[2].powi(3) - x[1].powi(2)) + 2.0 * (x[3].powi(3) - x[2].powi(2)),
            3.0 * (x[3].powi(3) - x[2].powi(2)) + 2.0 * (x[4].powi(3) - x[3].powi(2)),
            3.0 * (x[4].powi(3) - x[3].powi(2)) + 2.0 * (x[5].powi(3) - x[4].powi(2)),
            3.0 * (x[5].powi(3) - x[4].powi(2)),
        ])
    }

    fn f3(x: &Vec<f64>) -> Result<f64, Error> {
        Ok(x[0] + x[1].powi(2) + x[2] * x[3].powi(2))
    }

    fn g(x: &Vec<f64>) -> Result<Vec<f64>, Error> {
        Ok(vec![1.0, 2.0 * x[1], x[3].powi(2), 2.0 * x[3] * x[2]])
    }

    fn x1() -> Vec<f64> {
        vec![1.0f64, 1.0f64]
    }

    fn x2() -> Vec<f64> {
        vec![1.0f64, 1.0, 1.0, 1.0, 1.0, 1.0]
    }

    fn x3() -> Vec<f64> {
        vec![1.0f64, 1.0, 1.0, 1.0]
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

    fn res2() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 2.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 2.0],
            vec![0.0, 0.0, 2.0, 2.0],
        ]
    }

    fn res3() -> Vec<f64> {
        vec![8.0, 22.0, 27.0, 32.0, 37.0, 24.0]
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

    fn p1() -> Vec<f64> {
        vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]
    }

    fn p2() -> Vec<f64> {
        vec![2.0, 3.0, 4.0, 5.0]
    }

    #[test]
    fn test_forward_diff_func() {
        let grad = forward_diff(&f1);
        let out = grad(&x1()).unwrap();
        let res = [1.0, 2.0];

        for i in 0..2 {
            assert!((res[i] - out[i]).abs() < COMP_ACC)
        }

        let p = vec![1.0, 2.0];
        let grad = forward_diff(&f1);
        let out = grad(&p).unwrap();
        let res = [1.0, 4.0];

        for i in 0..2 {
            assert!((res[i] - out[i]).abs() < COMP_ACC)
        }
    }

    #[test]
    fn test_central_diff_func() {
        let grad = central_diff(&f1);
        let out = grad(&x1()).unwrap();
        let res = [1.0f64, 2.0];

        for i in 0..2 {
            assert!((res[i] - out[i]).abs() < COMP_ACC)
        }

        let p = vec![1.0f64, 2.0f64];
        let grad = central_diff(&f1);
        let out = grad(&p).unwrap();
        let res = [1.0f64, 4.0];

        for i in 0..2 {
            assert!((res[i] - out[i]).abs() < COMP_ACC)
        }
    }

    #[test]
    fn test_forward_jacobian_func() {
        let jacobian = forward_jacobian(&f2);
        let out = jacobian(&x2()).unwrap();
        let res = res1();
        // println!("{:?}", out);
        // println!("{:?}", res);
        for i in 0..6 {
            for j in 0..6 {
                assert!((res[i][j] - out[i][j]).abs() < COMP_ACC)
            }
        }
    }

    #[test]
    fn test_central_jacobian_vec_f64_trait() {
        let jacobian = central_jacobian(&f2);
        let out = jacobian(&x2()).unwrap();
        let res = res1();
        // println!("{:?}", jacobian);
        for i in 0..6 {
            for j in 0..6 {
                assert!((res[i][j] - out[i][j]).abs() < COMP_ACC)
            }
        }
    }

    #[test]
    fn test_forward_jacobian_vec_prod_vec_func() {
        let jacobian = forward_jacobian_vec_prod(&f2);
        let out = jacobian(&x2(), &p1()).unwrap();
        let res = res3();
        // println!("{:?}", jacobian);
        // the accuracy for this is pretty bad!!
        for i in 0..6 {
            assert!((res[i] - out[i]).abs() < 5.5 * COMP_ACC)
        }
    }

    #[test]
    fn test_central_jacobian_vec_prod_vec_func() {
        let jacobian = central_jacobian_vec_prod(&f2);
        let out = jacobian(&x2(), &p1()).unwrap();
        let res = res3();
        // println!("{:?}", jacobian);
        for i in 0..6 {
            assert!((res[i] - out[i]).abs() < COMP_ACC)
        }
    }

    #[test]
    fn test_forward_jacobian_pert_func() {
        let jacobian = forward_jacobian_pert(&f2);
        let out = jacobian(&x2(), &pert()).unwrap();
        let res = res1();
        // println!("jacobian:\n{:?}", jacobian);
        // println!("res:\n{:?}", res);
        for i in 0..6 {
            for j in 0..6 {
                assert!((res[i][j] - out[i][j]).abs() < COMP_ACC)
            }
        }
    }

    #[test]
    fn test_central_jacobian_pert_func() {
        let jacobian = central_jacobian_pert(&f2);
        let out = jacobian(&x2(), &pert()).unwrap();
        let res = res1();
        // println!("jacobian:\n{:?}", jacobian);
        // println!("res:\n{:?}", res);
        for i in 0..6 {
            for j in 0..6 {
                assert!((res[i][j] - out[i][j]).abs() < COMP_ACC)
            }
        }
    }

    #[test]
    fn test_forward_hessian_func() {
        let hessian = forward_hessian(&g);
        let out = hessian(&x3()).unwrap();
        let res = res2();
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            for j in 0..4 {
                assert!((res[i][j] - out[i][j]).abs() < COMP_ACC)
            }
        }
    }

    #[test]
    fn test_central_hessian_func() {
        let hessian = central_hessian(&g);
        let out = hessian(&x3()).unwrap();
        let res = res2();
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            for j in 0..4 {
                assert!((res[i][j] - out[i][j]).abs() < COMP_ACC)
            }
        }
    }

    #[test]
    fn test_forward_hessian_vec_prod_func() {
        let hessian = forward_hessian_vec_prod(&g);
        let out = hessian(&x3(), &p2()).unwrap();
        let res = [0.0, 6.0, 10.0, 18.0];
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            assert!((res[i] - out[i]).abs() < COMP_ACC)
        }
    }

    #[test]
    fn test_central_hessian_vec_prod_func() {
        let hessian = central_hessian_vec_prod(&g);
        let out = hessian(&x3(), &p2()).unwrap();
        let res = [0.0, 6.0, 10.0, 18.0];
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            assert!((res[i] - out[i]).abs() < COMP_ACC)
        }
    }

    #[test]
    fn test_forward_hessian_nograd_func() {
        let hessian = forward_hessian_nograd(&f3);
        let out = hessian(&x3()).unwrap();
        let res = res2();
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            for j in 0..4 {
                assert!((res[i][j] - out[i][j]).abs() < COMP_ACC)
            }
        }
    }

    #[test]
    fn test_forward_hessian_nograd_sparse_func() {
        let indices = vec![[1, 1], [2, 3], [3, 3]];
        let hessian = forward_hessian_nograd_sparse(&f3);
        let out = hessian(&x3(), indices).unwrap();
        let res = res2();
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            for j in 0..4 {
                assert!((res[i][j] - out[i][j]).abs() < COMP_ACC)
            }
        }
    }
}
