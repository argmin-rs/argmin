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
use diff::{central_diff_const, forward_diff_const};
use hessian::{
    central_hessian_const, central_hessian_vec_prod_const, forward_hessian_const,
    forward_hessian_nograd_const, forward_hessian_nograd_sparse_const,
    forward_hessian_vec_prod_const,
};
use jacobian::{
    central_jacobian_const, central_jacobian_pert_const, central_jacobian_vec_prod_const,
    forward_jacobian_const, forward_jacobian_pert_const, forward_jacobian_vec_prod_const,
};

pub(crate) type CostFn<'a, const N: usize, F> = &'a dyn Fn(&[F; N]) -> Result<F, Error>;
pub(crate) type GradientFn<'a, const N: usize, F> = &'a dyn Fn(&[F; N]) -> Result<[F; N], Error>;
pub(crate) type OpFn<'a, const N: usize, const M: usize, F> =
    &'a dyn Fn(&[F; N]) -> Result<[F; M], Error>;

#[inline(always)]
pub fn forward_diff<const N: usize, F>(
    f: CostFn<'_, N, F>,
) -> impl Fn(&[F; N]) -> Result<[F; N], Error> + '_
where
    F: Float + FromPrimitive,
{
    move |p: &[F; N]| forward_diff_const(p, &f)
}

#[inline(always)]
pub fn central_diff<const N: usize, F>(
    f: CostFn<'_, N, F>,
) -> impl Fn(&[F; N]) -> Result<[F; N], Error> + '_
where
    F: Float + FromPrimitive,
{
    move |p: &[F; N]| central_diff_const(p, &f)
}

#[inline(always)]
pub fn forward_jacobian<const N: usize, const M: usize, F>(
    f: OpFn<'_, N, M, F>,
) -> impl Fn(&[F; N]) -> Result<[[F; N]; M], Error> + '_
where
    F: Float + FromPrimitive,
{
    move |p: &[F; N]| forward_jacobian_const(p, &f)
}

#[inline(always)]
pub fn central_jacobian<const N: usize, const M: usize, F>(
    f: OpFn<'_, N, M, F>,
) -> impl Fn(&[F; N]) -> Result<[[F; N]; M], Error> + '_
where
    F: Float + FromPrimitive,
{
    move |p: &[F; N]| central_jacobian_const(p, &f)
}

#[inline(always)]
pub fn forward_jacobian_vec_prod<const N: usize, const M: usize, F>(
    f: OpFn<'_, N, M, F>,
) -> impl Fn(&[F; N], &[F; N]) -> Result<[F; M], Error> + '_
where
    F: Float + FromPrimitive,
{
    move |p: &[F; N], v: &[F; N]| forward_jacobian_vec_prod_const(p, f, v)
}

#[inline(always)]
pub fn central_jacobian_vec_prod<const N: usize, const M: usize, F>(
    f: OpFn<'_, N, M, F>,
) -> impl Fn(&[F; N], &[F; N]) -> Result<[F; M], Error> + '_
where
    F: Float + FromPrimitive,
{
    move |p: &[F; N], v: &[F; N]| central_jacobian_vec_prod_const(p, f, v)
}

#[inline(always)]
pub fn forward_jacobian_pert<const N: usize, const M: usize, F>(
    f: OpFn<'_, N, M, F>,
) -> impl Fn(&[F; N], &PerturbationVectors) -> Result<[[F; N]; M], Error> + '_
where
    F: Float + FromPrimitive + AddAssign,
{
    move |p: &[F; N], pert: &PerturbationVectors| forward_jacobian_pert_const(p, f, pert)
}

#[inline(always)]
pub fn central_jacobian_pert<const N: usize, const M: usize, F>(
    f: OpFn<'_, N, M, F>,
) -> impl Fn(&[F; N], &PerturbationVectors) -> Result<[[F; N]; M], Error> + '_
where
    F: Float + FromPrimitive + AddAssign,
{
    move |p: &[F; N], pert: &PerturbationVectors| central_jacobian_pert_const(p, f, pert)
}

#[inline(always)]
pub fn forward_hessian<const N: usize, F>(
    f: GradientFn<'_, N, F>,
) -> impl Fn(&[F; N]) -> Result<[[F; N]; N], Error> + '_
where
    F: Float + FromPrimitive,
{
    move |p: &[F; N]| forward_hessian_const(p, f)
}

#[inline(always)]
pub fn central_hessian<const N: usize, F>(
    f: GradientFn<'_, N, F>,
) -> impl Fn(&[F; N]) -> Result<[[F; N]; N], Error> + '_
where
    F: Float + FromPrimitive,
{
    move |p: &[F; N]| central_hessian_const(p, f)
}

#[inline(always)]
pub fn forward_hessian_vec_prod<const N: usize, F>(
    f: GradientFn<'_, N, F>,
) -> impl Fn(&[F; N], &[F; N]) -> Result<[F; N], Error> + '_
where
    F: Float + FromPrimitive,
{
    move |p: &[F; N], v: &[F; N]| forward_hessian_vec_prod_const(p, f, v)
}

#[inline(always)]
pub fn central_hessian_vec_prod<const N: usize, F>(
    f: GradientFn<'_, N, F>,
) -> impl Fn(&[F; N], &[F; N]) -> Result<[F; N], Error> + '_
where
    F: Float + FromPrimitive,
{
    move |p: &[F; N], v: &[F; N]| central_hessian_vec_prod_const(p, f, v)
}

#[inline(always)]
pub fn forward_hessian_nograd<const N: usize, F>(
    f: CostFn<'_, N, F>,
) -> impl Fn(&[F; N]) -> Result<[[F; N]; N], Error> + '_
where
    F: Float + FromPrimitive + AddAssign,
{
    move |p: &[F; N]| forward_hessian_nograd_const(p, f)
}

#[inline(always)]
pub fn forward_hessian_nograd_sparse<const N: usize, F>(
    f: CostFn<'_, N, F>,
) -> impl Fn(&[F; N], Vec<[usize; 2]>) -> Result<[[F; N]; N], Error> + '_
where
    F: Float + FromPrimitive + AddAssign,
{
    move |p: &[F; N], indices: Vec<[usize; 2]>| forward_hessian_nograd_sparse_const(p, f, indices)
}

#[cfg(test)]
mod tests {
    use crate::{PerturbationVector, PerturbationVectors};

    use super::*;

    const COMP_ACC: f64 = 1e-6;

    fn f1(x: &[f64; 2]) -> Result<f64, Error> {
        Ok(x[0] + x[1].powi(2))
    }

    fn f2(x: &[f64; 6]) -> Result<[f64; 6], Error> {
        Ok([
            2.0 * (x[1].powi(3) - x[0].powi(2)),
            3.0 * (x[1].powi(3) - x[0].powi(2)) + 2.0 * (x[2].powi(3) - x[1].powi(2)),
            3.0 * (x[2].powi(3) - x[1].powi(2)) + 2.0 * (x[3].powi(3) - x[2].powi(2)),
            3.0 * (x[3].powi(3) - x[2].powi(2)) + 2.0 * (x[4].powi(3) - x[3].powi(2)),
            3.0 * (x[4].powi(3) - x[3].powi(2)) + 2.0 * (x[5].powi(3) - x[4].powi(2)),
            3.0 * (x[5].powi(3) - x[4].powi(2)),
        ])
    }

    fn f3(x: &[f64; 4]) -> Result<f64, Error> {
        Ok(x[0] + x[1].powi(2) + x[2] * x[3].powi(2))
    }

    fn g(x: &[f64; 4]) -> Result<[f64; 4], Error> {
        Ok([1.0, 2.0 * x[1], x[3].powi(2), 2.0 * x[3] * x[2]])
    }

    fn x1() -> [f64; 2] {
        [1.0f64, 1.0f64]
    }

    fn x2() -> [f64; 6] {
        [1.0f64, 1.0, 1.0, 1.0, 1.0, 1.0]
    }

    fn x3() -> [f64; 4] {
        [1.0f64, 1.0, 1.0, 1.0]
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

    fn res2() -> [[f64; 4]; 4] {
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 2.0, 2.0],
        ]
    }

    fn res3() -> [f64; 6] {
        [8.0, 22.0, 27.0, 32.0, 37.0, 24.0]
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

    fn p1() -> [f64; 6] {
        [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]
    }

    fn p2() -> [f64; 4] {
        [2.0, 3.0, 4.0, 5.0]
    }

    #[test]
    fn test_forward_diff_func() {
        let grad = forward_diff(&f1);
        let out = grad(&x1()).unwrap();
        let res = [1.0, 2.0];

        for i in 0..2 {
            assert!((res[i] - out[i]).abs() < COMP_ACC)
        }

        let p = [1.0, 2.0];
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

        let p = [1.0f64, 2.0f64];
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
