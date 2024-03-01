// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

mod diff;
mod hessian;
mod jacobian;

use std::ops::AddAssign;

pub use diff::{central_diff_vec, forward_diff_vec};
pub use hessian::{
    central_hessian_vec, central_hessian_vec_prod_vec, forward_hessian_nograd_sparse_vec,
    forward_hessian_nograd_vec, forward_hessian_vec, forward_hessian_vec_prod_vec,
};
pub use jacobian::{
    central_jacobian_pert_vec, central_jacobian_vec, central_jacobian_vec_prod_vec,
    forward_jacobian_pert_vec, forward_jacobian_vec, forward_jacobian_vec_prod_vec,
};
use num::{Float, FromPrimitive};

use crate::{FiniteDiff, PerturbationVectors};

impl<F> FiniteDiff for Vec<F>
where
    F: Float + FromPrimitive + AddAssign,
{
    type Scalar = F;
    type Jacobian = Vec<Vec<Self::Scalar>>;
    type Hessian = Vec<Vec<Self::Scalar>>;
    type OperatorOutput = Vec<Self::Scalar>;

    fn forward_diff(&self, f: &dyn Fn(&Self) -> Self::Scalar) -> Self {
        forward_diff_vec(self, f)
    }

    fn central_diff(&self, f: &dyn Fn(&Self) -> Self::Scalar) -> Self {
        central_diff_vec(self, f)
    }

    fn forward_jacobian(&self, fs: &dyn Fn(&Self) -> Self::OperatorOutput) -> Self::Jacobian {
        forward_jacobian_vec(self, fs)
    }

    fn central_jacobian(&self, fs: &dyn Fn(&Self) -> Self::OperatorOutput) -> Self::Jacobian {
        central_jacobian_vec(self, fs)
    }

    fn forward_jacobian_vec_prod(
        &self,
        fs: &dyn Fn(&Self) -> Self::OperatorOutput,
        p: &Self,
    ) -> Self {
        forward_jacobian_vec_prod_vec(self, fs, p)
    }

    fn central_jacobian_vec_prod(
        &self,
        fs: &dyn Fn(&Self) -> Self::OperatorOutput,
        p: &Self,
    ) -> Self {
        central_jacobian_vec_prod_vec(self, fs, p)
    }

    fn forward_jacobian_pert(
        &self,
        fs: &dyn Fn(&Self) -> Self::OperatorOutput,
        pert: &PerturbationVectors,
    ) -> Self::Jacobian {
        forward_jacobian_pert_vec(self, fs, pert)
    }

    fn central_jacobian_pert(
        &self,
        fs: &dyn Fn(&Self) -> Self::OperatorOutput,
        pert: &PerturbationVectors,
    ) -> Self::Jacobian {
        central_jacobian_pert_vec(self, fs, pert)
    }

    fn forward_hessian(&self, g: &dyn Fn(&Self) -> Self::OperatorOutput) -> Self::Hessian {
        forward_hessian_vec(self, g)
    }

    fn central_hessian(&self, g: &dyn Fn(&Self) -> Self::OperatorOutput) -> Self::Hessian {
        central_hessian_vec(self, g)
    }

    fn forward_hessian_vec_prod(
        &self,
        g: &dyn Fn(&Self) -> Self::OperatorOutput,
        p: &Self,
    ) -> Self {
        forward_hessian_vec_prod_vec(self, g, p)
    }

    fn central_hessian_vec_prod(
        &self,
        g: &dyn Fn(&Self) -> Self::OperatorOutput,
        p: &Self,
    ) -> Self {
        central_hessian_vec_prod_vec(self, g, p)
    }

    fn forward_hessian_nograd(&self, f: &dyn Fn(&Self) -> F) -> Self::Hessian {
        forward_hessian_nograd_vec(self, f)
    }

    fn forward_hessian_nograd_sparse(
        &self,
        f: &dyn Fn(&Self) -> F,
        indices: Vec<[usize; 2]>,
    ) -> Self::Hessian {
        forward_hessian_nograd_sparse_vec(self, f, indices)
    }
}

#[cfg(test)]
mod tests {
    use crate::PerturbationVector;

    use super::*;

    const COMP_ACC: f64 = 1e-6;

    fn f1(x: &Vec<f64>) -> f64 {
        x[0] + x[1].powi(2)
    }

    fn f2(x: &Vec<f64>) -> Vec<f64> {
        vec![
            2.0 * (x[1].powi(3) - x[0].powi(2)),
            3.0 * (x[1].powi(3) - x[0].powi(2)) + 2.0 * (x[2].powi(3) - x[1].powi(2)),
            3.0 * (x[2].powi(3) - x[1].powi(2)) + 2.0 * (x[3].powi(3) - x[2].powi(2)),
            3.0 * (x[3].powi(3) - x[2].powi(2)) + 2.0 * (x[4].powi(3) - x[3].powi(2)),
            3.0 * (x[4].powi(3) - x[3].powi(2)) + 2.0 * (x[5].powi(3) - x[4].powi(2)),
            3.0 * (x[5].powi(3) - x[4].powi(2)),
        ]
    }

    fn f3(x: &Vec<f64>) -> f64 {
        x[0] + x[1].powi(2) + x[2] * x[3].powi(2)
    }

    fn g(x: &Vec<f64>) -> Vec<f64> {
        vec![1.0, 2.0 * x[1], x[3].powi(2), 2.0 * x[3] * x[2]]
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
            vec![-4.0, -6.0, 0.0, 0.0, 0.0, 0.0],
            vec![6.0, 5.0, -6.0, 0.0, 0.0, 0.0],
            vec![0.0, 6.0, 5.0, -6.0, 0.0, 0.0],
            vec![0.0, 0.0, 6.0, 5.0, -6.0, 0.0],
            vec![0.0, 0.0, 0.0, 6.0, 5.0, -6.0],
            vec![0.0, 0.0, 0.0, 0.0, 6.0, 9.0],
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
    fn test_forward_diff_vec_f64_trait() {
        let grad = x1().forward_diff(&f1);
        let res = [1.0f64, 2.0];

        for i in 0..2 {
            assert!((res[i] - grad[i]).abs() < COMP_ACC)
        }

        let p = vec![1.0f64, 2.0f64];
        let grad = p.forward_diff(&f1);
        let res = [1.0f64, 4.0];

        for i in 0..2 {
            assert!((res[i] - grad[i]).abs() < COMP_ACC)
        }
    }

    #[test]
    fn test_central_diff_vec_f64_trait() {
        let grad = x1().central_diff(&f1);
        let res = [1.0f64, 2.0];

        for i in 0..2 {
            assert!((res[i] - grad[i]).abs() < COMP_ACC)
        }

        let p = vec![1.0f64, 2.0f64];
        let grad = p.central_diff(&f1);
        let res = [1.0f64, 4.0];

        for i in 0..2 {
            assert!((res[i] - grad[i]).abs() < COMP_ACC)
        }
    }

    #[test]
    fn test_forward_jacobian_vec_f64_trait() {
        let jacobian = x2().forward_jacobian(&f2);
        let res = res1();
        // println!("{:?}", jacobian);
        for i in 0..6 {
            for j in 0..6 {
                assert!((res[i][j] - jacobian[i][j]).abs() < COMP_ACC)
            }
        }
    }

    #[test]
    fn test_central_jacobian_vec_f64_trait() {
        let jacobian = x2().central_jacobian(&f2);
        let res = res1();
        // println!("{:?}", jacobian);
        for i in 0..6 {
            for j in 0..6 {
                assert!((res[i][j] - jacobian[i][j]).abs() < COMP_ACC)
            }
        }
    }

    #[test]
    fn test_forward_jacobian_vec_prod_vec_f64_trait() {
        let jacobian = x2().forward_jacobian_vec_prod(&f2, &p1());
        let res = res3();
        // println!("{:?}", jacobian);
        // the accuracy for this is pretty bad!!
        for i in 0..6 {
            assert!((res[i] - jacobian[i]).abs() < 5.5 * COMP_ACC)
        }
    }

    #[test]
    fn test_central_jacobian_vec_prod_vec_f64_trait() {
        let jacobian = x2().central_jacobian_vec_prod(&f2, &p1());
        let res = res3();
        // println!("{:?}", jacobian);
        for i in 0..6 {
            assert!((res[i] - jacobian[i]).abs() < COMP_ACC)
        }
    }

    #[test]
    fn test_forward_jacobian_pert_vec_f64_trait() {
        let jacobian = x2().forward_jacobian_pert(&f2, &pert());
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
    fn test_central_jacobian_pert_vec_f64_trait() {
        let jacobian = x2().central_jacobian_pert(&f2, &pert());
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
    fn test_forward_hessian_vec_f64_trait() {
        let hessian = x3().forward_hessian(&g);
        let res = res2();
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            for j in 0..4 {
                assert!((res[i][j] - hessian[i][j]).abs() < COMP_ACC)
            }
        }
    }

    #[test]
    fn test_central_hessian_vec_f64_trait() {
        let hessian = x3().central_hessian(&g);
        let res = res2();
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            for j in 0..4 {
                assert!((res[i][j] - hessian[i][j]).abs() < COMP_ACC)
            }
        }
    }

    #[test]
    fn test_forward_hessian_vec_prod_vec_f64_trait() {
        let hessian = x3().forward_hessian_vec_prod(&g, &p2());
        let res = [0.0, 6.0, 10.0, 18.0];
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            assert!((res[i] - hessian[i]).abs() < COMP_ACC)
        }
    }

    #[test]
    fn test_central_hessian_vec_prod_vec_f64_trait() {
        let hessian = x3().central_hessian_vec_prod(&g, &p2());
        let res = [0.0, 6.0, 10.0, 18.0];
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            assert!((res[i] - hessian[i]).abs() < COMP_ACC)
        }
    }

    #[test]
    fn test_forward_hessian_nograd_vec_f64_trait() {
        let hessian = x3().forward_hessian_nograd(&f3);
        let res = res2();
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            for j in 0..4 {
                assert!((res[i][j] - hessian[i][j]).abs() < COMP_ACC)
            }
        }
    }

    #[test]
    fn test_forward_hessian_nograd_sparse_vec_f64_trait() {
        let indices = vec![[1, 1], [2, 3], [3, 3]];
        let hessian = x3().forward_hessian_nograd_sparse(&f3, indices);
        let res = res2();
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            for j in 0..4 {
                assert!((res[i][j] - hessian[i][j]).abs() < COMP_ACC)
            }
        }
    }
}
