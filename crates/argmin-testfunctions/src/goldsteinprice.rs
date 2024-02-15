// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Goldstein-Price test function
//!
//! Defined as
//!
//! `f(x_1, x_2) = [1 + (x_1 + x_2 + 1)^2 * (19 - 14*x_1 + 3*x_1^2 - 14*x_2 + 6*x_1*x_2 + 3*x_2^2)]
//!                * [30 + (2*x_1 - 3*x_2)^2(18 - 32 * x_1 + 12* x_1^2 + 48 * x_2 -
//!                   36 * x_1 * x_2 + 27 * x_2^2) ]`
//!
//! where `x_i \in [-2, 2]`.
//!
//! The global minimum is at `f(x_1, x_2) = f(0, -1) = 3`.

use num::{Float, FromPrimitive};

/// Goldstein-Price test function
///
/// Defined as
///
/// `f(x_1, x_2) = [1 + (x_1 + x_2 + 1)^2 * (19 - 14*x_1 + 3*x_1^2 - 14*x_2 + 6*x_1*x_2 + 3*x_2^2)]
///                * [30 + (2*x_1 - 3*x_2)^2(18 - 32 * x_1 + 12* x_1^2 + 48 * x_2 -
///                   36 * x_1 * x_2 + 27 * x_2^2) ]`
///
/// where `x_i \in [-2, 2]`.
///
/// The global minimum is at `f(x_1, x_2) = f(0, -1) = 3`.
pub fn goldsteinprice<T>(param: &[T; 2]) -> T
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;
    let n1 = T::from_f64(1.0).unwrap();
    let n2 = T::from_f64(2.0).unwrap();
    let n3 = T::from_f64(3.0).unwrap();
    let n6 = T::from_f64(6.0).unwrap();
    let n12 = T::from_f64(12.0).unwrap();
    let n14 = T::from_f64(14.0).unwrap();
    let n18 = T::from_f64(18.0).unwrap();
    let n19 = T::from_f64(19.0).unwrap();
    let n27 = T::from_f64(27.0).unwrap();
    let n30 = T::from_f64(30.0).unwrap();
    let n32 = T::from_f64(32.0).unwrap();
    let n36 = T::from_f64(36.0).unwrap();
    let n48 = T::from_f64(48.0).unwrap();
    (n1 + (x1 + x2 + n1).powi(2)
        * (n19 - n14 * (x1 + x2) + n3 * (x1.powi(2) + x2.powi(2)) + n6 * x1 * x2))
        * (n30
            + (n2 * x1 - n3 * x2).powi(2)
                * (n18 - n32 * x1 + n12 * x1.powi(2) + n48 * x2 - n36 * x1 * x2 + n27 * x2.powi(2)))
}

/// Derivative of Goldstein-Price test function
pub fn goldsteinprice_derivative<T>(param: &[T; 2]) -> [T; 2]
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;

    let n1 = T::from_f64(1.0).unwrap();
    let n2 = T::from_f64(2.0).unwrap();
    let n3 = T::from_f64(3.0).unwrap();
    let n4 = T::from_f64(4.0).unwrap();
    let n6 = T::from_f64(6.0).unwrap();
    let n12 = T::from_f64(12.0).unwrap();
    let n14 = T::from_f64(14.0).unwrap();
    let n18 = T::from_f64(18.0).unwrap();
    let n19 = T::from_f64(19.0).unwrap();
    let n24 = T::from_f64(24.0).unwrap();
    let n27 = T::from_f64(27.0).unwrap();
    let n30 = T::from_f64(30.0).unwrap();
    let n32 = T::from_f64(32.0).unwrap();
    let n36 = T::from_f64(36.0).unwrap();
    let n48 = T::from_f64(48.0).unwrap();
    let n54 = T::from_f64(54.0).unwrap();

    let x1s = x1.powi(2);
    let x2s = x2.powi(2);

    [
        (n2 * (x1 + x2 + n1) * (n3 * x1s + n6 * x2 * x1 - n14 * x1 + n3 * x2s - n14 * x2 + n19)
            + (x1 + x2 + n1).powi(2) * (n6 * x1 + n6 * x2 - n14))
            * ((n2 * x1 - n3 * x2).powi(2)
                * (n12 * x1s - n36 * x2 * x1 - n32 * x1 + n27 * x2s + n48 * x2 + n18)
                + n30)
            + ((x1 + x2 + n1).powi(2)
                * (n3 * x1s + n6 * x2 * x1 - n14 * x1 + n3 * x2s - n14 * x2 + n19)
                + n1)
                * (n4
                    * (n2 * x1 - n3 * x2)
                    * (n12 * x1s - n36 * x2 * x1 - n32 * x1 + n27 * x2s + n48 * x2 + n18)
                    + (n2 * x1 - n3 * x2).powi(2) * (n24 * x1 - n36 * x2 - n32)),
        ((x2 + x1 + n1).powi(2) * (n3 * x2s + n6 * x1 * x2 - n14 * x2 + n3 * x1s - n14 * x1 + n19)
            + n1)
            * ((n2 * x1 - n3 * x2).powi(2) * (n54 * x2 - n36 * x1 + n48)
                - n6 * (n2 * x1 - n3 * x2)
                    * (n27 * x2s - n36 * x1 * x2 + n48 * x2 + n12 * x1s - n32 * x1 + n18))
            + (n2
                * (x2 + x1 + n1)
                * (n3 * x2s + n6 * x1 * x2 - n14 * x2 + n3 * x1s - n14 * x1 + n19)
                + (x2 + x1 + n1).powi(2) * (n6 * x2 + n6 * x1 - n14))
                * ((n2 * x1 - n3 * x2).powi(2)
                    * (n27 * x2s - n36 * x1 * x2 + n48 * x2 + n12 * x1s - n32 * x1 + n18)
                    + n30),
    ]
}

/// Hessian of Goldstein-Price test function
pub fn goldsteinprice_hessian<T>(param: &[T; 2]) -> [[T; 2]; 2]
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;

    let n840 = T::from_f64(840.0).unwrap();
    let n1296 = T::from_f64(1296.0).unwrap();
    let n2016 = T::from_f64(2016.0).unwrap();
    let n2520 = T::from_f64(2520.0).unwrap();
    let n2916 = T::from_f64(2916.0).unwrap();
    let n3360 = T::from_f64(3360.0).unwrap();
    let n4680 = T::from_f64(4680.0).unwrap();
    let n5184 = T::from_f64(5184.0).unwrap();
    let n5940 = T::from_f64(5940.0).unwrap();
    let n6120 = T::from_f64(6120.0).unwrap();
    let n6432 = T::from_f64(6432.0).unwrap();
    let n6804 = T::from_f64(6804.0).unwrap();
    let n7344 = T::from_f64(7344.0).unwrap();
    let n7440 = T::from_f64(7440.0).unwrap();
    let n7776 = T::from_f64(7776.0).unwrap();
    let n8064 = T::from_f64(8064.0).unwrap();
    let n10080 = T::from_f64(10080.0).unwrap();
    let n10740 = T::from_f64(10740.0).unwrap();
    let n11016 = T::from_f64(11016.0).unwrap();
    let n11160 = T::from_f64(11160.0).unwrap();
    let n11664 = T::from_f64(11664.0).unwrap();
    let n12096 = T::from_f64(12096.0).unwrap();
    let n14688 = T::from_f64(14688.0).unwrap();
    let n15552 = T::from_f64(15552.0).unwrap();
    let n15660 = T::from_f64(15660.0).unwrap();
    let n17352 = T::from_f64(17352.0).unwrap();
    let n17460 = T::from_f64(17460.0).unwrap();
    let n17496 = T::from_f64(17496.0).unwrap();
    let n18360 = T::from_f64(18360.0).unwrap();
    let n19440 = T::from_f64(19440.0).unwrap();
    let n19680 = T::from_f64(19680.0).unwrap();
    let n20880 = T::from_f64(20880.0).unwrap();
    let n23760 = T::from_f64(23760.0).unwrap();
    let n24480 = T::from_f64(24480.0).unwrap();
    let n25920 = T::from_f64(25920.0).unwrap();
    let n26880 = T::from_f64(26880.0).unwrap();
    let n27216 = T::from_f64(27216.0).unwrap();
    let n27540 = T::from_f64(27540.0).unwrap();
    let n28560 = T::from_f64(28560.0).unwrap();
    let n29448 = T::from_f64(29448.0).unwrap();
    let n30240 = T::from_f64(30240.0).unwrap();
    let n30720 = T::from_f64(30720.0).unwrap();
    let n31104 = T::from_f64(31104.0).unwrap();
    let n32256 = T::from_f64(32256.0).unwrap();
    let n34704 = T::from_f64(34704.0).unwrap();
    let n36720 = T::from_f64(36720.0).unwrap();
    let n38592 = T::from_f64(38592.0).unwrap();
    let n38880 = T::from_f64(38880.0).unwrap();
    let n40320 = T::from_f64(40320.0).unwrap();
    let n40824 = T::from_f64(40824.0).unwrap();
    let n41760 = T::from_f64(41760.0).unwrap();
    let n42960 = T::from_f64(42960.0).unwrap();
    let n43740 = T::from_f64(43740.0).unwrap();
    let n47520 = T::from_f64(47520.0).unwrap();
    let n48960 = T::from_f64(48960.0).unwrap();
    let n51840 = T::from_f64(51840.0).unwrap();
    let n58320 = T::from_f64(58320.0).unwrap();
    let n59040 = T::from_f64(59040.0).unwrap();
    let n64440 = T::from_f64(64440.0).unwrap();
    let n69840 = T::from_f64(69840.0).unwrap();
    let n70848 = T::from_f64(70848.0).unwrap();
    let n73440 = T::from_f64(73440.0).unwrap();
    let n73728 = T::from_f64(73728.0).unwrap();
    let n92160 = T::from_f64(92160.0).unwrap();
    let n104760 = T::from_f64(104760.0).unwrap();
    let n132840 = T::from_f64(132840.0).unwrap();
    let n142560 = T::from_f64(142560.0).unwrap();
    let n141696 = T::from_f64(141696.0).unwrap();
    let n172152 = T::from_f64(172152.0).unwrap();

    let x1p2 = x1.powi(2);
    let x1p3 = x1.powi(3);
    let x1p4 = x1.powi(4);
    let x1p5 = x1.powi(5);
    let x1p6 = x1.powi(6);
    let x2p2 = x2.powi(2);
    let x2p3 = x2.powi(3);
    let x2p4 = x2.powi(4);
    let x2p5 = x2.powi(5);
    let x2p6 = x2.powi(6);

    let a = n8064 * x1p6
        + (-n12096 * x2 - n32256) * x1p5
        + (-n19440 * x2p2 + n40320 * x2 + n28560) * x1p4
        + (n24480 * x2p3 + n51840 * x2p2 - n3360 * x2 + n26880) * x1p3
        + (n15660 * x2p4 - n48960 * x2p3 - n64440 * x2p2 - n92160 * x2 - n29448) * x1p2
        + (-n11016 * x2p5 - n20880 * x2p4 + n7440 * x2p3 + n59040 * x2p2 + n34704 * x2 - n6432)
            * x1
        - n2916 * x2p6
        + n7344 * x2p5
        + n17460 * x2p4
        + n10080 * x2p3
        + n15552 * x2p2
        + n14688 * x2
        + n2520;

    let b = n40824 * x2p6
        + (n40824 * x1 - n27216) * x2p5
        + (-n43740 * x1p2 + n58320 * x1 - n132840) * x2p4
        + (-n36720 * x1p3 + n73440 * x1p2 - n23760 * x1 + n38880) * x2p3
        + (n15660 * x1p4 - n41760 * x1p3 + n104760 * x1p2 - n142560 * x1 + n172152) * x2p2
        + (n7344 * x1p5 - n24480 * x1p4 + n7440 * x1p3 + n30240 * x1p2 - n141696 * x1 + n73728)
            * x2
        - n1296 * x1p6
        + n5184 * x1p5
        - n10740 * x1p4
        + n19680 * x1p3
        + n15552 * x1p2
        - n38592 * x1
        + n6120;

    let offdiag = n6804 * x2p6
        + (n11664 - n17496 * x1) * x2p5
        + (-n27540 * x1p2 + n36720 * x1 - n5940) * x2p4
        + (n20880 * x1p3 - n41760 * x1p2 + n69840 * x1 - n47520) * x2p3
        + (n18360 * x1p4 - n48960 * x1p3 + n11160 * x1p2 + n30240 * x1 - n70848) * x2p2
        + (-n7776 * x1p5 + n25920 * x1p4 - n42960 * x1p3 + n59040 * x1p2 + n31104 * x1 - n38592)
            * x2
        - n2016 * x1p6
        + n8064 * x1p5
        - n840 * x1p4
        - n30720 * x1p3
        + n17352 * x1p2
        + n14688 * x1
        - n4680;

    [[a, offdiag], [offdiag, b]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use finitediff::FiniteDiff;
    use proptest::prelude::*;
    use std::{f32, f64};

    #[test]
    fn test_goldsteinprice_optimum() {
        assert_relative_eq!(
            goldsteinprice(&[0.0_f32, -1.0_f32]),
            3_f32,
            epsilon = f32::EPSILON
        );
        assert_relative_eq!(
            goldsteinprice(&[0.0_f64, -1.0_f64]),
            3_f64,
            epsilon = f64::EPSILON
        );

        let deriv = goldsteinprice_derivative(&[0.0_f64, -1.0_f64]);
        for i in 0..2 {
            assert_relative_eq!(deriv[i], 0.0, epsilon = f64::EPSILON);
        }
    }

    proptest! {
        #[test]
        fn test_goldsteinprice_derivative_finitediff(a in -2.0..2.0, b in -2.0..2.0) {
            let param = [a, b];
            let derivative = goldsteinprice_derivative(&param);
            let derivative_fd = Vec::from(param).central_diff(&|x| goldsteinprice(&[x[0], x[1]]));
            // println!("1: {derivative:?} at {a}/{b}");
            // println!("2: {derivative_fd:?} at {a}/{b}");
            for i in 0..derivative.len() {
                assert_relative_eq!(
                    derivative[i],
                    derivative_fd[i],
                    epsilon = 1e-3,
                    max_relative = 1e-1
                );
            }
        }
    }

    proptest! {
        #[test]
        fn test_goldsteinprice_derivative_finitediff_narrow(a in -0.5..0.5, b in -0.5..0.5) {
            // This evaluates the function on a narrower domain, which allows us to have a lower
            // epsilon, as the function is pretty steep at the boundary, which isn't great for
            // accuracy when using finite differentiation.
            let param = [a, b];
            let derivative = goldsteinprice_derivative(&param);
            let derivative_fd = Vec::from(param).central_diff(&|x| goldsteinprice(&[x[0], x[1]]));
            // println!("1: {derivative:?} at {a}/{b}");
            // println!("2: {derivative_fd:?} at {a}/{b}");
            for i in 0..derivative.len() {
                assert_relative_eq!(
                    derivative[i],
                    derivative_fd[i],
                    epsilon = 1e-3,
                    max_relative = 1e-2
                );
            }
        }
    }

    proptest! {
        #[test]
        fn test_goldsteinprice_hessian_finitediff(a in -2.0..2.0, b in -2.0..2.0) {
            let param = [a, b];
            let hessian = goldsteinprice_hessian(&param);
            let hessian_fd =
                Vec::from(param).central_hessian(&|x| goldsteinprice_derivative(&[x[0], x[1]]).to_vec());
            let n = hessian.len();
            // println!("1: {hessian:?} at {a}/{b}");
            // println!("2: {hessian_fd:?} at {a}/{b}");
            for i in 0..n {
                assert_eq!(hessian[i].len(), n);
                for j in 0..n {
                    if hessian_fd[i][j].is_finite() {
                        assert_relative_eq!(
                            hessian[i][j],
                            hessian_fd[i][j],
                            epsilon = 1e-5,
                            max_relative = 1e-1
                        );
                    }
                }
            }
        }
    }

    proptest! {
        #[test]
        fn test_goldsteinprice_hessian_finitediff_narrow(a in -0.5..0.5, b in -0.5..0.5) {
            // This evaluates the function on a narrower domain, which allows us to have a lower
            // epsilon, as the function is pretty steep at the boundary, which isn't great for
            // accuracy when using finite differentiation.
            let param = [a, b];
            let hessian = goldsteinprice_hessian(&param);
            let hessian_fd =
                Vec::from(param).central_hessian(&|x| goldsteinprice_derivative(&[x[0], x[1]]).to_vec());
            let n = hessian.len();
            // println!("1: {hessian:?} at {a}/{b}");
            // println!("2: {hessian_fd:?} at {a}/{b}");
            for i in 0..n {
                assert_eq!(hessian[i].len(), n);
                for j in 0..n {
                    if hessian_fd[i][j].is_finite() {
                        assert_relative_eq!(
                            hessian[i][j],
                            hessian_fd[i][j],
                            epsilon = 1e-5,
                            max_relative = 1e-2
                        );
                    }
                }
            }
        }
    }
}
