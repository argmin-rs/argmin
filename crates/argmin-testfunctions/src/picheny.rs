// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Picheny test function
//!
//! Variation of the Goldstein-Price test function.
//!
//! Defined as
//!
//! `f(x_1, x_2) = (1/2.427) * log([1 + (\bar{x}_1 + \bar{x}_2 + 1)^2 * (19 - 14*\bar{x}_2 +
//!                3*\bar{x}_1^2 - 14*\bar{x}_2 + 6*\bar{x}_1*\bar{x}_2 + 3*\bar{x}_2^2)]
//!                * [30 + (2*\bar{x}_1 - 3*\bar{x}_2)^2(18 - 32 * \bar{x}_1 + 12* \bar{x}_1^2 +
//!                48 * \bar{x}_2 - 36 * \bar{x}_1 * \bar{x}_2 + 27 * \bar{x}_2^2) ] - 8.693)`
//!
//! where `\bar{x}_i = 4*x_i - 2` and `x_i \in [0, 1]`.
//!
//! The global minimum is at `f(x_1, x_2) = f(0.5, 0.25) = 3.3851993182036826`.

//  (1/2.427) * (log_10([1 + ((4*x_1-2) + (4*x_2-2) + 1)^2 * (19 - 14*(4*x_1-2) + 3*(4*x_1-2)^2 - 14*(4*x_2-2) + 6*(4*x_1-2)*(4*x_2-2) + 3*(4*x_2-2)^2)]    * [30 + (2*(4*x_1-2) - 3*(4*x_2-2))^2*(18 - 32 * (4*x_1-2) + 12* (4*x_1-2)^2 +   48 * (4*x_2-2) - 36 * (4*x_1-2) * (4*x_2-2) + 27 * (4*x_2-2)^2) ]) - 8.693)

use num::{Float, FromPrimitive};

/// Picheny test function
///
/// Variation of the Goldstein-Price test function.
///
/// Defined as
///
/// `f(x_1, x_2) = (1/2.427) * log([1 + (\bar{x}_1 + \bar{x}_2 + 1)^2 * (19 - 14*\bar{x}_2 +
///                3*\bar{x}_1^2 - 14*\bar{x}_2 6*\bar{x}_1*\bar{x}_2 + 3*\bar{x}_2^2)]
///                * [30 + (2*\bar{x}_1 - 3*\bar{x}_2)^2(18 - 32 * \bar{x}_1 + 12* \bar{x}_1^2 +
///                48 * \bar{x}_2 - 36 * \bar{x}_1 * \bar{x}_2 + 27 * \bar{x}_2^2) ] - 8.693)`
///
/// where `\bar{x}_i = 4*x_i - 2` and `x_i \in [0, 1]`.
///
/// The global minimum is at `f(x_1, x_2) = f(0.5, 0.25) = 3.3851993182036826`.
pub fn picheny<T>(param: &[T; 2]) -> T
where
    T: Float + FromPrimitive,
{
    let n1 = T::from_f64(1.0).unwrap();
    let n2 = T::from_f64(2.0).unwrap();
    let n3 = T::from_f64(3.0).unwrap();
    let n4 = T::from_f64(4.0).unwrap();
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

    let [x1, x2] = param.map(|x| n4 * x - n2);

    T::from_f64(1.0 / 2.427).unwrap()
        * (((n1
            + (x1 + x2 + n1).powi(2)
                * (n19 - n14 * (x1 + x2) + n3 * (x1.powi(2) + x2.powi(2)) + n6 * x1 * x2))
            * (n30
                + (n2 * x1 - n3 * x2).powi(2)
                    * (n18 - n32 * x1 + n12 * x1.powi(2) + n48 * x2 - n36 * x1 * x2
                        + n27 * x2.powi(2))))
        .log10()
            - T::from_f64(8.693).unwrap())
}
/// Derivative of Picheny test function
pub fn picheny_derivative<T>(param: &[T; 2]) -> [T; 2]
where
    T: Float + FromPrimitive,
{
    let n1 = T::from_f64(1.0).unwrap();
    let n2 = T::from_f64(2.0).unwrap();
    let n3 = T::from_f64(3.0).unwrap();
    let n4 = T::from_f64(4.0).unwrap();
    let n6 = T::from_f64(6.0).unwrap();
    let n8 = T::from_f64(8.0).unwrap();
    let n10 = T::from_f64(10.0).unwrap();
    let n12 = T::from_f64(12.0).unwrap();
    let n14 = T::from_f64(14.0).unwrap();
    let n16 = T::from_f64(16.0).unwrap();
    let n18 = T::from_f64(18.0).unwrap();
    let n19 = T::from_f64(19.0).unwrap();
    let n24 = T::from_f64(24.0).unwrap();
    let n27 = T::from_f64(27.0).unwrap();
    let n30 = T::from_f64(30.0).unwrap();
    let n32 = T::from_f64(32.0).unwrap();
    let n36 = T::from_f64(36.0).unwrap();
    let n48 = T::from_f64(48.0).unwrap();
    let n56 = T::from_f64(56.0).unwrap();
    let n96 = T::from_f64(96.0).unwrap();
    let n128 = T::from_f64(128.0).unwrap();
    let n144 = T::from_f64(144.0).unwrap();
    let n192 = T::from_f64(192.0).unwrap();
    let n216 = T::from_f64(216.0).unwrap();
    let factor = T::from_f64(1.0 / 2.427).unwrap();

    let [x1, x2] = *param;

    let a = (factor
        * ((n8
            * (n4 * x1 + n4 * x2 - n3)
            * (n3 * (n4 * x1 - n2).powi(2) + n6 * (n4 * x2 - n2) * (n4 * x1 - n2)
                - n14 * (n4 * x1 - n2)
                + n3 * (n4 * x2 - n2).powi(2)
                - n14 * (n4 * x2 - n2)
                + n19)
            + (n4 * x1 + n4 * x2 - n3).powi(2)
                * (n24 * (n4 * x1 - n2) + n24 * (n4 * x2 - n2) - n56))
            * ((n2 * (n4 * x1 - n2) - n3 * (n4 * x2 - n2)).powi(2)
                * (n12 * (n4 * x1 - n2).powi(2)
                    - n36 * (n4 * x2 - n2) * (n4 * x1 - n2)
                    - n32 * (n4 * x1 - n2)
                    + n27 * (n4 * x2 - n2).powi(2)
                    + n48 * (n4 * x2 - n2)
                    + n18)
                + n30)
            + ((n4 * x1 + n4 * x2 - n3).powi(2)
                * (n3 * (n4 * x1 - n2).powi(2) + n6 * (n4 * x2 - n2) * (n4 * x1 - n2)
                    - n14 * (n4 * x1 - n2)
                    + n3 * (n4 * x2 - n2).powi(2)
                    - n14 * (n4 * x2 - n2)
                    + n19)
                + n1)
                * (n16
                    * (n2 * (n4 * x1 - n2) - n3 * (n4 * x2 - n2))
                    * (n12 * (n4 * x1 - n2).powi(2)
                        - n36 * (n4 * x2 - n2) * (n4 * x1 - n2)
                        - n32 * (n4 * x1 - n2)
                        + n27 * (n4 * x2 - n2).powi(2)
                        + n48 * (n4 * x2 - n2)
                        + n18)
                    + (n2 * (n4 * x1 - n2) - n3 * (n4 * x2 - n2)).powi(2)
                        * (n96 * (n4 * x1 - n2) - n144 * (n4 * x2 - n2) - n128))))
        / (n10.ln()
            * ((n4 * x1 + n4 * x2 - n3).powi(2)
                * (n3 * (n4 * x1 - n2).powi(2) + n6 * (n4 * x2 - n2) * (n4 * x1 - n2)
                    - n14 * (n4 * x1 - n2)
                    + n3 * (n4 * x2 - n2).powi(2)
                    - n14 * (n4 * x2 - n2)
                    + n19)
                + n1)
            * ((n2 * (n4 * x1 - n2) - n3 * (n4 * x2 - n2)).powi(2)
                * (n12 * (n4 * x1 - n2).powi(2)
                    - n36 * (n4 * x2 - n2) * (n4 * x1 - n2)
                    - n32 * (n4 * x1 - n2)
                    + n27 * (n4 * x2 - n2).powi(2)
                    + n48 * (n4 * x2 - n2)
                    + n18)
                + n30));

    let b = (factor
        * ((n8
            * (n4 * x2 + n4 * x1 - n3)
            * (n3 * (n4 * x2 - n2).powi(2) + n6 * (n4 * x1 - n2) * (n4 * x2 - n2)
                - n14 * (n4 * x2 - n2)
                + n3 * (n4 * x1 - n2).powi(2)
                - n14 * (n4 * x1 - n2)
                + n19)
            + (n4 * x2 + n4 * x1 - n3).powi(2)
                * (n24 * (n4 * x2 - n2) + n24 * (n4 * x1 - n2) - n56))
            * ((n2 * (n4 * x1 - n2) - n3 * (n4 * x2 - n2)).powi(2)
                * (n27 * (n4 * x2 - n2).powi(2) - n36 * (n4 * x1 - n2) * (n4 * x2 - n2)
                    + n48 * (n4 * x2 - n2)
                    + n12 * (n4 * x1 - n2).powi(2)
                    - n32 * (n4 * x1 - n2)
                    + n18)
                + n30)
            + ((n4 * x2 + n4 * x1 - n3).powi(2)
                * (n3 * (n4 * x2 - n2).powi(2) + n6 * (n4 * x1 - n2) * (n4 * x2 - n2)
                    - n14 * (n4 * x2 - n2)
                    + n3 * (n4 * x1 - n2).powi(2)
                    - n14 * (n4 * x1 - n2)
                    + n19)
                + n1)
                * ((n2 * (n4 * x1 - n2) - n3 * (n4 * x2 - n2)).powi(2)
                    * (n216 * (n4 * x2 - n2) - n144 * (n4 * x1 - n2) + n192)
                    - n24
                        * (n2 * (n4 * x1 - n2) - n3 * (n4 * x2 - n2))
                        * (n27 * (n4 * x2 - n2).powi(2) - n36 * (n4 * x1 - n2) * (n4 * x2 - n2)
                            + n48 * (n4 * x2 - n2)
                            + n12 * (n4 * x1 - n2).powi(2)
                            - n32 * (n4 * x1 - n2)
                            + n18))))
        / (n10.ln()
            * ((n4 * x2 + n4 * x1 - n3).powi(2)
                * (n3 * (n4 * x2 - n2).powi(2) + n6 * (n4 * x1 - n2) * (n4 * x2 - n2)
                    - n14 * (n4 * x2 - n2)
                    + n3 * (n4 * x1 - n2).powi(2)
                    - n14 * (n4 * x1 - n2)
                    + n19)
                + n1)
            * ((n2 * (n4 * x1 - n2) - n3 * (n4 * x2 - n2)).powi(2)
                * (n27 * (n4 * x2 - n2).powi(2) - n36 * (n4 * x1 - n2) * (n4 * x2 - n2)
                    + n48 * (n4 * x2 - n2)
                    + n12 * (n4 * x1 - n2).powi(2)
                    - n32 * (n4 * x1 - n2)
                    + n18)
                + n30));

    [a, b]
}

/// Hessian of Picheny test function
pub fn picheny_hessian<T>(param: &[T; 2]) -> [[T; 2]; 2]
where
    T: Float + FromPrimitive,
{
    // oh dear...
    let n2 = T::from_f64(2.0).unwrap();
    let n3 = T::from_f64(3.0).unwrap();
    let n4 = T::from_f64(4.0).unwrap();
    let n5 = T::from_f64(5.0).unwrap();
    let n6 = T::from_f64(6.0).unwrap();
    let n10 = T::from_f64(10.0).unwrap();
    let n11 = T::from_f64(11.0).unwrap();
    let n96 = T::from_f64(96.0).unwrap();
    let n144 = T::from_f64(144.0).unwrap();
    let n192 = T::from_f64(192.0).unwrap();
    let n277 = T::from_f64(277.0).unwrap();
    let n432 = T::from_f64(432.0).unwrap();
    let n576 = T::from_f64(576.0).unwrap();
    let n768 = T::from_f64(768.0).unwrap();
    let n864 = T::from_f64(864.0).unwrap();
    let n896 = T::from_f64(896.0).unwrap();
    let n1080 = T::from_f64(1080.0).unwrap();
    let n1152 = T::from_f64(1152.0).unwrap();
    let n1167 = T::from_f64(1167.0).unwrap();
    let n1512 = T::from_f64(1512.0).unwrap();
    let n1603 = T::from_f64(1603.0).unwrap();
    let n2048 = T::from_f64(2048.0).unwrap();
    let n2304 = T::from_f64(2304.0).unwrap();
    let n2688 = T::from_f64(2688.0).unwrap();
    let n3024 = T::from_f64(3024.0).unwrap();
    let n5376 = T::from_f64(5376.0).unwrap();
    let n5594 = T::from_f64(5594.0).unwrap();
    let n5874 = T::from_f64(5874.0).unwrap();
    let n6144 = T::from_f64(6144.0).unwrap();
    let n6912 = T::from_f64(6912.0).unwrap();
    let n9216 = T::from_f64(9216.0).unwrap();
    let n13824 = T::from_f64(13824.0).unwrap();
    let n20736 = T::from_f64(20736.0).unwrap();
    let n21546 = T::from_f64(21546.0).unwrap();
    let n24576 = T::from_f64(24576.0).unwrap();
    let n27648 = T::from_f64(27648.0).unwrap();
    let n31104 = T::from_f64(31104.0).unwrap();
    let n36864 = T::from_f64(36864.0).unwrap();
    let n77456 = T::from_f64(77456.0).unwrap();
    let n82944 = T::from_f64(82944.0).unwrap();
    let n85088 = T::from_f64(85088.0).unwrap();
    let n98304 = T::from_f64(98304.0).unwrap();
    let n118688 = T::from_f64(118688.0).unwrap();
    let n124416 = T::from_f64(124416.0).unwrap();
    let n127904 = T::from_f64(127904.0).unwrap();
    let n155024 = T::from_f64(155024.0).unwrap();
    let n163936 = T::from_f64(163936.0).unwrap();
    let n165888 = T::from_f64(165888.0).unwrap();
    let n248832 = T::from_f64(248832.0).unwrap();
    let n255264 = T::from_f64(255264.0).unwrap();
    let n255808 = T::from_f64(255808.0).unwrap();
    let n327872 = T::from_f64(327872.0).unwrap();
    let n331776 = T::from_f64(331776.0).unwrap();
    let n393216 = T::from_f64(393216.0).unwrap();
    let n442368 = T::from_f64(442368.0).unwrap();
    let n497664 = T::from_f64(497664.0).unwrap();
    let n516096 = T::from_f64(516096.0).unwrap();
    let n679936 = T::from_f64(679936.0).unwrap();
    let n688128 = T::from_f64(688128.0).unwrap();
    let n705280 = T::from_f64(705280.0).unwrap();
    let n995328 = T::from_f64(995328.0).unwrap();
    let n1057536 = T::from_f64(1057536.0).unwrap();
    let n1057920 = T::from_f64(1057920.0).unwrap();
    let n1136640 = T::from_f64(1136640.0).unwrap();
    let n1253376 = T::from_f64(1253376.0).unwrap();
    let n1290240 = T::from_f64(1290240.0).unwrap();
    let n1298688 = T::from_f64(1298688.0).unwrap();
    let n1327104 = T::from_f64(1327104.0).unwrap();
    let n1435136 = T::from_f64(1435136.0).unwrap();
    let n1441280 = T::from_f64(1441280.0).unwrap();
    let n1490944 = T::from_f64(1490944.0).unwrap();
    let n1515520 = T::from_f64(1515520.0).unwrap();
    let n1556736 = T::from_f64(1556736.0).unwrap();
    let n1781760 = T::from_f64(1781760.0).unwrap();
    let n1854464 = T::from_f64(1854464.0).unwrap();
    let n1880064 = T::from_f64(1880064.0).unwrap();
    let n1990656 = T::from_f64(1990656.0).unwrap();
    let n2088960 = T::from_f64(2088960.0).unwrap();
    let n2161920 = T::from_f64(2161920.0).unwrap();
    let n2322432 = T::from_f64(2322432.0).unwrap();
    let n2363520 = T::from_f64(2363520.0).unwrap();
    let n2520320 = T::from_f64(2520320.0).unwrap();
    let n2550112 = T::from_f64(2550112.0).unwrap();
    let n2580480 = T::from_f64(2580480.0).unwrap();
    let n2654208 = T::from_f64(2654208.0).unwrap();
    let n2752512 = T::from_f64(2752512.0).unwrap();
    let n2985984 = T::from_f64(2985984.0).unwrap();
    let n3113472 = T::from_f64(3113472.0).unwrap();
    let n3133440 = T::from_f64(3133440.0).unwrap();
    let n3896064 = T::from_f64(3896064.0).unwrap();
    let n4079616 = T::from_f64(4079616.0).unwrap();
    let n4231680 = T::from_f64(4231680.0).unwrap();
    let n4323840 = T::from_f64(4323840.0).unwrap();
    let n4523520 = T::from_f64(4523520.0).unwrap();
    let n4546560 = T::from_f64(4546560.0).unwrap();
    let n5040640 = T::from_f64(5040640.0).unwrap();
    let n5287680 = T::from_f64(5287680.0).unwrap();
    let n5492736 = T::from_f64(5492736.0).unwrap();
    let n5971968 = T::from_f64(5971968.0).unwrap();
    let n6266880 = T::from_f64(6266880.0).unwrap();
    let n6785280 = T::from_f64(6785280.0).unwrap();
    let n6850560 = T::from_f64(6850560.0).unwrap();
    let n7127040 = T::from_f64(7127040.0).unwrap();
    let n7175680 = T::from_f64(7175680.0).unwrap();
    let n7399680 = T::from_f64(7399680.0).unwrap();
    let n7728640 = T::from_f64(7728640.0).unwrap();
    let n8515584 = T::from_f64(8515584.0).unwrap();
    let n9089280 = T::from_f64(9089280.0).unwrap();
    let n9134080 = T::from_f64(9134080.0).unwrap();
    let n9400320 = T::from_f64(9400320.0).unwrap();
    let n9454080 = T::from_f64(9454080.0).unwrap();
    let n10081280 = T::from_f64(10081280.0).unwrap();
    let n13284864 = T::from_f64(13284864.0).unwrap();
    let n13570560 = T::from_f64(13570560.0).unwrap();
    let n13731840 = T::from_f64(13731840.0).unwrap();
    let n13934592 = T::from_f64(13934592.0).unwrap();
    let n14799360 = T::from_f64(14799360.0).unwrap();
    let n23185920 = T::from_f64(23185920.0).unwrap();
    let n27463680 = T::from_f64(27463680.0).unwrap();
    let n29598720 = T::from_f64(29598720.0).unwrap();
    let n27402240 = T::from_f64(27402240.0).unwrap();

    let factor = T::from_f64(9.88875154511743).unwrap();

    let [x_1, x_2] = *param;

    let offdiag = -(factor
        * (n768 * x_2.powi(3) + x_1 * (n2304 * x_2.powi(2) - n5376 * x_2 + n3024)
            - n2688 * x_2.powi(2)
            + x_1.powi(2) * (n2304 * x_2 - n2688)
            + n3024 * x_2
            + n768 * x_1.powi(3)
            - n1080)
        * (n331776 * x_2.powi(7) - n497664 * x_2.powi(6)
            + x_1
                * (-n995328 * x_2.powi(6) + n5492736 * x_2.powi(5) - n7399680 * x_2.powi(4)
                    + n1441280 * x_2.powi(3)
                    + n1556736 * x_2.powi(2)
                    - n255808 * x_2
                    + n5594)
            - n1057536 * x_2.powi(5)
            + x_1.powi(2)
                * (-n1880064 * x_2.powi(5) + n1136640 * x_2.powi(4) + n7728640 * x_2.powi(3)
                    - n6785280 * x_2.powi(2)
                    - n255264 * x_2
                    + n77456)
            + x_1.powi(3)
                * (n1781760 * x_2.powi(4) - n9134080 * x_2.powi(3)
                    + n5040640 * x_2.powi(2)
                    + n4231680 * x_2
                    - n118688)
            + n2363520 * x_2.powi(4)
            + x_1.powi(4)
                * (n2088960 * x_2.powi(3) + n1290240 * x_2.powi(2) - n7175680 * x_2 - n705280)
            - n1298688 * x_2.powi(3)
            + n163936 * x_2.powi(2)
            + x_1.powi(5) * (-n1327104 * x_2.powi(2) + n4079616 * x_2 + n1854464)
            + n5874 * x_2
            + x_1.powi(6) * (-n688128 * x_2 - n1490944)
            + n393216 * x_1.powi(7)
            - n1603))
        / (n10.ln()
            * (n192 * x_2.powi(4)
                + x_1 * (n768 * x_2.powi(3) - n2688 * x_2.powi(2) + n3024 * x_2 - n1080)
                - n896 * x_2.powi(3)
                + x_1.powi(2) * (n1152 * x_2.powi(2) - n2688 * x_2 + n1512)
                + n1512 * x_2.powi(2)
                + x_1.powi(3) * (n768 * x_2 - n896)
                - n1080 * x_2
                + n192 * x_1.powi(4)
                + n277)
                .powi(2)
            * (n31104 * x_2.powi(4) - n6912 * x_2.powi(3)
                + x_1 * (-n82944 * x_2.powi(3) + n13824 * x_2.powi(2) + n576 * x_2 - n96)
                + x_1.powi(2) * (n82944 * x_2.powi(2) - n9216 * x_2 - n192)
                - n432 * x_2.powi(2)
                + n144 * x_2
                + x_1.powi(3) * (n2048 - n36864 * x_2)
                + n6144 * x_1.powi(4)
                + n11))
        - (factor
            * (n124416 * x_2.powi(3) - n20736 * x_2.powi(2)
                + x_1 * (-n248832 * x_2.powi(2) + n27648 * x_2 + n576)
                + x_1.powi(2) * (n165888 * x_2 - n9216)
                - n864 * x_2
                - n36864 * x_1.powi(3)
                + n144)
            * (n331776 * x_2.powi(7) - n497664 * x_2.powi(6)
                + x_1
                    * (-n995328 * x_2.powi(6) + n5492736 * x_2.powi(5) - n7399680 * x_2.powi(4)
                        + n1441280 * x_2.powi(3)
                        + n1556736 * x_2.powi(2)
                        - n255808 * x_2
                        + n5594)
                - n1057536 * x_2.powi(5)
                + x_1.powi(2)
                    * (-n1880064 * x_2.powi(5) + n1136640 * x_2.powi(4) + n7728640 * x_2.powi(3)
                        - n6785280 * x_2.powi(2)
                        - n255264 * x_2
                        + n77456)
                + x_1.powi(3)
                    * (n1781760 * x_2.powi(4) - n9134080 * x_2.powi(3)
                        + n5040640 * x_2.powi(2)
                        + n4231680 * x_2
                        - n118688)
                + n2363520 * x_2.powi(4)
                + x_1.powi(4)
                    * (n2088960 * x_2.powi(3) + n1290240 * x_2.powi(2)
                        - n7175680 * x_2
                        - n705280)
                - n1298688 * x_2.powi(3)
                + n163936 * x_2.powi(2)
                + x_1.powi(5) * (-n1327104 * x_2.powi(2) + n4079616 * x_2 + n1854464)
                + n5874 * x_2
                + x_1.powi(6) * (-n688128 * x_2 - n1490944)
                + n393216 * x_1.powi(7)
                - n1603))
            / (n10.ln()
                * (n192 * x_2.powi(4)
                    + x_1 * (n768 * x_2.powi(3) - n2688 * x_2.powi(2) + n3024 * x_2 - n1080)
                    - n896 * x_2.powi(3)
                    + x_1.powi(2) * (n1152 * x_2.powi(2) - n2688 * x_2 + n1512)
                    + n1512 * x_2.powi(2)
                    + x_1.powi(3) * (n768 * x_2 - n896)
                    - n1080 * x_2
                    + n192 * x_1.powi(4)
                    + n277)
                * (n31104 * x_2.powi(4) - n6912 * x_2.powi(3)
                    + x_1 * (-n82944 * x_2.powi(3) + n13824 * x_2.powi(2) + n576 * x_2 - n96)
                    + x_1.powi(2) * (n82944 * x_2.powi(2) - n9216 * x_2 - n192)
                    - n432 * x_2.powi(2)
                    + n144 * x_2
                    + x_1.powi(3) * (n2048 - n36864 * x_2)
                    + n6144 * x_1.powi(4)
                    + n11)
                    .powi(2))
        + (factor
            * (n2322432 * x_2.powi(6) - n2985984 * x_2.powi(5)
                + x_1
                    * (-n5971968 * x_2.powi(5) + n27463680 * x_2.powi(4)
                        - n29598720 * x_2.powi(3)
                        + n4323840 * x_2.powi(2)
                        + n3113472 * x_2
                        - n255808)
                - n5287680 * x_2.powi(4)
                + x_1.powi(2)
                    * (-n9400320 * x_2.powi(4)
                        + n4546560 * x_2.powi(3)
                        + n23185920 * x_2.powi(2)
                        - n13570560 * x_2
                        - n255264)
                + x_1.powi(3)
                    * (n7127040 * x_2.powi(3) - n27402240 * x_2.powi(2)
                        + n10081280 * x_2
                        + n4231680)
                + n9454080 * x_2.powi(3)
                + x_1.powi(4) * (n6266880 * x_2.powi(2) + n2580480 * x_2 - n7175680)
                - n3896064 * x_2.powi(2)
                + n327872 * x_2
                + x_1.powi(5) * (n4079616 - n2654208 * x_2)
                - n688128 * x_1.powi(6)
                + n5874))
            / (n10.ln()
                * (n192 * x_2.powi(4)
                    + x_1 * (n768 * x_2.powi(3) - n2688 * x_2.powi(2) + n3024 * x_2 - n1080)
                    - n896 * x_2.powi(3)
                    + x_1.powi(2) * (n1152 * x_2.powi(2) - n2688 * x_2 + n1512)
                    + n1512 * x_2.powi(2)
                    + x_1.powi(3) * (n768 * x_2 - n896)
                    - n1080 * x_2
                    + n192 * x_1.powi(4)
                    + n277)
                * (n31104 * x_2.powi(4) - n6912 * x_2.powi(3)
                    + x_1 * (-n82944 * x_2.powi(3) + n13824 * x_2.powi(2) + n576 * x_2 - n96)
                    + x_1.powi(2) * (n82944 * x_2.powi(2) - n9216 * x_2 - n192)
                    - n432 * x_2.powi(2)
                    + n144 * x_2
                    + x_1.powi(3) * (n2048 - n36864 * x_2)
                    + n6144 * x_1.powi(4)
                    + n11));

    let b = -(factor
        * (n768 * x_2.powi(3)
            + n3 * (n768 * x_1 - n896) * x_2.powi(2)
            + n2 * (n1152 * x_1.powi(2) - n2688 * x_1 + n1512) * x_2
            + n768 * x_1.powi(3)
            - n2688 * x_1.powi(2)
            + n3024 * x_1
            - n1080)
        * (n1990656 * x_2.powi(7)
            + (n2322432 * x_1 - n8515584) * x_2.powi(6)
            + (-n2985984 * x_1.powi(2) - n2985984 * x_1 + n13284864) * x_2.powi(5)
            + (-n3133440 * x_1.powi(3) + n13731840 * x_1.powi(2) - n5287680 * x_1 - n9089280)
                * x_2.powi(4)
            + (n1781760 * x_1.powi(4) + n1515520 * x_1.powi(3) - n14799360 * x_1.powi(2)
                + n9454080 * x_1
                + n2550112)
                * x_2.powi(3)
            + (n1253376 * x_1.powi(5) - n6850560 * x_1.powi(4)
                + n7728640 * x_1.powi(3)
                + n2161920 * x_1.powi(2)
                - n3896064 * x_1
                - n155024)
                * x_2.powi(2)
            + (-n442368 * x_1.powi(6) + n516096 * x_1.powi(5) + n2520320 * x_1.powi(4)
                - n4523520 * x_1.powi(3)
                + n1556736 * x_1.powi(2)
                + n327872 * x_1
                - n21546)
                * x_2
            - n98304 * x_1.powi(7)
            + n679936 * x_1.powi(6)
            - n1435136 * x_1.powi(5)
            + n1057920 * x_1.powi(4)
            - n85088 * x_1.powi(3)
            - n127904 * x_1.powi(2)
            + n5874 * x_1
            + n1167))
        / (n10.ln()
            * (n192 * x_2.powi(4)
                + (n768 * x_1 - n896) * x_2.powi(3)
                + (n1152 * x_1.powi(2) - n2688 * x_1 + n1512) * x_2.powi(2)
                + (n768 * x_1.powi(3) - n2688 * x_1.powi(2) + n3024 * x_1 - n1080) * x_2
                + n192 * x_1.powi(4)
                - n896 * x_1.powi(3)
                + n1512 * x_1.powi(2)
                - n1080 * x_1
                + n277)
                .powi(2)
            * (n31104 * x_2.powi(4)
                + (-n82944 * x_1 - n6912) * x_2.powi(3)
                + (n82944 * x_1.powi(2) + n13824 * x_1 - n432) * x_2.powi(2)
                + (-n36864 * x_1.powi(3) - n9216 * x_1.powi(2) + n576 * x_1 + n144) * x_2
                + n6144 * x_1.powi(4)
                + n2048 * x_1.powi(3)
                - n192 * x_1.powi(2)
                - n96 * x_1
                + n11))
        - (factor
            * (n124416 * x_2.powi(3)
                + n3 * (-n82944 * x_1 - n6912) * x_2.powi(2)
                + n2 * (n82944 * x_1.powi(2) + n13824 * x_1 - n432) * x_2
                - n36864 * x_1.powi(3)
                - n9216 * x_1.powi(2)
                + n576 * x_1
                + n144)
            * (n1990656 * x_2.powi(7)
                + (n2322432 * x_1 - n8515584) * x_2.powi(6)
                + (-n2985984 * x_1.powi(2) - n2985984 * x_1 + n13284864) * x_2.powi(5)
                + (-n3133440 * x_1.powi(3) + n13731840 * x_1.powi(2) - n5287680 * x_1 - n9089280)
                    * x_2.powi(4)
                + (n1781760 * x_1.powi(4) + n1515520 * x_1.powi(3) - n14799360 * x_1.powi(2)
                    + n9454080 * x_1
                    + n2550112)
                    * x_2.powi(3)
                + (n1253376 * x_1.powi(5) - n6850560 * x_1.powi(4)
                    + n7728640 * x_1.powi(3)
                    + n2161920 * x_1.powi(2)
                    - n3896064 * x_1
                    - n155024)
                    * x_2.powi(2)
                + (-n442368 * x_1.powi(6) + n516096 * x_1.powi(5) + n2520320 * x_1.powi(4)
                    - n4523520 * x_1.powi(3)
                    + n1556736 * x_1.powi(2)
                    + n327872 * x_1
                    - n21546)
                    * x_2
                - n98304 * x_1.powi(7)
                + n679936 * x_1.powi(6)
                - n1435136 * x_1.powi(5)
                + n1057920 * x_1.powi(4)
                - n85088 * x_1.powi(3)
                - n127904 * x_1.powi(2)
                + n5874 * x_1
                + n1167))
            / (n10.ln()
                * (n192 * x_2.powi(4)
                    + (n768 * x_1 - n896) * x_2.powi(3)
                    + (n1152 * x_1.powi(2) - n2688 * x_1 + n1512) * x_2.powi(2)
                    + (n768 * x_1.powi(3) - n2688 * x_1.powi(2) + n3024 * x_1 - n1080) * x_2
                    + n192 * x_1.powi(4)
                    - n896 * x_1.powi(3)
                    + n1512 * x_1.powi(2)
                    - n1080 * x_1
                    + n277)
                * (n31104 * x_2.powi(4)
                    + (-n82944 * x_1 - n6912) * x_2.powi(3)
                    + (n82944 * x_1.powi(2) + n13824 * x_1 - n432) * x_2.powi(2)
                    + (-n36864 * x_1.powi(3) - n9216 * x_1.powi(2) + n576 * x_1 + n144) * x_2
                    + n6144 * x_1.powi(4)
                    + n2048 * x_1.powi(3)
                    - n192 * x_1.powi(2)
                    - n96 * x_1
                    + n11)
                    .powi(2))
        + (factor
            * (n13934592 * x_2.powi(6)
                + n6 * (n2322432 * x_1 - n8515584) * x_2.powi(5)
                + n5 * (-n2985984 * x_1.powi(2) - n2985984 * x_1 + n13284864) * x_2.powi(4)
                + n4 * (-n3133440 * x_1.powi(3) + n13731840 * x_1.powi(2)
                    - n5287680 * x_1
                    - n9089280)
                    * x_2.powi(3)
                + n3 * (n1781760 * x_1.powi(4) + n1515520 * x_1.powi(3)
                    - n14799360 * x_1.powi(2)
                    + n9454080 * x_1
                    + n2550112)
                    * x_2.powi(2)
                + n2 * (n1253376 * x_1.powi(5) - n6850560 * x_1.powi(4)
                    + n7728640 * x_1.powi(3)
                    + n2161920 * x_1.powi(2)
                    - n3896064 * x_1
                    - n155024)
                    * x_2
                - n442368 * x_1.powi(6)
                + n516096 * x_1.powi(5)
                + n2520320 * x_1.powi(4)
                - n4523520 * x_1.powi(3)
                + n1556736 * x_1.powi(2)
                + n327872 * x_1
                - n21546))
            / (n10.ln()
                * (n192 * x_2.powi(4)
                    + (n768 * x_1 - n896) * x_2.powi(3)
                    + (n1152 * x_1.powi(2) - n2688 * x_1 + n1512) * x_2.powi(2)
                    + (n768 * x_1.powi(3) - n2688 * x_1.powi(2) + n3024 * x_1 - n1080) * x_2
                    + n192 * x_1.powi(4)
                    - n896 * x_1.powi(3)
                    + n1512 * x_1.powi(2)
                    - n1080 * x_1
                    + n277)
                * (n31104 * x_2.powi(4)
                    + (-n82944 * x_1 - n6912) * x_2.powi(3)
                    + (n82944 * x_1.powi(2) + n13824 * x_1 - n432) * x_2.powi(2)
                    + (-n36864 * x_1.powi(3) - n9216 * x_1.powi(2) + n576 * x_1 + n144) * x_2
                    + n6144 * x_1.powi(4)
                    + n2048 * x_1.powi(3)
                    - n192 * x_1.powi(2)
                    - n96 * x_1
                    + n11));

    let a = -(factor
        * (n768 * x_1.powi(3)
            + n3 * (n768 * x_2 - n896) * x_1.powi(2)
            + n2 * (n1152 * x_2.powi(2) - n2688 * x_2 + n1512) * x_1
            + n768 * x_2.powi(3)
            - n2688 * x_2.powi(2)
            + n3024 * x_2
            - n1080)
        * (n393216 * x_1.powi(7)
            + (-n688128 * x_2 - n1490944) * x_1.powi(6)
            + (-n1327104 * x_2.powi(2) + n4079616 * x_2 + n1854464) * x_1.powi(5)
            + (n2088960 * x_2.powi(3) + n1290240 * x_2.powi(2) - n7175680 * x_2 - n705280)
                * x_1.powi(4)
            + (n1781760 * x_2.powi(4) - n9134080 * x_2.powi(3)
                + n5040640 * x_2.powi(2)
                + n4231680 * x_2
                - n118688)
                * x_1.powi(3)
            + (-n1880064 * x_2.powi(5) + n1136640 * x_2.powi(4) + n7728640 * x_2.powi(3)
                - n6785280 * x_2.powi(2)
                - n255264 * x_2
                + n77456)
                * x_1.powi(2)
            + (-n995328 * x_2.powi(6) + n5492736 * x_2.powi(5) - n7399680 * x_2.powi(4)
                + n1441280 * x_2.powi(3)
                + n1556736 * x_2.powi(2)
                - n255808 * x_2
                + n5594)
                * x_1
            + n331776 * x_2.powi(7)
            - n497664 * x_2.powi(6)
            - n1057536 * x_2.powi(5)
            + n2363520 * x_2.powi(4)
            - n1298688 * x_2.powi(3)
            + n163936 * x_2.powi(2)
            + n5874 * x_2
            - n1603))
        / (n10.ln()
            * (n192 * x_1.powi(4)
                + (n768 * x_2 - n896) * x_1.powi(3)
                + (n1152 * x_2.powi(2) - n2688 * x_2 + n1512) * x_1.powi(2)
                + (n768 * x_2.powi(3) - n2688 * x_2.powi(2) + n3024 * x_2 - n1080) * x_1
                + n192 * x_2.powi(4)
                - n896 * x_2.powi(3)
                + n1512 * x_2.powi(2)
                - n1080 * x_2
                + n277)
                .powi(2)
            * (n6144 * x_1.powi(4)
                + (n2048 - n36864 * x_2) * x_1.powi(3)
                + (n82944 * x_2.powi(2) - n9216 * x_2 - n192) * x_1.powi(2)
                + (-n82944 * x_2.powi(3) + n13824 * x_2.powi(2) + n576 * x_2 - n96) * x_1
                + n31104 * x_2.powi(4)
                - n6912 * x_2.powi(3)
                - n432 * x_2.powi(2)
                + n144 * x_2
                + n11))
        - (factor
            * (n24576 * x_1.powi(3)
                + n3 * (n2048 - n36864 * x_2) * x_1.powi(2)
                + n2 * (n82944 * x_2.powi(2) - n9216 * x_2 - n192) * x_1
                - n82944 * x_2.powi(3)
                + n13824 * x_2.powi(2)
                + n576 * x_2
                - n96)
            * (n393216 * x_1.powi(7)
                + (-n688128 * x_2 - n1490944) * x_1.powi(6)
                + (-n1327104 * x_2.powi(2) + n4079616 * x_2 + n1854464) * x_1.powi(5)
                + (n2088960 * x_2.powi(3) + n1290240 * x_2.powi(2) - n7175680 * x_2 - n705280)
                    * x_1.powi(4)
                + (n1781760 * x_2.powi(4) - n9134080 * x_2.powi(3)
                    + n5040640 * x_2.powi(2)
                    + n4231680 * x_2
                    - n118688)
                    * x_1.powi(3)
                + (-n1880064 * x_2.powi(5) + n1136640 * x_2.powi(4) + n7728640 * x_2.powi(3)
                    - n6785280 * x_2.powi(2)
                    - n255264 * x_2
                    + n77456)
                    * x_1.powi(2)
                + (-n995328 * x_2.powi(6) + n5492736 * x_2.powi(5) - n7399680 * x_2.powi(4)
                    + n1441280 * x_2.powi(3)
                    + n1556736 * x_2.powi(2)
                    - n255808 * x_2
                    + n5594)
                    * x_1
                + n331776 * x_2.powi(7)
                - n497664 * x_2.powi(6)
                - n1057536 * x_2.powi(5)
                + n2363520 * x_2.powi(4)
                - n1298688 * x_2.powi(3)
                + n163936 * x_2.powi(2)
                + n5874 * x_2
                - n1603))
            / (n10.ln()
                * (n192 * x_1.powi(4)
                    + (n768 * x_2 - n896) * x_1.powi(3)
                    + (n1152 * x_2.powi(2) - n2688 * x_2 + n1512) * x_1.powi(2)
                    + (n768 * x_2.powi(3) - n2688 * x_2.powi(2) + n3024 * x_2 - n1080) * x_1
                    + n192 * x_2.powi(4)
                    - n896 * x_2.powi(3)
                    + n1512 * x_2.powi(2)
                    - n1080 * x_2
                    + n277)
                * (n6144 * x_1.powi(4)
                    + (n2048 - n36864 * x_2) * x_1.powi(3)
                    + (n82944 * x_2.powi(2) - n9216 * x_2 - n192) * x_1.powi(2)
                    + (-n82944 * x_2.powi(3) + n13824 * x_2.powi(2) + n576 * x_2 - n96) * x_1
                    + n31104 * x_2.powi(4)
                    - n6912 * x_2.powi(3)
                    - n432 * x_2.powi(2)
                    + n144 * x_2
                    + n11)
                    .powi(2))
        + (factor
            * (n2752512 * x_1.powi(6)
                + n6 * (-n688128 * x_2 - n1490944) * x_1.powi(5)
                + n5 * (-n1327104 * x_2.powi(2) + n4079616 * x_2 + n1854464) * x_1.powi(4)
                + n4 * (n2088960 * x_2.powi(3) + n1290240 * x_2.powi(2)
                    - n7175680 * x_2
                    - n705280)
                    * x_1.powi(3)
                + n3 * (n1781760 * x_2.powi(4) - n9134080 * x_2.powi(3)
                    + n5040640 * x_2.powi(2)
                    + n4231680 * x_2
                    - n118688)
                    * x_1.powi(2)
                + n2 * (-n1880064 * x_2.powi(5)
                    + n1136640 * x_2.powi(4)
                    + n7728640 * x_2.powi(3)
                    - n6785280 * x_2.powi(2)
                    - n255264 * x_2
                    + n77456)
                    * x_1
                - n995328 * x_2.powi(6)
                + n5492736 * x_2.powi(5)
                - n7399680 * x_2.powi(4)
                + n1441280 * x_2.powi(3)
                + n1556736 * x_2.powi(2)
                - n255808 * x_2
                + n5594))
            / (n10.ln()
                * (n192 * x_1.powi(4)
                    + (n768 * x_2 - n896) * x_1.powi(3)
                    + (n1152 * x_2.powi(2) - n2688 * x_2 + n1512) * x_1.powi(2)
                    + (n768 * x_2.powi(3) - n2688 * x_2.powi(2) + n3024 * x_2 - n1080) * x_1
                    + n192 * x_2.powi(4)
                    - n896 * x_2.powi(3)
                    + n1512 * x_2.powi(2)
                    - n1080 * x_2
                    + n277)
                * (n6144 * x_1.powi(4)
                    + (n2048 - n36864 * x_2) * x_1.powi(3)
                    + (n82944 * x_2.powi(2) - n9216 * x_2 - n192) * x_1.powi(2)
                    + (-n82944 * x_2.powi(3) + n13824 * x_2.powi(2) + n576 * x_2 - n96) * x_1
                    + n31104 * x_2.powi(4)
                    - n6912 * x_2.powi(3)
                    - n432 * x_2.powi(2)
                    + n144 * x_2
                    + n11));

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
    fn test_picheny_optimum() {
        assert_relative_eq!(
            picheny(&[0.5_f32, 0.25_f32]),
            -3.3851993182,
            epsilon = f32::EPSILON
        );
        assert_relative_eq!(
            picheny(&[0.5_f64, 0.25_f64]),
            -3.3851993182036826,
            epsilon = f64::EPSILON
        );

        let deriv = picheny_derivative(&[0.5_f64, 0.25_f64]);
        // println!("1: {deriv:?}");
        for i in 0..2 {
            assert_relative_eq!(deriv[i], 0.0, epsilon = f64::EPSILON);
        }
    }

    proptest! {
        #[test]
        fn test_picheny_derivative(a in 0.0..1.0, b in 0.0..1.0) {
            let param = [a, b];
            let derivative = picheny_derivative(&param);
            let derivative_fd = Vec::from(param).central_diff(&|x| picheny(&[x[0], x[1]]));
            // println!("1: {derivative:?} at {a}/{b}");
            // println!("2: {derivative_fd:?} at {a}/{b}");
            for i in 0..derivative.len() {
                assert_relative_eq!(
                    derivative[i],
                    derivative_fd[i],
                    epsilon = 1e-5,
                    max_relative = 1e-2
                );
            }
        }
    }

    proptest! {
        #[test]
        fn test_picheny_hessian_finitediff(a in 0.0..1.0, b in 0.0..1.0) {
            let param = [a, b];
            let hessian = picheny_hessian(&param);
            let hessian_fd =
                Vec::from(param).central_hessian(&|x| picheny_derivative(&[x[0], x[1]]).to_vec());
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
