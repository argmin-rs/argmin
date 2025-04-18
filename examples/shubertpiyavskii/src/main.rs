// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use argmin::{
    core::{observers::ObserverMode, CostFunction, Error, Executor},
    solver::shubertpiyavskii::ShubertPiyavskii,
};
use argmin_observer_slog::SlogLogger;
use std::f64::consts::PI;

const MIN_BOUND: f64 = 0.0;
const MAX_BOUND: f64 = 30.0;
// Random amplitudes sampled from Uniform(-0.1, 0.1)
const EPSILONS: [f64; 5] = [-0.02509198, 0.09014286, 0.04639879, 0.0197317, -0.06879627];
// Random frequency shifts sampled from Uniform(0.9, 1.1)
const DELTAS: [f64; 5] = [0.9311989, 0.91161672, 1.07323523, 1.020223, 1.04161452];
const MAX_ITER: u64 = 500;

struct SuharevZilinskasNoisy {
    epsilons: [f64; 5],
    deltas: [f64; 5],
}

impl SuharevZilinskasNoisy {
    fn new(epsilons: [f64; 5], deltas: [f64; 5]) -> Self {
        SuharevZilinskasNoisy { epsilons, deltas }
    }

    /// Compute a tight upper bound on the function's Lipschitz constant.
    ///
    /// The original noise-free Suharev-Zilinskas' function has a Lipschitz constant of precisely
    /// 70. The contribution of random noise is bounded above by Σ_{k=1}^5 2π|εₖ||δₖ|.
    fn lipschitz_const(&self) -> f64 {
        70.0 + self
            .epsilons
            .iter()
            .zip(self.deltas.iter())
            .map(|(&epsilon, &delta)| 2.0 * PI * epsilon.abs() * delta.abs())
            .sum::<f64>()
    }
}

impl CostFunction for SuharevZilinskasNoisy {
    // One-dimensional problem; no vector needed
    type Param = f64;
    type Output = f64;

    /// Suharev-Zilinskas' function, with random noise added.
    ///
    /// Given by f(x) ≔ Σ_{k=1}^5 [k * sin((k + 1)x + k) + εₖ * sin(2πδₖx)], where the εₖ's are
    /// random amplitudes and the δₖ's are random frequency shifts.
    ///
    /// The function is highly multimodal. When εₖ = δₖ = 0 for all k (the original noise-free
    /// function), local minima occur approximately every 0.975 to 0.995 units, and global minima
    /// repeat precisely every 2π units. The addition of random noise makes the function even more
    /// challenging to optimize; with generic random εₖ's and δₖ's, we almost certainly end up with
    /// only one global minimum with many incredibly close local minima,as each trough is shifted by
    /// a different amount.
    ///
    /// Finding the true unique global minimum (or an arbitrarily close approximation) despite the
    /// addition of noise is a good stress test for the Shubert-Piyavskii method.
    fn cost(&self, x: &Self::Param) -> Result<Self::Output, Error> {
        let sum = self
            .epsilons
            .iter()
            .zip(self.deltas.iter())
            .enumerate()
            .map(|(k, (&epsilon, &delta))| {
                let k = (k + 1) as f64;
                let term = k * ((k + 1.0) * x + k).sin(); // Original term
                let noise = epsilon * (2.0 * PI * delta * x).sin(); // Added noise
                term + noise
            })
            .sum();

        Ok(sum)
    }
}

/// Test the Shubert-Piyavskii method on Suharev-Zilinskas' function with random noise. Given our
/// predefined `EPSILONS` and `DELTAS`, there should be 29 local minima over the interval [0, 30],
/// with the most promising candidates being:
/// - x ≈ 5.16954, f(x) ≈ -14.9424
/// - x ≈ 11.45501, f(x) ≈ -14.728
/// - x ≈ 17.73181, f(x) ≈ -14.74988
/// - x ≈ 24.01868, f(x) ≈ -14.96562 (the true global minimum)
///
/// Given our default tolerance of 0.01, the solver should correctly identify a point close to the
/// true minimum (assuming convergence occurs before the maximum number of iterations is reached).
fn main() -> Result<(), Error> {
    let cost = SuharevZilinskasNoisy::new(EPSILONS, DELTAS);
    let lipschitz_const = cost.lipschitz_const();
    // let solver = ShubertPiyavskii::new(MIN_BOUND, MAX_BOUND, lipschitz_const)?;
    let solver = ShubertPiyavskii::new(MIN_BOUND, MAX_BOUND, lipschitz_const)?;

    let res = Executor::new(cost, solver)
        .configure(|state| state.max_iters(MAX_ITER))
        .add_observer(SlogLogger::term(), ObserverMode::Always)
        .run()
        .unwrap();

    println!("Result of Shubert-Piyavskii method:\n{res}");
    Ok(())
}
