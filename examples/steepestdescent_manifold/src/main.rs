// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#![allow(unused_imports)]

use argmin::core::observers::ObserverMode;
use argmin::core::{CostFunction, Error, Executor, Gradient};
use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::linesearch::condition::{ArmijoCondition, LineSearchCondition};
use argmin::solver::linesearch::BacktrackingLineSearch;
use argmin_math::ArgminScaledAdd;
use argmin_observer_slog::SlogLogger;

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug)]
struct ClosestPointOnCircle {
    x: f64,
    y: f64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct CirclePoint {
    angle: f64,
}

impl CostFunction for ClosestPointOnCircle {
    type Param = CirclePoint;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let x_circ = p.angle.cos();
        let y_circ = p.angle.sin();
        let x_diff = x_circ - self.x;
        let y_diff = y_circ - self.y;
        Ok(x_diff.powi(2) + y_diff.powi(2))
    }
}

impl Gradient for ClosestPointOnCircle {
    type Param = CirclePoint;
    type Gradient = f64;

    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
        Ok(2.0 * (p.angle.cos() - self.x) * (-p.angle.sin())
            + 2.0 * (p.angle.sin() - self.y) * p.angle.cos())
    }
}

impl ArgminScaledAdd<f64, f64, CirclePoint> for CirclePoint {
    fn scaled_add(&self, alpha: &f64, delta: &f64) -> Self {
        CirclePoint {
            angle: self.angle + alpha * delta,
        }
    }
}

fn run() -> Result<(), Error> {
    // Define cost function (must implement `CostFunction` and `Gradient`)
    let cost = ClosestPointOnCircle { x: 1.0, y: 1.0 };

    // Define initial parameter vector
    let init_param = CirclePoint { angle: 0.0 };

    // Pick a line search.
    let cond = ArmijoCondition::new(0.5)?;
    let linesearch = BacktrackingLineSearch::new(cond);

    // Set up solver
    let solver = SteepestDescent::new(linesearch);

    // Run solver
    let res = Executor::new(cost, solver)
        .configure(|state| state.param(init_param).max_iters(10))
        .add_observer(SlogLogger::term(), ObserverMode::Always)
        .run()?;

    // Wait a second (lets the logger flush everything first)
    std::thread::sleep(std::time::Duration::from_secs(1));

    // print result
    println!("{res}");
    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("{e}");
        std::process::exit(1);
    }
}
