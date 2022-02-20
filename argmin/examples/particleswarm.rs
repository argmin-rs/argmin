// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use argmin::core::{ArgminOp, CostFunction, Error, Executor};
#[cfg(feature = "visualizer")]
use argmin::core::{ObserverMode, Surface, Visualizer3d};
use argmin::solver::particleswarm::ParticleSwarm;
use argmin_testfunctions::himmelblau;

struct Himmelblau {}

impl ArgminOp for Himmelblau {
    type Param = Vec<f64>;
    type Output = f64;
    type Hessian = ();
    type Jacobian = ();
    type Float = f64;
}

impl CostFunction for Himmelblau {
    type Param = Vec<f64>;
    type Output = f64;
    type Float = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        Ok(himmelblau(param))
    }
}

fn run() -> Result<(), Error> {
    // Define inital parameter vector
    let init_param: Vec<f64> = vec![0.1, 0.1];

    let cost_function = Himmelblau {};

    #[cfg(feature = "visualizer")]
    let visualizer = Visualizer3d::new()
        .delay(std::time::Duration::from_secs(1))
        .surface(Surface::new(Himmelblau {}, (-4.0, -4.0, 4.0, 4.0), 0.1));

    {
        let solver = ParticleSwarm::new((vec![-4.0, -4.0], vec![4.0, 4.0]), 100, 0.5, 0.0, 0.5)?;

        let executor = Executor::new(cost_function, solver)
            .configure(|config| config.param(init_param).max_iters(15));

        #[cfg(feature = "visualizer")]
        let executor = executor.add_observer(visualizer, ObserverMode::Always);

        let res = executor.run()?;

        // Wait a second (lets the logger flush everything before printing again)
        std::thread::sleep(std::time::Duration::from_secs(1));

        // Print Result
        println!("{}", res);
    }

    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("{}", e);
    }
}
