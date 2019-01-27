// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

extern crate argmin;
use argmin::prelude::*;
use argmin::solver::particleswarm::*;

use argmin_testfunctions::himmelblau;


struct PhonyOperator
{

}


impl ArgminOperator for PhonyOperator {
    type Parameters = Vec<f64>;
    type OperatorOutput = f64;
    type Hessian = ();

    fn apply(&self, param: &Self::Parameters) -> Result<Self::OperatorOutput, Error> {
        Ok(himmelblau(param))
    }
}





fn run() -> Result<(), Error> {
    // Define inital parameter vector
    let init_param: Vec<f64> = vec![0.1, 0.1];

    let cost_function = PhonyOperator {};

    // Set up line search method
    let mut solver = ParticleSwarm::new(&cost_function, init_param)?;

    // Attach a logger
    solver.add_logger(ArgminSlogLogger::term());

    solver.set_max_iters(10);

    // Run solver
    solver.run()?;

    // Wait a second (lets the logger flush everything before printing again)
    std::thread::sleep(std::time::Duration::from_secs(1));

    // Print Result
    println!("{:?}", solver.result());
    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("{} {}", e.as_fail(), e.backtrace());
    }
}
