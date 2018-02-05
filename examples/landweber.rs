// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#![allow(non_snake_case)]
extern crate argmin;
extern crate ndarray;
use ndarray::{arr1, arr2};
use argmin::prelude::*;
use argmin::{ArgminOperator, Landweber};

fn run() -> Result<(), Box<std::error::Error>> {
    // Set up problem
    let A = arr2(&[[4., 1.], [1., 3.]]);
    let y = arr1(&[1., 2.]);
    let mut prob = ArgminOperator::new(&A, &y);
    prob.target_cost(0.01);

    // Set up Newton solver
    let mut solver = Landweber::new(0.01);

    // Initialize the solver
    let init_param = arr1(&[0., 0.]);
    solver.init(&prob, &init_param)?;

    let mut par;
    loop {
        par = solver.next_iter()?;
        // println!("{:?}", par);
        if par.terminated {
            break;
        };
    }

    println!("{:?}", par);

    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("error: {}", e);
    }
}
