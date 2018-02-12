// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

extern crate argmin;
extern crate ndarray;
use ndarray::Array1;
use argmin::{ArgminProblem, BacktrackingLineSearch, GDGammaUpdate, GradientDescent, MultiStart};
use argmin::testfunctions::{rosenbrock_derivative_nd, rosenbrock_nd};

fn run() -> Result<(), Box<std::error::Error>> {
    let cost = |x: &Array1<f64>| -> f64 { rosenbrock_nd(x, 1_f64, 100_f64) };
    let gradient = |x: &Array1<f64>| -> Array1<f64> { rosenbrock_derivative_nd(x, 1_f64, 100_f64) };

    let mut prob: ArgminProblem<_, _, ()> = ArgminProblem::new(&cost);
    prob.gradient(&gradient);

    let mut solver1 = GradientDescent::new();
    solver1.max_iters(10_000);
    solver1.gamma_update(GDGammaUpdate::BarzilaiBorwein);

    let mut solver2 = GradientDescent::new();
    solver2.max_iters(10_000);
    solver2.gamma_update(GDGammaUpdate::Constant(0.0001));

    let mut solver3 = GradientDescent::new();
    solver3.max_iters(10_000);
    let mut linesearch = BacktrackingLineSearch::new(&cost, &gradient);
    linesearch.alpha(1.0);
    solver3.gamma_update(GDGammaUpdate::BacktrackingLineSearch(linesearch));

    // define inital parameter vector
    let init_param1: Array1<f64> = Array1::from_vec(vec![1.5, 1.5]);
    let init_param2: Array1<f64> = Array1::from_vec(vec![1.5, 1.5]);
    let init_param3: Array1<f64> = Array1::from_vec(vec![1.5, 1.5]);

    let mut multi_start = MultiStart::new();
    multi_start
        .push(solver1, &prob, init_param1)
        .push(solver2, &prob, init_param2)
        .push(solver3, &prob, init_param3);

    let res = multi_start.run();

    for r in res {
        println!("{:?}", r);
    }

    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("error: {}", e);
    }
}
