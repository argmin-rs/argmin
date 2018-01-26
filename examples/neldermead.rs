#![allow(unused_imports)]
extern crate argmin;
use argmin::ArgminSolver;
use argmin::problem::Problem;
use argmin::neldermead::NelderMead;
use argmin::testfunctions::{rosenbrock, rosenbrock_derivative, sphere, sphere_derivative};

fn run() -> Result<(), Box<std::error::Error>> {
    // Define cost function
    let cost = |x: &Vec<f64>| -> f64 { rosenbrock(x, 1_f64, 100_f64) };
    // let cost = |x: &Vec<f64>| -> f64 { sphere(x) };

    // Define bounds
    // Bounds are not satisfied yet.
    let lower_bound: Vec<f64> = vec![-1.5, -0.5];
    let upper_bound: Vec<f64> = vec![2.0, 3.0];

    // Set up problem
    let prob = Problem::new(&cost, &lower_bound, &upper_bound);

    // Set up GradientDecent solver
    let mut solver = NelderMead::new();
    solver.max_iters(100);

    // Choose the starting points.
    // Randomly chosen starting points may make Nelder Mead converge very slowly.
    // let init_params = vec![
    //     prob.random_param()?,
    //     prob.random_param()?,
    //     prob.random_param()?,
    // ];
    let init_params = vec![vec![0.0, 0.1], vec![2.0, 1.5], vec![2.0, -1.0]];

    let result = solver.run(&prob, &init_params)?;

    // print result
    println!("{:?}", result);
    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("error: {}", e);
    }
}
