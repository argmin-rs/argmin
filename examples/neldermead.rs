#![allow(unused_imports)]
extern crate argmin;
use argmin::problem::Problem;
use argmin::neldermead::NelderMead;
use argmin::testfunctions::{rosenbrock, rosenbrock_derivative, sphere, sphere_derivative};

fn run() -> Result<(), Box<std::error::Error>> {
    // Define cost function
    let cost = |x: &Vec<f64>| -> f64 { rosenbrock(x, 1_f64, 100_f64).unwrap() };
    // let gradient = |x: &Vec<f64>| -> Vec<f64> { rosenbrock_derivative(x, 1_f64, 100_f64).unwrap() };
    // let cost = |x: &Vec<f64>| -> f64 { sphere(x).unwrap() };
    // let gradient = |x: &Vec<f64>| -> Vec<f64> { sphere_derivative(x).unwrap() };

    // Define bounds
    let lower_bound: Vec<f64> = vec![-1.5, -0.5];
    let upper_bound: Vec<f64> = vec![2.0, 3.0];

    // Set up problem
    let mut prob = Problem::new(&cost, &lower_bound, &upper_bound);
    // prob.gradient(&gradient);

    // Set up GradientDecent solver
    let mut solver = NelderMead::new();
    solver.max_iters(10_000);

    // solver.init(&prob)?;
    let result = solver.run(&prob)?;

    // print result
    println!("{:?}", result);
    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("error: {}", e);
    }
}
