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
    let prob = Problem::new(&cost, &lower_bound, &upper_bound);

    // Set up GradientDecent solver
    let mut solver = NelderMead::new();
    solver.max_iters(10_000);

    // let init_params = vec![
    //     prob.random_param()?,
    //     prob.random_param()?,
    //     prob.random_param()?,
    // ];
    let init_params = vec![vec![0.0, 0.1], vec![2.0, 1.5], vec![2.0, -1.0]];

    // solver.init(&prob)?;
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
