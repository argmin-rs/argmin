#![allow(unused_imports)]
extern crate argmin;
use argmin::problem::Problem;
use argmin::backtracking::BacktrackingLineSearch;
use argmin::testfunctions::{rosenbrock, rosenbrock_derivative, sphere, sphere_derivative};

fn run() -> Result<(), Box<std::error::Error>> {
    // Define cost function
    let cost = |x: &Vec<f64>| -> f64 { rosenbrock(x, 1_f64, 100_f64) };
    let gradient = |x: &Vec<f64>| -> Vec<f64> { rosenbrock_derivative(x, 1_f64, 100_f64) };
    // let cost = |x: &Vec<f64>| -> f64 { sphere(x) };
    // let gradient = |x: &Vec<f64>| -> Vec<f64> { sphere_derivative(x) };

    // Set up GradientDecent solver
    let solver = BacktrackingLineSearch::new(&cost, &gradient);
    // solver.max_iters(10_000);

    let x = vec![4.1, 3.0];
    let p = gradient(&x);

    let result = solver.run(&(p.iter().map(|x| -x).collect::<Vec<f64>>()), &x)?;

    // print result
    println!("{:?}", result);
    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("error: {}", e);
    }
}
