#![allow(unused_imports)]
extern crate argmin;
extern crate ndarray;
use ndarray::{Array1, Array2};
use ndarray::prelude::*;
use argmin::problem::Problem;
use argmin::newton::Newton;
use argmin::testfunctions::{rosenbrock, rosenbrock_derivative, rosenbrock_derivative_nd,
                            rosenbrock_hessian, rosenbrock_hessian_nd, rosenbrock_nd, sphere,
                            sphere_derivative};

fn run() -> Result<(), Box<std::error::Error>> {
    // Define cost function
    // Choose either `Rosenbrock` or `Sphere` function.
    // let cost = |x: &Vec<f64>| -> f64 { rosenbrock(x, 1_f64, 100_f64) };
    // let gradient = |x: &Vec<f64>| -> Vec<f64> { rosenbrock_derivative(x, 1_f64, 100_f64) };
    // let hessian = |x: &Vec<f64>| -> Vec<f64> { rosenbrock_hessian(x, 1_f64, 100_f64) };
    // using ndarray
    let cost = |x: &Array1<f64>| -> f64 { rosenbrock_nd(x, 1_f64, 100_f64) };
    let gradient =
        |x: &Array1<f64>| -> Array<f64, _> { rosenbrock_derivative_nd(x, 1_f64, 100_f64) };
    let hessian = |x: &Array1<f64>| -> Array<f64, _> { rosenbrock_hessian_nd(x, 1_f64, 100_f64) };
    // let cost = |x: &Vec<f64>| -> f64 { sphere(x) };
    // let gradient = |x: &Vec<f64>| -> Vec<f64> { sphere_derivative(x) };

    // Define bounds
    // Note: Gradient Descent currently does not enforce these bounds which is why we can set them
    // to -1000 and +1000
    let lower_bound: Array1<f64> = Array1::from_vec(vec![-1000.0, -1000.0]);
    let upper_bound: Array1<f64> = Array1::from_vec(vec![1000.0, 1000.0]);
    // Unfortunately, setting them to +/- Infinity does not work (yet)
    // let lower_bound: Vec<f64> = vec![std::f64::NEG_INFINITY, std::f64::NEG_INFINITY];
    // let upper_bound: Vec<f64> = vec![std::f64::INFINITY, std::f64::INFINITY];

    // Set up problem
    // The problem requires a cost function, lower and upper bounds and takes an optional gradient.
    let mut prob = Problem::new(&cost, &lower_bound, &upper_bound);
    prob.gradient(&gradient);
    prob.hessian(&hessian);

    // Set up Newton solver
    let mut solver = Newton::new();

    // define inital parameter vector
    // `Problem` allows to create random parameter vectors which satisfies `lower_bound` and
    // `upper_bound`.
    let init_param: Array1<f64> = prob.random_param()?;
    // let init_param: Array1<f64> = Array1::from_vec(vec![1.5, 1.5]);
    println!("{:?}", init_param);

    // Manually solve it
    solver.init(&prob, &init_param)?;

    let mut par;
    loop {
        par = solver.next_iter()?;
        println!("{:?}", par);
        if par.iters >= 10 {
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
