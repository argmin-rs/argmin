extern crate argmin;
use argmin::problem::Problem;
use argmin::gradientdescent::GradientDescent;
use argmin::testfunctions::{rosenbrock, rosenbrock_derivative};

fn run() -> Result<(), Box<std::error::Error>> {
    // Define cost function
    let cost = |x: &Vec<f64>| -> f64 { rosenbrock(x, 1_f64, 100_f64).unwrap() };
    let gradient = |x: &Vec<f64>| -> Vec<f64> { rosenbrock_derivative(x, 1_f64, 100_f64).unwrap() };

    // Define bounds
    let lower_bound: Vec<f64> = vec![-1.5, -0.5];
    let upper_bound: Vec<f64> = vec![2.0, 3.0];

    // Set up problem
    let mut prob = Problem::new(&cost, &lower_bound, &upper_bound);
    prob.gradient(&gradient);

    // Set up GradientDecent solver
    let mut solver = GradientDescent::new();
    solver.max_iters(10);

    // definie inital parameter vector
    // let init_param: Vec<f64> = vec![-1.0, 2.0];

    let result = solver.run(&prob, &prob.random_param()?)?;

    // print result
    println!("{:?}", result);
    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("error: {}", e);
    }
}
