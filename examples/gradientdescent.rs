#![allow(unused_imports)]
extern crate argmin;
use argmin::problem::Problem;
use argmin::gradientdescent::{GDGammaUpdate, GradientDescent};
use argmin::testfunctions::{rosenbrock, rosenbrock_derivative, sphere, sphere_derivative};
use argmin::backtracking::BacktrackingLineSearch;

fn run() -> Result<(), Box<std::error::Error>> {
    // Define cost function
    let cost = |x: &Vec<f64>| -> f64 { rosenbrock(x, 1_f64, 100_f64).unwrap() };
    let gradient = |x: &Vec<f64>| -> Vec<f64> { rosenbrock_derivative(x, 1_f64, 100_f64).unwrap() };
    // let cost = |x: &Vec<f64>| -> f64 { sphere(x).unwrap() };
    // let gradient = |x: &Vec<f64>| -> Vec<f64> { sphere_derivative(x).unwrap() };

    // Define bounds
    let lower_bound: Vec<f64> = vec![-1.5, -0.5];
    let upper_bound: Vec<f64> = vec![2.0, 3.0];

    // Set up problem
    let mut prob = Problem::new(&cost, &lower_bound, &upper_bound);
    prob.gradient(&gradient);

    // Set up GradientDecent solver
    let mut solver = GradientDescent::new();
    solver.max_iters(10_000);
    // solver.gamma_update(GDGammaUpdate::Constant(0.0001));
    solver.gamma_update(GDGammaUpdate::BarzilaiBorwein);

    // define inital parameter vector
    let init_param: Vec<f64> = vec![1.5, 1.5];

    // let result = solver.run(&prob, &prob.random_param()?)?;
    let result1 = solver.run(&prob, &init_param)?;

    // print result
    println!("{:?}", result1);

    let mut solver = GradientDescent::new();
    solver.max_iters(10_000);

    let mut linesearch = BacktrackingLineSearch::new(&cost, &gradient);
    linesearch.alpha(1.0);

    solver.gamma_update(GDGammaUpdate::BacktrackingLineSearch(linesearch));

    // let result = solver.run(&prob, &prob.random_param()?)?;
    let result2 = solver.run(&prob, &init_param)?;

    // print result
    println!("{:?}", result2);

    // Manually solve it
    let mut solver = GradientDescent::new();
    solver.init(&prob, &init_param)?;

    loop {
        let par = solver.next_iter();
        println!("{:?}", par);
        if par.1 >= result1.iters {
            break;
        };
    }

    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("error: {}", e);
    }
}
