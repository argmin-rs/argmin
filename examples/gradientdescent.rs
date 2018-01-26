#![allow(unused_imports)]
extern crate argmin;
use argmin::problem::Problem;
use argmin::gradientdescent::{GDGammaUpdate, GradientDescent};
use argmin::testfunctions::{rosenbrock, rosenbrock_derivative, sphere, sphere_derivative};
use argmin::backtracking::BacktrackingLineSearch;

fn run() -> Result<(), Box<std::error::Error>> {
    // Define cost function
    // Choose either `Rosenbrock` or `Sphere` function.
    let cost = |x: &Vec<f64>| -> f64 { rosenbrock(x, 1_f64, 100_f64) };
    let gradient = |x: &Vec<f64>| -> Vec<f64> { rosenbrock_derivative(x, 1_f64, 100_f64) };
    // let cost = |x: &Vec<f64>| -> f64 { sphere(x) };
    // let gradient = |x: &Vec<f64>| -> Vec<f64> { sphere_derivative(x) };

    // Define bounds
    // Note: Gradient Descent currently does not enforce these bounds which is why we can set them
    // to -1000 and +1000
    let lower_bound: Vec<f64> = vec![-1000.0, -1000.0];
    let upper_bound: Vec<f64> = vec![1000.0, 1000.0];
    // Unfortunately, setting them to +/- Infinity does not work (yet)
    // let lower_bound: Vec<f64> = vec![std::f64::NEG_INFINITY, std::f64::NEG_INFINITY];
    // let upper_bound: Vec<f64> = vec![std::f64::INFINITY, std::f64::INFINITY];

    // Set up problem
    // The problem requires a cost function, lower and upper bounds and takes an optional gradient.
    let mut prob = Problem::new(&cost, &lower_bound, &upper_bound);
    prob.gradient(&gradient);

    // Set up GradientDecent solver
    let mut solver = GradientDescent::new();
    // Set the maximum number of iterations to 10000
    solver.max_iters(10_000);
    // Choose the method which calculates the step width. `GDGammaUpdate::Constant(0.0001)` sets a
    // constant step width of 0.0001 while `GDGammaUpdate::BarzilaiBorwein` updates the step width
    // according to TODO
    // solver.gamma_update(GDGammaUpdate::Constant(0.0001));
    solver.gamma_update(GDGammaUpdate::BarzilaiBorwein);

    // define inital parameter vector
    // `Problem` allows to create random parameter vectors which satisfies `lower_bound` and
    // `upper_bound`.
    let init_param: Vec<f64> = prob.random_param()?;
    // let init_param: Vec<f64> = vec![1.5, 1.5];
    println!("{:?}", init_param);

    // Actually run the solver on the problem.
    let result1 = solver.run(&prob, &init_param)?;

    // print result
    println!("{:?}", result1);

    // Define new solver
    let mut solver = GradientDescent::new();
    solver.max_iters(10_000);

    // Initialize a backtracking line search method to use for the gamma update method
    let mut linesearch = BacktrackingLineSearch::new(&cost, &gradient);
    linesearch.alpha(1.0);
    solver.gamma_update(GDGammaUpdate::BacktrackingLineSearch(linesearch));

    // Run solver
    let result2 = solver.run(&prob, &init_param)?;

    // print result
    println!("{:?}", result2);

    // Manually solve it
    // `GradientDescent` also allows you to initialize the solver yourself and run each iteration
    // manually. This is particularly useful if you need to get intermediate results or if your
    // need to print out information that is otherwise not accessible to you.
    let mut solver = GradientDescent::new();
    solver.init(&prob, &init_param)?;

    loop {
        let par = solver.next_iter();
        // println!("{:?}", par);
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
