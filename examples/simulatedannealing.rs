extern crate argmin;
use argmin::ArgminSolver;
use argmin::problem::Problem;
use argmin::sa::{SATempFunc, SimulatedAnnealing};
use argmin::testfunctions::rosenbrock;

fn run() -> Result<(), Box<std::error::Error>> {
    // Define cost function
    let cost = |x: &Vec<f64>| -> f64 { rosenbrock(x, 1_f64, 100_f64) };

    // Define bounds
    let lower_bound: Vec<f64> = vec![-1.5, -0.5];
    let upper_bound: Vec<f64> = vec![2.0, 3.0];

    // Set up problem
    let prob: Problem<_, _, f64> = Problem::new(&cost, &lower_bound, &upper_bound);
    // let prob: Problem = Problem::new(&cost, &lower_bound, &upper_bound);

    // Set up simulated annealing solver
    let mut solver = SimulatedAnnealing::new(10.0, 1_000_000)?;
    solver.temp_func(SATempFunc::Exponential(0.8));

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
