extern crate argmin;
use argmin::problem::Problem;
use argmin::sa::{SATempFunc, SimulatedAnnealing};
use argmin::testfunctions::rosenbrock;

fn main() {
    // Define cost function
    let cost = |x: &Vec<f64>| -> f64 { rosenbrock(x, 1_f64, 100_f64).unwrap() };

    // Define bounds
    let lower_bound: Vec<f64> = vec![-1.5, -0.5];
    let upper_bound: Vec<f64> = vec![2.0, 3.0];

    // Set up problem
    let prob = Problem::new(&cost, lower_bound, upper_bound);

    // Set up simulated annealing solver
    let mut solver = SimulatedAnnealing::new(10.0, 10_000_000).unwrap();
    solver.temp_func(SATempFunc::Exponential(0.8));

    // definie inital parameter vector
    let init_param: Vec<f64> = vec![-1.0, 2.0];

    // run optimization
    let result = solver.run(&prob, &init_param).unwrap();

    // print result
    println!("{:?}", result);
}
