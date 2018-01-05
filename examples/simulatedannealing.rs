extern crate argmin;
use argmin::problem::Problem;
use argmin::sa::{SATempFunc, SimulatedAnnealing};
use argmin::testfunctions::rosenbrock;

fn main() {
    // todo
    let init_param: Vec<f64> = vec![-1.0, 2.0];
    let lower_bound: Vec<f64> = vec![-1.5, -0.5];
    let upper_bound: Vec<f64> = vec![2.0, 3.0];
    let cost = |x: &Vec<f64>| -> f64 { rosenbrock(x, 1_f64, 100_f64).unwrap() };

    let prob = Problem::new(&cost, lower_bound, upper_bound);

    let mut solver = SimulatedAnnealing::new(prob, 10.0, 10_000_000).unwrap();
    solver.temp_func(SATempFunc::Exponential(0.8));

    println!("{:?}", solver.run(&init_param).unwrap());
}
