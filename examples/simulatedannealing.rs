extern crate argmin;
use argmin::sa::SimulatedAnnealing;
use argmin::testfunctions::rosenbrock;

fn main() {
    // todo
    let init_param: Vec<f64> = vec![-1.0, 2.0];
    let lower_bound: Vec<f64> = vec![-1.5, -0.5];
    let upper_bound: Vec<f64> = vec![2.0, 3.0];
    let cost = |x: &Vec<f64>| -> f64 { rosenbrock(x, 1_f64, 100_f64).unwrap() };
    let prob = SimulatedAnnealing::new(
        100.0,
        50_000_000,
        init_param,
        &cost,
        lower_bound,
        upper_bound,
    ).unwrap();
    println!("{:?}", prob.run().unwrap());
}
