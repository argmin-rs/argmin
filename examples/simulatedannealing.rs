extern crate argmin;
use argmin::sa::SimulatedAnnealing;
use argmin::testfunctions::rosenbrock;

fn main() {
    // todo
    let init_param: Vec<f64> = vec![0.1, 0.2];
    let cost = |x: Vec<f64>| {
        rosenbrock(x, 1_f64, 100_f64).unwrap();
    };
    let mut prob = SimulatedAnnealing::new(100.0, 20, init_param, &cost).unwrap();
    prob.run().unwrap();
}
