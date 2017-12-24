extern crate argmin;
use argmin::sa::SimulatedAnnealing;

fn main() {
    // todo
    let prob = SimulatedAnnealing::new(100.0, 20);
    prob.run();
}
