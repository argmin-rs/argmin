/// TODO DOCUMENTATION
///
use std::fmt::Debug;
use rand;
use rand::distributions::{IndependentSample, Range};

/// This trait needs to be implemented for every parameter fed into the solvers.
/// This is highly *UNSTABLE* and will change in the future.
pub trait ArgminParameter<T: Clone>: Clone + Debug {
    /// Defines a modification of the parameter vector.
    ///
    /// The parameters:
    ///
    /// `&self`: reference to the object of type `T`
    /// `lower_bound`: Lower bound of the parameter vector. Same type as parameter vector (`T`)
    /// `upper_bound`: Upper bound of the parameter vector. Same type as parameter vector (`T`)
    /// `constraint`: Additional (non)linear constraint whith the signature `&Fn(&T) -> bool`. The
    /// provided function takes a parameter as input and returns `true` if the parameter vector
    /// satisfies the constraints and `false` otherwise.
    fn modify(&self, &T, &T, &Fn(&T) -> bool) -> T;
}

impl ArgminParameter<Vec<f64>> for Vec<f64> {
    fn modify(
        &self,
        lower_bound: &Vec<f64>,
        upper_bound: &Vec<f64>,
        constraint: &Fn(&Vec<f64>) -> bool,
    ) -> Vec<f64> {
        let step = Range::new(0, self.len());
        let range = Range::new(-1.0_f64, 1.0_f64);
        let mut rng = rand::thread_rng();
        let mut param = self.clone();
        loop {
            let idx = step.ind_sample(&mut rng);
            param[idx] = self[idx] + range.ind_sample(&mut rng);
            if param[idx] < lower_bound[idx] {
                param[idx] = lower_bound[idx];
            }
            if param[idx] > upper_bound[idx] {
                param[idx] = upper_bound[idx];
            }
            if constraint(&param) {
                break;
            }
        }
        param
    }
}
