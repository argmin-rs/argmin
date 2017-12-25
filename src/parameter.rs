use rand;
use rand::distributions::{IndependentSample, Range};

pub trait ArgminParameter<T: Clone> {
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
