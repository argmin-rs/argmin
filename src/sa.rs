// TODO
// * Set initial parameter vector
// * set bounds
use errors::*;
use result::ArgminResult;
use parameter::ArgminParameter;
// use num::{Num, NumCast};

/// Simulated Annealing struct (duh)
pub struct SimulatedAnnealing<'a, T: ArgminParameter<T> + Clone + 'a, U: Ord + 'a> {
    /// Initial temperature
    pub init_temp: f64,
    /// Maximum number of iterations
    pub max_iters: u64,
    /// Initial parameter vector
    pub init_param: T,
    /// cost function
    pub cost_function: &'a Fn(&T) -> U,
}

impl<'a, T: ArgminParameter<T> + Clone + 'a, U: Ord + 'a> SimulatedAnnealing<'a, T, U> {
    pub fn new(
        init_temp: f64,
        max_iters: u64,
        init_param: T,
        cost_function: &'a Fn(&T) -> U,
    ) -> Result<Self> {
        if init_temp <= 0f64 {
            Err(
                ErrorKind::InvalidParameter("SimulatedAnnealing: Temperature must be > 0.".into())
                    .into(),
            )
        } else {
            Ok(SimulatedAnnealing {
                init_temp: init_temp,
                max_iters: max_iters,
                init_param: init_param,
                cost_function: cost_function,
            })
        }
    }

    /// Run simulated annealing solver
    pub fn run(&self) -> Result<ArgminResult<T, U>> {
        let mut param = self.init_param.clone();
        let mut cost = (self.cost_function)(&param);
        let mut temp = self.init_temp;
        for i in 0..self.max_iters {
            temp /= i as f64;
            let param_new = param.modify();
            let new_cost = (self.cost_function)(&param_new);
            if new_cost < cost {
                cost = new_cost;
                param = param_new;
            }
        }
        let res = ArgminResult::new(param, cost, self.max_iters);
        Ok(res)
    }
}
