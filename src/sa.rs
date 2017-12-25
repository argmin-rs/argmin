// TODO
// * Set initial parameter vector
// * set bounds
use errors::*;
use result::ArgminResult;
use parameter::ArgminParameter;
use std::fmt::Display;
// use num::{Num, NumCast};

/// Simulated Annealing struct (duh)
pub struct SimulatedAnnealing<'a, T: ArgminParameter<T> + Clone + 'a, U: PartialOrd + Display + 'a>
{
    /// Initial temperature
    pub init_temp: f64,
    /// Maximum number of iterations
    pub max_iters: u64,
    /// Initial parameter vector
    pub init_param: T,
    /// cost function
    pub cost_function: &'a Fn(&T) -> U,
    /// lower and upper bound. currently same type as init_param, could be changed in the future.
    pub lower_bound: T,
    pub upper_bound: T,
    /// (non)linear constraint which is `true` if a parameter vector lies within the bounds
    pub constraint: &'a Fn(&T) -> bool,
}

impl<'a, T: ArgminParameter<T> + Clone + 'a, U: PartialOrd + Display + 'a>
    SimulatedAnnealing<'a, T, U> {
    pub fn new(
        init_temp: f64,
        max_iters: u64,
        init_param: T,
        cost_function: &'a Fn(&T) -> U,
        lower_bound: T,
        upper_bound: T,
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
                lower_bound: lower_bound,
                upper_bound: upper_bound,
                constraint: &|_x| true,
            })
        }
    }

    // pub fn lower_bound(&mut self, lower_bound: T) -> &mut Self {
    //     self.lower_bound = lower_bound;
    //     self
    // }
    //
    // pub fn upper_bound(&mut self, upper_bound: T) -> &mut Self {
    //     self.upper_bound = upper_bound;
    //     self
    // }

    pub fn constraint(&mut self, constraint: &'a Fn(&T) -> bool) -> &mut Self {
        self.constraint = constraint;
        self
    }

    /// Run simulated annealing solver
    pub fn run(&self) -> Result<ArgminResult<T, U>> {
        let mut param = self.init_param.clone();
        let mut cost = (self.cost_function)(&param);
        let mut _temp = self.init_temp;
        for i in 0..self.max_iters {
            _temp /= i as f64;
            let param_new = param.modify(&self.lower_bound, &self.upper_bound, &self.constraint);
            let new_cost = (self.cost_function)(&param_new);
            // println!("iter: {}; cost: {}", i, new_cost);
            if new_cost < cost {
                cost = new_cost;
                param = param_new;
            }
        }
        let res = ArgminResult::new(param, cost, self.max_iters);
        Ok(res)
    }
}
