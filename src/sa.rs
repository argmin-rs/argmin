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
    /// lower and upper bound. currently same type as init_param, could be changed in the future.
    pub lower_bound: Option<T>,
    pub upper_bound: Option<T>,
    /// (non)linear constraint which is `true` if a parameter vector lies within the bounds
    pub constraint: Option<&'a Fn(&T) -> bool>,
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
                lower_bound: None,
                upper_bound: None,
                constraint: None,
            })
        }
    }

    pub fn lower_bound(&mut self, lower_bound: T) -> &mut Self {
        self.lower_bound = Some(lower_bound);
        self
    }

    pub fn upper_bound(&mut self, upper_bound: T) -> &mut Self {
        self.upper_bound = Some(upper_bound);
        self
    }

    pub fn constraint(&mut self, constraint: &'a Fn(&T) -> bool) -> &mut Self {
        self.constraint = Some(constraint);
        self
    }

    /// Run simulated annealing solver
    pub fn run(&self) -> Result<ArgminResult<T, U>> {
        let mut param = self.init_param.clone();
        let mut cost = (self.cost_function)(&param);
        let mut temp = self.init_temp;
        for i in 0..self.max_iters {
            temp /= i as f64;
            let param_new = param.modify(&self.lower_bound, &self.upper_bound, &self.constraint);
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
