// TODO
// * Set initial parameter vector
// * set bounds
use errors::*;
use result::ArgminResult;
use parameter::ArgminParameter;
use std::fmt::{Debug, Display};
use rand;
use rand::distributions::{IndependentSample, Range};
use num::{Float, FromPrimitive, NumCast};

pub enum SATempFunc {
    TemperatureFast,
    Boltzmann,
    Exponential,
    Custom,
}

/// Simulated Annealing struct (duh)
pub struct SimulatedAnnealing<
    'a,
    T: ArgminParameter<T> + Debug + Clone + 'a,
    U: Float + FromPrimitive + Display + 'a,
> {
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
    /// which temperature function?
    pub temp_func: SATempFunc,
    /// Custom temperature function
    pub custom_temp_func: Option<&'a Fn(f64, u64) -> f64>,
}

impl<'a, T: ArgminParameter<T> + Debug + Clone + 'a, U: Float + FromPrimitive + Display + 'a>
    SimulatedAnnealing<'a, T, U> {
    pub fn new(
        init_temp: f64,
        max_iters: u64,
        init_param: T,
        cost_function: &'a Fn(&T) -> U,
        lower_bound: T,
        upper_bound: T,
    ) -> Result<Self> {
        if init_temp <= FromPrimitive::from_f64(0_f64).unwrap() {
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
                constraint: &|_x: &T| true,
                temp_func: SATempFunc::TemperatureFast,
                custom_temp_func: None,
            })
        }
    }

    pub fn constraint(&mut self, constraint: &'a Fn(&T) -> bool) -> &mut Self {
        self.constraint = constraint;
        self
    }

    pub fn temp_func(&mut self, temperature_func: SATempFunc) -> &mut Self {
        self.temp_func = temperature_func;
        self
    }

    pub fn custom_temp_func(&mut self, func: &'a Fn(f64, u64) -> f64) -> &mut Self {
        self.temp_func = SATempFunc::Custom;
        self.custom_temp_func = Some(func);
        self
    }

    fn accept(&self, temp: f64, prev_cost: U, next_cost: U) -> bool {
        let step = Range::new(0.0, 1.0);
        let mut rng = rand::thread_rng();
        let prob: f64 = step.ind_sample(&mut rng);
        let _1: U = NumCast::from(1_f64).unwrap();
        (next_cost < prev_cost)
            || (1_f64 / (1_f64 + ((next_cost - prev_cost).to_f64().unwrap() / temp).exp()) > prob)
    }

    fn update_temperature(&self, iter: u64) -> Result<f64> {
        match self.temp_func {
            SATempFunc::TemperatureFast => Ok(self.init_temp / ((iter + 1) as f64)),
            SATempFunc::Boltzmann => Ok(self.init_temp / ((iter + 1) as f64).ln()),
            SATempFunc::Exponential => Ok(self.init_temp * 0.95.powf((iter + 1) as f64)),
            SATempFunc::Custom => match self.custom_temp_func {
                Some(func) => Ok(func(self.init_temp, iter)),
                None => Err(ErrorKind::InvalidParameter(
                    "SimulatedAnnealing: No custom temperature update function provided.".into(),
                ).into()),
            },
        }
    }

    /// Run simulated annealing solver
    pub fn run(&self) -> Result<ArgminResult<T, U>> {
        let mut param = self.init_param.clone();
        let mut cost = (self.cost_function)(&param);
        let mut temp = self.init_temp;
        let mut param_best = self.init_param.clone();
        let mut cost_best = cost;
        for i in 0..self.max_iters {
            let mut param_new = param.clone();
            for _ in 0..((temp.floor() as u64) + 1) {
                param_new =
                    param_new.modify(&self.lower_bound, &self.upper_bound, &self.constraint);
            }
            let new_cost = (self.cost_function)(&param_new);
            if self.accept(temp, cost, new_cost) {
                // println!("{} {} {:?}", i, temp, param_new);
                cost = new_cost;
                param = param_new;
                if cost < cost_best {
                    cost_best = cost;
                    param_best = param.clone();
                }
            }
            temp = self.update_temperature(i)?;
        }
        let res = ArgminResult::new(param_best, cost_best, self.max_iters);
        Ok(res)
    }
}
