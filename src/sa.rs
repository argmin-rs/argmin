/// TODO
///
/// * [ ] Different acceptance functions
/// * [ ] Exponential temperature function should take a parameter
/// * [ ] Early stopping criterions
use errors::*;
use problem::Problem;
use result::ArgminResult;
use parameter::ArgminParameter;
use std::fmt::{Debug, Display};
use rand;
use rand::distributions::{IndependentSample, Range};
use num::{Float, FromPrimitive, NumCast};

/// Definition of build in temperature functions for Simulated Annealing.
///
/// Given the initial temperature `t_init` and the iteration number `i`, the current temperature
/// `t_i` is given as follows:
///
/// `SATempFunc::TemperatureFast`: `t_i = t_init / i`
/// `SATempFunc::Boltzmann`: `t_i = t_init / ln(i)`
/// `SATempFunc::Exponential`: `t_i = t_init * 0.95^i`
/// `SATempFunc::Custom`: User provided temperature update function which has to implement the
/// function signature `&Fn(init_temp: f64, iteration_number: u64) -> f64`. See
/// `SimulatedAnnealing::custom_temp_func()` for details on how to provide a custom temperature
/// update function.
pub enum SATempFunc {
    /// `t_i = t_init / i`
    TemperatureFast,
    /// `t_i = t_init / ln(i)`
    Boltzmann,
    /// `t_i = t_init * x^i`
    Exponential(f64),
    /// User-provided temperature function. See `SimulatedAnnealing::custom_temp_func()` for
    /// details.
    Custom,
}

/// Simulated Annealing struct (duh)
pub struct SimulatedAnnealing<
    'a,
    T: ArgminParameter<T> + Debug + Clone + 'a,
    U: Float + FromPrimitive + Display + 'a,
> {
    pub problem: Problem<'a, T, U>,
    /// Initial temperature
    pub init_temp: f64,
    /// Maximum number of iterations
    pub max_iters: u64,
    /// Initial parameter vector
    pub init_param: T,
    /// which temperature function?
    pub temp_func: SATempFunc,
    /// Custom temperature function
    pub custom_temp_func: Option<&'a Fn(f64, u64) -> f64>,
}

impl<'a, T: ArgminParameter<T> + Debug + Clone + 'a, U: Float + FromPrimitive + Display + 'a>
    SimulatedAnnealing<'a, T, U> {
    /// Constructor
    ///
    /// Returns an `SimulatedAnnealing` struct where all entries of the struct are set according to
    /// the parameters provided, apart from `constraint`, `temp_func` and `custom_temp_func` which
    /// are set to default values (`&|_x| true`, `SATempFunc::TemperatureFast` and `None`,
    /// respectively.
    ///
    /// Parameters:
    ///
    /// `problem`: problem definition
    /// `init_tmep`: Initial temperature
    /// `max_iters`: Maximum number of iterations
    /// `init_param`: Initial parameter vector
    pub fn new(
        problem: Problem<'a, T, U>,
        init_temp: f64,
        max_iters: u64,
        init_param: T,
    ) -> Result<Self> {
        if init_temp <= FromPrimitive::from_f64(0_f64).unwrap() {
            Err(
                ErrorKind::InvalidParameter("SimulatedAnnealing: Temperature must be > 0.".into())
                    .into(),
            )
        } else {
            Ok(SimulatedAnnealing {
                problem: problem,
                init_temp: init_temp,
                max_iters: max_iters,
                init_param: init_param,
                temp_func: SATempFunc::TemperatureFast,
                custom_temp_func: None,
            })
        }
    }

    /// Change temperature function to one of the options in `SATempFunc`.
    ///
    /// This will overwrite any custom temperature functions provided by `custom_temp_func()`.
    pub fn temp_func(&mut self, temperature_func: SATempFunc) -> &mut Self {
        self.temp_func = temperature_func;
        self
    }

    /// Provide a custom temperature function.
    ///
    /// The function has to implement the function signature `&Fn(init_temp: f64, iteration_number:
    /// u64) -> f64` and return the current temperature.
    /// This will overwrite any changes done by a call to `temp_func()`.
    pub fn custom_temp_func(&mut self, func: &'a Fn(f64, u64) -> f64) -> &mut Self {
        self.temp_func = SATempFunc::Custom;
        self.custom_temp_func = Some(func);
        self
    }

    /// Acceptance function
    ///
    /// Any solution where `next_cost < prev_cost` will be accepted. Whenever a new solution is
    /// worse than the previous one, the acceptance probability is calculated as:
    ///
    ///     `1 / (1 + exp((next_cost - prev_cost) / current_temperature))`,
    ///
    /// which will always be between 0 and 0.5.
    fn accept(&self, temp: f64, prev_cost: U, next_cost: U) -> bool {
        let step = Range::new(0.0, 1.0);
        let mut rng = rand::thread_rng();
        let prob: f64 = step.ind_sample(&mut rng);
        let _1: U = NumCast::from(1_f64).unwrap();
        (next_cost < prev_cost)
            || (1_f64 / (1_f64 + ((next_cost - prev_cost).to_f64().unwrap() / temp).exp()) > prob)
    }

    /// Update the temperature based on the current iteration number.
    ///
    /// Updates are performed based on specific update functions. See `SATempFunc` for details.
    fn update_temperature(&self, iter: u64) -> Result<f64> {
        match self.temp_func {
            SATempFunc::TemperatureFast => Ok(self.init_temp / ((iter + 1) as f64)),
            SATempFunc::Boltzmann => Ok(self.init_temp / ((iter + 1) as f64).ln()),
            SATempFunc::Exponential(x) => if x < 1_f64 && x > 0_f64 {
                Ok(self.init_temp * x.powf((iter + 1) as f64))
            } else {
                Err(ErrorKind::InvalidParameter(
                    "SimulatedAnnealing: Parameter for exponential \
                     temperature update function needs to be >0 and <1."
                        .into(),
                ).into())
            },
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

        // Evaluate cost function of starting point
        let mut cost = (self.problem.cost_function)(&param);

        // Initialize temperature
        let mut temp = self.init_temp;

        // Set first best solution to initial parameter vector
        let mut param_best = self.init_param.clone();
        let mut cost_best = cost;

        // Start annealing
        for i in 0..self.max_iters {
            // Start off with current parameter vector and mutate it with the mutation proportional
            // to the current temperature
            let mut param_new = param.clone();
            for _ in 0..((temp.floor() as u64) + 1) {
                param_new = param_new.modify(
                    &self.problem.lower_bound,
                    &self.problem.upper_bound,
                    &self.problem.constraint,
                );
            }

            // Evaluate cost function with new parameter vector
            let new_cost = (self.problem.cost_function)(&param_new);

            // Decide whether new parameter vector should be accepted.
            // If no, move on with old parameter vector.
            if self.accept(temp, cost, new_cost) {
                // println!("{} {} {:?}", i, temp, param_new);
                // If yes, update the parameter vector for the next iteration.
                cost = new_cost;
                param = param_new;

                // In case the new solution is better than the current best, update best as well.
                if cost < cost_best {
                    cost_best = cost;
                    param_best = param.clone();
                }
            }

            // Update temperature for next iteration.
            temp = self.update_temperature(i)?;
        }

        // Return the result.
        Ok(ArgminResult::new(param_best, cost_best, self.max_iters))
    }
}
