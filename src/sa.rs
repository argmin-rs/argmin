// TODO
// * Set initial parameter vector
// * set bounds
use errors::*;
use result::ArgminResult;
use parameter::ArgminParameter;
// use num::{Num, NumCast};

/// Simulated Annealing struct (duh)
pub struct SimulatedAnnealing<'a, T: ArgminParameter, U: 'a, W: 'a> {
    /// Initial temperature
    init_temp: f64,
    /// Maximum number of iterations
    max_iters: u64,
    /// Initial parameter vector
    init_param: T,
    /// cost function
    cost_function: &'a Fn(U) -> W,
}

impl<'a, T: ArgminParameter, U, W> SimulatedAnnealing<'a, T, U, W> {
    pub fn new(
        init_temp: f64,
        max_iters: u64,
        init_param: T,
        cost_function: &'a Fn(U) -> W,
    ) -> Result<Self> {
        if init_temp <= 0f64 {
            Err(
                ErrorKind::InvalidParameter("SimulatedAnnealing: Temperature must be >= 0.".into())
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
    // pub fn run<P, C: Num + NumCast>() -> Result<ArgminResult<P, C>> {
    pub fn run(&mut self) -> Result<ArgminResult> {
        let param: Vec<f64> = vec![0.1, 0.2];
        let cost: f64 = 0.2;
        let res = ArgminResult::new(param, cost, 10);
        Ok(res)
    }
}
