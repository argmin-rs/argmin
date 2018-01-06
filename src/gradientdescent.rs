/// Gradient Descent
///
/// TODO
use std;
use errors::*;
use problem::Problem;
use result::ArgminResult;
// use parameter::ArgminParameter;
// use ArgminCostValue;

/// Gradient Descent struct (duh)
pub struct GradientDescent {
    step_size: f64,
    /// Maximum number of iterations
    max_iters: u64,
    /// Precision
    precision: f64,
}

impl GradientDescent {
    /// Return a GradientDescent struct
    pub fn new(step_size: f64) -> Self {
        GradientDescent {
            step_size: step_size,
            max_iters: std::u64::MAX,
            precision: 0.00000001,
        }
    }

    /// Set maximum number of iterations
    pub fn max_iters(&mut self, max_iters: u64) -> &mut Self {
        self.max_iters = max_iters;
        self
    }

    /// Set precision
    pub fn precision(&mut self, precision: f64) -> &mut Self {
        self.precision = precision;
        self
    }

    /// Run gradient descent method
    pub fn run(
        &self,
        problem: &Problem<Vec<f64>, f64>,
        init_param: &[f64],
    ) -> Result<ArgminResult<Vec<f64>, f64>> {
        let mut idx = 0;
        let mut param = init_param.to_owned();
        let mut prev_step_size;
        let gradient = problem.gradient.unwrap();

        loop {
            let prev_param = param.clone();
            let update = (gradient)(&prev_param);
            for i in 0..param.len() {
                param[i] -= update[i] * self.step_size;
            }
            prev_step_size = ((param[0] - prev_param[0]).powf(2.0)
                + (param[1] - prev_param[1]).powf(2.0))
                .sqrt();
            idx += 1;
            if idx >= self.max_iters {
                break;
            }
            if prev_step_size < self.precision {
                break;
            }
        }
        let fin_cost = (problem.cost_function)(&param);
        Ok(ArgminResult::new(param, fin_cost, idx))
    }
}
