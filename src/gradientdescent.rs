/// Gradient Descent
///
/// TODO
use std;
use errors::*;
use problem::Problem;
use result::ArgminResult;
// use parameter::ArgminParameter;
// use ArgminCostValue;

pub enum GDGammaUpdate {
    Constant(f64),
    BarzilaiBorwein,
}

/// Gradient Descent struct (duh)
pub struct GradientDescent {
    /// step size
    gamma: GDGammaUpdate,
    /// Maximum number of iterations
    max_iters: u64,
    /// Precision
    precision: f64,
}

impl GradientDescent {
    /// Return a GradientDescent struct
    pub fn new() -> Self {
        GradientDescent {
            gamma: GDGammaUpdate::BarzilaiBorwein,
            max_iters: std::u64::MAX,
            precision: 0.00000001,
        }
    }

    /// Set gradient descent gamma update method
    pub fn gamma_update(&mut self, gamma_update_method: GDGammaUpdate) -> &mut Self {
        self.gamma = gamma_update_method;
        self
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

    fn update_gamma(
        &self,
        cur_param: &[f64],
        prev_param: &[f64],
        cur_grad: &[f64],
        prev_grad: &[f64],
    ) -> f64 {
        match self.gamma {
            GDGammaUpdate::Constant(g) => g,
            GDGammaUpdate::BarzilaiBorwein => {
                let mut grad_diff: f64;
                let mut top: f64 = 0.0;
                let mut bottom: f64 = 0.0;
                for idx in 0..cur_grad.len() {
                    grad_diff = cur_grad[idx] - prev_grad[idx];
                    top += (cur_param[idx] - prev_param[idx]) * grad_diff;
                    bottom += grad_diff.powf(2.0);
                }
                // if bottom == 0.0 {
                //     bottom = 0.00001;
                // }
                top / bottom
            }
        }
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
        let mut cur_grad = vec![0.0, 0.0];
        let mut gamma = match self.gamma {
            GDGammaUpdate::Constant(g) => g,
            GDGammaUpdate::BarzilaiBorwein => 0.0000001,
        };

        loop {
            let prev_param = param.clone();
            let prev_grad = cur_grad.clone();
            cur_grad = (gradient)(&prev_param);

            for i in 0..param.len() {
                param[i] -= cur_grad[i] * gamma;
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
            gamma = self.update_gamma(&param, &prev_param, &cur_grad, &prev_grad);
            println!("{}", gamma);
        }
        let fin_cost = (problem.cost_function)(&param);
        Ok(ArgminResult::new(param, fin_cost, idx))
    }
}
