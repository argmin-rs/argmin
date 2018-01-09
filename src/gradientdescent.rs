/// Gradient Descent
///
/// TODO
use std;
use errors::*;
use problem::Problem;
use result::ArgminResult;
use backtracking;
// use parameter::ArgminParameter;
// use ArgminCostValue;

/// Gradient Descent gamma update method
///
/// Missing:
///   * Line search
pub enum GDGammaUpdate<'a> {
    /// Constant gamma
    Constant(f64),
    /// Gamma updated according to TODO
    /// Apparently this only works if the cost function is convex and the derivative of the cost
    /// function is Lipschitz.
    /// TODO: More detailed description (formula)
    BarzilaiBorwein,
    /// Backtracking line search
    BacktrackingLineSearch(backtracking::BacktrackingLineSearch<'a>),
}

/// Gradient Descent struct (duh)
pub struct GradientDescent<'a> {
    /// step size
    gamma: GDGammaUpdate<'a>,
    /// Maximum number of iterations
    max_iters: u64,
    /// Precision
    precision: f64,
}

impl<'a> GradientDescent<'a> {
    /// Return a GradientDescent struct
    pub fn new() -> Self {
        GradientDescent {
            gamma: GDGammaUpdate::BarzilaiBorwein,
            max_iters: std::u64::MAX,
            precision: 0.00000001,
        }
    }

    /// Set gradient descent gamma update method
    pub fn gamma_update(&mut self, gamma_update_method: GDGammaUpdate<'a>) -> &mut Self {
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
                top / bottom
            }
            GDGammaUpdate::BacktrackingLineSearch(ref bls) => {
                let result = bls.run(
                    &(cur_grad.iter().map(|x| -x).collect::<Vec<f64>>()),
                    cur_param,
                ).unwrap();
                result.0
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
        // let mut cur_grad = vec![0.0, 0.0];
        let mut cur_grad = (gradient)(&param);
        let mut gamma = match self.gamma {
            GDGammaUpdate::Constant(g) => g,
            GDGammaUpdate::BarzilaiBorwein | GDGammaUpdate::BacktrackingLineSearch(_) => 0.0001,
        };

        loop {
            let prev_param = param.clone();
            let prev_grad = cur_grad.clone();

            // Move to next point
            for i in 0..param.len() {
                param[i] -= cur_grad[i] * gamma;
            }

            // Stop if maximum number of iterations is reached
            idx += 1;
            if idx >= self.max_iters {
                break;
            }

            // Stop if current solution is good enough
            // This checks whether the current move has been smaller than `self.precision`
            prev_step_size = ((param[0] - prev_param[0]).powf(2.0)
                + (param[1] - prev_param[1]).powf(2.0))
                .sqrt();
            if prev_step_size < self.precision {
                break;
            }

            // Calculate next gradient
            cur_grad = (gradient)(&param);

            // Update gamma
            gamma = self.update_gamma(&param, &prev_param, &cur_grad, &prev_grad);
        }
        let fin_cost = (problem.cost_function)(&param);
        Ok(ArgminResult::new(param, fin_cost, idx))
    }
}

impl<'a> Default for GradientDescent<'a> {
    fn default() -> Self {
        Self::new()
    }
}
