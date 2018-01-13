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
    /// current state
    state: Option<&'a GradientDescentState<'a>>,
}

struct GradientDescentState<'a> {
    problem: &'a Problem<'a, Vec<f64>, f64>,
    prev_param: Vec<f64>,
    param: Vec<f64>,
    iter: u64,
    prev_gamma: f64,
    gamma: f64,
    prev_grad: Vec<f64>,
    cur_grad: Vec<f64>,
}

impl<'a> GradientDescent<'a> {
    /// Return a GradientDescent struct
    pub fn new() -> Self {
        GradientDescent {
            gamma: GDGammaUpdate::BarzilaiBorwein,
            max_iters: std::u64::MAX,
            precision: 0.00000001,
            state: None,
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

    fn update_gamma2(&self) -> f64 {
        match self.gamma {
            GDGammaUpdate::Constant(g) => g,
            GDGammaUpdate::BarzilaiBorwein => {
                let mut grad_diff: f64;
                let mut top: f64 = 0.0;
                let mut bottom: f64 = 0.0;
                for idx in 0..self.state.unwrap().cur_grad.len() {
                    grad_diff =
                        self.state.unwrap().cur_grad[idx] - self.state.unwrap().prev_grad[idx];
                    top += (self.state.unwrap().param[idx] - self.state.unwrap().prev_param[idx])
                        * grad_diff;
                    bottom += grad_diff.powf(2.0);
                }
                top / bottom
            }
            GDGammaUpdate::BacktrackingLineSearch(ref bls) => {
                let result = bls.run(
                    &(self.state
                        .unwrap()
                        .cur_grad
                        .iter()
                        .map(|x| -x)
                        .collect::<Vec<f64>>()),
                    &self.state.unwrap().param,
                ).unwrap();
                result.0
            }
        }
    }

    pub fn init(
        &mut self,
        problem: &'a Problem<'a, Vec<f64>, f64>,
        // problem: Problem<Vec<f64>, f64>,
        init_param: &[f64],
    ) -> Result<()> {
        let state = GradientDescentState {
            problem: problem,
            prev_param: vec![0_f64; init_param.len()],
            param: init_param.to_owned(),
            iter: 0_u64,
            prev_gamma: 0_f64,
            gamma: match self.gamma {
                GDGammaUpdate::Constant(g) => g,
                GDGammaUpdate::BarzilaiBorwein | GDGammaUpdate::BacktrackingLineSearch(_) => 0.0001,
            },
            prev_grad: vec![0_f64; init_param.len()],
            cur_grad: (problem.gradient.unwrap())(&init_param.to_owned()),
        };
        self.state = Some(&state);
        Ok(())
    }

    pub fn next(&mut self) {
        let mut state = self.state.unwrap();
        let gradient = self.state.unwrap().problem.gradient.unwrap();
        state.iter += 1;
    }

    fn terminate(&self) -> bool {
        // use zip here...
        let prev_step_size = ((self.state.unwrap().param[0] - self.state.unwrap().prev_param[0])
            .powf(2.0)
            + (self.state.unwrap().param[1] - self.state.unwrap().prev_param[1]).powf(2.0))
            .sqrt();
        if prev_step_size < self.precision {
            return true;
        }
        if self.state.unwrap().iter >= self.max_iters {
            return true;
        }
        false
    }

    pub fn run2(
        &mut self,
        problem: &'a Problem<'a, Vec<f64>, f64>,
        init_param: &[f64],
    ) -> Result<ArgminResult<Vec<f64>, f64>> {
        // initialize
        self.init(problem, init_param);

        loop {
            self.next();
            if self.terminate() {
                break;
            }
            self.update_gamma2();
        }
        let fin_cost = (problem.cost_function)(&self.state.unwrap().param);
        Ok(ArgminResult::new(
            self.state.unwrap().param.to_vec(),
            fin_cost,
            self.state.unwrap().iter,
        ))
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
