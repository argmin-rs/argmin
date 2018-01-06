/// Nelder-Mead method
///
/// TODO
use std;
use errors::*;
use problem::Problem;
use result::ArgminResult;

/// Nelder Mead method
pub struct NelderMead {
    /// Maximum number of iterations
    max_iters: u64,
    /// alpha
    alpha: f64,
    /// gamma
    gamma: f64,
    /// rho
    rho: f64,
    /// sigma
    sigma: f64,
}

impl NelderMead {
    /// Return a GradientDescent struct
    pub fn new() -> Self {
        NelderMead {
            max_iters: std::u64::MAX,
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
        }
    }

    /// Set maximum number of iterations
    pub fn max_iters(&mut self, max_iters: u64) -> &mut Self {
        self.max_iters = max_iters;
        self
    }

    /// alpha
    pub fn alpha(&mut self, alpha: f64) -> &mut Self {
        self.alpha = alpha;
        self
    }

    /// gamma
    pub fn gamma(&mut self, gamma: f64) -> &mut Self {
        self.gamma = gamma;
        self
    }

    /// rho
    pub fn rho(&mut self, rho: f64) -> &mut Self {
        self.rho = rho;
        self
    }

    /// sigma
    pub fn sigma(&mut self, sigma: f64) -> &mut Self {
        self.sigma = sigma;
        self
    }

    /// Run Nelder Mead optimization
    pub fn run(&self, problem: &Problem<Vec<f64>, f64>) -> Result<ArgminResult<Vec<f64>, f64>> {
        println!("{:?}", problem.lower_bound);
        Ok(ArgminResult::new(vec![0.0, 0.0], 1.0, 0))
    }
}

impl Default for NelderMead {
    fn default() -> Self {
        Self::new()
    }
}
