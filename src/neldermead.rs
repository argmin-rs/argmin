/// Nelder-Mead method
///
/// TODO
use std;
use errors::*;
use problem::Problem;
use result::ArgminResult;

/// Nelder Mead method
pub struct NelderMead<'a> {
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
    /// current state
    state: NelderMeadState<'a>,
}

#[derive(Clone)]
struct NelderMeadParam {
    param: Vec<f64>,
    cost: f64,
}

struct NelderMeadState<'a> {
    problem: Option<&'a Problem<'a, Vec<f64>, f64>>,
    param_vecs: Vec<NelderMeadParam>,
    iter: u64,
}

impl<'a> NelderMead<'a> {
    /// Return a GradientDescent struct
    pub fn new() -> Self {
        NelderMead {
            max_iters: std::u64::MAX,
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
            state: NelderMeadState {
                problem: None,
                param_vecs: vec![],
                iter: 0_u64,
            },
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

    fn sort_param_vecs(&mut self) {
        self.state.param_vecs.sort_by(|a, b| {
            a.cost
                .partial_cmp(&b.cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// initialization with predefined parameter vectors
    pub fn init(
        &mut self,
        problem: &Problem<Vec<f64>, f64>,
        param_vecs: &Vec<Vec<f64>>,
    ) -> Result<()> {
        for param in param_vecs.iter() {
            self.state.param_vecs.push(NelderMeadParam {
                param: param.to_vec(),
                cost: (problem.cost_function)(&param),
            });
        }
        self.sort_param_vecs();
        Ok(())
    }

    /// Calculate centroid of all but the worst vectors
    fn calculate_centroid(&self) -> Vec<f64> {
        let num_param = self.state.param_vecs.len() - 1;
        let mut x0: Vec<f64> = self.state.param_vecs[0].clone().param;
        for idx in 1..num_param {
            x0 = x0.iter()
                .zip(self.state.param_vecs[idx].param.iter())
                .map(|(a, b)| a + b)
                .collect();
        }
        x0 = x0.iter().map(|a| a / (num_param as f64)).collect();
        x0
    }

    /// Compute next iteration
    pub fn next_iter(&mut self) -> Result<ArgminResult<Vec<f64>, f64>> {
        self.sort_param_vecs();
        let x0 = self.calculate_centroid();
        // do stuff
        self.sort_param_vecs();
        let param = self.state.param_vecs[0].clone();
        Ok(ArgminResult::new(param.param, param.cost, self.state.iter))
    }

    /// Stopping criterions
    fn terminate(&mut self) -> bool {
        if self.state.iter >= self.max_iters {
            return true;
        } else {
            return false;
        }
    }

    /// Run Nelder Mead optimization
    pub fn run(
        &mut self,
        problem: &Problem<Vec<f64>, f64>,
        param_vecs: &Vec<Vec<f64>>,
    ) -> Result<ArgminResult<Vec<f64>, f64>> {
        self.init(&problem, &param_vecs)?;

        loop {
            self.next_iter()?;
            if self.terminate() {
                break;
            }
        }
        self.sort_param_vecs();
        let param = self.state.param_vecs[0].clone();
        Ok(ArgminResult::new(param.param, param.cost, self.state.iter))
    }
}

impl<'a> Default for NelderMead<'a> {
    fn default() -> Self {
        Self::new()
    }
}
