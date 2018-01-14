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

#[derive(Clone, Debug)]
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
        problem: &'a Problem<'a, Vec<f64>, f64>,
        param_vecs: &[Vec<f64>],
    ) -> Result<()> {
        self.state.problem = Some(problem);
        for param in param_vecs.iter() {
            self.state.param_vecs.push(NelderMeadParam {
                param: param.to_vec(),
                cost: (problem.cost_function)(param),
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

    fn reflect(&self, x0: &[f64], x: &[f64]) -> Vec<f64> {
        x0.iter()
            .zip(x.iter())
            .map(|(a, b)| a + self.alpha * (a - b))
            .collect()
    }

    fn expand(&self, x0: &[f64], x: &[f64]) -> Vec<f64> {
        x0.iter()
            .zip(x.iter())
            .map(|(a, b)| a + self.gamma * (b - a))
            .collect()
    }

    fn contract(&self, x0: &[f64], x: &[f64]) -> Vec<f64> {
        x0.iter()
            .zip(x.iter())
            .map(|(a, b)| a + self.rho * (b - a))
            .collect()
    }

    fn shrink(&mut self) {
        for idx in 1..self.state.param_vecs.len() {
            self.state.param_vecs[idx].param = self.state
                .param_vecs
                .first()
                .unwrap()
                .param
                .iter()
                .zip(self.state.param_vecs[idx].param.iter())
                .map(|(a, b)| a + self.sigma * (b - a))
                .collect();
            self.state.param_vecs[idx].cost =
                (self.state.problem.unwrap().cost_function)(&self.state.param_vecs[idx].param);
        }
    }

    /// Compute next iteration
    pub fn next_iter(&mut self) -> Result<ArgminResult<Vec<f64>, f64>> {
        self.sort_param_vecs();
        println!("{:?}", self.state.param_vecs.first().unwrap());
        let x0 = self.calculate_centroid();
        let xr = self.reflect(&x0, &self.state.param_vecs[0].param);
        let xr_cost = (self.state.problem.unwrap().cost_function)(&xr);
        if xr_cost < self.state.param_vecs.last().unwrap().cost
            && xr_cost >= self.state.param_vecs[0].cost
        {
            // reflection
            // println!("reflection");
            self.state.param_vecs.last_mut().unwrap().param = xr;
            self.state.param_vecs.last_mut().unwrap().cost = xr_cost;
        } else if xr_cost < self.state.param_vecs[0].cost {
            // expansion
            // println!("expansion");
            let xe = self.expand(&x0, &self.state.param_vecs[0].param);
            let xe_cost = (self.state.problem.unwrap().cost_function)(&xe);
            if xe_cost < xr_cost {
                self.state.param_vecs.last_mut().unwrap().param = xe;
                self.state.param_vecs.last_mut().unwrap().cost = xe_cost;
            } else {
                self.state.param_vecs.last_mut().unwrap().param = xr;
                self.state.param_vecs.last_mut().unwrap().cost = xr_cost;
            }
        } else if xr_cost >= self.state.param_vecs[self.state.param_vecs.len() - 1].cost {
            // contraction
            // println!("contraction");
            let xc = self.contract(&x0, &self.state.param_vecs.last().unwrap().param);
            let xc_cost = (self.state.problem.unwrap().cost_function)(&xc);
            if xc_cost < self.state.param_vecs.last().unwrap().cost {
                self.state.param_vecs.last_mut().unwrap().param = xc;
                self.state.param_vecs.last_mut().unwrap().cost = xc_cost;
            }
        } else {
            // println!("shrink");
            self.shrink()
        }

        self.state.iter += 1;

        self.sort_param_vecs();
        let param = self.state.param_vecs[0].clone();
        Ok(ArgminResult::new(param.param, param.cost, self.state.iter))
    }

    /// Stopping criterions
    fn terminate(&mut self) -> bool {
        // if self.state.iter >= self.max_iters {
        //     return true;
        // } else {
        //     return false;
        // }
        self.state.iter >= self.max_iters
    }

    /// Run Nelder Mead optimization
    pub fn run(
        &mut self,
        problem: &'a Problem<'a, Vec<f64>, f64>,
        param_vecs: &[Vec<f64>],
    ) -> Result<ArgminResult<Vec<f64>, f64>> {
        self.init(problem, &param_vecs.to_owned())?;
        let mut out;

        loop {
            out = self.next_iter()?;
            if self.terminate() {
                break;
            }
        }
        // self.sort_param_vecs();
        // let param = self.state.param_vecs[0].clone();
        // Ok(ArgminResult::new(param.param, param.cost, self.state.iter))
        Ok(out)
    }
}

impl<'a> Default for NelderMead<'a> {
    fn default() -> Self {
        Self::new()
    }
}
