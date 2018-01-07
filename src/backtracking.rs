/// Backtracking Line Search
///
/// TODO
use errors::*;

/// Reasons why it stopped -- shouldnt be here probably
#[derive(Debug)]
pub enum TerminationReason {
    /// Maximum number of iterations reached
    MaxNumberIterations,
    /// It converged before reaching the maximum number of iterations.
    Converged,
    /// dont know
    Unkown,
}

/// Backtracking Line Search
pub struct BacktrackingLineSearch<'a> {
    /// Reference to cost function.
    cost_function: &'a Fn(&Vec<f64>) -> f64,
    /// Gradient
    gradient: &'a Fn(&Vec<f64>) -> Vec<f64>,
    /// Maximum number of iterations
    max_iters: u64,
    /// Parameter `tau`
    tau: f64,
    /// Parameter `c`
    c: f64,
}

impl<'a> BacktrackingLineSearch<'a> {
    /// Initialize Backtracking Line Search
    ///
    /// Requires the cost function and gradient to be passed as parameter. The parameters
    /// `max_iters`, `tau`, and `c` are set to 100, 0.5 and 0.5, respectively.
    pub fn new(
        cost_function: &'a Fn(&Vec<f64>) -> f64,
        gradient: &'a Fn(&Vec<f64>) -> Vec<f64>,
    ) -> Self {
        BacktrackingLineSearch {
            cost_function: cost_function,
            gradient: gradient,
            max_iters: 100,
            tau: 0.5,
            c: 0.5,
        }
    }

    /// Set the maximum number of iterations
    pub fn max_iters(&mut self, max_iters: u64) -> &mut Self {
        self.max_iters = max_iters;
        self
    }

    /// Set c to a desired value between 0 and 1
    pub fn c(&mut self, c: f64) -> Result<&mut Self> {
        if c >= 1.0 || c <= 0.0 {
            return Err(ErrorKind::InvalidParameter(
                "BacktrackingLineSearch: Parameter `c` must satisfy 0 < c < 1.".into(),
            ).into());
        }
        self.c = c;
        Ok(self)
    }

    /// Set tau to a desired value between 0 and 1
    pub fn tau(&mut self, tau: f64) -> Result<&mut Self> {
        if tau >= 1.0 || tau <= 0.0 {
            return Err(ErrorKind::InvalidParameter(
                "BacktrackingLineSearch: Parameter `tau` must satisfy 0 < tau < 1.".into(),
            ).into());
        }
        self.tau = tau;
        Ok(self)
    }

    /// Run backtracking line search
    pub fn run(&self, p: &[f64], x: &[f64]) -> Result<(f64, u64, TerminationReason)> {
        // TODO:
        // * generics

        // ensure that p is a unit vector
        let p_mag: f64 = p.iter().map(|a| a.powf(2.0)).sum::<f64>().sqrt();
        let p: Vec<f64> = p.iter().map(|a| a / p_mag).collect();

        // compute m
        let m: f64 = p.iter()
            .zip((self.gradient)(&(x.to_owned())).iter())
            .map(|(a, b)| a * b)
            .sum();

        let t = -self.c * m;
        let fx = (self.cost_function)(&(x.to_owned()));
        let termination_reason;
        let mut idx = 0;
        // should this be a parameter?
        let mut alpha = 10.0;
        loop {
            let param = p.iter().zip(x.iter()).map(|(a, b)| b + alpha * a).collect();
            if fx - (self.cost_function)(&param) >= alpha * t {
                termination_reason = TerminationReason::Converged;
                break;
            }
            if idx > self.max_iters {
                termination_reason = TerminationReason::MaxNumberIterations;
                break;
            }
            idx += 1;
            alpha *= self.tau;
        }
        Ok((alpha, idx, termination_reason))
    }
}
