/// Landweber algorithm
///
/// TODO
use std;
use ndarray::{Array1, Array2};
use errors::*;
use operator::ArgminOperator;
use result::ArgminResult;
use ArgminSolver;

/// Newton method struct (duh)
pub struct Landweber<'a> {
    /// relaxation factor
    /// must satisfy 0 < omega < 2/sigma_1^2 where sigma_1 is the largest singular value of the
    /// matrix.
    omega: f64,
    /// Maximum number of iterations
    max_iters: u64,
    /// current state
    state: LandweberState<'a>,
}

/// Indicates the current state of the Landweber algorithm
struct LandweberState<'a> {
    /// Reference to the problem. This is an Option<_> because it is initialized as `None`
    operator: Option<&'a ArgminOperator<'a>>,
    /// Current parameter vector
    param: Array1<f64>,
    /// Current number of iteration
    iter: u64,
}

impl<'a> LandweberState<'a> {
    /// Constructor for `LandweberState`
    pub fn new() -> Self {
        LandweberState {
            operator: None,
            param: Array1::default(1),
            iter: 0_u64,
        }
    }
}

impl<'a> Landweber<'a> {
    /// Return a `Newton` struct
    pub fn new(omega: f64) -> Self {
        Landweber {
            omega: omega,
            max_iters: std::u64::MAX,
            state: LandweberState::new(),
        }
    }

    /// Set maximum number of iterations
    pub fn max_iters(&mut self, max_iters: u64) -> &mut Self {
        self.max_iters = max_iters;
        self
    }
}

impl<'a> ArgminSolver<'a> for Landweber<'a> {
    type A = Array1<f64>;
    type B = f64;
    type C = Array2<f64>;
    type D = Array1<f64>;
    type E = ArgminOperator<'a>;

    /// Initialize with a given problem and a starting point
    fn init(&mut self, operator: &'a Self::E, init_param: &Array1<f64>) -> Result<()> {
        self.state = LandweberState {
            operator: Some(operator),
            param: init_param.to_owned(),
            iter: 0_u64,
        };
        Ok(())
    }

    /// Compute next point
    fn next_iter(&mut self) -> Result<ArgminResult<Array1<f64>, f64>> {
        // TODO: Move to next point
        let prev_param = self.state.param.clone();
        let diff = self.state.operator.unwrap().apply(&prev_param) - self.state.operator.unwrap().y;
        self.state.param =
            prev_param - self.omega * self.state.operator.unwrap().apply_transpose(&diff);
        self.state.iter += 1;
        let norm: f64 = diff.iter().map(|a| a.powf(2.0)).sum::<f64>().sqrt();
        Ok(ArgminResult::new(
            self.state.param.clone(),
            norm,
            self.state.iter,
        ))
    }

    /// Indicates whether any of the stopping criteria are met
    fn terminate(&self) -> bool {
        false
    }

    /// Run Landweber method
    fn run(
        &mut self,
        operator: &'a Self::E,
        init_param: &Array1<f64>,
    ) -> Result<ArgminResult<Array1<f64>, f64>> {
        // initialize
        self.init(operator, init_param)?;

        let mut res;
        loop {
            res = self.next_iter()?;
            if self.terminate() {
                break;
            }
        }
        Ok(res)
    }
}

impl<'a> Default for Landweber<'a> {
    fn default() -> Self {
        Self::new(1.0)
    }
}
