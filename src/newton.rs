/// Gradient Descent
///
/// TODO
use std;
use errors::*;
use problem::Problem;
use result::ArgminResult;

/// Gradient Descent struct (duh)
pub struct Newton<'a> {
    /// step size
    gamma: f64,
    /// Maximum number of iterations
    max_iters: u64,
    /// current state
    state: NewtonState<'a>,
}

/// Indicates the current state of the Newton
struct NewtonState<'a> {
    /// Reference to the problem. This is an Option<_> because it is initialized as `None`
    problem: Option<&'a Problem<'a, Vec<f64>, f64>>,
    /// Current number of iteration
    param: Vec<f64>,
    /// Current number of iteration
    iter: u64,
}

impl<'a> NewtonState<'a> {
    /// Constructor for `GradientDescentState`
    pub fn new() -> Self {
        NewtonState {
            problem: None,
            param: vec![0.0],
            iter: 0_u64,
        }
    }
}

impl<'a> Newton<'a> {
    /// Return a GradientDescent struct
    pub fn new() -> Self {
        Newton {
            gamma: 1.0,
            max_iters: std::u64::MAX,
            state: NewtonState::new(),
        }
    }

    /// Set maximum number of iterations
    pub fn max_iters(&mut self, max_iters: u64) -> &mut Self {
        self.max_iters = max_iters;
        self
    }

    /// Initialize with a given problem and a starting point
    pub fn init(
        &mut self,
        problem: &'a Problem<'a, Vec<f64>, f64>,
        init_param: &[f64],
    ) -> Result<()> {
        self.state = NewtonState {
            problem: Some(problem),
            param: init_param.to_owned(),
            iter: 0_u64,
        };
        Ok(())
    }

    /// Compute next point
    pub fn next_iter(&mut self) -> (Vec<f64>, u64) {
        // TODO: Move to next point
        (vec![0.0, 0.0], 0)
    }

    /// Indicates whether any of the stopping criteria are met
    fn terminate(&self) -> bool {
        false
    }

    /// Run gradient descent method
    pub fn run(
        &mut self,
        problem: &'a Problem<'a, Vec<f64>, f64>,
        init_param: &[f64],
    ) -> Result<ArgminResult<Vec<f64>, f64>> {
        // initialize
        self.init(problem, init_param)?;

        loop {
            self.next_iter();
            if self.terminate() {
                break;
            }
        }
        let fin_cost = (problem.cost_function)(&self.state.param);
        Ok(ArgminResult::new(
            self.state.param.to_vec(),
            fin_cost,
            self.state.iter,
        ))
    }
}

impl<'a> Default for Newton<'a> {
    fn default() -> Self {
        Self::new()
    }
}
