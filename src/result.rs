/// `ArgminResult`
///
/// TODO
use parameter::ArgminParameter;

/// Return struct for all solvers.
#[derive(Debug)]
pub struct ArgminResult<T: ArgminParameter<T> + Clone, U: PartialOrd> {
    /// Final parameter vector
    pub param: T,
    /// Final cost value
    pub cost: U,
    /// Number of iterations
    pub iters: u64,
}

impl<T: ArgminParameter<T> + Clone, U: PartialOrd> ArgminResult<T, U> {
    /// Constructor
    ///
    /// `param`: Final (best) parameter vector
    /// `cost`: Final (best) cost function value
    /// `iters`: Number of iterations
    pub fn new(param: T, cost: U, iters: u64) -> Self {
        ArgminResult { param, cost, iters }
    }
}
