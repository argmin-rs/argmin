/// TODO DOCUMENTATION
///
use parameter::ArgminParameter;
use ArgminCostValue;

/// This struct hold all information that describes the optimization problem.
pub struct Problem<'a, T: ArgminParameter<T> + 'a, U: ArgminCostValue + 'a> {
    /// reference to a function which computes the cost/fitness for a given parameter vector
    pub cost_function: &'a Fn(&T) -> U,
    /// optional reference to a function which provides the gradient at a given point in parameter
    /// space
    pub gradient: Option<&'a Fn(&T) -> T>,
    /// lower bound of the parameter vector
    pub lower_bound: T,
    /// upper bound of the parameter vector
    pub upper_bound: T,
    /// (non)linear constraint which is `true` if a parameter vector lies within the bounds
    pub constraint: &'a Fn(&T) -> bool,
}

impl<'a, T: ArgminParameter<T> + 'a, U: ArgminCostValue + 'a> Problem<'a, T, U> {
    /// Create a new `Problem` struct.
    ///
    /// The field `gradient` is automatically set to `None`, but can be manually set by the
    /// `gradient` function. The (non) linear constraint `constraint` is set to a closure which
    /// evaluates to `true` everywhere. This can be overwritten with the `constraint` function.
    ///
    /// `cost_function`: Reference to a cost function
    /// `lower_bound`: lower bound for the parameter vector
    /// `upper_bound`: upper bound for the parameter vector
    pub fn new(cost_function: &'a Fn(&T) -> U, lower_bound: T, upper_bound: T) -> Self {
        Problem {
            cost_function: cost_function,
            gradient: None,
            lower_bound: lower_bound,
            upper_bound: upper_bound,
            constraint: &|_x: &T| true,
        }
    }

    /// Provide the gradient
    ///
    /// The function has to have the signature `&Fn(&T) -> T` where `T` is the type of
    /// the parameter vector. The function returns the gradient for a given parameter vector.
    pub fn gradient(&mut self, gradient: &'a Fn(&T) -> T) -> &mut Self {
        self.gradient = Some(gradient);
        self
    }

    /// Provide additional (non) linear constraint.
    ///
    /// The function has to have the signature `&Fn(&T) -> bool` where `T` is the type of
    /// the parameter vector. The function returns `true` if all constraints are satisfied and
    /// `false` otherwise.
    pub fn constraint(&mut self, constraint: &'a Fn(&T) -> bool) -> &mut Self {
        self.constraint = constraint;
        self
    }
}
