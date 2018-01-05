/// TODO DOCUMENTATION
///
use std::fmt::{Debug, Display};
use num::{Float, FromPrimitive};
use parameter::ArgminParameter;

pub struct Problem<
    'a,
    T: ArgminParameter<T> + Debug + Clone + 'a,
    U: Float + FromPrimitive + Display + 'a,
> {
    pub cost_function: &'a Fn(&T) -> U,
    pub gradient: Option<&'a Fn(&T) -> T>,
    /// lower bound of the parameter vector
    pub lower_bound: T,
    /// upper bound of the parameter vector
    pub upper_bound: T,
    /// (non)linear constraint which is `true` if a parameter vector lies within the bounds
    pub constraint: &'a Fn(&T) -> bool,
}

impl<'a, T: ArgminParameter<T> + Debug + Clone + 'a, U: Float + FromPrimitive + Display + 'a>
    Problem<'a, T, U> {
    pub fn new(cost_function: &'a Fn(&T) -> U, lower_bound: T, upper_bound: T) -> Self {
        Problem {
            cost_function: cost_function,
            gradient: None,
            lower_bound: lower_bound,
            upper_bound: upper_bound,
            constraint: &|_x: &T| true,
        }
    }

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
