// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

/// TODO DOCUMENTATION
///
use errors::*;
use parameter::ArgminParameter;
use ArgminCostValue;

/// This struct hold all information that describes the optimization problem.
pub struct Problem<'a, T: ArgminParameter + 'a, U: ArgminCostValue + 'a, V: 'a> {
    /// reference to a function which computes the cost/fitness for a given parameter vector
    pub cost_function: &'a Fn(&T) -> U,
    /// optional reference to a function which provides the gradient at a given point in parameter
    /// space
    pub gradient: Option<&'a Fn(&T) -> T>,
    /// optional reference to a function which provides the Hessian at a given point in parameter
    /// space
    pub hessian: Option<&'a Fn(&T) -> V>,
    /// lower bound of the parameter vector
    pub lower_bound: T,
    /// upper bound of the parameter vector
    pub upper_bound: T,
    /// (non)linear constraint which is `true` if a parameter vector lies within the bounds
    pub constraint: &'a Fn(&T) -> bool,
}

impl<'a, T: ArgminParameter + 'a, U: ArgminCostValue + 'a, V: 'a> Problem<'a, T, U, V> {
    /// Create a new `Problem` struct.
    ///
    /// The field `gradient` is automatically set to `None`, but can be manually set by the
    /// `gradient` function. The (non) linear constraint `constraint` is set to a closure which
    /// evaluates to `true` everywhere. This can be overwritten with the `constraint` function.
    ///
    /// `cost_function`: Reference to a cost function
    /// `lower_bound`: lower bound for the parameter vector
    /// `upper_bound`: upper bound for the parameter vector
    pub fn new(cost_function: &'a Fn(&T) -> U, lower_bound: &T, upper_bound: &T) -> Self {
        Problem {
            cost_function: cost_function,
            gradient: None,
            hessian: None,
            lower_bound: lower_bound.clone(),
            upper_bound: upper_bound.clone(),
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

    /// Provide the Hessian
    ///
    /// The function has to have the signature `&Fn(&T) -> T` where `T` is the type of
    /// the parameter vector. The function returns the gradient for a given parameter vector.
    pub fn hessian(&mut self, hessian: &'a Fn(&T) -> V) -> &mut Self {
        self.hessian = Some(hessian);
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

    /// Create a random parameter vector
    ///
    /// The parameter vector satisfies the `lower_bound` and `upper_bound`.
    pub fn random_param(&self) -> Result<T> {
        Ok(T::random(&self.lower_bound, &self.upper_bound)?)
    }
}
