// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! TODO DOCUMENTATION
//!

use errors::*;
use prelude::*;
use parameter::ArgminParameter;

/// This struct hold all information that describes the optimization problem.
#[derive(Clone)]
pub struct ArgminProblem<'a, T: ArgminParameter + 'a, U: ArgminCostValue + 'a, V: 'a> {
    /// reference to a function which computes the cost/fitness for a given parameter vector
    pub cost_function: &'a Fn(&T) -> U,
    /// optional reference to a function which provides the gradient at a given point in parameter
    /// space
    pub gradient: Option<&'a Fn(&T) -> T>,
    /// optional reference to a function which provides the Hessian at a given point in parameter
    /// space
    pub hessian: Option<&'a Fn(&T) -> V>,
    /// lower bound of the parameter vector
    pub lower_bound: Option<T>,
    /// upper bound of the parameter vector
    pub upper_bound: Option<T>,
    /// (non)linear constraint which is `true` if a parameter vector lies within the bounds
    pub constraint: &'a Fn(&T) -> bool,
    /// Target cost function value. The optimization will stop once this value is reached.
    pub target_cost: U,
}

impl<'a, T: ArgminParameter + 'a, U: ArgminCostValue + 'a, V: 'a> ArgminProblem<'a, T, U, V> {
    /// Create a new `ArgminProblem` struct.
    ///
    /// The field `gradient` is automatically set to `None`, but can be manually set by the
    /// `gradient` function. The (non) linear constraint `constraint` is set to a closure which
    /// evaluates to `true` everywhere. This can be overwritten with the `constraint` function.
    ///
    /// `cost_function`: Reference to a cost function
    pub fn new(cost_function: &'a Fn(&T) -> U) -> Self {
        ArgminProblem {
            cost_function,
            gradient: None,
            hessian: None,
            lower_bound: None,
            upper_bound: None,
            constraint: &|_x: &T| true,
            target_cost: U::min_value(),
        }
    }

    /// Set lower and upper bounds
    ///
    /// `lower_bound`: lower bound for the parameter vector
    /// `upper_bound`: upper bound for the parameter vector
    pub fn bounds(&mut self, lower_bound: &T, upper_bound: &T) -> &mut Self {
        self.lower_bound = Some(lower_bound.clone());
        self.upper_bound = Some(upper_bound.clone());
        self
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

    /// Set target cost function value
    ///
    /// If the optimization reaches this value, it will be stopped.
    pub fn target_cost(&mut self, target_cost: U) -> &mut Self {
        self.target_cost = target_cost;
        self
    }

    /// Create a random parameter vector
    ///
    /// The parameter vector satisfies the `lower_bound` and `upper_bound`.
    pub fn random_param(&self) -> Result<T> {
        match (self.lower_bound.as_ref(), self.upper_bound.as_ref()) {
            (Some(l), Some(u)) => Ok(T::random(l, u)),
            _ => Err(ErrorKind::InvalidParameter(
                "Parameter: lower_bound and upper_bound must be provided.".into(),
            ).into()),
        }
    }
}

unsafe impl<'a, T: ArgminParameter + 'a, U: ArgminCostValue + 'a, V: 'a> Send
    for ArgminProblem<'a, T, U, V>
{
}
unsafe impl<'a, T: ArgminParameter + 'a, U: ArgminCostValue + 'a, V: 'a> Sync
    for ArgminProblem<'a, T, U, V>
{
}
