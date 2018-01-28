// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

/// TODO DOCUMENTATION
///
use ndarray::{Array1, Array2};

/// ArgminOperator
pub struct ArgminOperator<'a> {
    /// Operator (for now a simple 2D matrix)
    pub operator: &'a Array2<f64>,
    /// y of Ax = y
    pub y: &'a Array1<f64>,
}

impl<'a> ArgminOperator<'a> {
    /// Constructor
    pub fn new(operator: &'a Array2<f64>, y: &'a Array1<f64>) -> Self {
        ArgminOperator {
            operator: operator,
            y: y,
        }
    }

    /// Forward application of the operator (A*x)
    pub fn apply(&self, x: &Array1<f64>) -> Array1<f64> {
        self.operator.dot(x)
    }

    /// Application of the transpose of the operator (A^T*x)
    pub fn apply_transpose(&self, x: &Array1<f64>) -> Array1<f64> {
        self.operator.t().dot(x)
    }
}
