/// TODO DOCUMENTATION
///
use errors::*;
use ndarray::{Array1, Array2};

pub struct ArgminOperator<'a> {
    /// Operator (for now a simple 2D matrix)
    operator: &'a Array2<f64>,
}

impl<'a> ArgminOperator<'a> {
    /// Constructor
    pub fn new(operator: &'a Array2<f64>) -> Self {
        ArgminOperator { operator: operator }
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
