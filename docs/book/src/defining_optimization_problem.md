# Defining an optimization problem

TODO TODO TODO TODO Rewrite! Also mention bulk methods.
A problem can be defined by implementing the `ArgminOp` trait which comes with the
associated types `Param`, `Output` and `Hessian`. `Param` is the type of your
parameter vector (i.e. the input to your cost function), `Output` is the type returned
by the cost function, `Hessian` is the type of the Hessian and `Jacobian` is the type of the
Jacobian.
The trait provides the following methods:

- `apply(&self, p: &Self::Param) -> Result<Self::Output, Error>`: Applys the cost
  function to parameters `p` of type `Self::Param` and returns the cost function value.
- `gradient(&self, p: &Self::Param) -> Result<Self::Param, Error>`: Computes the
  gradient at `p`.
- `hessian(&self, p: &Self::Param) -> Result<Self::Hessian, Error>`: Computes the Hessian
  at `p`.
- `jacobian(&self, p: &Self::Param) -> Result<Self::Jacobian, Error>`: Computes the Jacobian
  at `p`.

The following code snippet shows an example of how to use the Rosenbrock test functions from
`argmin-testfunctions` in argmin:

```rust
# extern crate argmin;
# extern crate argmin_testfunctions;
use argmin_testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative, rosenbrock_2d_hessian};
use argmin::core::{Error, CostFunction, Gradient, Hessian};

/// First, create a struct for your problem
struct Rosenbrock {
    a: f64,
    b: f64,
}

/// Implement `CostFunction` for `Rosenbrock`
impl CostFunction for Rosenbrock {
    /// Type of the parameter vector
    type Param = Vec<f64>;
    /// Type of the return value computed by the cost function
    type Output = f64;

    /// Apply the cost function to a parameter `p`
    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(rosenbrock_2d(p, 1.0, 100.0))
    }
}

/// Implement `Gradient` for `Rosenbrock`
impl Gradient for Rosenbrock {
    /// Type of the parameter vector
    type Param = Vec<f64>;
    /// Type of the return value computed by the cost function
    type Gradient = Vec<f64>;

    /// Compute the gradient at parameter `p`.
    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
        Ok(rosenbrock_2d_derivative(p, 1.0, 100.0))
    }
}

/// Implement `Hessian` for `Rosenbrock`
impl Hessian for Rosenbrock {
    /// Type of the parameter vector
    type Param = Vec<f64>;
    /// Type of the Hessian. Can be `()` if not needed.
    type Hessian = Vec<Vec<f64>>;

    /// Compute the Hessian at parameter `p`.
    fn hessian(&self, p: &Self::Param) -> Result<Self::Hessian, Error> {
        let t = rosenbrock_2d_hessian(p, 1.0, 100.0);
        Ok(vec![vec![t[0], t[1]], vec![t[2], t[3]]])
    }
}
```

It is optional to implement any of these methods, as there are default implementations which
will return an `Err` when called. What needs to be implemented is defined by the requirements
of the solver that is to be used.
