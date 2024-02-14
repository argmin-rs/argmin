# Defining an optimization problem

Depending on the requirements of the solver that is to be used, the optimization problem needs to implement a subset of the traits

- [`CostFunction`](https://docs.rs/argmin/latest/argmin/core/trait.CostFunction.html): Computes the cost or fitness for a parameter vector `p`
- [`Gradient`](https://docs.rs/argmin/latest/argmin/core/trait.Gradient.html): Computes the gradient for a parameter vector `p`
- [`Jacobian`](https://docs.rs/argmin/latest/argmin/core/trait.Jacobian.html): Computes the Jacobian for a parameter vector `p`
- [`Hessian`](https://docs.rs/argmin/latest/argmin/core/trait.Hessian.html): Computes the Hessian for a parameter vector `p`
- [`Operator`](https://docs.rs/argmin/latest/argmin/core/trait.Operator.html): Applies an operator to the parameter vector `p`
- [`Anneal`](https://docs.rs/argmin/latest/argmin/solver/simulatedannealing/trait.Anneal.html): Create a new parameter vector by "annealing" of the current parameter vector `p` (needed for SimulatedAnnealing).

Which subset is needed is given in the documentation of each solver.

## Example

The following code snippet shows how to use the Rosenbrock test functions from `argmin-testfunctions` in argmin.
For the sake of simplicity, this example will use the `vec` math backend.


```rust
# extern crate argmin;
# extern crate argmin_testfunctions;
use argmin_testfunctions::{
    rosenbrock, rosenbrock_derivative, rosenbrock_hessian
};
use argmin::core::{Error, CostFunction, Gradient, Hessian};

/// First, we create a struct called `Rosenbrock` for your problem
struct Rosenbrock {}

/// Implement `CostFunction` for `Rosenbrock`
///
/// First, we need to define the types which we will be using. Our parameter
/// vector will be a `Vec` of `f64` values and our cost function value will 
/// be a 64 bit floating point value.
/// This is reflected in the associated types `Param` and `Output`, respectively.
///
/// The method `cost` then defines how the cost function is computed for a
/// parameter vector `p`. Note that we have access to the fields `a` and `b`
/// of `Rosenbrock`.
impl CostFunction for Rosenbrock {
    /// Type of the parameter vector
    type Param = Vec<f64>;
    /// Type of the return value computed by the cost function
    type Output = f64;

    /// Apply the cost function to a parameter `p`
    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        // Evaluate Rosenbrock function
        Ok(rosenbrock(p))
    }
}

/// Implement `Gradient` for `Rosenbrock`
///
/// Similarly to `CostFunction`, we need to define the type of our parameter
/// vectors and of the gradient we are computing. Since the gradient is also
/// a vector, it is of type `Vec<f64>` just like `Param`.
impl Gradient for Rosenbrock {
    /// Type of the parameter vector
    type Param = Vec<f64>;
    /// Type of the gradient
    type Gradient = Vec<f64>;

    /// Compute the gradient at parameter `p`.
    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
        // Compute gradient of the Rosenbrock function
        Ok(rosenbrock_derivative(p))
    }
}

/// Implement `Hessian` for `Rosenbrock`
///
/// Again the types of the involved parameter vector and the Hessian needs to
/// be defined. Since the Hessian is a 2D matrix, we use `Vec<Vec<f64>>` here.
impl Hessian for Rosenbrock {
    /// Type of the parameter vector
    type Param = Vec<f64>;
    /// Type of the Hessian
    type Hessian = Vec<Vec<f64>>;

    /// Compute the Hessian at parameter `p`.
    fn hessian(&self, p: &Self::Param) -> Result<Self::Hessian, Error> {
        // Compute Hessian of the Rosenbrock function
        Ok(rosenbrock_hessian(p))
    }
}
```

The following example shows how to implement the `Operator` trait when the operator is the 2x2 matrix `[4.0, 1.0; 1.0, 3.0]`.

```rust
# extern crate argmin;
# extern crate argmin_testfunctions;
use argmin::core::{Error, Operator};

struct MyProblem {}

impl Operator for MyProblem {
    /// Type of the parameter vector
    type Param = Vec<f64>;
    /// Type of the output
    type Output = Vec<f64>;

    fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(vec![4.0 * p[0] + 1.0 * p[1], 1.0 * p[0] + 3.0 * p[1]])
    }
}
```

## Parallel evaluation with `bulk_*` methods

> **NOTE**
>
> So far only Particle Swarm Optimization allows parallel evaluations of parameter vectors.
> Therefore no increase in performance should be expected for other solvers. 

All of the above mentioned traits come with additional methods which enable processing several parameter vectors at once.
These methods require the `rayon` feature to be enabled. Without this feature, the methods resort to sequental processing of all inputs.
The methods are shown in the following table:

Trait        | Single input      | Multiple inputs |
-------------|-------------------|-----------------|
`CostFunction` | `cost`              | `bulk_cost`       |
`Gradient`     | `gradient`          | `bulk_gradient`   |
`Jacobian`     | `jacobian`          | `bulk_jacobian`   |
`Hessian`      | `hessian`           | `bulk_hessian`    |
`Operator`     | `apply`             | `bulk_apply`      |
`Anneal`       | `anneal`            | `bulk_anneal`     |

These `bulk_*` methods come with a default implementation, which essentially looks like this:

```rust
# extern crate argmin;
# extern crate argmin_testfunctions;
# use argmin_testfunctions::{rosenbrock, rosenbrock_derivative, rosenbrock_hessian};
use argmin::core::{Error, CostFunction, SyncAlias, SendAlias};
# struct Rosenbrock {}

impl CostFunction for Rosenbrock {
    type Param = Vec<f64>;
    type Output = f64;

    /// Conventional cost function which only processes a single parameter vector
    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        // [ ... ]
        # Ok(rosenbrock(p))
    }
    
    ////////////////////////////////////////////////////////////////////////////
    /// Bulk cost function which processes an array of parameter vectors     ///
    ////////////////////////////////////////////////////////////////////////////
    fn bulk_cost<'a, P>(&self, params: &'a [P]) -> Result<Vec<Self::Output>, Error>
    where
        P: std::borrow::Borrow<Self::Param> + SyncAlias,
        Self::Output: SendAlias,
        Self: SyncAlias,
    {
        #[cfg(feature = "rayon")]
        {
            if self.parallelize() {
                params.par_iter().map(|p| self.cost(p.borrow())).collect()
            } else {
                params.iter().map(|p| self.cost(p.borrow())).collect()
            }
        }
        #[cfg(not(feature = "rayon"))]
        {
            params.iter().map(|p| self.cost(p.borrow())).collect()
        }
    }
    
    /// Indicates whether parameter vectors passed to `bulk_cost` should be
    /// evaluated in parallel or sequentially.
    /// This allows one to turn off parallelization for individual traits, even if
    /// the `rayon` feature is enabled.
    fn parallelize(&self) -> bool {
        true
    }
}
```

If needed, the `bulk_*` methods can be overwritten by custom implementations.

Besides the `bulk_*` method, there is also a method `parallelize` which by default returns `true`.
This is a convenient way to turn parallelization off, even if the `rayon` feature is enabled.
It allows one to for instance turn parallel processing on for `CostFunction` and off for `Gradient`. 
To turn it off, simply overwrite `parallelize` such that it returns `false`. 

