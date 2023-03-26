# Implementing a solver

In this section we are going to implement the Landweber solver, which essentially is a special form of gradient descent.
In iteration \\( k \\), the new parameter vector \\( x\_{k+1} \\) is calculated from the previous parameter vector \\( x_k \\) and the gradient at \\( x_k \\) according to the following update rule:

\\[
x_{k+1} = x_k - \omega * \nabla f(x_k)
\\]

In order to implement this using argmin, one first needs to define the struct `Landweber` which holds data specific to the solver (the step length \\( \omega \\)).
Then, the [`Solver`](https://docs.rs/argmin/latest/argmin/core/trait.Solver.html) trait needs to be implemented for this struct.

The `Solver` trait consists of several methods; however, not all of them need to be implemented since most come with default implementations.

- `NAME`: a `&'static str` which holds the solvers name (mainly needed for the observers).
- `init(...)`: Run before the the actual iterations and initializes the solver. Does nothing by default.
- `next_iter(...)`: One iteration of the solver. Will be executed by the `Executor` until a stopping criterion is met.
- `terminate(...)`: Solver specific stopping criteria. This method is run after every iteration. Note that one can also terminate from within `next_iter` if necessary.
- `terminate_internal(...)`: By default calls `terminate` and in addition checks if the maximum number of iterations was reached or if the best cost function value is below the target cost value. Should only be overwritten if absolutely necessary.

Both `init` and `next_iter` have access to the optimization problem (`problem`) as well as the internal state (`state`).
The methods `terminate` and `terminate_internal` only have access to `state`.

The function parameter `problem` is a wrapped version of the optimization problem and as such gives access to the cost function, gradient, Hessian, Jacobian,...).
It also keeps track of how often each of these is called.

Via `state` the solver has access to the current parameter vector, the current best parameter vector, gradient, Hessian, Jacobian, population, the current iteration number, and so on.
The `state` can be modified (for instance a new parameter vector is set) and is then returned by both `init` and `next_iter`.
The `Executor` then takes care of updating the state properly, for instance by updating the current best parameter vector if the new parameter vector is better than the previous best.

It is advisable to design the solver such that it is generic over the actual type of the parameter vector, gradient, and so on.

The current naming convention for generics in argmin is as follows:

- `O`: Optimization problem
- `P`: Parameter vector
- `G`: Gradient
- `J`: Jacobian
- `H`: Hessian
- `F`: Floats (`f32` or `f64`)

These individual generic parameters are then constrained by type constraints.
For instance, the Landweber iteration requires the problem `O` to implement `Gradient`, therefore a trait bound of the form `O: Gradient<Param = P, Gradient = G>` is necessary.

From the Landweber update formula, we know that a scaled subtraction of two vectors is required.
This must be represented in form of a trait bound as well: `P: ArgminScaledSub<G, F, P>`.
`ArgminScaledSub` is a trait from `argmin-math` which represents a scaled subtraction.
With this trait bound, we require that it must be possible to subtract a value of type `G` scaled with a value of type `F` from a value of type `P`, resulting in a value of type `P`.

The generic type `F` represents floating point value and therefore allows users to choose which precision they want.

Implementing the algorithm is straightforward: First we get the current parameter vector `xk` from the state via `state.take_param()`.
Note that `take_param` moves the parameter vector from the `state` into `xk`, therefore one needs to make sure to move the updated parameter vector into `state` at the end of `next_iter` via `state.param(...)`.
Landweber requires the user to provide an initial parameter vector.
If this is not the case than we return an error to inform the user.
Then the gradient `grad` is computed by calling `problem.gradient(...)` on the parameter vector.
This will return the gradient and internally increase the gradient function evaluation count.
We compute the updated parameter vector `xkp1` by computing `xk.scaled_sub(&self.omega, &grad)` (which is possible because of the `ArgminScaledSub` trait bound introduced before).
Finally, the state is updated via `state.param(xkp1)` and returned by the function.

```rust
# extern crate argmin;
#
use argmin::core::{
    ArgminFloat, KV, Error, Gradient, IterState, Problem, Solver, State
};
use argmin::argmin_error_closure;
use serde::{Deserialize, Serialize};
use argmin_math::ArgminScaledSub;

// Define a struct which holds any parameters/data which are needed during the
// execution of the solver. Note that this does not include parameter vectors,
// gradients, Hessians, cost function values and so on, as those will be
// handled by the `Executor` and its internal state.
#[derive(Serialize, Deserialize)]
pub struct Landweber<F> {
    /// Step length
    omega: F,
}

impl<F> Landweber<F> {
    /// Constructor
    pub fn new(omega: F) -> Self {
        Landweber { omega }
    }
}

impl<O, F, P, G> Solver<O, IterState<P, G, (), (), F>> for Landweber<F>
where
    // The Landweber solver requires `O` to implement `Gradient`.
    // `P` and `G` indicate the types of the parameter vector and gradient,
    // respectively.
    O: Gradient<Param = P, Gradient = G>,
    // The parameter vector of type `P` needs to implement `ArgminScaledSub`
    // because of the update formula
    P: Clone + ArgminScaledSub<G, F, P>,
    // `F` is the floating point type (`f32` or `f64`)
    F: ArgminFloat,
{
    // This gives the solver a name which will be used for logging
    fn name(&self) -> &str { "Landweber" }

    // Defines the computations performed in a single iteration.
    fn next_iter(
        &mut self,
        // This gives access to the problem supplied to the `Executor`. `O` implements
        // `Gradient` and `Problem` takes care of counting the calls to the respective
        // functions.
        problem: &mut Problem<O>,
        // Current state of the optimization. This gives access to the parameter
        // vector, gradient, Hessian and cost function value of the current,
        // previous and best iteration as well as current iteration number, and
        // many more.
        mut state: IterState<P, G, (), (), F>,
    ) -> Result<(IterState<P, G, (), (), F>, Option<KV>), Error> {
        // First we obtain the current parameter vector from the `state` struct (`x_k`).
        // Landweber requires an initial parameter vector. Return an error if this was
        // not provided by the user.
        let xk = state.take_param().ok_or_else(argmin_error_closure!(
            NotInitialized,
            "Initial parameter vector required!"
        ))?;

        // Then we compute the gradient at `x_k` (`\nabla f(x_k)`)
        let grad = problem.gradient(&xk)?;

        // Now subtract `\nabla f(x_k)` scaled by `omega` from `x_k`
        // to compute `x_{k+1}`
        let xkp1 = xk.scaled_sub(&self.omega, &grad);

        // Return new the updated `state`
        Ok((state.param(xkp1), None))
    }
}
```

Another example with a more complex solver which requires `init` as well as returning a `KV` is (hopefully) coming soon!
