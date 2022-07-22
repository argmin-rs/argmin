# Basic concept

There are three components needed for solving an optimization problem in argmin:

* A definition of the optimization problem / model
* A solver
* An executor

The [Executor](https://docs.rs/argmin/latest/argmin/core/struct.Executor.html) applies the solver to the optimization problem.
It also accepts observers and checkpointing mechanisms, as well as an initial guess of the parameter vector, the cost function value at that initial guess, gradient, and so on.

A solver is anything that implements the [Solver](https://docs.rs/argmin/latest/argmin/core/trait.Solver.html) trait.
This trait defines how the optimization algorithm is initialized, how a single iteration is performed and when and how to terminate the iterations.

The optimization problem needs to implement a subset of the traits
[`CostFunction`](https://docs.rs/argmin/latest/argmin/core/trait.CostFunction.html),
[`Gradient`](https://docs.rs/argmin/latest/argmin/core/trait.Gradient.html),
[`Jacobian`](https://docs.rs/argmin/latest/argmin/core/trait.Jacobian.html),
[`Hessian`](https://docs.rs/argmin/latest/argmin/core/trait.Hessian.html), and
[`Operator`](https://docs.rs/argmin/latest/argmin/core/trait.Operator.html).
Which subset is needed is given by the requirements of the solver.
For example, `SteepestDescent`, as a gradient descent method, requires `CostFunction` and `Gradient`, while Newton's method expects `Gradient` and `Hessian`.

