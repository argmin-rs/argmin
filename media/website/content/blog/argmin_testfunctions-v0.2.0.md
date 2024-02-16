+++
title = "argmin_testfunctions 0.2.0 and argmin-testfunctions-py v0.0.1 released"
description = ""
date = 2024-02-16T00:00:00+00:00
updated = 2024-02-16T00:00:00+00:00
draft = false
template = "blog/page.html"

[taxonomies]
authors = ["Stefan Kroboth"]

[extra]
+++

<b>argmin</b> is a Rust library which offers a range of numerical optimization methods and is a framework for 
developing optimization algorithms.  Details about the design and features of argmin can be found on
[the website](https://argmin-rs.org),
in the [book](https://argmin-rs.org/book),
on [Github](https://github.com/argmin-rs/argmin),
on [crates.io](https://crates.io/crates/argmin) and
on [lib.rs](https://lib.rs/crates/argmin).

<b>argmin_testfunctions</b> is one of the Rust libraries in the argmin ecosystem and provides a wide range of
test functions for optimization problems (not just for argmin). For many years it has been sitting around in its repo without
getting much attention.
I decided to change that by pulling the code into the argmin monorepo, updating it to the current Rust edition
and started implementing new features. The main addition is the calculation of the derivatives and Hessians
for all test functions. 

As input, the functions now take parameters either as `&[T]`, `&[T; 2]`, or `&[T; N]`, where `N` is a const generic and
`T: Float` (`f64` and `f32`).
Functions which only accept two dimensional input accept `&[T; 2]` and return `T` (test function), `[T; 2]` (derivative), or `[[T; 2]; 2]`
(Hessian).
An example for this is the Himmelblau test function:

```rust
use argmin_testfunctions::{himmelblau, himmelblau_derivative, himmelblau_hessian};

let p: [f64; 2] = [0.1, 0.2];

let c: f64 = himmelblau(&p);
let d: [f64; 2] = himmelblau_derivative(&p);
let h: [[f64; 2]; 2] = himmelblau_hessian(&p);
```

Functions which accept an arbitrary number of parameters come in two forms. One variant accepts `&[T]` and returns
`T` (test function), `Vec<T>` (derivative), or `Vec<Vec<T>>` (Hessian), for instance:

```rust
use argmin_testfunctions::{rosenbrock, rosenbrock_derivative, rosenbrock_hessian};

let p = [0.1, 0.2, 0.3];

let c: f64 = rosenbrock(&p);
let d: Vec<f64> = rosenbrock_derivative(&p);
let h: Vec<Vec<f64>> = rosenbrock_hessian(&p);
```

This obviously allocates due to the `Vec`s and as such isn't the most efficient. If the number of parameters are known
at compile-time, one can use the const generics versions of the functions which take `&[T; N]` and return 
`[T; N]` (derivative) or `[[T; N]; N]` (Hessian):

```rust
use argmin_testfunctions::{
    rosenbrock, rosenbrock_derivative_const, rosenbrock_hessian_const
};

let p: [f64; 3] = [0.1, 0.2, 0.3];

let c: f64 = rosenbrock(&p);
let d: [f64; 3] = rosenbrock_derivative_const(&p);
let h: [[f64; 3]; 3] = rosenbrock_hessian_const(&p);
```

This does not allocate and hence is faster, but requires the number of parameters to be known at compile-time.

The Rosenbrock test function also has additional optional parameters `a` and `b` which can be adjusted to one's needs:

```rust
use argmin_testfunctions::{
    rosenbrock_ab, rosenbrock_ab_derivative, rosenbrock_ab_derivative_const,
    rosenbrock_ab_hessian, rosenbrock_ab_hessian_const
};

let p = [0.1, 0.2, 0.3];
let a = 5.0;
let b = 200.0;

let c: f64 = rosenbrock_ab(&p, a, b);
let d: Vec<f64> = rosenbrock_ab_derivative(&p, a, b);
let h: Vec<Vec<f64>> = rosenbrock_ab_hessian(&p, a, b);

// There are also const generics versions
let d: [f64; 3] = rosenbrock_ab_derivative_const(&p, a, b);
let h: [[f64; 3]; 3] = rosenbrock_ab_hessian_const(&p, a, b);
```

Similar functions exist for `ackley` (`ackley_abc`) and `rastrigin` (`rastrigin_a`).


## Python interface

The new Python module `argmin-testfunctions-py` makes the library also available in Python.
This module is essentially a thin wrapper around the Rust library implemented using PyO3.
Apart from the missing const generics versions, it features the same functionality as the Rust library.
A notable difference is that optional parameters such as `a` and `b` of the Rosenbrock test function
are implemented as optional arguments in Python:

```python
from argmin_testfunctions_py import *

# Adjusting `a` and `b`
c = rosenbrock([0.1, 0.2, 0.3, 0.4], a = 5.0, b = 200.0)
g = rosenbrock_derivative([0.1, 0.2, 0.3, 0.4], a = 5.0, b = 200.0)
h = rosenbrock_hessian([0.1, 0.2, 0.3, 0.4], a = 5.0, b = 200.0)

# When `a` and `b` are omitted, they will default to `a = 1.0` and `b = 100.0`
c = rosenbrock([0.1, 0.2, 0.3, 0.4])
g = rosenbrock_derivative([0.1, 0.2, 0.3, 0.4])
h = rosenbrock_hessian([0.1, 0.2, 0.3, 0.4])
```

This module is still a bit experimental, which is expressed with the version number 0.0.1. 
Eventually the version of the Python module may end up in lockstep with `argmin_testfunctions`' version number.

It can be found on [PyPI](https://pypi.org/project/argmin-testfunctions-py/) and installed via
`pip install argmin-testfunctions-py`.


## Tests and Benchmarks

All test functions are tested against known points and the derivative and Hessian calculations
are tested against finite differences (using [finitediff](https://crates.io/crates/finitediff)).
For each test, multiple parameter vectors are evaluated using [proptest](https://crates.io/crates/proptest). 
As usual, tests can be executed with `cargo test`. 

The benchmarks now use [criterion.rs](https://crates.io/crates/criterion). `cargo bench` will benchmark all
functions and produce a report in `target/criterion/report/index.html`. 


## Contributing

This library's usefulness is directly proportional to the number of test functions it provides.
Therefore all contributions to this library, in particular additional test functions, are highly welcome.
Interested people should have a look
at [this issue](https://github.com/argmin-rs/argmin/issues/450) which provides inspiration for what
to implement as well as a couple of hints on how to implement it such that it best fits into the library.

While I believe that most implementations are quite efficient, I am sure that there are some which could
be improved. If anyone wants to get the last bit of performance out of this, feel free to open a PR.

Ideally this library would also print the formulas in the docs
[as actual formulas](https://github.com/argmin-rs/argmin/issues/418) and having
[visualizations of the test functions](https://github.com/argmin-rs/argmin/issues/451) in the docs would
be a nice addition.

## Conclusion

With this release the `argmin_testfunctions` crate finally got derivative and Hessian calculations for 
all test functions. 
Thanks to `argmin-testfunctions-py` the functionality is now also available in Python.
Hopefully this crate will be of use to some. If you find bugs or have suggestions or feedback, feel free to get
in touch, either via Github or the Discord.

Details on the individual test functions can be found in the documentation, either for the
[latest release](https://docs.rs/argmin_testfunctions) or
[the current main branch](https://argmin-rs.github.io/argmin/argmin_testfunctions/index.html).


## Discord server

If you're interested you're invited to join the  [Discord](https://discord.gg/fYB8AwxxMW)!


<br>
<script async defer src="https://buttons.github.io/buttons.js"></script>
<p align="center">
<a class="github-button" href="https://github.com/argmin-rs/argmin" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star argmin-rs/argmin on GitHub">Star</a>
<a class="github-button" href="https://github.com/argmin-rs/argmin/subscription" data-icon="octicon-eye" data-size="large" data-show-count="true" aria-label="Watch argmin-rs/argmin on GitHub">Watch</a>
<a class="github-button" href="https://github.com/argmin-rs/argmin/fork" data-icon="octicon-repo-forked" data-size="large" data-show-count="true" aria-label="Fork argmin-rs/argmin on GitHub">Fork</a>
<a class="github-button" href="https://github.com/sponsors/stefan-k" data-icon="octicon-heart" data-size="large" aria-label="Sponsor @stefan-k on GitHub">Sponsor</a>
</p>
