// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! A (hopefully) simple example of using Nelder-Mead to find the roots of a
//! cubic polynomial.
//!
//! You can run this example with:
//! `cargo run --example neldermead-cubic --features slog-logger`

use argmin::core::observers::ObserverMode;
use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::neldermead::NelderMead;
use argmin_observer_slog::SlogLogger;

/// Coefficients describing a cubic `f(x) = ax^3 + bx^2 + cx + d`
#[derive(Clone, Copy)]
struct Cubic {
    /// Coefficient of the `x^3` term
    a: f64,
    /// Coefficient of the `x^2` term
    b: f64,
    /// Coefficient of the `x` term
    c: f64,
    /// Coefficient of the `x^0` term
    d: f64,
}

impl Cubic {
    /// Evaluate the cubic at `x`.
    fn eval(self, x: f64) -> f64 {
        self.a * x.powi(3) + self.b * x.powi(2) + self.c * x + self.d
    }
}

impl CostFunction for Cubic {
    type Param = f64;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        // The cost function is the evaluation of the polynomial with our
        // parameter, squared. The parameter is a guess of `x`, and the
        // objective is to minimize `x` (i.e. find a polynomial root). The
        // square value can be considered an error. We want the error to (1)
        // always be positive and (2) bigger the further it is from a polynomial
        // root.
        Ok(self.eval(*p).powi(2))
    }
}

fn run() -> Result<(), Error> {
    // Define the cost function. This needs to be something with an
    // implementation of `CostFunction`; in this case, the impl is right
    // above. Here, our cubic is `(x-2)(x+2)(x-5)`; see
    // <https://www.wolframalpha.com/input?i=%28x-2%29%28x%2B2%29%28x-5%29> for
    // more info.
    let cost = Cubic {
        a: 1.0,
        b: -5.0,
        c: -4.0,
        d: 20.0,
    };

    // Let's find a root of the cubic (+5).
    {
        // Set up solver -- note that the proper choice of the vertices is very
        // important! This example should find 5, because our vertices are 6 and 7.
        let solver = NelderMead::new(vec![6.0, 7.0]).with_sd_tolerance(0.0001)?;

        // Run solver
        let res = Executor::new(cost, solver)
            .configure(|state| state.max_iters(100))
            .add_observer(SlogLogger::term(), ObserverMode::Always)
            .run()?;

        // Wait a second (lets the logger flush everything before printing again)
        std::thread::sleep(std::time::Duration::from_secs(1));

        // Print result
        println!(
            "Polynomial root: {}",
            res.state.get_best_param().expect("Found a root")
        );
    }

    // Now find -2.
    {
        let solver = NelderMead::new(vec![-3.0, -4.0]).with_sd_tolerance(0.0001)?;
        let res = Executor::new(cost, solver)
            .configure(|state| state.max_iters(100))
            .add_observer(SlogLogger::term(), ObserverMode::Always)
            .run()?;
        std::thread::sleep(std::time::Duration::from_secs(1));
        println!("{res}");
        println!(
            "Polynomial root: {}",
            res.state.get_best_param().expect("Found a root")
        );
    }

    // This example will find +2, even though it might look like we're trying to
    // find +5.
    {
        let solver = NelderMead::new(vec![4.0, 6.0]).with_sd_tolerance(0.0001)?;
        let res = Executor::new(cost, solver)
            .configure(|state| state.max_iters(100))
            .add_observer(SlogLogger::term(), ObserverMode::Always)
            .run()?;
        std::thread::sleep(std::time::Duration::from_secs(1));
        println!("{res}");
        println!(
            "Polynomial root: {}",
            res.state.get_best_param().expect("Found a root")
        );
    }

    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("{e}");
        std::process::exit(1);
    }
}
