// Copyright 2018-2023 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

// TODO: docs

use std::path::Iter;

use pyo3::prelude::*;

use argmin::{core, solver};

use crate::{
    problem::Problem,
    types::{IterState, Scalar},
};

#[pyclass]
#[derive(Clone)]
pub enum Solver {
    Newton,
}

pub enum DynamicSolver {
    // NOTE: I tried using a Box<dyn Solver<> here, but Solver is not object safe.
    Newton(solver::newton::Newton<Scalar>),
}

impl From<Solver> for DynamicSolver {
    fn from(solver: Solver) -> Self {
        match solver {
            Solver::Newton => Self::Newton(solver::newton::Newton::new()),
        }
    }
}

impl core::Solver<Problem, IterState> for DynamicSolver {
    // TODO: make this a trait method so we can return a dynamic
    const NAME: &'static str = "Dynamic Solver";

    fn name(&self) -> &str {
        match self {
            DynamicSolver::Newton(inner) => {
                <argmin::solver::newton::Newton<f64> as argmin::core::Solver<Problem, IterState>>
                ::name(inner)
            }
        }
    }

    fn next_iter(
        &mut self,
        problem: &mut core::Problem<Problem>,
        state: IterState,
    ) -> Result<(IterState, Option<core::KV>), core::Error> {
        match self {
            DynamicSolver::Newton(inner) => inner.next_iter(problem, state),
        }
    }
}
