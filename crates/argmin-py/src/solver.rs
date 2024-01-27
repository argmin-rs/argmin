// Copyright 2018-2023 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

// TODO: docs

use pyo3::prelude::*;

use argmin::{core, solver};

use crate::{problem::Problem, types::IterState};

#[pyclass]
#[derive(Clone)]
pub enum Solver {
    Newton,
}

pub struct DynamicSolver(Box<dyn core::Solver<Problem, IterState> + Send>);

impl From<Solver> for DynamicSolver {
    fn from(value: Solver) -> Self {
        let inner = match value {
            Solver::Newton => solver::newton::Newton::new(),
        };
        Self(Box::new(inner))
    }
}

impl core::Solver<Problem, IterState> for DynamicSolver {
    // TODO: make this a trait method so we can return a dynamic
    fn name(&self) -> &str {
        self.0.name()
    }

    fn next_iter(
        &mut self,
        problem: &mut core::Problem<Problem>,
        state: IterState,
    ) -> Result<(IterState, Option<core::KV>), core::Error> {
        self.0.next_iter(problem, state)
    }
}
