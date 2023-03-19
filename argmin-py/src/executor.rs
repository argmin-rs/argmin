// Copyright 2018-2023 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

// TODO: docs

use pyo3::{prelude::*, types::PyDict};

use argmin::core;

use crate::problem::Problem;
use crate::solver::{DynamicSolver, Solver};
use crate::types::{IterState, PyArray1};

#[pyclass]
pub struct Executor(Option<core::Executor<Problem, DynamicSolver, IterState>>);

impl Executor {
    /// Consumes the inner executor.
    ///
    /// PyObjects do not allow methods that consume the object itself, so this is a workaround
    /// for using methods like `configure` and `run`.
    fn take(&mut self) -> anyhow::Result<core::Executor<Problem, DynamicSolver, IterState>> {
        let Some(inner) = self.0.take() else {
            return Err(anyhow::anyhow!("Executor was already run."));
        };
        Ok(inner)
    }
}

#[pymethods]
impl Executor {
    #[new]
    fn new(problem: Problem, solver: Solver) -> Self {
        Self(Some(core::Executor::new(problem, solver.into())))
    }

    #[pyo3(signature = (**kwargs))]
    fn configure(&mut self, kwargs: Option<&PyDict>) -> PyResult<()> {
        if let Some(kwargs) = kwargs {
            let new_self = self.take()?.configure(|mut state| {
                if let Some(param) = kwargs.get_item("param") {
                    let param: &PyArray1 = param.extract().unwrap();
                    state = state.param(param.to_owned_array());
                }
                if let Some(max_iters) = kwargs.get_item("max_iters") {
                    state = state.max_iters(max_iters.extract().unwrap());
                }
                state
            });
            self.0 = Some(new_self);
        }
        Ok(())
    }

    fn run(&mut self) -> PyResult<String> {
        // TODO: return usable OptimizationResult
        let res = self.take()?.run();
        Ok(res?.to_string())
    }
}
