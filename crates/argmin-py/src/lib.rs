// Copyright 2018-2023 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

// TODO: docs
mod executor;
mod problem;
mod solver;
mod types;

use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "argmin")]
fn argmin_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<executor::Executor>()?;
    m.add_class::<problem::Problem>()?;
    m.add_class::<solver::Solver>()?;

    Ok(())
}
