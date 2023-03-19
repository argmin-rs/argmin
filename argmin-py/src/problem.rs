// Copyright 2018-2023 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

// TODO: docs

use numpy::ToPyArray;
use pyo3::{prelude::*, types::PyTuple};

use argmin::core;

use crate::types::{Array1, Array2, Scalar};

#[pyclass]
#[derive(Clone)]
pub struct Problem {
    gradient: PyObject,
    hessian: PyObject,
    // TODO: jacobian
}

#[pymethods]
impl Problem {
    #[new]
    fn new(gradient: PyObject, hessian: PyObject) -> Self {
        Self { gradient, hessian }
    }
}

impl core::Gradient for Problem {
    type Param = Array1;
    type Gradient = Array1;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        call(&self.gradient, param)
    }
}

impl argmin::core::Hessian for Problem {
    type Param = Array1;

    type Hessian = Array2;

    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, core::Error> {
        call(&self.hessian, param)
    }
}

fn call<InputDimension, OutputDimension>(
    callable: &PyObject,
    param: &ndarray::Array<Scalar, InputDimension>,
) -> Result<ndarray::Array<Scalar, OutputDimension>, argmin::core::Error>
where
    InputDimension: ndarray::Dimension,
    OutputDimension: ndarray::Dimension,
{
    // TODO: prevent dynamic dispatch for every call
    Python::with_gil(|py| {
        let args = PyTuple::new(py, [param.to_pyarray(py)]);
        let pyresult = callable.call(py, args, Default::default())?;
        let pyarray = pyresult.extract::<&numpy::PyArray<Scalar, OutputDimension>>(py)?;
        // TODO: try to get ownership instead of cloning
        Ok(pyarray.to_owned_array())
    })
}
