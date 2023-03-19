// Copyright 2018-2023 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Base types for the Python extension.

pub type Scalar = f64; // TODO: allow complex numbers
pub type Array1 = ndarray::Array1<Scalar>;
pub type Array2 = ndarray::Array2<Scalar>;
pub type PyArray1 = numpy::PyArray1<Scalar>;

pub type IterState = argmin::core::IterState<Array1, Array1, (), ndarray::Array2<Scalar>, Scalar>;
