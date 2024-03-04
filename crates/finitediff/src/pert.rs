// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

/// Perturbation Vector for the accelerated computation of the Jacobian.
#[derive(Clone, Default)]
pub struct PerturbationVector {
    /// x indices
    pub x_idx: Vec<usize>,
    /// corresponding function indices
    pub r_idx: Vec<Vec<usize>>,
}

impl PerturbationVector {
    /// Create a new empty `PerturbationVector`
    pub fn new() -> Self {
        PerturbationVector {
            x_idx: vec![],
            r_idx: vec![],
        }
    }

    /// Add an index `x_idx` and the corresponding function indices `r_idx`
    pub fn add(mut self, x_idx: usize, r_idx: Vec<usize>) -> Self {
        self.x_idx.push(x_idx);
        self.r_idx.push(r_idx);
        self
    }
}

/// A collection of `PerturbationVector`s
pub type PerturbationVectors = Vec<PerturbationVector>;
