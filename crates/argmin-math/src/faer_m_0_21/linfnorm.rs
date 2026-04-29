// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminLInfNorm;
use faer::mat::AsMatRef;
use faer::{Mat, MatRef};
use faer_traits::ComplexField;

impl<E: ComplexField> ArgminLInfNorm<E::Real> for MatRef<'_, E> {
    fn linf_norm(&self) -> E::Real {
        self.norm_max()
    }
}

impl<E: ComplexField> ArgminLInfNorm<E::Real> for Mat<E> {
    fn linf_norm(&self) -> E::Real {
        self.as_mat_ref().norm_max()
    }
}
