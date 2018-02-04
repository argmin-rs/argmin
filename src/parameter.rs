// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! TODO DOCUMENTATION
//!

use std::fmt::Debug;
use std::cmp::PartialOrd;
use std::default::Default;
use std::ops::{Index, IndexMut};
use rand;
use rand::distributions::{IndependentSample, Range};
use ndarray::Array1;

/// This trait needs to be implemented for every parameter fed into the solvers.
/// This is highly *UNSTABLE* and will change in the future.
pub trait ArgminParameter
    : Clone
    + Default
    + Debug
    + Index<usize, Output = <Self as ArgminParameter>::Element>
    + IndexMut<usize> {
    /// Type of a single element of the parameter vector
    type Element: PartialOrd + Clone;
    /// Defines a single modification of the parameter vector.
    ///
    /// The parameters:
    ///
    /// `&self`: reference to the object of type `Self`
    fn modify(&self) -> (Self, usize);

    /// Returns a completely random parameter vector
    ///
    /// The resulting parameter vector satisfies `lower_bound`, `upper_bound`.
    fn random(&Self, &Self) -> Self;
}

/// Create a random parameter vector within lower and upper bound.
macro_rules! random_vec_iter {
    ($type:ty) => {
        fn random(lower_bound: &$type, upper_bound: &$type) -> $type {
            let mut rng = rand::thread_rng();
            let out: $type = lower_bound
                .iter()
                .zip(upper_bound.iter())
                .map(|(l, u)| {
                    if l >= u {
                        panic!("Parameter: lower_bound must be lower than upper_bound.");
                    }
                    let range = Range::new(*l, *u);
                    range.ind_sample(&mut rng)
                })
                .collect();
            out
        }
    }
}

/// Modify one parameter of the parameter vector
macro_rules! modify_one_parameter {
    () => {
        fn modify(&self) -> (Self, usize) {
            let pos = Range::new(0, self.len());
            let range = Range::new(-1.0, 1.0);
            let mut rng = rand::thread_rng();
            let mut param = self.clone();
            let idx = pos.ind_sample(&mut rng);
            param[idx] = self[idx] + range.ind_sample(&mut rng);
            (param, idx)
        }
    }
}

/// Implement `ArgminParameter`
macro_rules! implement_argmin_parameter {
    ($param:ty, $element:ty) => {
        impl ArgminParameter for $param {
            type Element = $element;
            modify_one_parameter!();
            random_vec_iter!($param);
        }
    }
}

implement_argmin_parameter!(Vec<f64>, f64);
implement_argmin_parameter!(Vec<f32>, f32);
implement_argmin_parameter!(Array1<f64>, f64);
implement_argmin_parameter!(Array1<f32>, f32);
