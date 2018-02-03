// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

/// TODO DOCUMENTATION
///
use std::fmt::Debug;
use std::default::Default;
use rand;
use rand::distributions::{IndependentSample, Range};
use ndarray::Array1;

/// This trait needs to be implemented for every parameter fed into the solvers.
/// This is highly *UNSTABLE* and will change in the future.
pub trait ArgminParameter: Clone + Default + Debug {
    /// Defines a modification of the parameter vector.
    ///
    /// The parameters:
    ///
    /// `&self`: reference to the object of type `Self`
    /// `lower_bound`: Lower bound of the parameter vector. Same type as parameter vector (`Self`)
    /// `upper_bound`: Upper bound of the parameter vector. Same type as parameter vector (`Self`)
    /// `constraint`: Additional (non)linear constraint whith the signature `&Fn(&Self) -> bool`.
    /// The provided function takes a parameter as input and returns `true` if the parameter vector
    /// satisfies the constraints and `false` otherwise.
    fn modify(&self, &Self, &Self, &Fn(&Self) -> bool) -> Self;

    /// Returns a completely random parameter vector
    ///
    /// The resulting parameter vector satisfies `lower_bound`, `upper_bound`.
    fn random(&Self, &Self) -> Self;
}

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

impl ArgminParameter for Vec<f64> {
    fn modify(
        &self,
        lower_bound: &Vec<f64>,
        upper_bound: &Vec<f64>,
        constraint: &Fn(&Vec<f64>) -> bool,
    ) -> Vec<f64> {
        let step = Range::new(0, self.len());
        let range = Range::new(-1.0_f64, 1.0_f64);
        let mut rng = rand::thread_rng();
        let mut param = self.clone();
        loop {
            let idx = step.ind_sample(&mut rng);
            param[idx] = self[idx] + range.ind_sample(&mut rng);
            if param[idx] < lower_bound[idx] {
                param[idx] = lower_bound[idx];
            }
            if param[idx] > upper_bound[idx] {
                param[idx] = upper_bound[idx];
            }
            if constraint(&param) {
                break;
            }
        }
        param
    }

    random_vec_iter!(Vec<f64>);
}

impl ArgminParameter for Array1<f64> {
    fn modify(
        &self,
        lower_bound: &Array1<f64>,
        upper_bound: &Array1<f64>,
        constraint: &Fn(&Array1<f64>) -> bool,
    ) -> Array1<f64> {
        let step = Range::new(0, self.len());
        let range = Range::new(-1.0_f64, 1.0_f64);
        let mut rng = rand::thread_rng();
        let mut param = self.clone();
        loop {
            let idx = step.ind_sample(&mut rng);
            param[idx] = self[idx] + range.ind_sample(&mut rng);
            if param[idx] < lower_bound[idx] {
                param[idx] = lower_bound[idx];
            }
            if param[idx] > upper_bound[idx] {
                param[idx] = upper_bound[idx];
            }
            if constraint(&param) {
                break;
            }
        }
        param
    }

    random_vec_iter!(Array1<f64>);
}
