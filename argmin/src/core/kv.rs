// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Key Value storage
//!
//! A very simple key-value storage.
//!
//! TODOs:
//!   * Either use something existing, or at least evaluate the performance and if necessary,
//!     improve performance.

use std;
use std::fmt::Display;
use std::rc::Rc;

/// A simple key-value storage
#[derive(Clone, Default)]
pub struct KV {
    /// The actual key value storage
    pub kv: Vec<(&'static str, Rc<dyn Display>)>,
}

impl std::fmt::Display for KV {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "KV")?;
        self.kv
            .iter()
            .map(|(key, val)| -> std::fmt::Result { writeln!(f, "   {}: {}", key, val) })
            .count();
        Ok(())
    }
}

impl KV {
    /// Constructor
    pub fn new() -> Self {
        KV { kv: vec![] }
    }

    /// Push a key-value pair to the `kv` vector.
    pub fn push(&mut self, key: &'static str, val: Rc<dyn Display>) -> &mut Self {
        self.kv.push((key, val));
        self
    }

    /// Merge another `kv` into `self.kv`
    #[must_use]
    pub fn merge(mut self, mut other: KV) -> Self {
        self.kv.append(&mut other.kv);
        self
    }
}

impl std::iter::FromIterator<(&'static str, Rc<dyn Display>)> for KV {
    fn from_iter<I: IntoIterator<Item = (&'static str, Rc<dyn Display>)>>(iter: I) -> Self {
        let mut c = KV::new();

        for i in iter {
            c.push(i.0, i.1);
        }

        c
    }
}

impl std::iter::Extend<(&'static str, Rc<dyn Display>)> for KV {
    fn extend<I: IntoIterator<Item = (&'static str, Rc<dyn Display>)>>(&mut self, iter: I) {
        for i in iter {
            self.push(i.0, i.1);
        }
    }
}
