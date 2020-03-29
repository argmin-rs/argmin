// Copyright 2018-2020 argmin developers
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

use serde::{Deserialize, Serialize};
use std;

/// A simple key-value storage
#[derive(Clone, Default, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Serialize, Deserialize)]
pub struct ArgminKV {
    /// The actual key value storage
    #[serde(borrow)]
    pub kv: Vec<(&'static str, String)>,
}

impl std::fmt::Display for ArgminKV {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "ArgminKV")?;
        self.kv
            .iter()
            .map(|(key, val)| -> std::fmt::Result { writeln!(f, "   {}: {}", key, val) })
            .count();
        Ok(())
    }
}

impl ArgminKV {
    /// Constructor
    pub fn new() -> Self {
        ArgminKV { kv: vec![] }
    }

    /// Push a key-value pair to the `kv` vector.
    ///
    /// This formats the `val` using `format!`. Therefore `T` has to implement `Display`.
    pub fn push<T: std::fmt::Display>(&mut self, key: &'static str, val: T) -> &mut Self {
        self.kv.push((key, format!("{}", val)));
        self
    }

    /// Merge another `kv` into `self.kv`
    pub fn merge(mut self, other: &mut ArgminKV) -> Self {
        self.kv.append(&mut other.kv);
        self
    }
}

impl std::iter::FromIterator<(&'static str, String)> for ArgminKV {
    fn from_iter<I: IntoIterator<Item = (&'static str, String)>>(iter: I) -> Self {
        let mut c = ArgminKV::new();

        for i in iter {
            c.push(i.0, i.1);
        }

        c
    }
}

impl std::iter::Extend<(&'static str, String)> for ArgminKV {
    fn extend<I: IntoIterator<Item = (&'static str, String)>>(&mut self, iter: I) {
        for i in iter {
            self.push(i.0, i.1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    send_sync_test!(argmin_kv, ArgminKV);
}
