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

#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std;

/// A simple key-value storage
#[derive(Clone, Default, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct KV {
    /// The actual key value storage
    #[cfg_attr(feature = "serde1", serde(borrow))]
    pub kv: Vec<(&'static str, String)>,
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
    ///
    /// This formats the `val` using `format!`. Therefore `T` has to implement `Display`.
    pub fn push<T: std::fmt::Display>(&mut self, key: &'static str, val: T) -> &mut Self {
        self.kv.push((key, format!("{}", val)));
        self
    }

    /// Merge another `kv` into `self.kv`
    #[must_use]
    pub fn merge(mut self, other: &mut KV) -> Self {
        self.kv.append(&mut other.kv);
        self
    }
}

impl std::iter::FromIterator<(&'static str, String)> for KV {
    fn from_iter<I: IntoIterator<Item = (&'static str, String)>>(iter: I) -> Self {
        let mut c = KV::new();

        for i in iter {
            c.push(i.0, i.1);
        }

        c
    }
}

impl std::iter::Extend<(&'static str, String)> for KV {
    fn extend<I: IntoIterator<Item = (&'static str, String)>>(&mut self, iter: I) {
        for i in iter {
            self.push(i.0, i.1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    send_sync_test!(argmin_kv, KV);
}
