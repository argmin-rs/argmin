// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use std::fmt;
use std::fmt::Display;
use std::rc::Rc;

/// A simple key-value storage
///
/// Keeps pairs of `(&'static str, Rc<dyn Display>)` and is used to pass key-value pairs to
/// [`Observers`](`crate::core::observers`) in each iteration of an optimization algorithm.
/// Typically constructed using the [`make_kv!`](`crate::make_kv`) macro.
///
/// # Example
///
/// ```
/// use argmin::make_kv;
///
/// let kv = make_kv!(
///     "key1" => "value1";
///     "key2" => "value2";
///     "key3" => 1234;
/// );
/// # assert_eq!(kv.kv.len(), 3);
/// # assert_eq!(kv.kv[0].0, "key1");
/// # assert_eq!(format!("{}", kv.kv[0].1), "value1");
/// # assert_eq!(kv.kv[1].0, "key2");
/// # assert_eq!(format!("{}", kv.kv[1].1), "value2");
/// # assert_eq!(kv.kv[2].0, "key3");
/// # assert_eq!(format!("{}", kv.kv[2].1), "1234");
/// ```
#[derive(Clone, Default)]
pub struct KV {
    /// The actual key value storage
    pub kv: Vec<(&'static str, Rc<dyn Display>)>,
}

impl Display for KV {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "KV")?;
        for (key, val) in self.kv.iter() {
            writeln!(f, "   {}: {}", key, val)?;
        }
        Ok(())
    }
}

impl KV {
    /// Constructor a new empty `KV`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::KV;
    /// #
    /// let kv = KV::new();
    /// # assert_eq!(kv.kv.len(), 0);
    /// ```
    pub fn new() -> Self {
        KV { kv: vec![] }
    }

    /// Push a key-value pair
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::KV;
    /// # use std::rc::Rc;
    ///
    /// let mut kv = KV::new();
    /// kv.push("key", Rc::new("value"));
    /// kv.push("key", Rc::new(1234));
    /// # assert_eq!(kv.kv.len(), 2);
    /// # assert_eq!(kv.kv[0].0, "key");
    /// # assert_eq!(format!("{}", kv.kv[0].1), "value");
    /// # assert_eq!(kv.kv[1].0, "key");
    /// # assert_eq!(format!("{}", kv.kv[1].1), "1234");
    /// ```
    pub fn push(&mut self, key: &'static str, val: Rc<dyn Display>) -> &mut Self {
        self.kv.push((key, val));
        self
    }

    /// Merge with another `KV`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::KV;
    /// # use std::rc::Rc;
    ///
    /// let mut kv1 = KV::new();
    /// kv1.push("key1", Rc::new("value1"));
    /// # assert_eq!(kv1.kv.len(), 1);
    /// # assert_eq!(kv1.kv[0].0, "key1");
    /// # assert_eq!(format!("{}", kv1.kv[0].1), "value1");
    ///
    /// let mut kv2 = KV::new();
    /// kv2.push("key2", Rc::new("value2"));
    /// # assert_eq!(kv2.kv.len(), 1);
    /// # assert_eq!(kv2.kv[0].0, "key2");
    /// # assert_eq!(format!("{}", kv2.kv[0].1), "value2");
    ///
    /// let kv1 = kv1.merge(kv2);
    /// # assert_eq!(kv1.kv.len(), 2);
    /// # assert_eq!(kv1.kv[0].0, "key1");
    /// # assert_eq!(format!("{}", kv1.kv[0].1), "value1");
    /// # assert_eq!(kv1.kv[1].0, "key2");
    /// # assert_eq!(format!("{}", kv1.kv[1].1), "value2");
    /// ```
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
