// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use std::fmt;
use std::fmt::{Debug, Display};

/// Types available for use in [`KV`](KV).
///
/// `Float`, `Int` and `UnsignedInt` are all 64bit. The corresponding 32bit variants must be
/// be converted to 64 bit. Preferably the `From` impls are used to create a `KVType`:
///
/// ```
/// # use argmin::core::KVType;
/// let x: KVType = 1u64.into();
/// assert_eq!(x, KVType::UnsignedInt(1u64));
///
/// let x: KVType = 2u32.into();
/// assert_eq!(x, KVType::UnsignedInt(2u64));
///
/// let x: KVType = 2i64.into();
/// assert_eq!(x, KVType::Int(2i64));
///
/// let x: KVType = 2i32.into();
/// assert_eq!(x, KVType::Int(2i64));
///
/// let x: KVType = 1.0f64.into();
/// assert_eq!(x, KVType::Float(1f64));
///
/// let x: KVType = 1.0f32.into();
/// assert_eq!(x, KVType::Float(1f64));
///
/// let x: KVType = true.into();
/// assert_eq!(x, KVType::Bool(true));
///
/// let x: KVType = "a str".into();
/// assert_eq!(x, KVType::Str("a str".to_string()));
///
/// let x: KVType = "a String".to_string().into();
/// assert_eq!(x, KVType::Str("a String".to_string()));
/// ```
#[derive(Clone, PartialEq, Debug)]
pub enum KVType {
    /// Floating point values
    Float(f64),
    /// Signed integers
    Int(i64),
    /// Unsigned integers
    UnsignedInt(u64),
    /// Boolean values
    Bool(bool),
    /// Strings
    Str(String),
}

impl From<f64> for KVType {
    fn from(x: f64) -> KVType {
        KVType::Float(x)
    }
}

impl From<f32> for KVType {
    fn from(x: f32) -> KVType {
        KVType::Float(f64::from(x))
    }
}

impl From<i64> for KVType {
    fn from(x: i64) -> KVType {
        KVType::Int(x)
    }
}

impl From<u64> for KVType {
    fn from(x: u64) -> KVType {
        KVType::UnsignedInt(x)
    }
}

impl From<i32> for KVType {
    fn from(x: i32) -> KVType {
        KVType::Int(i64::from(x))
    }
}

impl From<u32> for KVType {
    fn from(x: u32) -> KVType {
        KVType::UnsignedInt(u64::from(x))
    }
}

impl From<bool> for KVType {
    fn from(x: bool) -> KVType {
        KVType::Bool(x)
    }
}

impl From<String> for KVType {
    fn from(x: String) -> KVType {
        KVType::Str(x)
    }
}

impl<'a> From<&'a str> for KVType {
    fn from(x: &'a str) -> KVType {
        KVType::Str(x.to_string())
    }
}

impl Display for KVType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            KVType::Float(x) => write!(f, "{}", x)?,
            KVType::Int(x) => write!(f, "{}", x)?,
            KVType::UnsignedInt(x) => write!(f, "{}", x)?,
            KVType::Bool(x) => write!(f, "{}", x)?,
            KVType::Str(x) => write!(f, "{}", x)?,
        };
        Ok(())
    }
}

/// A simple key-value storage
///
/// Keeps pairs of `(&'static str, KVType)` and is used to pass key-value pairs to
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
    pub kv: Vec<(&'static str, KVType)>,
}

impl Debug for KV {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "{}", self)?;
        Ok(())
    }
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
    /// # use argmin::core::{KV, KVType};
    ///
    /// let mut kv = KV::new();
    /// kv.push("key", KVType::Str("value".to_string()));
    /// kv.push("key", KVType::Int(1234));
    /// # assert_eq!(kv.kv.len(), 2);
    /// # assert_eq!(kv.kv[0].0, "key");
    /// # assert_eq!(format!("{}", kv.kv[0].1), "value");
    /// # assert_eq!(kv.kv[1].0, "key");
    /// # assert_eq!(format!("{}", kv.kv[1].1), "1234");
    /// ```
    pub fn push(&mut self, key: &'static str, val: KVType) -> &mut Self {
        self.kv.push((key, val));
        self
    }

    /// Merge with another `KV`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{KV, KVType};
    ///
    /// let mut kv1 = KV::new();
    /// kv1.push("key1", KVType::Str("value1".to_string()));
    /// # assert_eq!(kv1.kv.len(), 1);
    /// # assert_eq!(kv1.kv[0].0, "key1");
    /// # assert_eq!(format!("{}", kv1.kv[0].1), "value1");
    ///
    /// let mut kv2 = KV::new();
    /// kv2.push("key2", KVType::Str("value2".to_string()));
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

impl std::iter::FromIterator<(&'static str, KVType)> for KV {
    fn from_iter<I: IntoIterator<Item = (&'static str, KVType)>>(iter: I) -> Self {
        let mut c = KV::new();
        for i in iter {
            c.push(i.0, i.1);
        }
        c
    }
}

impl std::iter::Extend<(&'static str, KVType)> for KV {
    fn extend<I: IntoIterator<Item = (&'static str, KVType)>>(&mut self, iter: I) {
        for i in iter {
            self.push(i.0, i.1);
        }
    }
}
