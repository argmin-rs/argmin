// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use std::collections::HashMap;
use std::fmt;
use std::fmt::{Debug, Display};

/// Types available for use in [`KV`](KV).
///
/// `Float`, `Int` and `Uint` are all 64bit. The corresponding 32bit variants must be
/// be converted to 64 bit. Preferably the `From` impls are used to create a `KVType`:
///
/// ```
/// # use argmin::core::KVType;
/// let x: KVType = 1u64.into();
/// assert_eq!(x, KVType::Uint(1u64));
///
/// let x: KVType = 2u32.into();
/// assert_eq!(x, KVType::Uint(2u64));
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
    Uint(u64),
    /// Boolean values
    Bool(bool),
    /// Strings
    Str(String),
}

impl KVType {
    /// Returns the kind of the `KVType`
    ///
    /// # Examples
    ///
    /// ```
    /// # use argmin::core::KVType;
    /// assert_eq!(KVType::Float(1.0).kind(), "Float");
    /// assert_eq!(KVType::Int(1).kind(), "Int");
    /// assert_eq!(KVType::Uint(1).kind(), "Uint");
    /// assert_eq!(KVType::Bool(true).kind(), "Bool");
    /// assert_eq!(KVType::Str("string".to_string()).kind(), "Str");
    /// ```
    pub fn kind(&self) -> &'static str {
        match self {
            KVType::Float(_) => "Float",
            KVType::Int(_) => "Int",
            KVType::Uint(_) => "Uint",
            KVType::Bool(_) => "Bool",
            KVType::Str(_) => "Str",
        }
    }

    /// Extract float from `KVType`
    ///
    /// Returns `Some(<float>)` if `KVType` is of kind `Float` and `None` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::KVType;
    /// assert_eq!(KVType::Float(1.0).get_float(), Some(1.0));
    /// assert_eq!(KVType::Int(1).get_float(), None);
    /// assert_eq!(KVType::Uint(1).get_float(), None);
    /// assert_eq!(KVType::Bool(true).get_float(), None);
    /// assert_eq!(KVType::Str("not a float".to_string()).get_float(), None);
    /// ```
    pub fn get_float(&self) -> Option<f64> {
        if let KVType::Float(x) = *self {
            Some(x)
        } else {
            None
        }
    }

    /// Extract int from `KVType`
    ///
    /// Returns `Some(<int>)` if `KVType` is of kind `Int` and `None` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::KVType;
    /// assert_eq!(KVType::Int(1).get_int(), Some(1i64));
    /// assert_eq!(KVType::Float(1.0).get_int(), None);
    /// assert_eq!(KVType::Uint(1).get_int(), None);
    /// assert_eq!(KVType::Bool(true).get_int(), None);
    /// assert_eq!(KVType::Str("not an int".to_string()).get_int(), None);
    /// ```
    pub fn get_int(&self) -> Option<i64> {
        if let KVType::Int(x) = *self {
            Some(x)
        } else {
            None
        }
    }

    /// Extract unsigned int from `KVType`
    ///
    /// Returns `Some(<unsigned int>)` if `KVType` is of kind `Uint` and `None` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::KVType;
    /// assert_eq!(KVType::Uint(1).get_uint(), Some(1u64));
    /// assert_eq!(KVType::Int(1).get_uint(), None);
    /// assert_eq!(KVType::Float(1.0).get_uint(), None);
    /// assert_eq!(KVType::Bool(true).get_uint(), None);
    /// assert_eq!(KVType::Str("not an uint".to_string()).get_uint(), None);
    /// ```
    pub fn get_uint(&self) -> Option<u64> {
        if let KVType::Uint(x) = *self {
            Some(x)
        } else {
            None
        }
    }

    /// Extract bool from `KVType`
    ///
    /// Returns `Some(<bool>)` if `KVType` is of kind `Bool` and `None` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::KVType;
    /// assert_eq!(KVType::Bool(true).get_bool(), Some(true));
    /// assert_eq!(KVType::Float(1.0).get_bool(), None);
    /// assert_eq!(KVType::Int(1).get_bool(), None);
    /// assert_eq!(KVType::Uint(1).get_bool(), None);
    /// assert_eq!(KVType::Str("not a bool".to_string()).get_bool(), None);
    /// ```
    pub fn get_bool(&self) -> Option<bool> {
        if let KVType::Bool(x) = *self {
            Some(x)
        } else {
            None
        }
    }

    /// Extract String from `KVType`
    ///
    /// Returns `Some(<string>)` if `KVType` is of kind `Str` and `None` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::KVType;
    /// assert_eq!(KVType::Str("a string".to_string()).get_string(), Some("a string".to_string()));
    /// assert_eq!(KVType::Float(1.0).get_string(), None);
    /// assert_eq!(KVType::Int(1).get_string(), None);
    /// assert_eq!(KVType::Uint(1).get_string(), None);
    /// assert_eq!(KVType::Bool(true).get_string(), None);
    /// ```
    pub fn get_string(&self) -> Option<String> {
        if let KVType::Str(x) = self {
            Some(x.clone())
        } else {
            None
        }
    }
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
        KVType::Uint(x)
    }
}

impl From<i32> for KVType {
    fn from(x: i32) -> KVType {
        KVType::Int(i64::from(x))
    }
}

impl From<u32> for KVType {
    fn from(x: u32) -> KVType {
        KVType::Uint(u64::from(x))
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
            KVType::Float(x) => write!(f, "{x}")?,
            KVType::Int(x) => write!(f, "{x}")?,
            KVType::Uint(x) => write!(f, "{x}")?,
            KVType::Bool(x) => write!(f, "{x}")?,
            KVType::Str(x) => write!(f, "{x}")?,
        };
        Ok(())
    }
}

/// A simple key-value storage
///
/// Keeps pairs of `(&'static str, KVType)` and is used to pass key-value pairs to
/// [`Observers`](`crate::core::observers`) in each iteration of an optimization algorithm.
/// Typically constructed using the [`kv!`](`crate::kv`) macro.
///
/// # Example
///
/// ```
/// use argmin::kv;
///
/// let kv = kv!(
///     "key1" => "value1";
///     "key2" => "value2";
///     "key3" => 1234;
/// );
/// # assert_eq!(kv.kv.len(), 3);
/// # assert_eq!(format!("{}", kv.get("key1").unwrap()), "value1");
/// # assert_eq!(format!("{}", kv.get("key2").unwrap()), "value2");
/// # assert_eq!(format!("{}", kv.get("key3").unwrap()), "1234");
/// ```
#[derive(Clone, Default, PartialEq)]
pub struct KV {
    /// The actual key value storage
    pub kv: HashMap<&'static str, KVType>,
}

impl Debug for KV {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "{self}")?;
        Ok(())
    }
}
impl Display for KV {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "KV")?;
        for (key, val) in self.kv.iter() {
            writeln!(f, "   {key}: {val}")?;
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
        KV { kv: HashMap::new() }
    }

    /// Insert a key-value pair
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{KV, KVType};
    /// let mut kv = KV::new();
    /// kv.insert("key1", KVType::Str("value".to_string()));
    /// kv.insert("key2", KVType::Int(1234));
    /// # assert_eq!(kv.kv.len(), 2);
    /// # assert_eq!(format!("{}", kv.get("key1").unwrap()), "value");
    /// # assert_eq!(format!("{}", kv.get("key2").unwrap()), "1234");
    /// ```
    pub fn insert(&mut self, key: &'static str, val: KVType) -> &mut Self {
        self.kv.insert(key, val);
        self
    }

    /// Retrieve an element from the KV by key
    ///
    /// Returns `Some(<reference to KVType>)` if `key` is present and `None` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{KV, KVType};
    /// let mut kv1 = KV::new();
    /// kv1.insert("key1", KVType::Float(12.0));
    ///
    /// assert_eq!(kv1.get("key1"), Some(&KVType::Float(12.0)));
    /// assert_eq!(kv1.get("non_existing"), None);
    /// ```
    pub fn get(&self, key: &'static str) -> Option<&KVType> {
        self.kv.get(key)
    }

    /// Returns all available keys and their `KVType` kind
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{KV, KVType};
    /// let mut kv1 = KV::new();
    /// kv1.insert("key1", KVType::Str("value1".to_string()));
    ///
    /// assert_eq!(kv1.keys(), vec![("key1", "Str")]);
    /// ```
    pub fn keys(&self) -> Vec<(&'static str, &'static str)> {
        self.kv.iter().map(|(&k, v)| (k, v.kind())).collect()
    }

    /// Merge with another `KV`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{KV, KVType};
    /// let mut kv1 = KV::new();
    /// kv1.insert("key1", KVType::Str("value1".to_string()));
    /// # assert_eq!(kv1.kv.len(), 1);
    /// # assert_eq!(format!("{}", kv1.get("key1").unwrap()), "value1");
    ///
    /// let mut kv2 = KV::new();
    /// kv2.insert("key2", KVType::Str("value2".to_string()));
    /// # assert_eq!(kv2.kv.len(), 1);
    /// # assert_eq!(format!("{}", kv2.get("key2").unwrap()), "value2");
    ///
    /// let kv1 = kv1.merge(kv2);
    /// # assert_eq!(kv1.kv.len(), 2);
    /// # assert_eq!(format!("{}", kv1.get("key1").unwrap()), "value1");
    /// # assert_eq!(format!("{}", kv1.get("key2").unwrap()), "value2");
    /// ```
    #[must_use]
    pub fn merge(mut self, other: KV) -> Self {
        self.kv.extend(other.kv);
        self
    }
}

impl std::iter::FromIterator<(&'static str, KVType)> for KV {
    fn from_iter<I: IntoIterator<Item = (&'static str, KVType)>>(iter: I) -> Self {
        let mut c = KV::new();
        for i in iter {
            c.insert(i.0, i.1);
        }
        c
    }
}

impl std::iter::Extend<(&'static str, KVType)> for KV {
    fn extend<I: IntoIterator<Item = (&'static str, KVType)>>(&mut self, iter: I) {
        for i in iter {
            self.insert(i.0, i.1);
        }
    }
}
