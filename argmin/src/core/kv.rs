// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use std::collections::HashMap;
use std::fmt;
use std::fmt::{Debug, Display};

#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// Types available for use in [`KV`](KV).
///
/// `Float`, `Int` and `Uint` are all 64bit. The corresponding 32bit variants must be
/// be converted to 64 bit. Preferably the `From` impls are used to create a `KvValue`:
///
/// ```
/// # use argmin::core::KvValue;
/// let x: KvValue = 1u64.into();
/// assert_eq!(x, KvValue::Uint(1u64));
///
/// let x: KvValue = 2u32.into();
/// assert_eq!(x, KvValue::Uint(2u64));
///
/// let x: KvValue = 2i64.into();
/// assert_eq!(x, KvValue::Int(2i64));
///
/// let x: KvValue = 2i32.into();
/// assert_eq!(x, KvValue::Int(2i64));
///
/// let x: KvValue = 1.0f64.into();
/// assert_eq!(x, KvValue::Float(1f64));
///
/// let x: KvValue = 1.0f32.into();
/// assert_eq!(x, KvValue::Float(1f64));
///
/// let x: KvValue = true.into();
/// assert_eq!(x, KvValue::Bool(true));
///
/// let x: KvValue = "a str".into();
/// assert_eq!(x, KvValue::Str("a str".to_string()));
///
/// let x: KvValue = "a String".to_string().into();
/// assert_eq!(x, KvValue::Str("a String".to_string()));
/// ```
#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub enum KvValue {
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

impl KvValue {
    /// Returns the kind of the `KvValue`
    ///
    /// # Examples
    ///
    /// ```
    /// # use argmin::core::KvValue;
    /// assert_eq!(KvValue::Float(1.0).kind(), "Float");
    /// assert_eq!(KvValue::Int(1).kind(), "Int");
    /// assert_eq!(KvValue::Uint(1).kind(), "Uint");
    /// assert_eq!(KvValue::Bool(true).kind(), "Bool");
    /// assert_eq!(KvValue::Str("string".to_string()).kind(), "Str");
    /// ```
    pub fn kind(&self) -> &'static str {
        match self {
            KvValue::Float(_) => "Float",
            KvValue::Int(_) => "Int",
            KvValue::Uint(_) => "Uint",
            KvValue::Bool(_) => "Bool",
            KvValue::Str(_) => "Str",
        }
    }

    /// Extract float from `KvValue`
    ///
    /// Returns `Some(<float>)` if `KvValue` is of kind `Float`.
    ///
    /// **Note:** For `KvValue::Int` and `KvValue::Uint`integer values are cast to f64, therefore
    /// this operation may be lossy!
    ///
    /// `KvValue::Bool` is turned into `1.0f64` if true and `0.0f64` if false.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::KvValue;
    /// assert_eq!(KvValue::Float(1.0).get_float(), Some(1.0));
    /// assert_eq!(KvValue::Int(1).get_float(), Some(1.0));
    /// assert_eq!(KvValue::Uint(1).get_float(), Some(1.0));
    /// assert_eq!(KvValue::Bool(true).get_float(), Some(1.0));
    /// assert_eq!(KvValue::Str("not a number".to_string()).get_float(), None);
    /// ```
    pub fn get_float(&self) -> Option<f64> {
        match self {
            KvValue::Float(x) => Some(*x),
            KvValue::Int(x) => Some(*x as f64),
            KvValue::Uint(x) => Some(*x as f64),
            KvValue::Bool(true) => Some(1.0),
            KvValue::Bool(false) => Some(0.0),
            _ => None,
        }
    }

    /// Extract int from `KvValue`
    ///
    /// Returns `Some(<int>)` if `KvValue` is of kind `Int` and `None` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::KvValue;
    /// assert_eq!(KvValue::Int(1).get_int(), Some(1i64));
    /// assert_eq!(KvValue::Float(1.0).get_int(), None);
    /// assert_eq!(KvValue::Uint(1).get_int(), None);
    /// assert_eq!(KvValue::Bool(true).get_int(), None);
    /// assert_eq!(KvValue::Str("not an int".to_string()).get_int(), None);
    /// ```
    pub fn get_int(&self) -> Option<i64> {
        if let KvValue::Int(x) = *self {
            Some(x)
        } else {
            None
        }
    }

    /// Extract unsigned int from `KvValue`
    ///
    /// Returns `Some(<unsigned int>)` if `KvValue` is of kind `Uint` and `None` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::KvValue;
    /// assert_eq!(KvValue::Uint(1).get_uint(), Some(1u64));
    /// assert_eq!(KvValue::Int(1).get_uint(), None);
    /// assert_eq!(KvValue::Float(1.0).get_uint(), None);
    /// assert_eq!(KvValue::Bool(true).get_uint(), None);
    /// assert_eq!(KvValue::Str("not an uint".to_string()).get_uint(), None);
    /// ```
    pub fn get_uint(&self) -> Option<u64> {
        if let KvValue::Uint(x) = *self {
            Some(x)
        } else {
            None
        }
    }

    /// Extract bool from `KvValue`
    ///
    /// Returns `Some(<bool>)` if `KvValue` is of kind `Bool` and `None` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::KvValue;
    /// assert_eq!(KvValue::Bool(true).get_bool(), Some(true));
    /// assert_eq!(KvValue::Float(1.0).get_bool(), None);
    /// assert_eq!(KvValue::Int(1).get_bool(), None);
    /// assert_eq!(KvValue::Uint(1).get_bool(), None);
    /// assert_eq!(KvValue::Str("not a bool".to_string()).get_bool(), None);
    /// ```
    pub fn get_bool(&self) -> Option<bool> {
        if let KvValue::Bool(x) = *self {
            Some(x)
        } else {
            None
        }
    }

    /// Extract String from `KvValue`
    ///
    /// Returns `Some(<string>)` if `KvValue` is of kind `Str` and `None` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::KvValue;
    /// assert_eq!(KvValue::Str("a string".to_string()).get_string(), Some("a string".to_string()));
    /// assert_eq!(KvValue::Float(1.0).get_string(), None);
    /// assert_eq!(KvValue::Int(1).get_string(), None);
    /// assert_eq!(KvValue::Uint(1).get_string(), None);
    /// assert_eq!(KvValue::Bool(true).get_string(), None);
    /// ```
    pub fn get_string(&self) -> Option<String> {
        if let KvValue::Str(x) = self {
            Some(x.clone())
        } else {
            None
        }
    }

    /// Get String representation of `KvValue`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::KvValue;
    /// assert_eq!(KvValue::Str("a string".to_string()).as_string(), "a string".to_string());
    /// assert_eq!(KvValue::Float(1.0).as_string(), "1".to_string());
    /// assert_eq!(KvValue::Int(1).as_string(), "1".to_string());
    /// assert_eq!(KvValue::Uint(1).as_string(), "1".to_string());
    /// assert_eq!(KvValue::Bool(true).as_string(), "true".to_string());
    /// ```
    pub fn as_string(&self) -> String {
        match self {
            KvValue::Str(x) => x.clone(),
            KvValue::Float(x) => format!("{x}"),
            KvValue::Bool(x) => format!("{x}"),
            KvValue::Int(x) => format!("{x}"),
            KvValue::Uint(x) => format!("{x}"),
        }
    }
}

impl From<f64> for KvValue {
    fn from(x: f64) -> KvValue {
        KvValue::Float(x)
    }
}

impl From<f32> for KvValue {
    fn from(x: f32) -> KvValue {
        KvValue::Float(f64::from(x))
    }
}

impl From<i64> for KvValue {
    fn from(x: i64) -> KvValue {
        KvValue::Int(x)
    }
}

impl From<u64> for KvValue {
    fn from(x: u64) -> KvValue {
        KvValue::Uint(x)
    }
}

impl From<i32> for KvValue {
    fn from(x: i32) -> KvValue {
        KvValue::Int(i64::from(x))
    }
}

impl From<u32> for KvValue {
    fn from(x: u32) -> KvValue {
        KvValue::Uint(u64::from(x))
    }
}

impl From<bool> for KvValue {
    fn from(x: bool) -> KvValue {
        KvValue::Bool(x)
    }
}

impl From<String> for KvValue {
    fn from(x: String) -> KvValue {
        KvValue::Str(x)
    }
}

impl<'a> From<&'a str> for KvValue {
    fn from(x: &'a str) -> KvValue {
        KvValue::Str(x.to_string())
    }
}

impl Display for KvValue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            KvValue::Float(x) => write!(f, "{x}")?,
            KvValue::Int(x) => write!(f, "{x}")?,
            KvValue::Uint(x) => write!(f, "{x}")?,
            KvValue::Bool(x) => write!(f, "{x}")?,
            KvValue::Str(x) => write!(f, "{x}")?,
        };
        Ok(())
    }
}

/// A simple key-value storage
///
/// Keeps pairs of `(&'static str, KvValue)` and is used to pass key-value pairs to
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
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct KV {
    /// The actual key value storage
    pub kv: HashMap<String, KvValue>,
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
    /// # use argmin::core::{KV, KvValue};
    /// let mut kv = KV::new();
    /// kv.insert("key1", KvValue::Str("value".to_string()));
    /// kv.insert("key2", KvValue::Int(1234));
    /// # assert_eq!(kv.kv.len(), 2);
    /// # assert_eq!(format!("{}", kv.get("key1").unwrap()), "value");
    /// # assert_eq!(format!("{}", kv.get("key2").unwrap()), "1234");
    /// ```
    pub fn insert<T: AsRef<str>>(&mut self, key: T, val: KvValue) -> &mut Self {
        self.kv.insert(key.as_ref().into(), val);
        self
    }

    /// Retrieve an element from the KV by key
    ///
    /// Returns `Some(<reference to KvValue>)` if `key` is present and `None` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{KV, KvValue};
    /// let mut kv1 = KV::new();
    /// kv1.insert("key1", KvValue::Float(12.0));
    ///
    /// assert_eq!(kv1.get("key1"), Some(&KvValue::Float(12.0)));
    /// assert_eq!(kv1.get("non_existing"), None);
    /// ```
    pub fn get<T: AsRef<str>>(&self, key: T) -> Option<&KvValue> {
        self.kv.get(key.as_ref())
    }

    /// Returns all available keys and their `KvValue` kind
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{KV, KvValue};
    /// let mut kv1 = KV::new();
    /// kv1.insert("key1", KvValue::Str("value1".to_string()));
    ///
    /// assert_eq!(kv1.keys(), vec![("key1".to_string(), "Str")]);
    /// ```
    pub fn keys(&self) -> Vec<(String, &'static str)> {
        self.kv.iter().map(|(k, v)| (k.clone(), v.kind())).collect()
    }

    /// Merge with another `KV`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{KV, KvValue};
    /// let mut kv1 = KV::new();
    /// kv1.insert("key1", KvValue::Str("value1".to_string()));
    /// # assert_eq!(kv1.kv.len(), 1);
    /// # assert_eq!(format!("{}", kv1.get("key1").unwrap()), "value1");
    ///
    /// let mut kv2 = KV::new();
    /// kv2.insert("key2", KvValue::Str("value2".to_string()));
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

impl std::iter::FromIterator<(&'static str, KvValue)> for KV {
    fn from_iter<I: IntoIterator<Item = (&'static str, KvValue)>>(iter: I) -> Self {
        let mut c = KV::new();
        for i in iter {
            c.insert(i.0, i.1);
        }
        c
    }
}

impl std::iter::Extend<(&'static str, KvValue)> for KV {
    fn extend<I: IntoIterator<Item = (&'static str, KvValue)>>(&mut self, iter: I) {
        for i in iter {
            self.insert(i.0, i.1);
        }
    }
}
