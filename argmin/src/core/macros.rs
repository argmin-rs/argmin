// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Macros

/// Creates an `KV` at compile time
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
#[macro_export]
macro_rules! make_kv {
    ($($k:expr =>  $v:expr;)*) => {
        $crate::core::KV { kv: vec![ $(($k, std::rc::Rc::new($v))),* ] }
    };
}

/// Release an `T` from an `Option<T>` if it is not `None`. If it is `None`, return an
/// `ArgminError` with a provided message.
#[macro_export]
macro_rules! check_param {
    ($param:expr, $msg:expr, $error:ident) => {
        match $param {
            None => {
                return Err($crate::core::ArgminError::$error {
                    text: $msg.to_string(),
                }
                .into());
            }
            Some(ref x) => x.clone(),
        }
    };
    ($param:expr, $msg:expr) => {
        check_param!($param, $msg, NotInitialized)
    };
}

/// Create an `ArgminError` with a provided message.
#[macro_export]
macro_rules! argmin_error {
    ($error_type:ident, $msg:expr) => {
        $crate::core::ArgminError::$error_type {
            text: $msg.to_string(),
        }
        .into()
    };
}

/// Create an `ArgminError` with a provided message wrapped in a closure for use in
/// `.ok_or_else(...)` methods on `Option`s.
#[macro_export]
macro_rules! argmin_error_closure {
    ($error_type:ident, $msg:expr) => {
        || -> $crate::core::Error { argmin_error!($error_type, $msg) }
    };
}

/// Convert a constant to a float of given precision
#[macro_export]
macro_rules! float {
    ($t:ident, $val:expr) => {
        $t::from_f64($val).unwrap()
    };
    ($val:expr) => {
        F::from_f64($val).unwrap()
    };
}

/// Implements a simple send and a simple sync test for a given type.
#[cfg(test)]
macro_rules! send_sync_test {
    ($n:ident, $t:ty) => {
        paste::item! {
            #[test]
            #[allow(non_snake_case)]
            fn [<test_send_ $n>]() {
                fn assert_send<T: Send>() {}
                assert_send::<$t>();
            }
        }

        paste::item! {
            #[test]
            #[allow(non_snake_case)]
            fn [<test_sync_ $n>]() {
                fn assert_sync<T: Sync>() {}
                assert_sync::<$t>();
            }
        }
    };
}

/// Reuse a list of trait bounds by giving it a name,
/// e.g. trait_bound!(CopyAndDefault; Copy, Default);
#[macro_export]
macro_rules! trait_bound {
    ($name:ident ; $head:path $(, $tail:path)*) => {
        #[allow(missing_docs)]
        pub trait $name : $head $(+ $tail)* {}
        impl<T> $name for T where T: $head $(+ $tail)* {}
    };
}

/// Creates tests for asserting that a struct implements `Send`, `Sync` and `Clone`
#[cfg(test)]
#[macro_export]
macro_rules! test_trait_impl {
    ($n:ident, $t:ty) => {
        paste::item! {
            #[test]
            #[allow(non_snake_case)]
            fn [<test_send_ $n>]() {
                fn assert_send<T: Send>() {}
                assert_send::<$t>();
            }
        }

        paste::item! {
            #[test]
            #[allow(non_snake_case)]
            fn [<test_sync_ $n>]() {
                fn assert_sync<T: Sync>() {}
                assert_sync::<$t>();
            }
        }

        paste::item! {
            #[test]
            #[allow(non_snake_case)]
            fn [<test_clone_ $n>]() {
                fn assert_clone<T: Clone>() {}
                assert_clone::<$t>();
            }
        }
    };
}

/// Asserts that expression $n leads to an error of type $t and text $s
#[cfg(test)]
#[macro_export]
macro_rules! assert_error {
    ($n:expr, $t:ty, $s:expr) => {
        assert_eq!(
            $n.err().unwrap().downcast_ref::<$t>().unwrap().to_string(),
            $s
        );
    };
}
