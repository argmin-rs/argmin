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
/// # assert_eq!(format!("{}", kv.get("key1").unwrap()), "value1");
/// # assert_eq!(format!("{}", kv.get("key2").unwrap()), "value2");
/// # assert_eq!(format!("{}", kv.get("key3").unwrap()), "1234");
/// ```
#[macro_export]
macro_rules! make_kv {
    ($($k:expr =>  $v:expr;)*) => {
        $crate::core::KV { kv: std::collections::HashMap::from([ $(($k, $v.into())),* ]) }
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
        || -> $crate::core::Error { $crate::argmin_error!($error_type, $msg) }
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

/// Creates the `bulk_X` methods.
#[macro_export]
macro_rules! bulk {
    ($method_name:tt, $input:ty, $output:ty) => {
        paste::item! {
            #[doc = concat!(
                "Compute `",
                stringify!($method_name),
                "` in bulk. ",
                "If the `rayon` feature is enabled, multiple calls to `",
                stringify!($method_name),
                "` will be run in parallel using `rayon`, otherwise they will execute ",
                "sequentially. If the `rayon` feature is enabled, parallelization can still be ",
                "turned off by overwriting `parallelize` to return `false`. This can be useful ",
                "in cases where it is preferable to parallelize only certain parts. ",
                "Note that even if `parallelize` is set to false, the parameter vectors and the ",
                "problem are still required to be `Send` and `Sync`. Those bounds are linked to ",
                "the `rayon` feature. This method can be overwritten.",
            )]
            fn [<bulk_ $method_name>]<'a, P>(&self, params: &'a [P]) -> Result<Vec<$output>, Error>
            where
                P: std::borrow::Borrow<$input> + SyncAlias,
                $output: SendAlias,
                Self: SyncAlias,
            {
                #[cfg(feature = "rayon")]
                {
                    if self.parallelize() {
                        params.par_iter().map(|p| self.$method_name(p.borrow())).collect()
                    } else {
                        params.iter().map(|p| self.$method_name(p.borrow())).collect()
                    }
                }
                #[cfg(not(feature = "rayon"))]
                {
                    params.iter().map(|p| self.$method_name(p.borrow())).collect()
                }
            }
        }

        #[doc = concat!(
                    "Indicates whether to parallelize calls to `",
                    stringify!($method_name),
                    "` when using `bulk_",
                    stringify!($method_name),
                    "`. By default returns true, but can be set manually to `false` if needed. ",
                    "This allows users to turn off parallelization for certain traits ",
                    "implemented on their problem. ",
                    "Note that parallelization requires the `rayon` feature to be enabled, ",
                    "otherwise calls to `",
                    stringify!($method_name),
                    "` will be executed sequentially independent of how `parallelize` is set."
                )]
        fn parallelize(&self) -> bool {
            true
        }
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
