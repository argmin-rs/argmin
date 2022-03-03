// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Macros

/// Creates an `KV` at compile time in order to avoid pushing to the `kv` vector.
#[macro_export]
macro_rules! make_kv {
    ($($k:expr =>  $v:expr;)*) => {
        KV { kv: vec![ $(($k, format!("{:?}", $v))),* ] }
    };
}

/// Release an `T` from an `Option<T>` if it is not `None`. If it is `None`, return an
/// `ArgminError` with a message that needs to be provided.
#[macro_export]
macro_rules! check_param {
    ($param:expr, $msg:expr, $error:ident) => {
        match $param {
            None => {
                return Err(ArgminError::$error {
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
