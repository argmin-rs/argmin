// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Macros

/// This macro crates a test for send an sync
#[cfg(test)]
#[macro_export]
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
