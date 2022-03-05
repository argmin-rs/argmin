// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Macros

/// Implements a (private) setter method for field `$name` of type `$type` via a mutable reference
/// to `Self`. Returns `&mut Self`.
#[macro_export]
macro_rules! setter {
    ($name:ident, $type:ty, $doc:tt) => {
        #[doc=$doc]
        fn $name(&mut self, $name: $type) -> &mut Self {
            self.$name = $name;
            self
        }
    };
}

/// Implements a (public) setter method for field `$name` of type `$type` via a mutable reference
/// to `Self`. Returns `&mut Self`.
#[macro_export]
macro_rules! pub_setter {
    ($name:ident, $type:ty, $doc:tt) => {
        #[doc=$doc]
        pub fn $name(&mut self, $name: $type) -> &mut Self {
            self.$name = $name;
            self
        }
    };
}

/// Implements a (private) getter method for field `$name` of type `Option<$type>` via an immutable
/// reference to `Self`. Clones when returning a `Option<$type>`.
#[macro_export]
macro_rules! getter_option {
    ($name:ident, $type:ty, $doc:tt) => {
        item! {
            #[doc=$doc]
            fn [<get_ $name>](&self) -> Option<$type> {
                self.$name.clone()
            }
        }
    };
}

/// Implements a (public) getter method for field `$name` of type `Option<$type>` via an immutable
/// reference to `Self`. Clones when returning a `Option<$type>`.
#[macro_export]
macro_rules! pub_getter_option {
    ($name:ident, $type:ty, $doc:tt) => {
        item! {
            #[doc=$doc]
            pub fn [<get_ $name>](&self) -> Option<$type> {
                self.$name.clone()
            }
        }
    };
}

/// Implements a (private) getter method for field `$name` of type `Option<$type>` via an immutable
/// reference to `Self`. Returns a reference to the inner `$type` as a `Option<&$type>`.
#[macro_export]
macro_rules! getter_option_ref {
    ($name:ident, $type:ty, $doc:tt) => {
        item! {
            #[doc=$doc]
            fn [<get_ $name _ref>](&self) -> Option<&$type> {
                self.$name.as_ref()
            }
        }
    };
}

/// Implements a (public) getter method for field `$name` of type `Option<$type>` via an immutable
/// reference to `Self`. Returns a reference to the inner `$type` as a `Option<&$type>`.
#[macro_export]
macro_rules! pub_getter_option_ref {
    ($name:ident, $type:ty, $doc:tt) => {
        item! {
            #[doc=$doc]
            pub fn [<get_ $name _ref>](&self) -> Option<&$type> {
                self.$name.as_ref()
            }
        }
    };
}

/// Implements a (private) getter method for field `$name` of type `Option<$type>` via a mutable
/// reference to `Self`. Returns the inner `$type` as a `Option<$type>` by `take`ing it (therefore
/// the content of the field will be replaced with `None`).
#[macro_export]
macro_rules! take {
    ($name:ident, $type:ty, $doc:tt) => {
        item! {
            #[doc=$doc]
            fn [<take_ $name>](&mut self) -> Option<$type> {
                self.$name.take()
            }
        }
    };
}

/// Implements a (public) getter method for field `$name` of type `Option<$type>` via a mutable
/// reference to `Self`. Returns the inner `$type` as a `Option<$type>` by `take`ing it (therefore
/// the content of the field will be replaced with `None`).
#[macro_export]
macro_rules! pub_take {
    ($name:ident, $type:ty, $doc:tt) => {
        item! {
            #[doc=$doc]
            pub fn [<take_ $name>](&mut self) -> Option<$type> {
                self.$name.take()
            }
        }
    };
}

/// Implements a (private) getter method for field `$name` of type `$type` via an immutable
/// reference to `Self`. Clones when returning a `$type`.
#[macro_export]
macro_rules! getter {
    ($name:ident, $type:ty, $doc:tt) => {
        item! {
            #[doc=$doc]
            fn [<get_ $name>](&self) -> $type {
                self.$name.clone()
            }
        }
    };
}

/// Implements a (public) getter method for field `$name` of type `$type` via an immutable
/// reference to `Self`. Clones when returning a `$type`.
#[macro_export]
macro_rules! pub_getter {
    ($name:ident, $type:ty, $doc:tt) => {
        item! {
            #[doc=$doc]
            pub fn [<get_ $name>](&self) -> $type {
                self.$name.clone()
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
