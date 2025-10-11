// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! This crate contains an observer based on the `tracing` crate.
//!
//! See [`TracingLogger`] for details regarding usage.
//!
//! # Usage
//!
//! Add the following line to your dependencies list:
//!
//! ```toml
//! [dependencies]
#![doc = concat!("argmin-observer-tracing = \"", env!("CARGO_PKG_VERSION"), "\"")]
//! ```
//!
//! # License
//!
//! Licensed under either of
//!
//!   * Apache License, Version 2.0,
//!     ([LICENSE-APACHE](https://github.com/argmin-rs/argmin/blob/main/LICENSE-APACHE) or
//!     <http://www.apache.org/licenses/LICENSE-2.0>)
//!   * MIT License ([LICENSE-MIT](https://github.com/argmin-rs/argmin/blob/main/LICENSE-MIT) or
//!     <http://opensource.org/licenses/MIT>)
//!
//! at your option.
//!
//! ## Contribution
//!
//! Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion
//! in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above,
//! without any additional terms or conditions.

use argmin::core::observers::Observe;
use argmin::core::{Error, State, KV};
use tracing::{info, span, Level};

/// A logger using the [`tracing`](https://crates.io/crates/tracing) crate as backend.
#[derive(Clone, Default)]
pub struct TracingLogger {}

impl TracingLogger {
    pub fn new() -> Self {
        TracingLogger {}
    }
}

impl<I> Observe<I> for TracingLogger
where
    I: State,
{
    /// Log basic information about the optimization after initialization.
    fn observe_init(&mut self, name: &str, _state: &I, kv: &KV) -> Result<(), Error> {
        let span = span!(Level::INFO, "init");
        let _enter = span.enter();
        info!(target: "observer_tracing", name, kv = ?kv.kv);
        Ok(())
    }

    /// Logs information about the progress of the optimization after every iteration.
    fn observe_iter(&mut self, state: &I, kv: &KV) -> Result<(), Error> {
        let span = span!(Level::INFO, "iter");
        let _enter = span.enter();
        info!(
            target: "observer_tracing",
            iter = state.get_iter(),
            cost = Into::<f64>::into(state.get_cost()),
            best_cost = Into::<f64>::into(state.get_best_cost()),
            func_counts = ?state.get_func_counts(),
            kv = ?kv.kv,
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // TODO
}
