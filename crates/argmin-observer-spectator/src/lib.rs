// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! This observer sends metrics and parameter vectors to a Spectator instance.
//!
//! ## Example
//!
//! ```
//! use argmin_observer_spectator::SpectatorBuilder;
//!
//! let observer = SpectatorBuilder::new()
//!     // Optional: Name the optimization run
//!     // Default: random uuid.
//!     .with_name("optimization_run_1")
//!     // Optional, defaults to 127.0.0.1
//!     .with_host("127.0.0.1")
//!     // Optional, defaults to 5498
//!     .with_port(5498)
//!     // Choose which metrics should automatically be selected.
//!     // If omitted, all metrics will be selected.
//!     .select(&["cost", "best_cost"])
//!     // Build Spectator observer
//!     .build();
//! ```
//!
//! The `observer`, when passed to `add_observer` of `Executor` sends metrics to a Spectator
//! instance running on `127.0.0.1:5498`. For details on how to configure the observer the reader
//! is referred to the documentation of [`SpectatorBuilder`].
//! Make sure a Spectator instance is running when calling `.build()` on [`SpectatorBuilder`].
//!
//! # Usage
//!
//! Add the following line to your dependencies list:
//!
//! ```toml
//! [dependencies]
#![doc = concat!("argmin-observer-spectator = \"", env!("CARGO_PKG_VERSION"), "\"")]
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

mod observer;
mod sender;

pub use observer::SpectatorBuilder;
