// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Errors
//!
//! Gigantic TODO: Remove `failure` dependency and implement own error handling.

use failure_derive::Fail;

/// Argmin error type
#[derive(Debug, Fail)]
pub enum ArgminError {
    /// Indicates and invalid parameter
    #[fail(display = "Invalid parameter: {}", text)]
    InvalidParameter {
        /// Text
        text: String,
    },

    /// Indicates that a function is not implemented
    #[fail(display = "Not implemented: {}", text)]
    NotImplemented {
        /// Text
        text: String,
    },

    /// Indicates that a function is not initialized
    #[fail(display = "Not initialized: {}", text)]
    NotInitialized {
        /// Text
        text: String,
    },

    /// Indicates that a condition is violated
    #[fail(display = "Condition violated: {}", text)]
    ConditionViolated {
        /// Text
        text: String,
    },

    /// Checkpoint was not found
    #[fail(display = "Checkpoint not found: {}", text)]
    CheckpointNotFound {
        /// Text
        text: String,
    },

    /// Indicates an impossible error
    #[fail(display = "Impossible Error: {}", text)]
    ImpossibleError {
        /// Text
        text: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    send_sync_test!(error, ArgminError);
}
