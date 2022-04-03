// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Errors

use thiserror::Error;

/// Argmin error type
#[derive(Debug, Error)]
pub enum ArgminError {
    /// Indicates and invalid parameter
    #[error("Invalid parameter: {text:?}")]
    InvalidParameter {
        /// Text
        text: String,
    },

    /// Indicates that a function is not implemented
    #[error("Not implemented: {text:?}")]
    NotImplemented {
        /// Text
        text: String,
    },

    /// Indicates that a function is not initialized
    #[error("Not initialized: {text:?}")]
    NotInitialized {
        /// Text
        text: String,
    },

    /// Indicates that a condition is violated
    #[error("Condition violated: {text:?}")]
    ConditionViolated {
        /// Text
        text: String,
    },

    /// Checkpoint was not found
    #[error("Checkpoint not found: {text:?}")]
    CheckpointNotFound {
        /// Text
        text: String,
    },

    /// For errors which are likely bugs.
    #[error("Potential bug: {text:?}. This is potentially a bug. Please file a report on https://github.com/argmin-rs/argmin/issues")]
    PotentialBug {
        /// Text
        text: String,
    },

    /// Indicates an impossible error
    #[error("Impossible Error: {text:?}")]
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
