// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use std::collections::{HashMap, HashSet};

use anyhow::Error;
use argmin::core::{TerminationStatus, KV};
use bytes::{Bytes, BytesMut};
use serde::{Deserialize, Serialize};
use time::Duration;

/// Enum used to encode information sent to spectator.
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub enum Message {
    /// Register a new run
    NewRun {
        /// Name of the run
        name: String,
        /// Name of the solver
        solver: String,
        /// Maximum number of iterations
        max_iter: u64,
        /// Target cost function value
        target_cost: f64,
        /// Initial parameter vector
        init_param: Option<Vec<f64>>,
        /// Solver-specific settings (returned by the `init` method of the `Solver` trait)
        settings: KV,
        /// Preselected metrics
        selected: HashSet<String>,
    },
    /// A set of metrics samples sent after an iteration
    Samples {
        /// Name of the run
        name: String,
        /// Current iteration
        iter: u64,
        /// Time needed for this iteration
        time: Duration,
        /// Current termination_status
        termination_status: TerminationStatus,
        /// Solver-specific metrics
        kv: KV,
    },
    /// Function evaluation counts (Cost function, gradient, Hessian, ...)
    FuncCounts {
        /// Name of the run
        name: String,
        /// Current iteration
        iter: u64,
        /// Function counts
        kv: HashMap<String, u64>,
    },
    /// Parameter vector
    Param {
        /// Name of the run
        name: String,
        /// Current iteration
        iter: u64,
        /// Current parameter vector
        param: Vec<f64>,
    },
    /// Current best parameter vector
    BestParam {
        /// Name of the run
        name: String,
        /// Current iteration
        iter: u64,
        /// Current best parameter vector
        param: Vec<f64>,
    },
    /// Termination
    Termination {
        /// Name of the run
        name: String,
        /// Termination status
        termination_status: TerminationStatus,
    },
}

impl Message {
    /// Serialize message
    #[allow(unused)]
    pub fn pack(&self) -> Result<Bytes, Error> {
        let buf = rmp_serde::encode::to_vec(&self)?;
        Ok(Bytes::from(buf))
    }

    /// Deserialize message
    #[allow(unused)]
    pub fn unpack(buf: &BytesMut) -> Result<Self, Error> {
        Ok(rmp_serde::from_slice::<Message>(buf)?)
    }
}
