// Copyright 2018-2023 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use std::collections::HashSet;

use time::Duration;

use anyhow::Error;
use argmin::core::{TerminationStatus, KV};
use bytes::{Bytes, BytesMut};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum Message {
    NewRun {
        name: String,
        solver: String,
        max_iter: u64,
        target_cost: f64,
        init_param: Option<Vec<f64>>,
        settings: KV,
        selected: HashSet<String>,
    },
    Samples {
        name: String,
        iter: u64,
        time: Duration,
        termination_status: TerminationStatus,
        kv: KV,
    },
    Param {
        name: String,
        iter: u64,
        param: Vec<f64>,
    },
    BestParam {
        name: String,
        iter: u64,
        param: Vec<f64>,
    },
    Termination {
        name: String,
        termination_status: TerminationStatus,
    },
}

impl Message {
    #[allow(unused)]
    pub fn pack(&self) -> Result<Bytes, Error> {
        let buf = rmp_serde::encode::to_vec(&self)?;
        Ok(Bytes::from(buf))
    }

    #[allow(unused)]
    pub fn unpack(buf: &BytesMut) -> Result<Self, Error> {
        Ok(rmp_serde::from_slice::<Message>(buf)?)
    }
}
