// Copyright 2018-2023 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use time::Duration;

use argmin::core::{TerminationStatus, KV};
use bytes::{Bytes, BytesMut};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum Message {
    NewRun {
        name: String,
        max_iter: u64,
        target_cost: f64,
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
}

impl Message {
    pub fn pack(&self) -> Result<Bytes, anyhow::Error> {
        let buf = rmp_serde::encode::to_vec(&self)?;
        Ok(Bytes::from(buf))
    }

    pub fn unpack(buf: &BytesMut) -> Result<Self, anyhow::Error> {
        Ok(rmp_serde::from_slice::<Message>(buf)?)
    }
}
