// Copyright 2018-2023 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use argmin::core::KV;
use bytes::{Bytes, BytesMut};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum Message {
    NewRun { name: String },
    Samples { name: String, kv: KV },
    Param { name: String, param: Vec<f64> },
    BestParam { name: String, param: Vec<f64> },
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
