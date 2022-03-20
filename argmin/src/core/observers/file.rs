// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Output parameter vectors to file

use crate::core::observers::Observe;
use crate::core::{ArgminFloat, Error, IterState, State, KV};
use serde::{Deserialize, Serialize};
use std::default::Default;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

/// Different kinds of serializers
#[derive(Copy, Clone, Serialize, Deserialize, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum WriteToFileSerializer {
    /// Bincode
    Bincode,
    /// JSON
    JSON,
}

impl Default for WriteToFileSerializer {
    fn default() -> Self {
        WriteToFileSerializer::Bincode
    }
}

/// Write parameter vectors to file
#[derive(Clone, Serialize, Deserialize, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct WriteToFile {
    /// Directory
    dir: String,
    /// File prefix
    prefix: String,
    /// Chosen serializer
    serializer: WriteToFileSerializer,
}

impl WriteToFile {
    /// Create a new `WriteToFile` struct
    pub fn new(dir: &str, prefix: &str) -> Self {
        WriteToFile {
            dir: dir.to_string(),
            prefix: prefix.to_string(),
            serializer: WriteToFileSerializer::Bincode,
        }
    }

    /// Set serializer
    #[must_use]
    pub fn serializer(mut self, serializer: WriteToFileSerializer) -> Self {
        self.serializer = serializer;
        self
    }
}

impl<P, G, J, H, F> Observe<IterState<P, G, J, H, F>> for WriteToFile
where
    P: Clone + Serialize,
    F: ArgminFloat,
{
    fn observe_iter(&mut self, state: &IterState<P, G, J, H, F>, _kv: &KV) -> Result<(), Error> {
        let param = state.get_param_ref().unwrap().clone();
        let iter = state.get_iter();
        let dir = Path::new(&self.dir);
        if !dir.exists() {
            std::fs::create_dir_all(&dir)?
        }

        let fname = dir.join(format!("{}_{}.arp", self.prefix, iter));

        let f = BufWriter::new(File::create(fname)?);
        match self.serializer {
            WriteToFileSerializer::Bincode => {
                bincode::serialize_into(f, &param)?;
            }
            WriteToFileSerializer::JSON => {
                serde_json::to_writer_pretty(f, &param)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    send_sync_test!(write_to_file, WriteToFile);
}
