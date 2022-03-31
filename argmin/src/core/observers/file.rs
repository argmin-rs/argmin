// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Write parameter vectors to a file during optimization.
//!
//! See documentation of [`WriteToFile`] and [`WriteToFileSerializer`] for details.

use crate::core::observers::Observe;
use crate::core::{Error, State, KV};
use serde::Serialize;
use std::default::Default;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

/// Write parameter vectors to a file during optimization.
///
/// This observer requires a directory to save the files to and a file prefix. Files will be
/// written to disk as `<directory>/<file_prefix>_<iteration_number>.arp`. For serialization
/// either `JSON` or [`bincode`](https://crates.io/crates/bincode) can be chosen via the enum
/// [`WriteToFileSerializer`].
///
/// This feature requires the `serde1` feature to be set.
///
/// # Example
///
/// Create an observer for saving the parameter vector into a JSON file.
///
/// ```
/// use argmin::core::observers::{WriteToFile, WriteToFileSerializer};
///
/// let observer = WriteToFile::new("directory", "file_prefix", WriteToFileSerializer::JSON);
/// ```
///
/// Create an observer for saving the parameter vector into a binary file using
/// [`bincode`](https://crates.io/crates/bincode).
///
/// ```
/// use argmin::core::observers::{WriteToFile, WriteToFileSerializer};
///
/// let observer = WriteToFile::new("directory", "file_prefix", WriteToFileSerializer::Bincode);
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WriteToFile {
    /// Directory where files are saved to
    dir: PathBuf,
    /// File prefix
    prefix: String,
    /// Chosen serializer
    serializer: WriteToFileSerializer,
}

impl WriteToFile {
    /// Create a new instance of `WriteToFile`.
    ///
    /// # Example
    /// ```
    /// # use argmin::core::observers::{WriteToFile, WriteToFileSerializer};
    /// let observer = WriteToFile::new("directory", "file_prefix", WriteToFileSerializer::JSON);
    /// ```
    pub fn new<N: AsRef<str>>(dir: N, prefix: N, serializer: WriteToFileSerializer) -> Self {
        WriteToFile {
            dir: PathBuf::from(dir.as_ref()),
            prefix: String::from(prefix.as_ref()),
            serializer,
        }
    }
}

/// `WriteToFile` only implements `observer_iter` and not `observe_init` to avoid saving the
/// initial parameter vector. It will only save if there is a parameter vector available in the
/// state, otherwise it will skip saving silently.
impl<I> Observe<I> for WriteToFile
where
    I: State,
    <I as State>::Param: Serialize,
{
    fn observe_iter(&mut self, state: &I, _kv: &KV) -> Result<(), Error> {
        if let Some(param) = state.get_param() {
            let iter = state.get_iter();
            if !self.dir.exists() {
                std::fs::create_dir_all(&self.dir)?
            }

            let fname = self.dir.join(format!("{}_{}.arp", self.prefix, iter));
            let f = BufWriter::new(File::create(fname)?);

            match self.serializer {
                WriteToFileSerializer::Bincode => {
                    bincode::serialize_into(f, param)?;
                }
                WriteToFileSerializer::JSON => {
                    serde_json::to_writer_pretty(f, param)?;
                }
            }
        }
        Ok(())
    }
}

/// Available serializers for [`WriteToFile`].
///
/// # Example
///
/// ```
/// use argmin::core::observers::WriteToFileSerializer;
///
/// let bincode = WriteToFileSerializer::Bincode;
/// let json = WriteToFileSerializer::JSON;
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum WriteToFileSerializer {
    /// Use [`bincode`](https://crates.io/crates/bincode) for creating binary files
    Bincode,
    /// Use [`serde_json`](https://crates.io/crates/serde_json) for creating JSON files
    JSON,
}

impl Default for WriteToFileSerializer {
    /// Defaults to `Bincode`
    ///
    /// # Example
    ///
    /// ```
    /// use argmin::core::observers::WriteToFileSerializer;
    /// let default = WriteToFileSerializer::default();
    /// assert_eq!(default, WriteToFileSerializer::Bincode);
    /// ```
    fn default() -> Self {
        WriteToFileSerializer::Bincode
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    send_sync_test!(write_to_file, WriteToFile);
}
