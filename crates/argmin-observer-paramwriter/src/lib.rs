// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Writes parameter vectors to a file during optimization.
//!
//! See documentation of [`ParamWriter`] and [`ParamWriterFormat`] for details.
//!
//! # Usage
//!
//! Add the following line to your dependencies list:
//!
//! ```toml
//! [dependencies]
#![doc = concat!("argmin-observer-paramwriter = \"", env!("CARGO_PKG_VERSION"), "\"")]
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
use serde::Serialize;
use std::default::Default;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

/// Write parameter vectors to a file during optimization.
///
/// This observer requires a directory to save the files to and a file prefix. Files will be
/// written to disk as `<directory>/<file_prefix>_<iteration_number>.<extension>`. For
/// serialization either `JSON` or `Binary` (via [bincode](https://crates.io/crates/bincode))
/// can be chosen via the enum [`ParamWriterFormat`].
///
/// # Example
///
/// Create an observer for saving the parameter vector into a JSON file.
///
/// ```
/// use argmin_observer_paramwriter::{ParamWriter, ParamWriterFormat};
///
/// let observer = ParamWriter::new("directory", "file_prefix", ParamWriterFormat::JSON);
/// ```
///
/// Create an observer for saving the parameter vector into a binary file using
/// [`bincode`](https://crates.io/crates/bincode).
///
/// ```
/// use argmin_observer_paramwriter::{ParamWriter, ParamWriterFormat};
///
/// let observer = ParamWriter::new("directory", "file_prefix", ParamWriterFormat::Binary);
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParamWriter {
    /// Directory where files are saved to
    dir: PathBuf,
    /// File prefix
    prefix: String,
    /// Chosen serializer
    serializer: ParamWriterFormat,
}

impl ParamWriter {
    /// Create a new instance of `ParamWriter`.
    ///
    /// # Example
    /// ```
    /// # use argmin_observer_paramwriter::{ParamWriter, ParamWriterFormat};
    /// let observer = ParamWriter::new("directory", "file_prefix", ParamWriterFormat::JSON);
    /// ```
    pub fn new<N: AsRef<str>>(dir: N, prefix: N, serializer: ParamWriterFormat) -> Self {
        ParamWriter {
            dir: PathBuf::from(dir.as_ref()),
            prefix: String::from(prefix.as_ref()),
            serializer,
        }
    }
}

/// `ParamWriter` only implements `observer_iter` and not `observe_init` to avoid saving the
/// initial parameter vector. It will only save if there is a parameter vector available in the
/// state, otherwise it will skip saving silently.
impl<I> Observe<I> for ParamWriter
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

            let fname = self.dir.join(format!(
                "{}_{}.{}",
                self.prefix,
                iter,
                self.serializer.extension()
            ));
            let f = BufWriter::new(File::create(fname)?);

            match self.serializer {
                ParamWriterFormat::Binary => {
                    bincode::serialize_into(f, param)?;
                }
                ParamWriterFormat::JSON => {
                    serde_json::to_writer_pretty(f, param)?;
                }
            }
        }
        Ok(())
    }
}

/// Available serializers for [`ParamWriter`].
///
/// # Extensions
///
/// * JSON: `.json`
/// * Binary: `.bin`
///
/// # Example
///
/// ```
/// use argmin_observer_paramwriter::ParamWriterFormat;
///
/// let bincode = ParamWriterFormat::Binary;
/// let json = ParamWriterFormat::JSON;
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum ParamWriterFormat {
    /// Use [`bincode`](https://crates.io/crates/bincode) for creating binary files
    #[default]
    Binary,
    /// Use [`serde_json`](https://crates.io/crates/serde_json) for creating JSON files
    JSON,
}

impl ParamWriterFormat {
    pub fn extension(&self) -> &str {
        match *self {
            ParamWriterFormat::Binary => "bin",
            ParamWriterFormat::JSON => "json",
        }
    }
}

#[cfg(test)]
mod tests {}
