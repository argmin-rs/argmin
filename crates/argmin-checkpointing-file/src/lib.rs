// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! This crate creates checkpoints on disk for an optimization run.
//!
//! Saves a checkpoint on disk from which an interrupted optimization run can be resumed.
//! For details on the usage please see the documentation of [`FileCheckpoint`] or have a look at
//! the [example](https://github.com/argmin-rs/argmin/tree/main/examples/checkpoint).
//!
//! # Usage
//!
//! Add the following line to your dependencies list:
//!
//! ```toml
//! [dependencies]
#![doc = concat!("argmin-checkpointing-file = \"", env!("CARGO_PKG_VERSION"), "\"")]
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

pub use argmin::core::checkpointing::{Checkpoint, CheckpointingFrequency};
use argmin::core::Error;
use serde::{de::DeserializeOwned, Serialize};
use std::default::Default;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;

/// Handles saving a checkpoint to disk as a binary file.
#[derive(Clone, Eq, PartialEq, Debug, Hash)]
pub struct FileCheckpoint {
    /// Indicates how often a checkpoint is created
    pub frequency: CheckpointingFrequency,
    /// Directory where the checkpoints are saved to
    pub directory: PathBuf,
    /// Name of the checkpoint files
    pub filename: PathBuf,
}

impl Default for FileCheckpoint {
    /// Create a default `FileCheckpoint` instance.
    ///
    /// This will save the checkpoint in the file `.checkpoints/checkpoint.arg`.
    ///
    /// # Example
    ///
    /// ```
    /// use argmin_checkpointing_file::FileCheckpoint;
    /// # use argmin::core::checkpointing::CheckpointingFrequency;
    /// # use std::path::PathBuf;
    ///
    /// let checkpoint = FileCheckpoint::default();
    /// # assert_eq!(checkpoint.frequency, CheckpointingFrequency::default());
    /// # assert_eq!(checkpoint.directory, PathBuf::from(".checkpoints"));
    /// # assert_eq!(checkpoint.filename, PathBuf::from("checkpoint.arg"));
    /// ```
    fn default() -> FileCheckpoint {
        FileCheckpoint {
            frequency: CheckpointingFrequency::default(),
            directory: PathBuf::from(".checkpoints"),
            filename: PathBuf::from("checkpoint.arg"),
        }
    }
}

impl FileCheckpoint {
    /// Create a new `FileCheckpoint` instance
    ///
    /// # Example
    ///
    /// ```
    /// use argmin_checkpointing_file::{FileCheckpoint, CheckpointingFrequency};
    /// # use std::path::PathBuf;
    ///
    /// let directory = "checkpoints";
    /// let filename = "optimization";
    ///
    /// // When passed to an `Executor`, this will save a checkpoint in the file
    /// // `checkpoints/optimization.arg` in every iteration.
    /// let checkpoint = FileCheckpoint::new(directory, filename, CheckpointingFrequency::Always);
    /// # assert_eq!(checkpoint.frequency, CheckpointingFrequency::Always);
    /// # assert_eq!(checkpoint.directory, PathBuf::from("checkpoints"));
    /// # assert_eq!(checkpoint.filename, PathBuf::from("optimization.arg"));
    /// ```
    pub fn new<N: AsRef<str>>(directory: N, name: N, frequency: CheckpointingFrequency) -> Self {
        FileCheckpoint {
            frequency,
            directory: PathBuf::from(directory.as_ref()),
            filename: PathBuf::from(format!("{}.arg", name.as_ref())),
        }
    }
}

impl<S, I> Checkpoint<S, I> for FileCheckpoint
where
    S: Serialize + DeserializeOwned,
    I: Serialize + DeserializeOwned,
{
    /// Writes checkpoint to disk.
    ///
    /// If the directory does not exist already, it will be created. It uses `bincode` to serialize
    /// the data.
    /// It will return an error if creating the directory or file or serialization failed.
    ///
    /// # Example
    ///
    /// ```
    /// use argmin_checkpointing_file::{FileCheckpoint, CheckpointingFrequency, Checkpoint};
    ///
    /// # use std::fs::File;
    /// # use std::io::BufReader;
    /// # let checkpoint = FileCheckpoint::new(".checkpoints", "save_test" , CheckpointingFrequency::Always);
    /// # let solver: u64 = 12;
    /// # let state: u64 = 21;
    /// # let _ = std::fs::remove_file(".checkpoints/save_test.arg");
    /// checkpoint.save(&solver, &state);
    /// # let (f_solver, f_state): (u64, u64) = bincode::deserialize_from(
    /// #     BufReader::new(File::open(".checkpoints/save_test.arg").unwrap())
    /// # ).unwrap();
    /// # assert_eq!(solver, f_solver);
    /// # assert_eq!(state, f_state);
    /// # let _ = std::fs::remove_file(".checkpoints/save_test.arg");
    /// ```
    fn save(&self, solver: &S, state: &I) -> Result<(), Error> {
        if !self.directory.exists() {
            std::fs::create_dir_all(&self.directory)?
        }
        let fname = self.directory.join(&self.filename);
        let f = BufWriter::new(File::create(fname)?);
        bincode::serialize_into(f, &(solver, state))?;
        Ok(())
    }

    /// Load a checkpoint from disk.
    ///
    ///
    /// If there is no checkpoint on disk, it will return `Ok(None)`.
    /// Returns an error if opening the file or deserialization failed.
    ///
    /// # Example
    ///
    /// ```
    /// use argmin_checkpointing_file::{FileCheckpoint, CheckpointingFrequency, Checkpoint};
    /// # use argmin::core::Error;
    ///
    /// # use std::fs::File;
    /// # use std::io::BufWriter;
    /// # fn main() -> Result<(), Error> {
    /// # std::fs::DirBuilder::new().recursive(true).create(".checkpoints").unwrap();
    /// # let f = BufWriter::new(File::create(".checkpoints/load_test.arg")?);
    /// # let f_solver: u64 = 12;
    /// # let f_state: u64 = 21;
    /// # bincode::serialize_into(f, &(f_solver, f_state))?;
    /// # let checkpoint = FileCheckpoint::new(".checkpoints", "load_test" , CheckpointingFrequency::Always);
    /// let (solver, state) = checkpoint.load()?.unwrap();
    /// # // Let the compiler know which types to expect.
    /// # let blah1: u64 = solver;
    /// # let blah2: u64 = state;
    /// # assert_eq!(solver, f_solver);
    /// # assert_eq!(state, f_state);
    /// # let _ = std::fs::remove_file(".checkpoints/load_test.arg");
    /// #
    /// # // Return none if File does not exist
    /// # let checkpoint = FileCheckpoint::new(".checkpoints", "certainly_does_not_exist" , CheckpointingFrequency::Always);
    /// # let loaded: Option<(u64, u64)> = checkpoint.load()?;
    /// # assert!(loaded.is_none());
    /// # Ok(())
    /// # }
    /// ```
    fn load(&self) -> Result<Option<(S, I)>, Error> {
        let path = &self.directory.join(&self.filename);
        if !path.exists() {
            return Ok(None);
        }
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Ok(Some(bincode::deserialize_from(reader)?))
    }

    /// Returns the how often a checkpoint is to be saved.
    ///
    /// Used internally by [`save_cond`](`argmin::core::checkpointing::Checkpoint::save_cond`).
    fn frequency(&self) -> CheckpointingFrequency {
        self.frequency
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use argmin::core::test_utils::TestSolver;
    use argmin::core::{IterState, State};

    #[test]
    #[allow(clippy::type_complexity)]
    fn test_save() {
        let solver = TestSolver::new();
        let state: IterState<Vec<f64>, (), (), (), (), f64> =
            IterState::new().param(vec![1.0f64, 0.0]);
        let check = FileCheckpoint::new("checkpoints", "solver", CheckpointingFrequency::Always);
        check.save_cond(&solver, &state, 20).unwrap();

        let _loaded: Option<(TestSolver, IterState<Vec<f64>, (), (), (), (), f64>)> =
            check.load().unwrap();
    }
}
