// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::checkpointing::{Checkpoint, CheckpointingFrequency};
use crate::core::{DeserializeOwnedAlias, Error, SerializeAlias};
use std::default::Default;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;

/// FileCheckpoint
///
/// Defines how often and where a checkpoint is saved.
#[derive(Clone, Eq, PartialEq, Debug, Hash)]
pub struct FileCheckpoint {
    /// Indicates how often a checkpoint is created
    frequency: CheckpointingFrequency,
    /// Directory where the checkpoints are saved to
    directory: PathBuf,
    /// Name of the checkpoint files
    filename: PathBuf,
}

impl Default for FileCheckpoint {
    fn default() -> FileCheckpoint {
        FileCheckpoint {
            frequency: CheckpointingFrequency::default(),
            directory: PathBuf::from(".checkpoints"),
            filename: PathBuf::from("checkpoint.arg"),
        }
    }
}

impl FileCheckpoint {
    /// Define a new checkpoint
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
    S: SerializeAlias + DeserializeOwnedAlias,
    I: SerializeAlias + DeserializeOwnedAlias,
{
    /// Write checkpoint to disk
    fn save(&self, solver: &S, state: &I) -> Result<(), Error> {
        if !self.directory.exists() {
            std::fs::create_dir_all(&self.directory)?
        }
        let fname = self.directory.join(&self.filename);
        let f = BufWriter::new(File::create(fname)?);
        bincode::serialize_into(f, &(solver, state))?;
        Ok(())
    }

    /// Load a checkpoint from disk
    fn load(&self) -> Result<Option<(S, I)>, Error> {
        let path = &self.directory.join(&self.filename);
        if !path.exists() {
            return Ok(None);
        }
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Ok(Some(bincode::deserialize_from(reader)?))
    }

    fn frequency(&self) -> CheckpointingFrequency {
        self.frequency
    }
}

#[cfg(test)]
#[cfg(feature = "serde1")]
mod tests {
    use super::*;
    use crate::core::test_utils::TestSolver;
    use crate::core::{IterState, State};

    #[test]
    #[allow(clippy::type_complexity)]
    fn test_save() {
        let solver = TestSolver::new();
        let state: IterState<Vec<f64>, (), (), (), f64> = IterState::new().param(vec![1.0f64, 0.0]);
        // let state: usize = 12;
        let check = FileCheckpoint::new("checkpoints", "solver", CheckpointingFrequency::Always);
        check.save_cond(&solver, &state, 20).unwrap();

        // let _loaded: Option<(TestSolver, usize)> = check.load().unwrap();
        let _loaded: Option<(TestSolver, IterState<Vec<f64>, (), (), (), f64>)> =
            check.load().unwrap();
    }
}
