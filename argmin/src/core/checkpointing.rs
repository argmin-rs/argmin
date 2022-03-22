// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::Error;
#[cfg(feature = "serde1")]
use crate::core::{DeserializeOwnedAlias, SerializeAlias};
use std::default::Default;
use std::fmt::Display;
#[cfg(feature = "serde1")]
use std::fs::File;
#[cfg(feature = "serde1")]
use std::io::{BufReader, BufWriter};
#[cfg(feature = "serde1")]
use std::path::Path;

/// TODO
pub trait Checkpoint<S, I> {
    /// TODO
    fn store(&self, solver: &S, state: &I) -> Result<(), Error>;

    /// TODO
    fn store_cond(&self, solver: &S, state: &I, iter: u64) -> Result<(), Error> {
        match self.frequency() {
            CheckpointingFrequency::Always => self.store(solver, state)?,
            CheckpointingFrequency::Every(it) if iter % it == 0 => self.store(solver, state)?,
            CheckpointingFrequency::Never | CheckpointingFrequency::Every(_) => {}
        };
        Ok(())
    }

    /// TODO
    fn load(&self) -> Result<Option<(S, I)>, Error>;

    /// TODO
    fn frequency(&self) -> CheckpointingFrequency;
}

/// Checkpoint
///
/// Defines how often and where a checkpoint is saved.
#[derive(Clone, Eq, PartialEq, Debug, Hash)]
#[cfg(feature = "serde1")]
pub struct FileCheckpoint {
    /// Indicates how often a checkpoint is created
    frequency: CheckpointingFrequency,
    /// Directory where the checkpoints are stored
    directory: String,
    /// Name of the checkpoint files
    filename: String,
}

#[cfg(feature = "serde1")]
impl Default for FileCheckpoint {
    fn default() -> FileCheckpoint {
        FileCheckpoint {
            frequency: CheckpointingFrequency::default(),
            directory: ".checkpoints".to_string(),
            filename: "default.arg".to_string(),
        }
    }
}

#[cfg(feature = "serde1")]
impl FileCheckpoint {
    /// Define a new checkpoint
    pub fn new(directory: &str, name: &str, frequency: CheckpointingFrequency) -> Self {
        FileCheckpoint {
            frequency,
            directory: directory.to_string(),
            filename: format!("{}.arg", name),
        }
    }
}

#[cfg(feature = "serde1")]
impl<S, I> Checkpoint<S, I> for FileCheckpoint
where
    S: SerializeAlias + DeserializeOwnedAlias,
    I: SerializeAlias + DeserializeOwnedAlias,
{
    /// Write checkpoint to disk
    fn store(&self, solver: &S, state: &I) -> Result<(), Error> {
        let dir = Path::new(&self.directory);
        if !dir.exists() {
            std::fs::create_dir_all(&dir)?
        }
        let fname = dir.join(Path::new(&self.filename));
        let f = BufWriter::new(File::create(fname)?);
        bincode::serialize_into(f, &(solver, state))?;
        Ok(())
    }

    /// Load a checkpoint from disk
    fn load(&self) -> Result<Option<(S, I)>, Error> {
        let path = Path::new(&self.directory).join(Path::new(&self.filename));
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

/// Defines at which intervals a checkpoint is saved.
///
/// # Example
///
/// ```
/// use argmin::core::checkpointing::CheckpointingFrequency;
///
/// // A checkpoint every 10 iterations
/// let every_10 = CheckpointingFrequency::Every(10);
///
/// // A checkpoint in each iteration
/// let always = CheckpointingFrequency::Always;
///
/// // The default is `CheckpointingFrequency::Never`
/// assert_eq!(CheckpointingFrequency::default(), CheckpointingFrequency::Never);
/// ```
#[derive(Clone, Eq, PartialEq, Debug, Hash, Copy)]
pub enum CheckpointingFrequency {
    /// Never create checkpoint
    Never,
    /// Create checkpoint every N iterations
    Every(u64),
    /// Create checkpoint in every iteration
    Always,
}

impl Default for CheckpointingFrequency {
    fn default() -> CheckpointingFrequency {
        CheckpointingFrequency::Never
    }
}

impl Display for CheckpointingFrequency {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            CheckpointingFrequency::Never => write!(f, "Never"),
            CheckpointingFrequency::Every(i) => write!(f, "Every({})", i),
            CheckpointingFrequency::Always => write!(f, "Always"),
        }
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
    fn test_store() {
        let solver = TestSolver::new();
        let state: IterState<Vec<f64>, (), (), (), f64> = IterState::new().param(vec![1.0f64, 0.0]);
        // let state: usize = 12;
        let check = FileCheckpoint::new("checkpoints", "solver", CheckpointingFrequency::Always);
        check.store_cond(&solver, &state, 20).unwrap();

        // let _loaded: Option<(TestSolver, usize)> = check.load().unwrap();
        let _loaded: Option<(TestSolver, IterState<Vec<f64>, (), (), (), f64>)> =
            check.load().unwrap();
    }
}
