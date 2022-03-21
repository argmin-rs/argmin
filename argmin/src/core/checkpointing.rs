// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{ArgminError, DeserializeOwnedAlias, Error, SerializeAlias};
use serde::{Deserialize, Serialize};
use std::default::Default;
use std::fmt::Display;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// Checkpoint
///
/// Defines how often and where a checkpoint is saved.
#[derive(Clone, Serialize, Deserialize, Eq, PartialEq, Debug, Hash)]
pub struct Checkpoint {
    /// Indicates how often a checkpoint is created
    mode: CheckpointingFrequency,
    /// Directory where the checkpoints are stored
    directory: String,
    /// Name of the checkpoint files
    filename: String,
}

impl Default for Checkpoint {
    fn default() -> Checkpoint {
        Checkpoint {
            mode: CheckpointingFrequency::default(),
            directory: ".checkpoints".to_string(),
            filename: "default.arg".to_string(),
        }
    }
}

impl Checkpoint {
    /// Define a new checkpoint
    pub fn new(directory: &str, name: &str, mode: CheckpointingFrequency) -> Self {
        Checkpoint {
            mode,
            directory: directory.to_string(),
            filename: format!("{}.arg", name),
        }
    }

    /// Write checkpoint to disk
    fn store<E: SerializeAlias, I: SerializeAlias>(
        &self,
        executor: &E,
        state: &I,
    ) -> Result<(), Error> {
        let dir = Path::new(&self.directory);
        if !dir.exists() {
            std::fs::create_dir_all(&dir)?
        }
        let fname = dir.join(Path::new(&self.filename));
        let f = BufWriter::new(File::create(fname)?);
        bincode::serialize_into(f, &(executor, state))?;
        Ok(())
    }

    /// Write checkpoint based on the desired `CheckpointingFrequency`
    pub fn store_cond<E: SerializeAlias, I: SerializeAlias>(
        &self,
        executor: &E,
        state: &I,
        iter: u64,
    ) -> Result<(), Error> {
        match self.mode {
            CheckpointingFrequency::Always => self.store(executor, state)?,
            CheckpointingFrequency::Every(it) if iter % it == 0 => self.store(executor, state)?,
            CheckpointingFrequency::Never | CheckpointingFrequency::Every(_) => {}
        };
        Ok(())
    }
}

/// Load a checkpoint from disk
pub fn load_checkpoint<T: DeserializeOwnedAlias, I: DeserializeOwnedAlias, P: AsRef<Path>>(
    path: P,
) -> Result<(T, I), Error> {
    let path = path.as_ref();
    if !path.exists() {
        return Err(ArgminError::CheckpointNotFound {
            text: path.to_str().unwrap().to_string(),
        }
        .into());
    }
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    Ok(bincode::deserialize_from(reader)?)
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
#[derive(Clone, Serialize, Deserialize, Eq, PartialEq, Debug, Hash, Copy)]
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
mod tests {
    use super::*;
    use crate::core::test_utils::{TestProblem, TestSolver};
    use crate::core::{Executor, IterState};

    #[test]
    #[allow(clippy::type_complexity)]
    fn test_store() {
        let problem = TestProblem::new();
        let solver = TestSolver::new();
        let mut exec: Executor<TestProblem, TestSolver, _> = Executor::new(problem, solver)
            .configure(|config: IterState<Vec<f64>, (), (), (), f64>| {
                config.param(vec![1.0f64, 0.0])
            });
        let state = exec.take_state().unwrap();
        let check = Checkpoint::new("checkpoints", "solver", CheckpointingFrequency::Always);
        check.store_cond(&exec, &state, 20).unwrap();

        let (_loaded, _state): (
            Executor<TestProblem, TestSolver, IterState<Vec<f64>, (), (), (), f64>>,
            IterState<Vec<f64>, (), (), (), f64>,
        ) = load_checkpoint("checkpoints/solver.arg").unwrap();
    }
}
