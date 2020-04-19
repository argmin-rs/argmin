// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{ArgminError, Error};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::default::Default;
use std::fmt::Display;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// Defines at which intervals a checkpoint is saved.
#[derive(Clone, Serialize, Deserialize, Eq, PartialEq, Ord, PartialOrd, Debug, Hash, Copy)]
pub enum CheckpointMode {
    /// Never create checkpoint
    Never,
    /// Create checkpoint every N iterations
    Every(u64),
    /// Create checkpoint in every iteration
    Always,
}

impl Default for CheckpointMode {
    fn default() -> CheckpointMode {
        CheckpointMode::Never
    }
}

impl Display for CheckpointMode {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            CheckpointMode::Never => write!(f, "Never"),
            CheckpointMode::Every(i) => write!(f, "Every({})", i),
            CheckpointMode::Always => write!(f, "Always"),
        }
    }
}

/// Checkpoint
///
/// Defines how often and where a checkpoint is saved.
#[derive(Clone, Serialize, Deserialize, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
pub struct ArgminCheckpoint {
    mode: CheckpointMode,
    directory: String,
    name: String,
}

impl Default for ArgminCheckpoint {
    fn default() -> ArgminCheckpoint {
        ArgminCheckpoint {
            mode: CheckpointMode::Never,
            directory: ".checkpoints".to_string(),
            name: "default".to_string(),
        }
    }
}

impl ArgminCheckpoint {
    /// Define a new checkpoint
    pub fn new(directory: &str, mode: CheckpointMode) -> Result<Self, Error> {
        match mode {
            CheckpointMode::Every(_) | CheckpointMode::Always => {
                std::fs::create_dir_all(&directory)?
            }
            _ => {}
        }
        let name = "solver".to_string();
        let directory = directory.to_string();
        Ok(ArgminCheckpoint {
            mode,
            directory,
            name,
        })
    }

    /// Set directory of checkpoint
    #[inline]
    pub fn set_dir(&mut self, dir: &str) {
        self.directory = dir.to_string();
    }

    /// Get directory of checkpoint
    #[inline]
    pub fn dir(&self) -> String {
        self.directory.clone()
    }

    /// Set name of checkpoint
    #[inline]
    pub fn set_name(&mut self, name: &str) {
        self.name = name.to_string();
    }

    /// Get name of checkpoint
    #[inline]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    /// Set mode of checkpoint
    #[inline]
    pub fn set_mode(&mut self, mode: CheckpointMode) {
        self.mode = mode
    }

    /// Write checkpoint to disk
    #[inline]
    pub fn store<T: Serialize>(&self, executor: &T, filename: String) -> Result<(), Error> {
        let dir = Path::new(&self.directory);
        if !dir.exists() {
            std::fs::create_dir_all(&dir)?
        }
        let fname = dir.join(Path::new(&filename));

        let f = BufWriter::new(File::create(fname)?);
        bincode::serialize_into(f, executor)?;
        Ok(())
    }

    /// Write checkpoint based on the desired `CheckpointMode`
    #[inline]
    pub fn store_cond<T: Serialize>(&self, executor: &T, iter: u64) -> Result<(), Error> {
        let mut filename = self.name();
        filename.push_str(".arg");
        match self.mode {
            CheckpointMode::Always => self.store(executor, filename)?,
            CheckpointMode::Every(it) if iter % it == 0 => self.store(executor, filename)?,
            CheckpointMode::Never | CheckpointMode::Every(_) => {}
        };
        Ok(())
    }
}

/// Load a checkpoint from disk
pub fn load_checkpoint<T: DeserializeOwned, P: AsRef<Path>>(path: P) -> Result<T, Error> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::nooperator::MinimalNoOperator;
    use crate::core::*;

    #[derive(Serialize, Deserialize, Clone, Debug)]
    pub struct PhonySolver {}

    impl PhonySolver {
        /// Constructor
        pub fn new() -> Self {
            PhonySolver {}
        }
    }

    impl<O> Solver<O> for PhonySolver
    where
        O: ArgminOp,
    {
        fn next_iter(
            &mut self,
            _op: &mut OpWrapper<O>,
            _state: &IterState<O>,
        ) -> Result<ArgminIterData<O>, Error> {
            unimplemented!()
        }
    }

    #[test]
    fn test_store() {
        let op: MinimalNoOperator = MinimalNoOperator::new();
        let solver = PhonySolver::new();
        let exec: Executor<MinimalNoOperator, PhonySolver> =
            Executor::new(op, solver, vec![0.0f64, 0.0]);
        let check = ArgminCheckpoint::new("checkpoints", CheckpointMode::Always).unwrap();
        check.store_cond(&exec, 20).unwrap();

        let _loaded: Executor<MinimalNoOperator, PhonySolver> =
            load_checkpoint("checkpoints/solver.arg").unwrap();
    }
}
