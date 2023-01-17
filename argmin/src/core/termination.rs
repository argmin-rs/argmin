// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// Status of optimization execution
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub enum TerminationStatus {
    /// Execution is terminated
    Terminated(TerminationReason),
    /// Execution is running
    NotTerminated,
}

impl TerminationStatus {
    /// Returns `true` if a solver terminated and `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use argmin::core::{TerminationStatus, TerminationReason};
    ///
    /// assert!(TerminationStatus::Terminated(TerminationReason::MaxItersReached).terminated());
    /// assert!(TerminationStatus::Terminated(TerminationReason::TargetCostReached).terminated());
    /// assert!(TerminationStatus::Terminated(TerminationReason::SolverConverged).terminated());
    /// assert!(TerminationStatus::Terminated(TerminationReason::KeyboardInterrupt).terminated());
    /// assert!(TerminationStatus::Terminated(TerminationReason::SolverExit("Exit reason".to_string())).terminated());
    /// ```
    pub fn terminated(&self) -> bool {
        matches!(self, TerminationStatus::Terminated(_))
    }
}

impl std::fmt::Display for TerminationStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            TerminationStatus::Terminated(reason) => f.write_str(reason.text()),
            TerminationStatus::NotTerminated => f.write_str("Running"),
        }
    }
}

impl Default for TerminationStatus {
    fn default() -> Self {
        TerminationStatus::NotTerminated
    }
}

/// Reasons for optimization algorithms to stop
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub enum TerminationReason {
    /// Reached maximum number of iterations
    MaxItersReached,
    /// Reached target cost function value
    TargetCostReached,
    /// Algorithm manually interrupted with Ctrl+C
    KeyboardInterrupt,
    /// Converged
    SolverConverged,
    /// Solver exit with given reason
    SolverExit(String),
}

impl TerminationReason {
    /// Returns a textual representation of what happened.
    ///
    /// # Example
    ///
    /// ```
    /// use argmin::core::TerminationReason;
    ///
    /// assert_eq!(
    ///     TerminationReason::MaxItersReached.text(),
    ///     "Maximum number of iterations reached"
    /// );
    /// assert_eq!(
    ///     TerminationReason::TargetCostReached.text(),
    ///     "Target cost value reached"
    /// );
    /// assert_eq!(
    ///     TerminationReason::KeyboardInterrupt.text(),
    ///     "Keyboard interrupt"
    /// );
    /// assert_eq!(
    ///     TerminationReason::SolverConverged.text(),
    ///     "Solver converged"
    /// );
    /// assert_eq!(
    ///     TerminationReason::SolverExit("Aborted".to_string()).text(),
    ///     "Aborted"
    /// );
    /// ```
    pub fn text(&self) -> &str {
        match self {
            TerminationReason::MaxItersReached => "Maximum number of iterations reached",
            TerminationReason::TargetCostReached => "Target cost value reached",
            TerminationReason::KeyboardInterrupt => "Keyboard interrupt",
            TerminationReason::SolverConverged => "Solver converged",
            TerminationReason::SolverExit(reason) => reason.as_ref(),
        }
    }
}

impl std::fmt::Display for TerminationReason {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.text())
    }
}

impl Default for TerminationReason {
    fn default() -> Self {
        TerminationReason::SolverExit("Undefined".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    send_sync_test!(termination_reason, TerminationReason);
}
