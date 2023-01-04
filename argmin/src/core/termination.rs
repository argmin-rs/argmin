// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// Status of optimization execution
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
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
    /// assert!(TerminationStatus::Terminated(TerminationReason::TargetPrecisionReached).terminated());
    /// assert!(TerminationStatus::Terminated(TerminationReason::NoChangeInCost).terminated());
    /// assert!(TerminationStatus::Terminated(TerminationReason::AcceptedStallIterExceeded).terminated());
    /// assert!(TerminationStatus::Terminated(TerminationReason::BestStallIterExceeded).terminated());
    /// assert!(TerminationStatus::Terminated(TerminationReason::LineSearchConditionMet).terminated());
    /// assert!(TerminationStatus::Terminated(TerminationReason::TargetToleranceReached).terminated());
    /// assert!(TerminationStatus::Terminated(TerminationReason::KeyboardInterrupt).terminated());
    /// assert!(TerminationStatus::Terminated(TerminationReason::Aborted).terminated());
    /// ```
    pub fn terminated(self) -> bool {
        !matches!(self, TerminationStatus::NotTerminated)
    }
}

impl std::fmt::Display for TerminationStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
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
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub enum TerminationReason {
    /// Reached maximum number of iterations
    MaxItersReached,
    /// Reached target cost function value
    TargetCostReached,
    /// Reached target precision
    TargetPrecisionReached,
    /// No change in cost function value
    NoChangeInCost,
    /// Accepted stall iter exceeded (Simulated Annealing)
    AcceptedStallIterExceeded,
    /// Best stall iter exceeded (Simulated Annealing)
    BestStallIterExceeded,
    /// Condition for line search met
    LineSearchConditionMet,
    /// Reached target tolerance
    TargetToleranceReached,
    /// Algorithm manually interrupted with Ctrl+C
    KeyboardInterrupt,
    /// Algorithm aborted
    Aborted,
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
    ///     TerminationReason::TargetPrecisionReached.text(),
    ///     "Target precision reached"
    /// );
    /// assert_eq!(
    ///     TerminationReason::NoChangeInCost.text(),
    ///     "No change in cost function value"
    /// );
    /// assert_eq!(
    ///     TerminationReason::AcceptedStallIterExceeded.text(),
    ///     "Accepted stall iterations exceeded"
    /// );
    /// assert_eq!(
    ///     TerminationReason::BestStallIterExceeded.text(),
    ///     "Best stall iterations exceeded"
    /// );
    /// assert_eq!(
    ///     TerminationReason::LineSearchConditionMet.text(),
    ///     "Line search condition met"
    /// );
    /// assert_eq!(
    ///     TerminationReason::TargetToleranceReached.text(),
    ///     "Target tolerance reached"
    /// );
    /// assert_eq!(
    ///     TerminationReason::KeyboardInterrupt.text(),
    ///     "Keyboard interrupt"
    /// );
    /// assert_eq!(
    ///     TerminationReason::Aborted.text(),
    ///     "Optimization aborted"
    /// );
    /// ```
    pub fn text(&self) -> &str {
        match *self {
            TerminationReason::MaxItersReached => "Maximum number of iterations reached",
            TerminationReason::TargetCostReached => "Target cost value reached",
            TerminationReason::TargetPrecisionReached => "Target precision reached",
            TerminationReason::NoChangeInCost => "No change in cost function value",
            TerminationReason::AcceptedStallIterExceeded => "Accepted stall iterations exceeded",
            TerminationReason::BestStallIterExceeded => "Best stall iterations exceeded",
            TerminationReason::LineSearchConditionMet => "Line search condition met",
            TerminationReason::TargetToleranceReached => "Target tolerance reached",
            TerminationReason::KeyboardInterrupt => "Keyboard interrupt",
            TerminationReason::Aborted => "Optimization aborted",
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
        TerminationReason::Aborted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    send_sync_test!(termination_reason, TerminationReason);
}
