// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Termination
//!
//! Defines reasons for termination.
//!
//! TODO:
//!   * Maybe it is better to define a trait (with `terminated` and `text` methods), because it
//!     would allow implementers of solvers to define their own `TerminationReason`s. However, this
//!     would require a lot of work.

#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// Indicates why the optimization algorithm stopped
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub enum TerminationReason {
    /// In case it has not terminated yet
    NotTerminated,
    /// Maximum number of iterations reached
    MaxItersReached,
    /// Target cost function value reached
    TargetCostReached,
    /// Target precision reached
    TargetPrecisionReached,
    /// Cost function value did not change
    NoChangeInCost,
    /// Acceped stall iter exceeded
    AcceptedStallIterExceeded,
    /// Best stall iter exceeded
    BestStallIterExceeded,
    /// Condition for Line search met
    LineSearchConditionMet,
    /// Target tolerance reached
    TargetToleranceReached,
    /// No improvement possible
    NoImprovementPossible,
    /// Aborted
    Aborted,
}

impl TerminationReason {
    /// Returns `true` if a solver terminated and `false` otherwise
    pub fn terminated(self) -> bool {
        !matches!(self, TerminationReason::NotTerminated)
    }

    /// Returns a texual representation of what happened
    pub fn text(&self) -> &str {
        match *self {
            TerminationReason::NotTerminated => "Not terminated",
            TerminationReason::MaxItersReached => "Maximum number of iterations reached",
            TerminationReason::TargetCostReached => "Target cost value reached",
            TerminationReason::TargetPrecisionReached => "Target precision reached",
            TerminationReason::NoChangeInCost => "No change in cost function value",
            TerminationReason::AcceptedStallIterExceeded => "Accepted stall iterations exceeded",
            TerminationReason::BestStallIterExceeded => "Best stall iterations exceeded",
            TerminationReason::LineSearchConditionMet => "Line search condition met",
            TerminationReason::TargetToleranceReached => "Target tolerance reached",
            TerminationReason::NoImprovementPossible => "No improvement possible",
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
        TerminationReason::NotTerminated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    send_sync_test!(termination_reason, TerminationReason);
}
