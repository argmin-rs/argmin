// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Multi-Start (TODO)

use ArgminSolver;
use ArgminResult;

/// Starts several optimization problems at once
pub struct MultiStart<'a, A>
where
    A: ArgminSolver<'a>,
{
    /// Vector of solvers
    solvers: Vec<A>,
    /// Corresponding vector of problem definitions
    problems: Vec<<A as ArgminSolver<'a>>::ProblemDefinition>,
    /// Correspondin vector of initial parameters
    init_params: Vec<<A as ArgminSolver<'a>>::StartingPoints>,
}

impl<'a, A> MultiStart<'a, A>
where
    A: ArgminSolver<'a>,
{
    /// Create a new empty instance of `MultiStart`
    pub fn new() -> Self {
        MultiStart {
            solvers: vec![],
            problems: vec![],
            init_params: vec![],
        }
    }

    /// Add another `solver` with corresponding `prob_def` (problem definition) and `init_param`
    /// (initial parameter).
    pub fn push(
        &mut self,
        solver: A,
        prob_def: A::ProblemDefinition,
        init_param: A::StartingPoints,
    ) -> &mut Self {
        self.solvers.push(solver);
        self.problems.push(prob_def);
        self.init_params.push(init_param);
        self
    }

    /// Run the solvers sequentially
    pub fn run(&mut self) -> Vec<ArgminResult<A::Parameter, A::CostValue>> {
        self.solvers
            .iter_mut()
            .zip(self.problems.clone().into_iter())
            .zip(self.init_params.iter())
            .map(|((s, p), i)| s.run(p, i).unwrap())
            .collect()
    }
}

impl<'a, A> Default for MultiStart<'a, A>
where
    A: ArgminSolver<'a>,
{
    fn default() -> Self {
        Self::new()
    }
}
