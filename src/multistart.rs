// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Multi-Start (TODO)

use futures::Future;
use futures::prelude::*;
use futures_cpupool::{CpuFuture, CpuPool};
use ArgminSolver;
use ArgminResult;
use parameter::ArgminParameter;
use ArgminCostValue;

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
    /// CPU pool
    pool: CpuPool,
}

impl<'a, A, B> Future for ArgminResult<A, B>
where
    A: ArgminParameter,
    B: ArgminCostValue,
{
    type Item = ArgminResult<A, B>;
    // type Error = Box<std::error::Error>;
    type Error = ();

    fn poll(&mut self) -> Result<Async<Self::Item>, Self::Error> {
        Ok(Async::Ready(self.clone()))
    }
}

impl<'a, A> MultiStart<'a, A>
where
    A: ArgminSolver<'a> + Send,
    <A as ArgminSolver<'a>>::Parameter: 'static,
    <A as ArgminSolver<'a>>::CostValue: 'static,
{
    /// Create a new empty instance of `MultiStart`
    pub fn new() -> Self {
        MultiStart {
            solvers: vec![],
            problems: vec![],
            init_params: vec![],
            pool: CpuPool::new(4),
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

    /// Run solvers in parallel
    pub fn run_parallel(&mut self) -> Vec<ArgminResult<A::Parameter, A::CostValue>> {
        let pool = self.pool.clone();
        let runs: Vec<CpuFuture<_, _>> = self.solvers
            .iter_mut()
            .zip(self.problems.clone().into_iter())
            .zip(self.init_params.iter())
            .map(|((s, p), i)| pool.spawn(s.run(p, i).unwrap()))
            .collect();
        runs.into_iter().map(|a| a.wait().unwrap()).collect()
    }
}

// impl<'a, A> Default for MultiStart<'a, A>
// where
//     A: ArgminSolver<'a> + Send,
//     <A as ArgminSolver<'a>>::Parameter: 'static,
//     <A as ArgminSolver<'a>>::CostValue: 'static,
// {
//     fn default() -> Self {
//         Self::new()
//     }
// }
