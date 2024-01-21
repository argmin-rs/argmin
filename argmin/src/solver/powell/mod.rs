use crate::core::{
    ArgminFloat, CostFunction, DeserializeOwnedAlias, Executor, IterState, LineSearch,
    OptimizationResult, SerializeAlias, Solver, State,
};
use argmin_math::{ArgminAdd, ArgminDot, ArgminSub, ArgminZeroLike};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::mem;

#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct PowellLineSearch<P, L> {
    search_vectors: Vec<P>,
    linesearch: L,
}

impl<P, L> PowellLineSearch<P, L> {
    pub fn new(initial_search_vectors: Vec<P>, linesearch: L) -> Self {
        PowellLineSearch {
            search_vectors: initial_search_vectors,
            linesearch,
        }
    }
}

impl<O, P, F, L> Solver<O, IterState<P, (), (), (), F>> for PowellLineSearch<P, L>
where
    O: CostFunction<Param = P, Output = F>,
    P: Clone
        + SerializeAlias
        + DeserializeOwnedAlias
        + ArgminAdd<P, P>
        + ArgminZeroLike
        + ArgminSub<P, P>
        + ArgminDot<P, F>,
    F: ArgminFloat,
    L: Clone + LineSearch<P, F> + Solver<O, IterState<P, (), (), (), F>>,
{
    const NAME: &'static str = "Powell-LS";

    fn next_iter(
        &mut self,
        problem: &mut crate::core::Problem<O>,
        mut state: IterState<P, (), (), (), F>,
    ) -> Result<(IterState<P, (), (), (), F>, Option<crate::core::KV>), anyhow::Error> {
        let param = state
            .take_param()
            .ok_or_else(argmin_error_closure!(NotInitialized, "not initialized"))?; // TODO add Error message

        // new displacement vector created from displacement vectors of line searches
        let new_displacement = param.zero_like();
        let mut best_direction: (usize, F) = (0, float!(0.0));

        // init line search
        let (ls_state, _) = self.linesearch.init(problem, state.clone())?;

        // Loop over all search vectors and perform line optimization along each search direction
        for (i, search_vector) in self.search_vectors.iter().enumerate() {
            self.linesearch.search_direction(search_vector.clone());

            let line_cost = ls_state.get_cost();

            // Run solver
            let OptimizationResult {
                problem: _sub_problem,
                state: sub_state,
                ..
            } = Executor::new(problem.take_problem().unwrap(), self.linesearch.clone())
                .configure(|state| state.param(param.clone()).cost(line_cost))
                .ctrlc(false)
                .run()?;

            // update new displacement vector
            let displacement = &sub_state.get_best_param().unwrap().sub(&param);
            let displacement_magnitude = displacement.dot(&displacement).sqrt();
            new_displacement.add(displacement);

            //store index of lowest cost displacement vector
            if best_direction.1 < displacement_magnitude {
                best_direction.0 = i;
                best_direction.1 = displacement_magnitude;
            }
        }

        // replace best performing search direction with new search direction
        let _ = mem::replace(
            &mut self.search_vectors[best_direction.0],
            new_displacement.clone(),
        );

        // set new parameters
        let param = param.add(&new_displacement);
        let cost = problem.cost(&param);

        Ok((state.param(param).cost(cost?), None))
    }
}
