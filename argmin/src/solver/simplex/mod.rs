// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Simplex Algorithm
//!
//! # References
//!
//! \[0\] [Wikipedia](https://en.wikipedia.org/wiki/Simplex_algorithm)

// use crate::core::{
//     ArgminError, ArgminFloat, ArgminIterData, ArgminKV, ArgminOp, DeserializeOwnedAlias, Error,
//     IterState, OpWrapper, SerializeAlias, Solver, TerminationReason,
// };
use crate::core::*;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// TODO
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct Simplex<F> {
    tableau: Option<Tableau<F>>,
}

impl<F> Simplex<F>
where
    F: ArgminFloat,
{
    /// Constructor
    pub fn new() -> Self {
        Simplex { tableau: None }
    }

    pub(super) fn cost(&self) -> F {
        self.tableau.as_ref().unwrap().cost()
    }

    // pub(super) fn variables(&self) -> Vec<F> {
    //     self.tableau.as_ref().unwrap().variables()
    // }

    pub(super) fn step(&mut self) {
        self.tableau.as_mut().unwrap().step()
    }
}

impl<F> Default for Simplex<F>
where
    F: ArgminFloat,
{
    fn default() -> Self {
        Simplex::new()
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
struct Tableau<F> {
    c: Vec<F>,
    constr: Vec<F>,
    b: Vec<F>,
    cost: F,
    num_constraints: usize,
    num_variables: usize,
    constraints_counter: usize,
}

impl<F> Tableau<F>
where
    F: ArgminFloat,
{
    #[allow(non_snake_case)]
    pub fn new(c: &[F], b: &[F], A: &[Vec<F>]) -> Result<Tableau<F>, Error> {
        let num_constraints = b.len();
        let num_variables = c.len();
        let mut constr =
            vec![F::from_f64(0.0).unwrap(); (num_constraints + num_variables) * num_constraints];
        let mut b_n = Vec::with_capacity(num_constraints);
        for (constraints_counter, (row, b_i)) in A.iter().zip(b.iter()).enumerate() {
            for (i, v) in row.iter().cloned().enumerate() {
                constr[constraints_counter * (num_constraints + num_variables) + i] = v;
            }
            constr[constraints_counter * (num_constraints + num_variables)
                + num_variables
                + constraints_counter] = F::from_f64(1.0).unwrap();
            b_n.push(*b_i);
        }

        let mut c_n = vec![F::from_f64(0.0).unwrap(); num_variables + num_constraints];
        c_n[..num_variables].clone_from_slice(c);

        Ok(Tableau {
            c: c_n,
            constr,
            b: b_n,
            cost: F::from_f64(0.0).unwrap(),
            num_constraints,
            num_variables,
            constraints_counter: 0,
        })
    }

    fn step(&mut self) {
        // Obtain next variable to enter basis
        let pivot_column = self.get_pivot_column_idx();
        println!("pivot_column = {:?}", pivot_column);
        let pivot_row = self.get_pivot_row_idx(pivot_column);
        println!("pivot_row = {:?}", pivot_row);
        let pivot_element = self.get_element(pivot_row, pivot_column);
        println!("pivot_element = {:?}", pivot_element);
        self.normalize_row(pivot_row, pivot_element);
        println!("Tableau = {:?}", self);
        println!("b = {:?}", self.b);
        println!("pivoting");
        self.transform_non_pivot_rows(pivot_row, pivot_column);
        println!("Tableau = {:?}", self);
        println!("b = {:?}", self.b);
    }

    fn get_column(&self, idx: usize) -> Vec<F> {
        let mut out = Vec::with_capacity(self.num_constraints);
        for i in 0..self.num_constraints {
            out.push(self.constr[idx + i * (self.num_constraints + self.num_variables)]);
        }
        out
    }

    fn get_element(&self, row: usize, column: usize) -> F {
        self.constr[row * (self.num_constraints + self.num_variables) + column]
    }

    fn normalize_row(&mut self, row: usize, factor: F) {
        for i in 0..(self.num_constraints + self.num_variables) {
            self.constr[row * (self.num_constraints + self.num_variables) + i] =
                self.constr[row * (self.num_constraints + self.num_variables) + i] / factor;
        }
        self.b[row] = self.b[row] / factor;
    }

    fn transform_non_pivot_rows(&mut self, pivot_row: usize, pivot_column: usize) {
        let row: Vec<F> = ((pivot_row * (self.num_constraints + self.num_variables))
            ..((pivot_row + 1) * (self.num_constraints + self.num_variables)))
            .into_iter()
            .map(|i| self.constr[i])
            .collect();
        let b_f = self.b[pivot_row];
        for i in 0..self.num_constraints {
            if i == pivot_row {
                continue;
            }
            let factor = self.get_element(i, pivot_column);
            for (j, &elem) in row
                .iter()
                .enumerate()
                .take(self.num_variables + self.num_constraints)
            {
                self.constr[(i * (self.num_constraints + self.num_variables)) + j] = self.constr
                    [(i * (self.num_constraints + self.num_variables)) + j]
                    - factor * elem;
            }
            self.b[i] = self.b[i] - factor * b_f;
        }
        let factor = self.c[pivot_column];
        for (j, &elem) in row
            .iter()
            .enumerate()
            .take(self.num_variables + self.num_constraints)
        {
            self.c[j] = self.c[j] - factor * elem;
        }
        self.cost = self.cost - factor * b_f;
    }

    fn get_pivot_row_idx(&self, column: usize) -> usize {
        let (i_leaves, _): (usize, F) = self
            .get_column(column)
            .iter()
            .zip(self.b.iter())
            .enumerate()
            .filter(|(_, (&pc, _))| pc > F::from_f64(0.0).unwrap())
            .map(|(i, (&pc, &bc)): (usize, (&F, &F))| (i, bc / pc))
            .fold((0, F::infinity()), |(iacc, acc), (i, x)| {
                if x < acc {
                    (i, x)
                } else {
                    (iacc, acc)
                }
            });
        i_leaves
    }

    fn get_pivot_column_idx(&self) -> usize {
        let (i_enter, _): (usize, F) =
            self.c
                .iter()
                .enumerate()
                .fold((0, F::infinity()), |(iacc, acc), (i, &x)| {
                    if x < acc {
                        (i, x)
                    } else {
                        (iacc, acc)
                    }
                });
        i_enter
    }

    fn cost(&self) -> F {
        -self.cost
    }
}

impl<O, F> Solver<LinearProgramState<O>> for Simplex<F>
where
    O: LinearProgram<Float = F, Param = Vec<F>>,
    F: ArgminFloat,
{
    const NAME: &'static str = "Simplex";

    fn init(
        &mut self,
        op: &mut OpWrapper<O>,
        _state: &mut LinearProgramState<O>,
    ) -> Result<Option<ArgminIterData<LinearProgramState<O>>>, Error> {
        self.tableau = Some(Tableau::new(op.c()?, op.b()?, op.A()?)?);
        Ok(Some(
            ArgminIterData::new()
                // .param(self.variables())
                .cost(self.cost()),
        ))
    }

    /// Perform one iteration of Simplex algorithm
    fn next_iter(
        &mut self,
        _op: &mut OpWrapper<O>,
        _state: &mut LinearProgramState<O>,
    ) -> Result<ArgminIterData<LinearProgramState<O>>, Error> {
        self.step();

        Ok(ArgminIterData::new()
            // .param(self.variables())
            .cost(self.cost()))
    }

    fn terminate(&mut self, _state: &LinearProgramState<O>) -> TerminationReason {
        // if self
        //     .tableau_ref()
        //     .slice(s![0, 1..])
        //     .iter()
        //     .cloned()
        //     .map(|x| x.is_sign_positive())
        //     .fold(true, |acc, x| acc & x)
        // {
        //     return TerminationReason::NoImprovementPossible;
        // }
        TerminationReason::NotTerminated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_trait_impl;

    test_trait_impl!(sa, Simplex<f64>);
}
