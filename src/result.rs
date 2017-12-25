/// `ArgminResult`
///
/// TODO
// use num::{Num, NumCast};
use parameter::ArgminParameter;

// pub struct ArgminResult<P, C: Num + NumCast> {
#[derive(Debug)]
pub struct ArgminResult<T: ArgminParameter<T> + Clone, U: PartialOrd> {
    pub param: T,
    pub cost: U,
    pub iters: u64,
}

impl<T: ArgminParameter<T> + Clone, U: PartialOrd> ArgminResult<T, U> {
    // pub fn new(param: P, cost: C, iters: u64) -> Self {
    pub fn new(param: T, cost: U, iters: u64) -> Self {
        ArgminResult { param, cost, iters }
    }
}
