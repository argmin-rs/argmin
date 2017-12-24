/// `ArgminResult`
///
/// TODO
// use num::{Num, NumCast};

// pub struct ArgminResult<P, C: Num + NumCast> {
pub struct ArgminResult {
    // param: P,
    // cost: C,
    param: Vec<f64>,
    cost: f64,
    iters: u64,
}

// impl<P, C: Num + NumCast> ArgminResult<P, C> {
impl ArgminResult {
    // pub fn new(param: P, cost: C, iters: u64) -> Self {
    pub fn new(param: Vec<f64>, cost: f64, iters: u64) -> Self {
        ArgminResult { param, cost, iters }
    }
}
