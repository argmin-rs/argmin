use errors::*;
/// Rosenbrock test function
///
/// Parameters are usually: `a = 1` and `b = 100`
/// TODO: make this multidimensional
pub fn rosenbrock(param: &Vec<f64>, a: f64, b: f64) -> Result<f64> {
    Ok((a - param[0]).powf(2.0) + b * (param[1] - param[0].powf(2.0)).powf(2.0))
}
