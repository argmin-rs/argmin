use errors::*;
/// Rosenbrock test function
///
/// Parameters are usually: `a = 1` and `b = 100`
/// TODO: make this multidimensional
pub fn rosenbrock(param: &[f64], a: f64, b: f64) -> Result<f64> {
    Ok((a - param[0]).powf(2.0) + b * (param[1] - param[0].powf(2.0)).powf(2.0))
}

/// Derivative of 2D Rosenbrock function
pub fn rosenbrock_derivative(param: &[f64], a: f64, b: f64) -> Result<Vec<f64>> {
    let x = param[0];
    let y = param[1];
    let mut out = vec![];
    out.push(-2_f64 * a + 4_f64 * b * x.powf(3_f64) - 4_f64 * b * x * y + 2_f64 * x);
    out.push(2_f64 * b * (y - x.powf(2_f64)));
    Ok(out)
}
