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
    out.push(-2.0 * a + 4.0 * b * x.powf(3.0) - 4.0 * b * x * y + 2.0 * x);
    out.push(2.0 * b * (y - x.powf(2.0)));
    Ok(out)
}

/// Hessian of 2D Rosenbrock function
pub fn rosenbrock_hessian(_param: &[f64], _a: f64, _b: f64) -> Result<Vec<f64>> {
    unimplemented!()
}

/// Sphere test function
pub fn sphere(param: &[f64]) -> Result<f64> {
    Ok(param.iter().map(|x| x.powf(2.0)).sum())
}

/// Derivative of sphere test function
pub fn sphere_derivative(param: &[f64]) -> Result<Vec<f64>> {
    Ok(param.iter().map(|x| 2.0 * x).collect())
}
