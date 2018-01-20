use ndarray::{Array1, Array2};
/// Rosenbrock test function
///
/// Parameters are usually: `a = 1` and `b = 100`
/// TODO: make this multidimensional
pub fn rosenbrock(param: &[f64], a: f64, b: f64) -> f64 {
    (a - param[0]).powf(2.0) + b * (param[1] - param[0].powf(2.0)).powf(2.0)
}

/// Derivative of 2D Rosenbrock function
pub fn rosenbrock_derivative(param: &[f64], a: f64, b: f64) -> Vec<f64> {
    let x = param[0];
    let y = param[1];
    let mut out = vec![];
    out.push(-2.0 * a + 4.0 * b * x.powf(3.0) - 4.0 * b * x * y + 2.0 * x);
    out.push(2.0 * b * (y - x.powf(2.0)));
    out
}

/// Hessian of 2D Rosenbrock function
pub fn rosenbrock_hessian(param: &[f64], _a: f64, b: f64) -> Vec<f64> {
    let x = param[0];
    let y = param[1];
    let mut out = vec![];
    // d/dxdx
    out.push(12.0 * b * x.powf(2.0) - 4.0 * b * y + 2.0);
    // d/dxdy
    out.push(-4.0 * b * x);
    // d/dydx
    out.push(-4.0 * b * x);
    // d/dydy
    out.push(2.0 * b);
    out
}

/// Rosenbrock test function, taking ndarray
///
/// Parameters are usually: `a = 1` and `b = 100`
/// TODO: make this multidimensional
pub fn rosenbrock_nd(param: &Array1<f64>, a: f64, b: f64) -> f64 {
    (a - param[0]).powf(2.0) + b * (param[1] - param[0].powf(2.0)).powf(2.0)
}

/// Derivative of 2D Rosenbrock function, returning an ndarray
pub fn rosenbrock_derivative_nd(param: &Array1<f64>, a: f64, b: f64) -> Array1<f64> {
    let x = param[0];
    let y = param[1];
    let mut out = Array1::zeros(2);
    out[[0]] = -2.0 * a + 4.0 * b * x.powf(3.0) - 4.0 * b * x * y + 2.0 * x;
    out[[1]] = 2.0 * b * (y - x.powf(2.0));
    out
}

/// Hessian of 2D Rosenbrock function, returning an ndarray
pub fn rosenbrock_hessian_nd(param: &Array1<f64>, _a: f64, b: f64) -> Array2<f64> {
    let x = param[0];
    let y = param[1];
    let mut out = Array2::zeros((2, 2));
    // d/dxdx
    out[[0, 0]] = 12.0 * b * x.powf(2.0) - 4.0 * b * y + 2.0;
    // d/dxdy
    out[[0, 1]] = -4.0 * b * x;
    // d/dydx
    out[[1, 0]] = -4.0 * b * x;
    // d/dydy
    out[[1, 1]] = 2.0 * b;
    out
}

/// Sphere test function
pub fn sphere(param: &[f64]) -> f64 {
    param.iter().map(|x| x.powf(2.0)).sum()
}

/// Derivative of sphere test function
pub fn sphere_derivative(param: &[f64]) -> Vec<f64> {
    param.iter().map(|x| 2.0 * x).collect()
}
