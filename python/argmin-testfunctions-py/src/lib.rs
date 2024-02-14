use argmin_testfunctions::*;
use paste::paste;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::stringify;

#[macro_export]
macro_rules! func {
    ($function:ident) => {
        func!(name = $function, function = $function,);
    };
    ($function:ident, num = $num:expr) => {
        func!(name = $function, function = $function, num = $num,);
    };
    (name = $name:ident, function = $function:ident, $($a:ident : $t:ty = $v:expr),* ) => {
        paste! {
            #[pyfunction(name = $name "", signature = (param, $($a = $v),*))]
            fn [<$name _py>](param: Vec<f64>, $($a: $t),*) -> f64 {
                $function(&param[..], $($a),*)
            }

            #[pyfunction(name = $name "_derivative", signature = (param, $($a = $v),*))]
            fn [<$name _derivative_py>](param: Vec<f64>, $($a: $t),*) -> Vec<f64> {
                [<$function _derivative>](&param[..], $($a),*)
            }

            #[pyfunction(name = $name "_hessian", signature = (param, $($a = $v),*))]
            fn [<$name _hessian_py>](param: Vec<f64>, $($a: $t),*) -> Vec<Vec<f64>> {
                [<$function _hessian>](&param[..], $($a),*)
            }
        }
    };
    (name = $name:ident, function = $function:ident, num = $num:expr, $($a:ident : $t:ty = $v:expr),* ) => {
        paste! {
            #[pyfunction(name = $name "", signature = (param, $($a = $v),*))]
            fn [<$name _py>](param: Vec<f64>, $($a: $t),*) -> PyResult<f64> {
                let n = param.len();
                if let Ok(param) = param.try_into() {
                    Ok($function(&param, $($a),*))
                } else {
                    Err(PyValueError::new_err(format!("incompatible number of parameters: expected {}, found {}", stringify!($num), n)))
                }
            }

            #[pyfunction(name = $name "_derivative", signature = (param, $($a = $v),*))]
            fn [<$name _derivative_py>](param: Vec<f64>, $($a: $t),*) -> PyResult<Vec<f64>> {
                let n = param.len();
                if let Ok(param) = param.try_into() {
                    Ok([<$function _derivative>](&param, $($a),*).to_vec())
                } else {
                    Err(PyValueError::new_err(format!("incompatible number of parameters: expected {}, found {}", stringify!($num), n)))
                }
            }

            #[pyfunction(name = $name "_hessian", signature = (param, $($a = $v),*))]
            fn [<$name _hessian_py>](param: Vec<f64>, $($a: $t),*) -> PyResult<Vec<Vec<f64>>> {
                let n = param.len();
                if let Ok(param) = param.try_into() {
                    Ok([<$function _hessian>](&param, $($a),*).iter().map(|r| r.to_vec()).collect::<Vec<_>>())
                } else {
                    Err(PyValueError::new_err(format!("incompatible number of parameters: expected {}, found {}", stringify!($num), n)))
                }
            }
        }
    };
}

#[macro_export]
macro_rules! add_function {
    ($m:ident, $function:ident) => {
        paste! {
            $m.add_function(wrap_pyfunction!([<$function _py>], $m)?)?;
            $m.add_function(wrap_pyfunction!([<$function _derivative_py>], $m)?)?;
            $m.add_function(wrap_pyfunction!([<$function _hessian_py>], $m)?)?;
        }
    };
}

func!(name = ackley, function = ackley_abc, a: f64 = 20.0, b: f64 = 0.2, c: f64 = 6.2831853071795864769252867665590057683943387987502116419498891846);
func!(beale, num = 2);
func!(booth, num = 2);
func!(bukin_n6, num = 2);
func!(cross_in_tray, num = 2);
func!(easom, num = 2);
func!(eggholder, num = 2);
func!(goldsteinprice, num = 2);
func!(himmelblau, num = 2);
func!(holder_table, num = 2);
func!(levy);
func!(levy_n13, num = 2);
func!(matyas, num = 2);
func!(mccorminck, num = 2);
func!(picheny, num = 2);
func!(name = rastrigin, function = rastrigin_a, a: f64 = 10.0);
func!(name = rosenbrock, function = rosenbrock_ab, a: f64 = 1.0, b: f64 = 100.0);
func!(schaffer_n2, num = 2);
func!(schaffer_n4, num = 2);
func!(sphere);
func!(styblinski_tang);
func!(threehumpcamel, num = 2);

#[pymodule]
fn argmin_testfunctions_py(_py: Python, m: &PyModule) -> PyResult<()> {
    add_function!(m, ackley);
    add_function!(m, beale);
    add_function!(m, booth);
    add_function!(m, bukin_n6);
    add_function!(m, cross_in_tray);
    add_function!(m, easom);
    add_function!(m, eggholder);
    add_function!(m, goldsteinprice);
    add_function!(m, himmelblau);
    add_function!(m, holder_table);
    add_function!(m, levy);
    add_function!(m, levy_n13);
    add_function!(m, matyas);
    add_function!(m, mccorminck);
    add_function!(m, picheny);
    add_function!(m, rastrigin);
    add_function!(m, rosenbrock);
    add_function!(m, schaffer_n2);
    add_function!(m, schaffer_n4);
    add_function!(m, sphere);
    add_function!(m, styblinski_tang);
    add_function!(m, threehumpcamel);
    Ok(())
}
