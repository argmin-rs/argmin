use anyhow::Error;
use finitediff_rust::ndarr;
use numpy::{ndarray::Array1, IntoPyArray, PyArray1, PyArray2};
use pyo3::{
    exceptions::PyTypeError,
    prelude::*,
    types::{PyCFunction, PyDict, PyTuple},
};

fn process_args(args: &PyTuple) -> PyResult<Array1<f64>> {
    Ok(args
        .get_item(0)
        .map_err(|_| PyErr::new::<PyTypeError, _>("Insufficient number of arguments"))?
        .downcast::<PyArray1<f64>>()?
        .to_owned_array())
}

macro_rules! not_callable {
    ($py:ident, $f:ident) => {
        Err(PyErr::new::<PyTypeError, _>(format!(
            "object {} not callable",
            $f.as_ref($py).get_type()
        )))
    };
}

/// Forward diff
#[pyfunction]
fn forward_diff<'py>(py: Python<'py>, f: Py<PyAny>) -> PyResult<&'py PyCFunction> {
    if f.as_ref(py).is_callable() {
        PyCFunction::new_closure(
            py,
            None,
            None,
            move |args: &PyTuple, _kwargs: Option<&PyDict>| -> PyResult<Py<PyArray1<f64>>> {
                Python::with_gil(|py| {
                    let out = (ndarr::forward_diff(&|x: &Array1<f64>| -> Result<f64, Error> {
                        let x = PyArray1::from_array(py, x);
                        Ok(f.call(py, (x,), None)?.extract(py)?)
                    }))(&process_args(args)?)?;
                    Ok(out.into_pyarray(py).into())
                })
            },
        )
    } else {
        not_callable!(py, f)
    }
}

/// Central diff
#[pyfunction]
fn central_diff<'py>(py: Python<'py>, f: Py<PyAny>) -> PyResult<&'py PyCFunction> {
    if f.as_ref(py).is_callable() {
        PyCFunction::new_closure(
            py,
            None,
            None,
            move |args: &PyTuple, _kwargs: Option<&PyDict>| -> PyResult<Py<PyArray1<f64>>> {
                Python::with_gil(|py| {
                    let out = (ndarr::central_diff(&|x: &Array1<f64>| -> Result<f64, Error> {
                        let x = PyArray1::from_array(py, x);
                        Ok(f.call(py, (x,), None)?.extract(py)?)
                    }))(&process_args(args)?)?;
                    Ok(out.into_pyarray(py).into())
                })
            },
        )
    } else {
        not_callable!(py, f)
    }
}

/// Forward Jacobian
#[pyfunction]
fn forward_jacobian<'py>(py: Python<'py>, f: Py<PyAny>) -> PyResult<&'py PyCFunction> {
    if f.as_ref(py).is_callable() {
        PyCFunction::new_closure(
            py,
            None,
            None,
            move |args: &PyTuple, _kwargs: Option<&PyDict>| -> PyResult<Py<PyArray2<f64>>> {
                Python::with_gil(|py| {
                    let out = (ndarr::forward_jacobian(
                        &|x: &Array1<f64>| -> Result<Array1<f64>, Error> {
                            let x = PyArray1::from_array(py, x);
                            Ok(f.call(py, (x,), None)?
                                .extract::<&PyArray1<f64>>(py)?
                                .to_owned_array())
                        },
                    ))(&process_args(args)?)?;
                    Ok(out.into_pyarray(py).into())
                })
            },
        )
    } else {
        not_callable!(py, f)
    }
}

/// Central Jacobian
#[pyfunction]
fn central_jacobian<'py>(py: Python<'py>, f: Py<PyAny>) -> PyResult<&'py PyCFunction> {
    if f.as_ref(py).is_callable() {
        PyCFunction::new_closure(
            py,
            None,
            None,
            move |args: &PyTuple, _kwargs: Option<&PyDict>| -> PyResult<Py<PyArray2<f64>>> {
                Python::with_gil(|py| {
                    let out = (ndarr::central_jacobian(
                        &|x: &Array1<f64>| -> Result<Array1<f64>, Error> {
                            let x = PyArray1::from_array(py, x);
                            Ok(f.call(py, (x,), None)?
                                .extract::<&PyArray1<f64>>(py)?
                                .to_owned_array())
                        },
                    ))(&process_args(args)?)?;
                    Ok(out.into_pyarray(py).into())
                })
            },
        )
    } else {
        not_callable!(py, f)
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn finitediff(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(forward_diff, m)?)?;
    m.add_function(wrap_pyfunction!(central_diff, m)?)?;
    m.add_function(wrap_pyfunction!(forward_jacobian, m)?)?;
    m.add_function(wrap_pyfunction!(central_jacobian, m)?)?;
    Ok(())
}
