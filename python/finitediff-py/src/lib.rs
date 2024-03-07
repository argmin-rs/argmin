use finitediff_rust::ndarr;
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1};
use pyo3::{
    exceptions::PyTypeError,
    prelude::*,
    types::{PyCFunction, PyDict, PyTuple},
};

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
                    let out = (ndarr::forward_diff(|x: &Array1<f64>| -> f64 {
                        let x = PyArray1::from_array(py, x);
                        f.call(py, (x,), None).unwrap().extract(py).unwrap()
                    }))(
                        &args
                            .get_item(0)?
                            .downcast::<PyArray1<f64>>()?
                            .to_owned_array(),
                    );
                    Ok(out.into_pyarray(py).into())
                })
            },
        )
    } else {
        Err(PyErr::new::<PyTypeError, _>(format!(
            "object {} not callable",
            f.as_ref(py).get_type()
        )))
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn finitediff(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(forward_diff, m)?)?;
    Ok(())
}
