use finitediff_rust::ndarr;
use numpy::ndarray::{Array1, ArrayD, ArrayView1, ArrayViewD, ArrayViewMutD, Zip};
use numpy::{
    Complex64, IntoPyArray, PyArray1, PyArrayDyn, PyReadonlyArray1, PyReadonlyArrayDyn,
    PyReadwriteArray1, PyReadwriteArrayDyn,
};
use pyo3::{
    exceptions::PyTypeError,
    prelude::*,
    types::{PyCFunction, PyDict, PyFunction, PyTuple},
};

/// Forward diff
#[pyfunction]
fn forward_diff<'py>(py: Python<'py>, f: Py<PyFunction>) -> PyResult<&'py PyCFunction> {
    PyCFunction::new_closure(
        py,
        None,
        None,
        move |args: &PyTuple, kwargs: Option<&PyDict>| -> Py<PyArray1<f64>> {
            Python::with_gil(|py| {
                let out = (ndarr::forward_diff(|x: &Array1<f64>| -> f64 {
                    let x = PyArray1::from_array(py, x);
                    f.call(py, (x,), None).unwrap().extract(py).unwrap()
                }))(
                    &args
                        .get_item(0)
                        .unwrap()
                        .downcast::<PyArray1<f64>>()
                        .unwrap()
                        .to_owned_array(),
                );
                out.into_pyarray(py).into()
            })
        },
    )
}

/// A Python module implemented in Rust.
#[pymodule]
fn finitediff(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(forward_diff, m)?)?;
    Ok(())
}
// /// Forward diff
// #[pyfunction]
// fn forward_diff<'py>(py: Python<'py>, f: Py<PyFunction>) -> PyResult<&'py PyCFunction> {
//     // fn forward_diff<'py>(py: Python<'py>) -> PyResult<&'py PyCFunction> {
//     // if f.is_callable() {
//     let inner_func = move |x: &Array1<f64>| -> f64 {
//         Python::with_gil(|py| {
//             let x = PyArray1::from_array(py, x);
//             // let f = f.borrow(py);
//             // f.call((x,), None).unwrap().extract().unwrap()
//             f.call(py, (x,), None).unwrap().extract(py).unwrap()
//         })
//     };
//     PyCFunction::new_closure(
//         py,
//         None,
//         None,
//         // |args: &PyTuple, _kwargs: Option<&PyDict>| -> &'py PyArray1<f64> {
//         // |args: &PyTuple, _kwargs: Option<&PyDict>| -> Py<PyArray1<f64>> {
//         // |args: &PyTuple, _kwargs: Option<&PyDict>| {
//         // Python::with_gil(|py| {
//         move |args: &PyTuple, kwargs: Option<&PyDict>| {
//             // (ndarr::forward_diff(|x: &Array1<f64>| {
//             Python::with_gil(|py| {
//                 (ndarr::forward_diff(
//                     // 0.0
//                     // inner_func(x), // f.call((x.into(),), None).unwrap().extract().unwrap()
//                     inner_func, // f.call((x.into(),), None).unwrap().extract().unwrap()
//                                // Python::with_gil(|py| f.call((x.into(),), kwargs).unwrap().extract().unwrap())
//                                // }
//                 ))(
//                     &args
//                         .get_item(0)
//                         .unwrap()
//                         .downcast::<PyArray1<f64>>()
//                         .unwrap()
//                         .to_owned_array(), // .to_owned(),
//                 )
//                 .into_pyarray(py)
//                 // .to_owned()
//             })
//         },
//     )
//     // } else {
//     //     Err(PyErr::new::<PyTypeError, _>(
//     //         "Provided Python object not callable",
//     //     ))
//     // }
// }
// // &PyArray1::from_owned_array(
// // py,
// // )
