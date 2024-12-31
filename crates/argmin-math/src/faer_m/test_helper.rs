use faer::{mat::from_column_major_slice_generic, zipped, Conjugate, Mat, SimpleEntity};

/// create a column vector (in Nx1 matrix form) from a Vec instance
pub fn column_vector_from_vec<E: SimpleEntity>(vec: Vec<E>) -> Mat<E> {
    Mat::<E>::from_fn(vec.len(), 1, |ir, _ic| vec[ir])
}
