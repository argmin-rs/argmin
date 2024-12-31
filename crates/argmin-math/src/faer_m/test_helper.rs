use faer::{mat::from_column_major_slice_generic, zipped, Conjugate, Mat, SimpleEntity};

/// create a column vector (in Nx1 matrix form) from a Vec instance
pub fn column_vector_from_vec<E: SimpleEntity>(vec: Vec<E>) -> Mat<E> {
    column_vector_from_slice(vec.as_slice())
}

/// create an owning column vector from a slice
pub fn column_vector_from_slice<E: SimpleEntity>(slice: &[E]) -> Mat<E> {
    Mat::<E>::from_fn(slice.len(), 1, |ir, _ic| slice[ir])
}

/// helper method to translate an nalgebra call Vector3::new(a,b,c) to the
/// equivalent faer matrix constructor
pub fn vector3_new<E: SimpleEntity>(a: E, b: E, c: E) -> Mat<E> {
    column_vector_from_slice(&[a, b, c])
}
