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

/// helper method to translate an nalgebra call Vector2::new(a,b) to the
/// equivalent faer matrix constructor
pub fn vector2_new<E: SimpleEntity>(a: E, b: E) -> Mat<E> {
    column_vector_from_slice(&[a, b])
}

/// helper method to translate an nalgebra call Matrix2x3::new(a,b,c, d,e,f) to the
/// equivalent faer matrix constructor
pub fn matrix2x3_new<E: SimpleEntity>(a: E, b: E, c: E, d: E, e: E, f: E) -> Mat<E> {
    faer::mat![[a, b, c], [d, e, f]]
}

/// helper method to translate an nalgebra call Matrix2::new(a,b, c,d) to the
/// equivalent faer matrix constructor
pub fn matrix2_new<E: SimpleEntity>(a: E, b: E, c: E, d: E) -> Mat<E> {
    faer::mat![[a, b], [c, d]]
}
