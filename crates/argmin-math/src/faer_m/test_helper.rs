use faer::{mat::from_column_major_slice_generic, Conjugate, Mat, SimpleEntity};

/// create a column vector (in Nx1 matrix form) from a Vec instance
pub fn column_vector_from_vec<E: SimpleEntity + Conjugate<Canonical = E>>(vec: Vec<E>) -> Mat<E> {
    from_column_major_slice_generic::<E, usize, usize>(vec.as_slice(), vec.len(), 1).to_owned()
}
