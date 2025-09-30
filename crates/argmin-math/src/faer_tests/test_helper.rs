use faer::Mat;

cfg_if::cfg_if! {
    if #[cfg(feature = "faer_v0_20")] {
        use faer::ComplexField;
    } else if #[cfg(feature = "faer_v0_21")] {
        use faer_traits::ComplexField;
    } else if #[cfg(feature = "faer_v0_23")] {
        use faer_traits::ComplexField;
    }
}

/// create a column vector (in Nx1 matrix form) from a Vec instance
/// equivalent to the nalgebra call DVector::from_vec
pub fn column_vector_from_vec<E: ComplexField + Copy>(vec: Vec<E>) -> Mat<E> {
    column_vector_from_slice(vec.as_slice())
}

/// create an owning column vector from a slice
pub fn column_vector_from_slice<E: ComplexField + Copy>(slice: &[E]) -> Mat<E> {
    Mat::<E>::from_fn(slice.len(), 1, |ir, _ic| slice[ir])
}

/// helper method to translate an nalgebra call Vector3::new(a,b,c) to the
/// equivalent faer matrix constructor
pub fn vector3_new<E: ComplexField + Copy>(a: E, b: E, c: E) -> Mat<E> {
    let v = column_vector_from_slice(&[a, b, c]);
    assert_eq!(v.nrows(), 3);
    assert_eq!(v.ncols(), 1);
    v
}

/// helper method to translate an nalgebra call RowVector3::new(a,b,c) to the
/// equivalent faer matrix constructor
pub fn row_vector3_new<E: ComplexField>(a: E, b: E, c: E) -> Mat<E> {
    let v = faer::mat![[a, b, c]];
    assert_eq!(v.nrows(), 1);
    assert_eq!(v.ncols(), 3);
    v
}

/// helper method to translate an nalgebra call Vector2::new(a,b) to the
/// equivalent faer matrix constructor
pub fn vector2_new<E: ComplexField + Copy>(a: E, b: E) -> Mat<E> {
    let v = column_vector_from_slice(&[a, b]);
    assert_eq!(v.nrows(), 2);
    assert_eq!(v.ncols(), 1);
    v
}

/// helper method to translate an nalgebra call Matrix2x3::new(a,b,c, d,e,f) to the
/// equivalent faer matrix constructor
pub fn matrix2x3_new<E: ComplexField>(a: E, b: E, c: E, d: E, e: E, f: E) -> Mat<E> {
    let m = faer::mat![[a, b, c], [d, e, f]];
    assert_eq!(m.nrows(), 2);
    assert_eq!(m.ncols(), 3);
    m
}

/// helper method to translate an nalgebra call Matrix2::new(a,b, c,d) to the
/// equivalent faer matrix constructor
pub fn matrix2_new<E: ComplexField>(a: E, b: E, c: E, d: E) -> Mat<E> {
    let m = faer::mat![[a, b], [c, d]];
    assert_eq!(m.nrows(), 2);
    assert_eq!(m.ncols(), 2);
    m
}

/// helper method to translate an nalgebra call Matrix3::new(a,b,c, d,e,f, g,h,i) to the
/// equivalent faer matrix constructor
pub fn matrix3_new<E: ComplexField>(
    a: E,
    b: E,
    c: E,
    d: E,
    e: E,
    f: E,
    g: E,
    h: E,
    i: E,
) -> Mat<E> {
    let m = faer::mat![[a, b, c], [d, e, f], [g, h, i]];
    assert_eq!(m.nrows(), 3);
    assert_eq!(m.ncols(), 3);
    m
}

/// helper method to get the matrix element at a certain index. Normally this
/// is not necessary, but due to a quirk in older faer versions for complex
/// matrices, this helps unify the logic for certain tests.
pub fn matrix_element_at<E: ComplexField + Copy>(
    mat: &Mat<E, usize, usize>,
    index: (usize, usize),
) -> E {
    cfg_if::cfg_if! {
        if #[cfg(feature = "faer_v0_20")] {
            mat.read(index.0,index.1)
        } else {
            mat[index]
        }
    }
}
