use crate::ArgminEye;
use faer::{ComplexField, Entity, Mat, Shape};

impl<E: Entity + ComplexField, R: Shape, C: Shape> ArgminEye for Mat<E, R, C>
where
    R: TryFrom<usize>,
    C: TryFrom<usize>,
{
    fn eye(n: usize) -> Self {
        let (nr, nc) = match (R::try_from(n), C::try_from(n)) {
            (Ok(nr), Ok(nc)) => (nr, nc),
            _ => panic!("invalid matrix size for index type"),
        };

        Mat::identity(nr, nc)
    }

    fn eye_like(&self) -> Self {
        let nr = self.nrows();
        let nc = self.ncols();
        //@note(geo-ant) in the nalgebra implementation we also enforce
        // that the matrix is square, which we don't enforce here
        Mat::identity(nr, nc)
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helper::*;
    use super::*;
    use approx::assert_relative_eq;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_eye_ $t>]() {
                    let e: Mat<$t> = <_ as ArgminEye>::eye(3);
                    let res = matrix3_new(
                        1 as $t, 0 as $t, 0 as $t,
                        0 as $t, 1 as $t, 0 as $t,
                        0 as $t, 0 as $t, 1 as $t
                    );
                    for i in 0..3 {
                        for j in 0..3 {
                            assert_relative_eq!(res[(i, j)] as f64, e[(i, j)] as f64, epsilon = f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_eye_like_ $t>]() {
                    let a = matrix3_new(
                        0 as $t, 2 as $t, 6 as $t,
                        3 as $t, 2 as $t, 7 as $t,
                        9 as $t, 8 as $t, 1 as $t
                    );
                    let e: Mat<$t> = a.eye_like();
                    let res = matrix3_new(
                        1 as $t, 0 as $t, 0 as $t,
                        0 as $t, 1 as $t, 0 as $t,
                        0 as $t, 0 as $t, 1 as $t
                    );
                    for i in 0..3 {
                        for j in 0..3 {
                            assert_relative_eq!(res[(i, j)] as f64, e[(i, j)] as f64, epsilon = f64::EPSILON);
                        }
                    }
                }
            }
        };
    }

    make_test!(f32);
    make_test!(f64);
}
