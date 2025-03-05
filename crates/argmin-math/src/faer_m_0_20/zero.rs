use crate::ArgminZeroLike;
use faer::{Entity, Mat, Shape};

impl<E, R, C> ArgminZeroLike for Mat<E, R, C>
where
    E: Entity,
    R: Shape,
    C: Shape,
{
    fn zero_like(&self) -> Self {
        Self::zeros(self.nrows(), self.ncols())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::faer_tests::test_helper::*;
    use approx::assert_relative_eq;
    use faer::mat;
    use faer::mat::{AsMatRef, MatRef};
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_zero_like_ $t>]() {
                    let t: Mat<$t> = column_vector_from_vec::<$t>(vec![]);
                    let a = t.zero_like();
                    assert_eq!(t, a);
                }
            }

            item! {
                #[test]
                fn [<test_zero_like_2_ $t>]() {
                    let a = column_vector_from_vec(vec![42 as $t, 42 as $t, 42 as $t, 42 as $t]).zero_like();
                    for i in 0..4 {
                        assert_relative_eq!(0 as f64, a[(i,0)] as f64, epsilon = f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_2d_zero_like_ $t>]() {
                    let t: Mat<$t> = column_vector_from_vec(vec![0 as $t ,0 as $t]);
                    let a = t.zero_like();
                    assert_eq!(t, a);
                }
            }

            item! {
                #[test]
                fn [<test_2d_zero_like_2_ $t>]() {
                    let a = mat![
                      [42 as $t, 42 as $t],
                      [42 as $t, 42 as $t]
                    ].zero_like();

                    for i in 0..2 {
                        for j in 0..2 {
                            assert_relative_eq!(0 as f64, a[(i, j)] as f64, epsilon = f64::EPSILON);
                        }
                    }
                }
            }
        };
    }

    //@note(geo-ant): partial equality does not work for integral types in faer
    // which is why we implement the tests for floating point numbers only
    make_test!(f32);
    make_test!(f64);
}
