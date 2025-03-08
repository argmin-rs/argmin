use crate::faer_tests::test_helper::*;
use crate::ArgminConj;
use approx::assert_relative_eq;
use faer::Mat;
use num_complex::Complex;
use num_complex::ComplexFloat;
use paste::item;

macro_rules! make_test {
    ($t:ty) => {
        item! {
            #[test]
            fn [<test_conj_complex_faer_ $t>]() {
                    let a : Mat<Complex<$t>> = vector3_new(
                    Complex::new(1 as $t, 2 as $t),
                    Complex::new(4 as $t, -3 as $t),
                    Complex::new(8 as $t, 0 as $t)
                );
                let b = vector3_new(
                    Complex::new(1 as $t, -2 as $t),
                    Complex::new(4 as $t, 3 as $t),
                    Complex::new(8 as $t, 0 as $t)
                );
                let res: Mat<_> = <Mat<Complex<$t>> as ArgminConj>::conj(&a);
                assert_eq!(res.nrows(),3);
                assert_eq!(res.ncols(),1);
                for i in 0..3 {
                    assert_relative_eq!(matrix_element_at(&b,(i,0)).re(), matrix_element_at(&res,(i,0)).re(), epsilon = $t::EPSILON);
                    assert_relative_eq!(matrix_element_at(&b,(i,0)).im(), matrix_element_at(&res,(i,0)).im(), epsilon = $t::EPSILON);
                }
            }
        }

        item! {
            #[test]
            fn [<test_conj_faer_ $t>]() {
                let a = vector3_new(1 as $t, 4 as $t, 8 as $t);
                let b = vector3_new(1 as $t, 4 as $t, 8 as $t);
                let res = <_ as ArgminConj>::conj(&a);
                for i in 0..3 {
                    assert_relative_eq!(b[(i,0)], res[(i,0)], epsilon = $t::EPSILON);
                }
            }
        }
    };
}

make_test!(f32);
make_test!(f64);
