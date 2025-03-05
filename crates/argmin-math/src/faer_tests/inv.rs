use crate::faer_m::InverseError;
use crate::faer_tests::test_helper::*;
use crate::ArgminInv;
use approx::assert_relative_eq;
use faer::mat::AsMatRef;
use paste::item;

macro_rules! make_test {
    ($t:ty) => {
        item! {
            #[test]
            fn [<test_inv_ $t>]() {
                let a = matrix2_new(
                    2 as $t, 5 as $t,
                    1 as $t, 3 as $t,
                );
                let target = matrix2_new(
                    3 as $t, -5 as $t,
                    -1 as $t, 2 as $t,
                );
                let res = <_ as ArgminInv<_>>::inv(&a).unwrap();
                let res1 = <_ as ArgminInv<_>>::inv(&a.as_mat_ref()).unwrap();
                assert_eq!(res,res1);
                assert_eq!(res.nrows(),2);
                assert_eq!(res.ncols(),2);
                for i in 0..2 {
                    for j in 0..2 {
                        //@note(geo-ant) the 20 epsilon are a bit arbitrary,
                        // but it's to avoid spurious errors due to numerical effects
                        // while keeping good accuracy.
                        assert_relative_eq!(res[(i, j)], target[(i, j)], epsilon = 20.*$t::EPSILON);
                    }
                }
            }
        }

        item! {
            #[test]
            fn [<test_inv_error $t>]() {
                let a = matrix2_new(
                    2 as $t, 5 as $t,
                    4 as $t, 10 as $t,
                );
                let err = <_ as ArgminInv<_>>::inv(&a).unwrap_err().downcast::<InverseError>().unwrap();
                assert_eq!(err, InverseError {});
            }
        }
    };
}

make_test!(f32);
make_test!(f64);
