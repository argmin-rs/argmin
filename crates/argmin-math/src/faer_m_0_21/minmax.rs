use crate::ArgminMinMax;
use faer::{mat, unzip, zip, Mat, MatRef};

impl<E: PartialOrd + Copy> ArgminMinMax for Mat<E> {
    #[inline]
    fn max(a: &Self, b: &Self) -> Self {
        faer::zip!(a, b).map(|faer::unzip!(a, b)| {
            let aa = *a;
            let bb = *b;
            //@note(geo-ant) directly cribbed from the nalgebra implementation
            // will have the same problems with NaN values.
            if aa > bb {
                aa
            } else {
                bb
            }
        })
    }

    #[inline]
    fn min(a: &Mat<E>, b: &Mat<E>) -> Mat<E> {
        faer::zip!(a, b).map(|faer::unzip!(a, b)| {
            let aa = *a;
            let bb = *b;
            //@note(geo-ant) directly cribbed from the nalgebra implementation
            // will have the same problems with NaN values.
            if aa < bb {
                aa
            } else {
                bb
            }
        })
    }
}
