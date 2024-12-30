use faer::{mat, unzipped, zipped, Entity, Mat, SimpleEntity};

use crate::ArgminMinMax;

impl<'a, E: SimpleEntity + PartialOrd> ArgminMinMax for Mat<E> {
    #[inline]
    fn max(a: &Self, b: &Self) -> Self {
        faer::zipped!(a, b).map(|faer::unzipped!(a, b)| {
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
        faer::zipped!(a, b).map(|faer::unzipped!(a, b)| {
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

#[cfg(test)]
mod test {
    #[test]
    fn add_tests() {
        todo!()
    }
}
