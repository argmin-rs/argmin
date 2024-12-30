use faer::{mat, unzipped, zipped, Entity, Mat};

use crate::ArgminMinMax;

impl<'a, E: Entity + PartialOrd> ArgminMinMax for Mat<E> {
    #[inline]
    fn max(a: &Self, b: &Self) -> Self {
        let mut res = Mat::<E>::zeros(a.nrows(), a.ncols());
        faer::zipped_rw!(res.as_mut(), a.as_ref(), b.as_ref()).for_each(
            |faer::unzipped!(mut res, a, b)| {
                let aa = a.read();
                let bb = b.read();
                //@note(geo-ant) directly cribbed from the nalgebra implementation
                // will have the same problems with NaN values.
                res.write(if aa > bb { aa } else { bb });
            },
        );
        res
    }

    #[inline]
    fn min(a: &Mat<E>, b: &Mat<E>) -> Mat<E> {
        let mut res = Mat::<E>::zeros(a.nrows(), a.ncols());
        faer::zipped_rw!(res.as_mut(), a.as_ref(), b.as_ref()).for_each(
            |faer::unzipped!(mut res, a, b)| {
                let aa = a.read();
                let bb = b.read();
                //@note(geo-ant) directly cribbed from the nalgebra implementation
                // will have the same problems with NaN values.
                res.write(if aa < bb { aa } else { bb });
            },
        );
        res
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn add_tests() {
        todo!()
    }
}
