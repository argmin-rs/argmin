use crate::ArgminRandom;
use faer::Mat;
use rand::distributions::uniform::SampleUniform;

impl<E: PartialOrd + SampleUniform + Copy> ArgminRandom for Mat<E> {
    fn rand_from_range<R: rand::Rng>(min: &Self, max: &Self, rng: &mut R) -> Self {
        assert!(
            min.nrows() != 0 || min.ncols() != 0,
            "internal error: empty matrix unexpected"
        );
        assert_eq!(
            min.shape(),
            max.shape(),
            "internal error: matrices of same shape expected"
        );

        faer::zip!(min, max).map(|faer::unzip!(min, max)| {
            let a = *min;
            let b = *max;
            //@note(geo-ant) code was copied from the nalgebra implementation
            // to get the exact same behaviour
            // Do not require a < b:
            // We do want to know if a and b are *exactly* the same.
            #[allow(clippy::float_cmp)]
            if a == b {
                a
            } else if a < b {
                rng.gen_range(a..b)
            } else {
                rng.gen_range(b..a)
            }
        })
    }
}
