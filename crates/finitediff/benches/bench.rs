// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Benches

#![feature(test)]

extern crate finitediff;
extern crate test;

const MASSIVENESS: usize = 256;

fn cost_vec_f64(x: &Vec<f64>) -> f64 {
    x.iter().fold(0.0, |a, acc| a + acc)
}

#[cfg(feature = "ndarray")]
fn cost_ndarray_f64(x: &ndarray::Array1<f64>) -> f64 {
    x.iter().fold(0.0, |a, acc| a + acc)
}

fn cost_multi_vec_f64(x: &Vec<f64>) -> Vec<f64> {
    x.clone()
}

#[cfg(feature = "ndarray")]
fn cost_multi_ndarray_f64(x: &ndarray::Array1<f64>) -> ndarray::Array1<f64> {
    x.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use finitediff::*;
    use test::{black_box, Bencher};

    #[bench]
    fn cost_func_vec_f64_1(b: &mut Bencher) {
        let x = vec![1.0f64; MASSIVENESS];
        b.iter(|| {
            black_box(cost_vec_f64(&x));
        });
    }

    #[bench]
    fn cost_func_vec_f64_np1(b: &mut Bencher) {
        let x = vec![1.0f64; MASSIVENESS];
        b.iter(|| {
            for _ in 0..(MASSIVENESS + 1) {
                black_box(cost_vec_f64(&x));
            }
        });
    }

    #[bench]
    fn cost_func_vec_f64_2n(b: &mut Bencher) {
        let x = vec![1.0f64; MASSIVENESS];
        b.iter(|| {
            for _ in 0..(2 * MASSIVENESS) {
                black_box(cost_vec_f64(&x));
            }
        });
    }

    #[bench]
    fn cost_func_multi_vec_f64_1(b: &mut Bencher) {
        let x = vec![1.0f64; MASSIVENESS];
        b.iter(|| {
            black_box(cost_multi_vec_f64(&x));
        });
    }

    #[bench]
    fn cost_func_multi_vec_f64_np1(b: &mut Bencher) {
        let x = vec![1.0f64; MASSIVENESS];
        b.iter(|| {
            for _ in 0..(MASSIVENESS + 1) {
                black_box(cost_multi_vec_f64(&x));
            }
        });
    }

    #[bench]
    fn cost_func_multi_vec_f64_2n(b: &mut Bencher) {
        let x = vec![1.0f64; MASSIVENESS];
        b.iter(|| {
            for _ in 0..(2 * MASSIVENESS) {
                black_box(cost_multi_vec_f64(&x));
            }
        });
    }

    #[bench]
    fn forward_diff_vec_f64(b: &mut Bencher) {
        let x = vec![1.0f64; MASSIVENESS];
        b.iter(|| {
            black_box(x.forward_diff(&cost_vec_f64));
        });
    }

    #[cfg(feature = "ndarray")]
    #[bench]
    fn forward_diff_ndarray_f64(b: &mut Bencher) {
        let x = ndarray::Array1::from(vec![1.0f64; MASSIVENESS]);
        b.iter(|| {
            black_box(x.forward_diff(&cost_ndarray_f64));
        });
    }

    #[bench]
    fn central_diff_vec_f64(b: &mut Bencher) {
        let x = vec![1.0f64; MASSIVENESS];
        b.iter(|| {
            black_box(x.central_diff(&cost_vec_f64));
        });
    }

    #[cfg(feature = "ndarray")]
    #[bench]
    fn central_diff_ndarray_f64(b: &mut Bencher) {
        let x = ndarray::Array1::from(vec![1.0f64; MASSIVENESS]);
        b.iter(|| {
            black_box(x.central_diff(&cost_ndarray_f64));
        });
    }

    #[bench]
    fn forward_jacobian_vec_f64(b: &mut Bencher) {
        let x = vec![1.0f64; MASSIVENESS];
        b.iter(|| {
            black_box(x.forward_jacobian(&cost_multi_vec_f64));
        });
    }

    #[cfg(feature = "ndarray")]
    #[bench]
    fn forward_jacobian_ndarray_f64(b: &mut Bencher) {
        let x = ndarray::Array1::from(vec![1.0f64; MASSIVENESS]);
        b.iter(|| {
            black_box(x.forward_jacobian(&cost_multi_ndarray_f64));
        });
    }

    #[bench]
    fn central_jacobian_vec_f64(b: &mut Bencher) {
        let x = vec![1.0f64; MASSIVENESS];
        b.iter(|| {
            black_box(x.central_jacobian(&cost_multi_vec_f64));
        });
    }

    #[cfg(feature = "ndarray")]
    #[bench]
    fn central_jacobian_ndarray_f64(b: &mut Bencher) {
        let x = ndarray::Array1::from(vec![1.0f64; MASSIVENESS]);
        b.iter(|| {
            black_box(x.central_jacobian(&cost_multi_ndarray_f64));
        });
    }

    #[bench]
    fn forward_jacobian_vec_prod_vec_f64(b: &mut Bencher) {
        let x = vec![1.0f64; MASSIVENESS];
        let p = vec![1.0f64; MASSIVENESS];
        b.iter(|| {
            black_box(x.forward_jacobian_vec_prod(&cost_multi_vec_f64, &p));
        });
    }

    #[cfg(feature = "ndarray")]
    #[bench]
    fn forward_jacobian_vec_prod_ndarray_f64(b: &mut Bencher) {
        let x = ndarray::Array1::from(vec![1.0f64; MASSIVENESS]);
        let p = ndarray::Array1::from(vec![1.0f64; MASSIVENESS]);
        b.iter(|| {
            black_box(x.forward_jacobian_vec_prod(&cost_multi_ndarray_f64, &p));
        });
    }

    #[bench]
    fn central_jacobian_vec_prod_vec_f64(b: &mut Bencher) {
        let x = vec![1.0f64; MASSIVENESS];
        let p = vec![1.0f64; MASSIVENESS];
        b.iter(|| {
            black_box(x.forward_jacobian_vec_prod(&cost_multi_vec_f64, &p));
        });
    }

    #[cfg(feature = "ndarray")]
    #[bench]
    fn central_jacobian_vec_prod_ndarray_f64(b: &mut Bencher) {
        let x = ndarray::Array1::from(vec![1.0f64; MASSIVENESS]);
        let p = ndarray::Array1::from(vec![1.0f64; MASSIVENESS]);
        b.iter(|| {
            black_box(x.forward_jacobian_vec_prod(&cost_multi_ndarray_f64, &p));
        });
    }

    #[bench]
    fn forward_jacobian_pert_vec_f64(b: &mut Bencher) {
        let pert = vec![
            PerturbationVector::new()
                .add(0, vec![0, 1])
                .add(3, vec![2, 3, 4]),
            PerturbationVector::new()
                .add(1, vec![0, 1, 2])
                .add(4, vec![3, 4, 5]),
            PerturbationVector::new()
                .add(2, vec![1, 2, 3])
                .add(5, vec![4, 5]),
        ];

        let x = vec![1.0f64; MASSIVENESS];

        b.iter(|| {
            let p2 = pert.clone();
            black_box(x.forward_jacobian_pert(&cost_multi_vec_f64, &p2));
        });
    }

    #[cfg(feature = "ndarray")]
    #[bench]
    fn forward_jacobian_pert_ndarray_f64(b: &mut Bencher) {
        let pert = vec![
            PerturbationVector::new()
                .add(0, vec![0, 1])
                .add(3, vec![2, 3, 4]),
            PerturbationVector::new()
                .add(1, vec![0, 1, 2])
                .add(4, vec![3, 4, 5]),
            PerturbationVector::new()
                .add(2, vec![1, 2, 3])
                .add(5, vec![4, 5]),
        ];

        let x = ndarray::Array1::from(vec![1.0f64; MASSIVENESS]);

        b.iter(|| {
            let p2 = pert.clone();
            black_box(x.forward_jacobian_pert(&cost_multi_ndarray_f64, &p2));
        });
    }

    #[bench]
    fn central_jacobian_pert_vec_f64(b: &mut Bencher) {
        let pert = vec![
            PerturbationVector::new()
                .add(0, vec![0, 1])
                .add(3, vec![2, 3, 4]),
            PerturbationVector::new()
                .add(1, vec![0, 1, 2])
                .add(4, vec![3, 4, 5]),
            PerturbationVector::new()
                .add(2, vec![1, 2, 3])
                .add(5, vec![4, 5]),
        ];

        let x = vec![1.0f64; MASSIVENESS];

        b.iter(|| {
            let p2 = pert.clone();
            black_box(x.central_jacobian_pert(&cost_multi_vec_f64, &p2));
        });
    }

    #[cfg(feature = "ndarray")]
    #[bench]
    fn central_jacobian_pert_ndarray_f64(b: &mut Bencher) {
        let pert = vec![
            PerturbationVector::new()
                .add(0, vec![0, 1])
                .add(3, vec![2, 3, 4]),
            PerturbationVector::new()
                .add(1, vec![0, 1, 2])
                .add(4, vec![3, 4, 5]),
            PerturbationVector::new()
                .add(2, vec![1, 2, 3])
                .add(5, vec![4, 5]),
        ];

        let x = ndarray::Array1::from(vec![1.0f64; MASSIVENESS]);

        b.iter(|| {
            let p2 = pert.clone();
            black_box(x.central_jacobian_pert(&cost_multi_ndarray_f64, &p2));
        });
    }

    #[bench]
    fn forward_hessian_vec_f64(b: &mut Bencher) {
        let x = vec![1.0f64; MASSIVENESS];
        b.iter(|| {
            black_box(x.forward_hessian(&cost_multi_vec_f64));
        });
    }

    #[cfg(feature = "ndarray")]
    #[bench]
    fn forward_hessian_ndarray_f64(b: &mut Bencher) {
        let x = ndarray::Array1::from(vec![1.0f64; MASSIVENESS]);
        b.iter(|| {
            black_box(x.forward_hessian(&cost_multi_ndarray_f64));
        });
    }

    #[bench]
    fn central_hessian_vec_f64(b: &mut Bencher) {
        let x = vec![1.0f64; MASSIVENESS];
        b.iter(|| {
            black_box(x.central_hessian(&cost_multi_vec_f64));
        });
    }

    #[cfg(feature = "ndarray")]
    #[bench]
    fn central_hessian_ndarray_f64(b: &mut Bencher) {
        let x = ndarray::Array1::from(vec![1.0f64; MASSIVENESS]);
        b.iter(|| {
            black_box(x.central_hessian(&cost_multi_ndarray_f64));
        });
    }

    #[bench]
    fn forward_hessian_vec_prod_vec_f64(b: &mut Bencher) {
        let x = vec![1.0f64; MASSIVENESS];
        let p = vec![1.0f64; MASSIVENESS];
        b.iter(|| {
            black_box(x.forward_hessian_vec_prod(&cost_multi_vec_f64, &p));
        });
    }

    #[cfg(feature = "ndarray")]
    #[bench]
    fn forward_hessian_vec_prod_ndarray_f64(b: &mut Bencher) {
        let x = ndarray::Array1::from(vec![1.0f64; MASSIVENESS]);
        let p = ndarray::Array1::from(vec![1.0f64; MASSIVENESS]);
        b.iter(|| {
            black_box(x.forward_hessian_vec_prod(&cost_multi_ndarray_f64, &p));
        });
    }

    #[bench]
    fn central_hessian_vec_prod_vec_f64(b: &mut Bencher) {
        let x = vec![1.0f64; MASSIVENESS];
        let p = vec![1.0f64; MASSIVENESS];
        b.iter(|| {
            black_box(x.central_hessian_vec_prod(&cost_multi_vec_f64, &p));
        });
    }

    #[cfg(feature = "ndarray")]
    #[bench]
    fn central_hessian_vec_prod_ndarray_f64(b: &mut Bencher) {
        let x = ndarray::Array1::from(vec![1.0f64; MASSIVENESS]);
        let p = ndarray::Array1::from(vec![1.0f64; MASSIVENESS]);
        b.iter(|| {
            black_box(x.central_hessian_vec_prod(&cost_multi_ndarray_f64, &p));
        });
    }

    #[bench]
    fn forward_hessian_nograd_vec_f64(b: &mut Bencher) {
        let x = vec![1.0f64; MASSIVENESS];
        b.iter(|| {
            black_box(x.forward_hessian_nograd(&cost_vec_f64));
        });
    }

    #[cfg(feature = "ndarray")]
    #[bench]
    fn forward_hessian_nograd_ndarray_f64(b: &mut Bencher) {
        let x = ndarray::Array1::from(vec![1.0f64; MASSIVENESS]);
        b.iter(|| {
            black_box(x.forward_hessian_nograd(&cost_ndarray_f64));
        });
    }

    #[bench]
    fn forward_hessian_nograd_sparse_vec_f64(b: &mut Bencher) {
        let x = vec![1.0f64; MASSIVENESS];
        b.iter(|| {
            let indices = vec![[1, 2], [23, 23], [128, 8]];
            black_box(x.forward_hessian_nograd_sparse(&cost_vec_f64, indices));
        });
    }

    #[cfg(feature = "ndarray")]
    #[bench]
    fn forward_hessian_nograd_sparse_ndarray_f64(b: &mut Bencher) {
        let x = ndarray::Array1::from(vec![1.0f64; MASSIVENESS]);
        b.iter(|| {
            let indices = vec![[1, 2], [23, 23], [128, 8]];
            black_box(x.forward_hessian_nograd_sparse(&cost_ndarray_f64, indices));
        });
    }
}
