// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use std::ops::{Add, IndexMut};

use anyhow::Error;
use num::{Float, FromPrimitive};

/// Panics when idx > x.len()
#[inline(always)]
pub fn mod_and_calc<F, C, T>(
    x: &mut C,
    f: &dyn Fn(&C) -> Result<T, Error>,
    idx: usize,
    y: F,
) -> Result<T, Error>
where
    F: Add<Output = F> + Copy,
    C: IndexMut<usize, Output = F>,
{
    let xtmp = x[idx];
    x[idx] = xtmp + y;
    let fx1 = (f)(x)?;
    x[idx] = xtmp;
    Ok(fx1)
}

/// Panics when idx > N
#[inline(always)]
pub fn mod_and_calc_const<const N: usize, F, T>(
    x: &mut [F; N],
    f: &dyn Fn(&[F; N]) -> Result<T, Error>,
    idx: usize,
    y: F,
) -> Result<T, Error>
where
    F: Add<Output = F> + Copy,
{
    assert!(idx < N);
    let xtmp = x[idx];
    x[idx] = xtmp + y;
    let fx1 = (f)(x)?;
    x[idx] = xtmp;
    Ok(fx1)
}

#[inline(always)]
pub fn restore_symmetry_vec<F>(mut mat: Vec<Vec<F>>) -> Vec<Vec<F>>
where
    F: Float + FromPrimitive,
{
    for i in 0..mat.len() {
        for j in (i + 1)..mat[i].len() {
            let t = (mat[i][j] + mat[j][i]) / F::from_f64(2.0).unwrap();
            mat[i][j] = t;
            mat[j][i] = t;
        }
    }
    mat
}

#[inline(always)]
pub fn restore_symmetry_const<const N: usize, F>(mut mat: [[F; N]; N]) -> [[F; N]; N]
where
    F: Float + FromPrimitive,
{
    for i in 0..mat.len() {
        for j in (i + 1)..mat[i].len() {
            let t = (mat[i][j] + mat[j][i]) / F::from_f64(2.0).unwrap();
            mat[i][j] = t;
            mat[j][i] = t;
        }
    }
    mat
}

/// Restore symmetry for an array of type `ndarray::Array2<f64>`
///
/// Unfortunately, this is *really* slow!
#[cfg(feature = "ndarray")]
#[inline(always)]
pub fn restore_symmetry_ndarray<F>(mut mat: ndarray::Array2<F>) -> ndarray::Array2<F>
where
    F: Float + FromPrimitive,
{
    let (nx, ny) = mat.dim();
    for i in 0..nx {
        for j in (i + 1)..ny {
            let t = (mat[(i, j)] + mat[(j, i)]) / F::from_f64(2.0).unwrap();
            mat[(i, j)] = t;
            mat[(j, i)] = t;
        }
    }
    mat
}

pub struct KV<F> {
    k: Vec<usize>,
    v: Vec<F>,
}

impl<F: Copy> KV<F> {
    pub fn new(capacity: usize) -> Self {
        KV {
            k: Vec::with_capacity(capacity),
            v: Vec::with_capacity(capacity),
        }
    }

    pub fn set(&mut self, k: usize, v: F) -> &mut Self {
        self.k.push(k);
        self.v.push(v);
        self
    }

    pub fn get(&self, k: usize) -> Option<F> {
        for (i, kk) in self.k.iter().enumerate() {
            if *kk == k {
                return Some(self.v[i]);
            }
            if *kk > k {
                return None;
            }
        }
        None
    }
}
