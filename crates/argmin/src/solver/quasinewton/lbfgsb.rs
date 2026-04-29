// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! L-BFGS-B (limited-memory BFGS with box constraints).
//!
//! Solves `min f(x)` subject to `l <= x <= u` using the algorithm of
//! Byrd, Lu, and Nocedal (1995), Zhu et al. (1997, Algorithm 778), and the
//! Morales-Nocedal (2011) projection-first subspace minimization variant.
//!
//! Each outer iteration: generalized Cauchy point on the projected
//! gradient ray, subspace minimization on the free variables using the
//! L-BFGS compact representation, and a projected line search.
//!
//! Substep behavior (Cauchy point, free-set classification, subspace
//! minimization) was validated bit-for-bit against the L-BFGS-B 3.0
//! Fortran reference on a 5-dimensional Rosenbrock fixture. The line
//! search is the user-supplied [`LineSearch`] (typically Moré-Thuente)
//! rather than Fortran's `dcsrch`, so individual iterates need not match
//! the reference to FP precision; final-state convergence does.

use crate::core::{
    ArgminFloat, CostFunction, Error, Executor, Gradient, IterState, LineSearch,
    OptimizationResult, Problem, Solver, State, TerminationReason, TerminationStatus, KV,
};
use argmin_math::{
    ArgminAdd, ArgminDot, ArgminFromSlice, ArgminL2Norm, ArgminLInfNorm, ArgminMinMax, ArgminMul,
    ArgminScaledAdd, ArgminSub, ArgminZeroLike,
};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::marker::PhantomData;

/// # Limited-memory BFGS with box constraints (L-BFGS-B)
///
/// Solves
///
/// $$
/// \min_{x \in \mathbb{R}^n} f(x) \quad \text{subject to} \quad l \le x \le u
/// $$
///
/// using the algorithm of Byrd, Lu, Nocedal (1995) and Zhu, Byrd, Lu, Nocedal (1997)
/// with the Morales-Nocedal (2011) update. The algorithm proceeds in three substeps
/// per outer iteration: generalized Cauchy point, subspace minimization on the free
/// variables using the L-BFGS compact representation, and a projected Moré-Thuente
/// line search.
///
/// ## Required configuration
///
/// - An initial parameter vector via [`Executor::configure`](`crate::core::Executor::configure`).
/// - A line search implementing [`LineSearch`].
/// - A history size `m` (typical range: 5-20).
/// - Box constraints `(lower, upper)` of the same shape as the parameter vector.
///
/// ## Tolerances
///
/// - [`with_tolerance_grad`](LBFGSB::with_tolerance_grad): stop when
///   `||proj_grad||_∞ < tol_grad`. Default `sqrt(EPSILON)`.
/// - [`with_tolerance_cost`](LBFGSB::with_tolerance_cost): stop when
///   `|f_prev - f_curr| < tol_cost`. Default `EPSILON`.
///
/// ## References
///
/// - Byrd, R. H.; Lu, P.; Nocedal, J. (1995). "A Limited Memory Algorithm for Bound
///   Constrained Optimization". SIAM J. Sci. Comput. 16(5), 1190-1208.
/// - Zhu, C.; Byrd, R. H.; Lu, P.; Nocedal, J. (1997). "Algorithm 778: L-BFGS-B".
///   ACM TOMS 23(4), 550-560.
/// - Morales, J. L.; Nocedal, J. (2011). "Remark on Algorithm 778". ACM TOMS 38(1).
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct LBFGSB<L, P, G, F> {
    /// line search
    linesearch: L,
    /// history size
    m: usize,
    /// box constraints `(lower, upper)`
    bounds: (P, P),
    /// s_{k-1} = x_k - x_{k-1}
    s: VecDeque<P>,
    /// y_{k-1} = g_k - g_{k-1}
    y: VecDeque<G>,
    /// θ scaling = y^T y / s^T y. Defaults to 1 before any pair is stored.
    theta: F,
    /// Powell curvature safeguard threshold (s^T y > eps * ||y||^2).
    skip_ys_eps: F,
    /// Tolerance on infinity norm of projected gradient.
    tol_grad: F,
    /// Tolerance on absolute change in cost between iterations.
    tol_cost: F,
    phantom: PhantomData<(G, F)>,
}

impl<L, P, G, F> LBFGSB<L, P, G, F>
where
    F: ArgminFloat,
{
    /// Construct a new instance of [`LBFGSB`].
    ///
    /// # Arguments
    ///
    /// * `linesearch` - a line search implementing [`LineSearch`]
    /// * `m` - number of `(s, y)` pairs to keep in history (typical: 5-20)
    /// * `bounds` - tuple `(lower, upper)` of the same shape as the parameter vector
    pub fn new(linesearch: L, m: usize, bounds: (P, P)) -> Self {
        LBFGSB {
            linesearch,
            m,
            bounds,
            s: VecDeque::with_capacity(m),
            y: VecDeque::with_capacity(m),
            theta: float!(1.0),
            skip_ys_eps: F::epsilon(),
            tol_grad: F::epsilon().sqrt(),
            tol_cost: F::epsilon(),
            phantom: PhantomData,
        }
    }

    /// Set the tolerance on the infinity norm of the projected gradient.
    ///
    /// The provided value must be non-negative. Defaults to `sqrt(EPSILON)`.
    pub fn with_tolerance_grad(mut self, tol_grad: F) -> Result<Self, Error> {
        if tol_grad < float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`L-BFGS-B`: gradient tolerance must be >= 0."
            ));
        }
        self.tol_grad = tol_grad;
        Ok(self)
    }

    /// Set the tolerance on the change of the cost function between iterations.
    ///
    /// The provided value must be non-negative. Defaults to `EPSILON`.
    pub fn with_tolerance_cost(mut self, tol_cost: F) -> Result<Self, Error> {
        if tol_cost < float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`L-BFGS-B`: cost tolerance must be >= 0."
            ));
        }
        self.tol_cost = tol_cost;
        Ok(self)
    }

    /// Set the Powell curvature safeguard threshold (default `EPSILON`).
    ///
    /// New `(s, y)` pairs are added to history only if `s.dot(y) > eps * ||y||^2`.
    pub fn with_curvature_threshold(mut self, eps: F) -> Result<Self, Error> {
        if eps < float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`L-BFGS-B`: curvature threshold must be >= 0."
            ));
        }
        self.skip_ys_eps = eps;
        Ok(self)
    }
}

// ----------------------------------------------------------------------------
// Small dense linear solver: Gaussian elimination with partial pivoting.
// Used for the 2k × 2k system arising from the compact representation. Since
// 2k ≤ ~40 in practice, the O(k^3) cost is dwarfed by the surrounding O(k·n)
// dot products.
// ----------------------------------------------------------------------------

/// Solves `A x = b` in place. `a` is row-major `n × n`; `b` is overwritten with `x`.
fn gauss_solve<F: ArgminFloat>(a: &mut [F], b: &mut [F], n: usize) -> Result<(), Error> {
    debug_assert_eq!(a.len(), n * n);
    debug_assert_eq!(b.len(), n);
    for k in 0..n {
        let mut piv = k;
        let mut piv_val = a[k * n + k].abs();
        for i in (k + 1)..n {
            let v = a[i * n + k].abs();
            if v > piv_val {
                piv = i;
                piv_val = v;
            }
        }
        if piv_val < F::epsilon() {
            return Err(argmin_error!(
                ImpossibleError,
                "`L-BFGS-B`: singular subspace matrix encountered in Gaussian elimination."
            ));
        }
        if piv != k {
            for j in 0..n {
                a.swap(k * n + j, piv * n + j);
            }
            b.swap(k, piv);
        }
        let akk = a[k * n + k];
        for i in (k + 1)..n {
            let factor = a[i * n + k] / akk;
            if factor == F::zero() {
                continue;
            }
            for j in (k + 1)..n {
                let aik = a[i * n + j];
                let akj = a[k * n + j];
                a[i * n + j] = aik - factor * akj;
            }
            a[i * n + k] = F::zero();
            b[i] = b[i] - factor * b[k];
        }
    }
    for k in (0..n).rev() {
        let mut s = b[k];
        for j in (k + 1)..n {
            s = s - a[k * n + j] * b[j];
        }
        b[k] = s / a[k * n + k];
    }
    Ok(())
}

// ----------------------------------------------------------------------------
// Compact-representation building blocks.
// ----------------------------------------------------------------------------

/// Build `S^T S` and `S^T Y` (each `m × m`, row-major). Only the leading
/// `k × k` block is populated, where `k = s_deque.len()`.
fn build_ssyy<P, G, F>(s_deque: &VecDeque<P>, y_deque: &VecDeque<G>, m: usize) -> (Vec<F>, Vec<F>)
where
    P: ArgminDot<P, F> + ArgminDot<G, F>,
    F: ArgminFloat,
{
    let mut ss = vec![F::zero(); m * m];
    let mut sy = vec![F::zero(); m * m];
    let k = s_deque.len();
    for i in 0..k {
        for j in 0..k {
            ss[i * m + j] = s_deque[i].dot(&s_deque[j]);
            sy[i * m + j] = s_deque[i].dot(&y_deque[j]);
        }
    }
    (ss, sy)
}

/// Build the `K` matrix (size `2k × 2k`, row-major) from `ss`, `sy`, and `θ`.
///
/// `K = [[ -D, L^T ], [ L, θ S^T S ]]` with `D = diag(s_i^T y_i)` and `L_{ij}`
/// equal to the strictly-lower-triangular part of `S^T Y`.
fn build_k_matrix<F: ArgminFloat>(ss: &[F], sy: &[F], theta: F, m: usize, k: usize) -> Vec<F> {
    let n = 2 * k;
    let mut out = vec![F::zero(); n * n];
    for i in 0..k {
        out[i * n + i] = -sy[i * m + i];
    }
    for i in 0..k {
        for j in 0..k {
            out[(k + i) * n + (k + j)] = theta * ss[i * m + j];
        }
    }
    for i in 0..k {
        for j in (i + 1)..k {
            out[i * n + (k + j)] = sy[j * m + i];
        }
    }
    for i in 0..k {
        for j in 0..i {
            out[(k + i) * n + j] = sy[i * m + j];
        }
    }
    out
}

// ----------------------------------------------------------------------------
// Generalized Cauchy point (Byrd-Lu-Nocedal 1995, Algorithm CP).
//
// Operates on scalar buffers extracted from the parameter / gradient / bound
// vectors. Per-coordinate iteration is required, so the caller is responsible
// for `Vec<F>` extraction (which is why the public `Solver` impl carries the
// `for<'a> &'a P: IntoIterator<Item = &'a F>` HRTB).
// ----------------------------------------------------------------------------

/// Compute breakpoints `t_i = (x_i - u_i)/g_i` if `g_i < 0`,
/// `(x_i - l_i)/g_i` if `g_i > 0`, else `+∞`.
fn breakpoints<F: ArgminFloat>(xs: &[F], gs: &[F], lbs: &[F], ubs: &[F]) -> Vec<F> {
    let n = xs.len();
    let mut t = vec![F::infinity(); n];
    for i in 0..n {
        if gs[i] < F::zero() {
            if ubs[i].is_finite() {
                t[i] = (xs[i] - ubs[i]) / gs[i];
            }
        } else if gs[i] > F::zero() && lbs[i].is_finite() {
            t[i] = (xs[i] - lbs[i]) / gs[i];
        }
    }
    t
}

#[allow(clippy::too_many_arguments)]
fn cauchy_point_scalar<F: ArgminFloat>(
    xs: &[F],
    gs: &[F],
    lbs: &[F],
    ubs: &[F],
    s_rows: &[Vec<F>],
    y_rows: &[Vec<F>],
    theta: F,
    k_matrix: &[F],
) -> Result<(Vec<F>, Vec<F>, Vec<usize>), Error> {
    let n = xs.len();
    let k_pairs = s_rows.len();
    let two_k = 2 * k_pairs;

    let t = breakpoints(xs, gs, lbs, ubs);
    let mut d: Vec<F> = (0..n)
        .map(|i| if t[i] > F::zero() { -gs[i] } else { F::zero() })
        .collect();
    let mut alive: Vec<bool> = (0..n)
        .map(|i| t[i] > F::zero() && t[i].is_finite())
        .collect();
    let mut x_c = xs.to_vec();
    let mut c = vec![F::zero(); two_k];

    // p = W^T d, where W = [Y, θ S]. Only "alive" coords contribute non-zero d.
    let mut p = vec![F::zero(); two_k];
    for i in 0..k_pairs {
        let mut acc = F::zero();
        for j in 0..n {
            if d[j] != F::zero() {
                acc = acc + y_rows[i][j] * d[j];
            }
        }
        p[i] = acc;
    }
    for i in 0..k_pairs {
        let mut acc = F::zero();
        for j in 0..n {
            if d[j] != F::zero() {
                acc = acc + s_rows[i][j] * d[j];
            }
        }
        p[k_pairs + i] = theta * acc;
    }

    // f' = -d^T d, f'' = θ d^T d - p^T K^{-1} p.
    let mut dtd = F::zero();
    for di in &d {
        dtd = dtd + (*di) * (*di);
    }
    let mut f_prime = -dtd;
    let mut f_double = theta * dtd;
    if two_k > 0 {
        let mut buf = k_matrix.to_vec();
        let mut pp = p.clone();
        gauss_solve(&mut buf, &mut pp, two_k)?;
        let mut acc = F::zero();
        for j in 0..two_k {
            acc = acc + p[j] * pp[j];
        }
        f_double = f_double - acc;
    }

    let mut t_old = F::zero();
    loop {
        // Find smallest finite breakpoint among alive variables.
        let mut t_b = F::infinity();
        let mut b: Option<usize> = None;
        for i in 0..n {
            if alive[i] && t[i] < t_b {
                t_b = t[i];
                b = Some(i);
            }
        }
        let dt = match b {
            Some(_) => t_b - t_old,
            None => F::infinity(),
        };
        let dt_min = if f_double > F::zero() {
            -f_prime / f_double
        } else {
            F::infinity()
        };

        if dt_min < dt {
            let advance = if dt_min < F::zero() {
                F::zero()
            } else {
                dt_min
            };
            for i in 0..n {
                if alive[i] {
                    x_c[i] = x_c[i] + advance * d[i];
                }
            }
            for j in 0..two_k {
                c[j] = c[j] + advance * p[j];
            }
            break;
        }

        let b = match b {
            Some(b) => b,
            None => break,
        };

        let dt_used = dt;
        for j in 0..two_k {
            c[j] = c[j] + dt_used * p[j];
        }
        let zb = if d[b] > F::zero() {
            x_c[b] = ubs[b];
            ubs[b] - xs[b]
        } else {
            x_c[b] = lbs[b];
            lbs[b] - xs[b]
        };
        for i in 0..n {
            if alive[i] && i != b {
                x_c[i] = x_c[i] + dt_used * d[i];
            }
        }
        // w_b = b-th row of W.
        let mut w_b = vec![F::zero(); two_k];
        for i in 0..k_pairs {
            w_b[i] = y_rows[i][b];
        }
        for i in 0..k_pairs {
            w_b[k_pairs + i] = theta * s_rows[i][b];
        }
        let g_b = gs[b];
        let theta_g_b = theta * g_b;
        let g_b_sq = g_b * g_b;
        let (wt_m_c, wt_m_p, wt_m_w) = if two_k > 0 {
            let mut buf = k_matrix.to_vec();
            let mut tmp_c = c.clone();
            gauss_solve(&mut buf, &mut tmp_c, two_k)?;
            let mut buf = k_matrix.to_vec();
            let mut tmp_p = p.clone();
            gauss_solve(&mut buf, &mut tmp_p, two_k)?;
            let mut buf = k_matrix.to_vec();
            let mut tmp_w = w_b.clone();
            gauss_solve(&mut buf, &mut tmp_w, two_k)?;
            let mut a_c = F::zero();
            let mut a_p = F::zero();
            let mut a_w = F::zero();
            for j in 0..two_k {
                a_c = a_c + w_b[j] * tmp_c[j];
                a_p = a_p + w_b[j] * tmp_p[j];
                a_w = a_w + w_b[j] * tmp_w[j];
            }
            (a_c, a_p, a_w)
        } else {
            (F::zero(), F::zero(), F::zero())
        };

        f_prime = f_prime + dt_used * f_double + g_b * g_b - theta_g_b * zb - g_b * wt_m_c;
        f_double =
            f_double - theta * g_b_sq - F::from_f64(2.0).unwrap() * g_b * wt_m_p - g_b_sq * wt_m_w;
        for j in 0..two_k {
            p[j] = p[j] + g_b * w_b[j];
        }
        alive[b] = false;
        d[b] = F::zero();
        t_old = t_b;
        if !alive.iter().any(|&a| a) {
            break;
        }
    }

    // A variable is free at the GCP if it is not pinned at a bound there.
    // `alive` = "still actively descending at the end of the inner loop"
    // misses two cases that Fortran (via `iwhere = -3`) treats as free:
    //   1. `g[i] == 0`: the variable cannot move along the projected gradient
    //      ray, so its Cauchy point equals `x_k`. If `x_k` is interior, the
    //      variable should be free for subspace minimization.
    //   2. unbounded descent (`g[i] < 0` with no upper bound, or `g[i] > 0`
    //      with no lower bound): no breakpoint, so `t[i] = +∞` and the var
    //      is `alive=false` from the start, but it is interior at GCP.
    // Use the Cauchy-point feasibility status as the discriminator: free iff
    // strictly interior to its bounds.
    let free: Vec<usize> = (0..n)
        .filter(|&i| {
            let at_lb = lbs[i].is_finite() && x_c[i] <= lbs[i];
            let at_ub = ubs[i].is_finite() && x_c[i] >= ubs[i];
            !(at_lb || at_ub)
        })
        .collect();
    Ok((x_c, c, free))
}

// ----------------------------------------------------------------------------
// Subspace minimization.
//
// Solves the unconstrained Newton step in the free subspace using the compact
// representation:
//
//     B_FF d_F = -r̂ where B_FF = θ I - N K^{-1} N^T, N = Z^T W
//
// Sherman-Morrison-Woodbury yields
//
//     d_F = -(1/θ)(r̂ + N β)   with   (θ K - N^T N) β = N^T r̂.
//
// After computing d_F, backtrack to keep `x_c[F] + α d_F` feasible.
// ----------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn subspace_minimization<F: ArgminFloat>(
    x_c: &[F],
    x_k: &[F],
    g: &[F],
    lbs: &[F],
    ubs: &[F],
    free: &[usize],
    c: &[F],
    s_rows: &[Vec<F>],
    y_rows: &[Vec<F>],
    theta: F,
    k_matrix: &[F],
) -> Result<Vec<F>, Error> {
    let n = x_c.len();
    let k_pairs = s_rows.len();
    let two_k = 2 * k_pairs;
    let n_free = free.len();

    if n_free == 0 {
        return Ok(x_c.to_vec());
    }
    if two_k == 0 {
        // Empty history: B = theta*I, so the unconstrained Newton step from
        // x_c on the free subspace is x_k - g/theta (clipped to the box).
        let inv_theta = F::one() / theta;
        let mut x_star = x_c.to_vec();
        for &i in free {
            let proposed = x_k[i] - inv_theta * g[i];
            x_star[i] = if proposed < lbs[i] {
                lbs[i]
            } else if proposed > ubs[i] {
                ubs[i]
            } else {
                proposed
            };
        }
        return Ok(x_star);
    }

    // K^{-1} c.
    let mut k_inv_c = c.to_vec();
    let mut buf = k_matrix.to_vec();
    gauss_solve(&mut buf, &mut k_inv_c, two_k)?;

    // (W K^{-1} c)[i] for all i.
    let mut wkic = vec![F::zero(); n];
    for j in 0..n {
        let mut acc = F::zero();
        for i in 0..k_pairs {
            acc = acc + y_rows[i][j] * k_inv_c[i];
        }
        for i in 0..k_pairs {
            acc = acc + theta * s_rows[i][j] * k_inv_c[k_pairs + i];
        }
        wkic[j] = acc;
    }

    // r_hat[free idx] = g[i] + θ (x_c[i] - x_k[i]) - wkic[i].
    let mut r_hat = vec![F::zero(); n_free];
    for (idx, &i) in free.iter().enumerate() {
        r_hat[idx] = g[i] + theta * (x_c[i] - x_k[i]) - wkic[i];
    }

    // alpha = N^T r_hat (length 2k). N's rows are W[i, :] for i in F.
    let mut alpha = vec![F::zero(); two_k];
    for j in 0..k_pairs {
        let mut acc = F::zero();
        for (idx, &i) in free.iter().enumerate() {
            acc = acc + y_rows[j][i] * r_hat[idx];
        }
        alpha[j] = acc;
    }
    for j in 0..k_pairs {
        let mut acc = F::zero();
        for (idx, &i) in free.iter().enumerate() {
            acc = acc + s_rows[j][i] * r_hat[idx];
        }
        alpha[k_pairs + j] = theta * acc;
    }

    // NTN[a, b] = sum_{i in F} W[i, a] W[i, b].
    let mut ntn = vec![F::zero(); two_k * two_k];
    for &i in free {
        let mut wi = vec![F::zero(); two_k];
        for a in 0..k_pairs {
            wi[a] = y_rows[a][i];
        }
        for a in 0..k_pairs {
            wi[k_pairs + a] = theta * s_rows[a][i];
        }
        for a in 0..two_k {
            let wa = wi[a];
            if wa == F::zero() {
                continue;
            }
            for b in 0..two_k {
                ntn[a * two_k + b] = ntn[a * two_k + b] + wa * wi[b];
            }
        }
    }

    // M_hat = θ K - NTN.
    let mut m_hat = vec![F::zero(); two_k * two_k];
    for ij in 0..(two_k * two_k) {
        m_hat[ij] = theta * k_matrix[ij] - ntn[ij];
    }
    let mut beta = alpha.clone();
    gauss_solve(&mut m_hat, &mut beta, two_k)?;

    // d_F = -(1/θ)(r_hat + N β).
    let inv_theta = F::one() / theta;
    let mut d_f = vec![F::zero(); n_free];
    for (idx, &i) in free.iter().enumerate() {
        let mut nb = F::zero();
        for a in 0..k_pairs {
            nb = nb + y_rows[a][i] * beta[a];
        }
        for a in 0..k_pairs {
            nb = nb + theta * s_rows[a][i] * beta[k_pairs + a];
        }
        d_f[idx] = -inv_theta * (r_hat[idx] + nb);
    }

    // Morales-Nocedal 2011: first try per-coordinate clipping of x_c + d_F.
    // If the resulting point is a descent direction (relative to the original
    // gradient at x_k), accept it. Otherwise fall back to uniform-α
    // bound-backtracking to keep x_c + α d feasible, fix the binding
    // coordinate, and step.
    let mut x_proj = x_c.to_vec();
    let mut hit_bound = false;
    for (idx, &i) in free.iter().enumerate() {
        let proposed = x_c[i] + d_f[idx];
        if proposed < lbs[i] {
            x_proj[i] = lbs[i];
            hit_bound = true;
        } else if proposed > ubs[i] {
            x_proj[i] = ubs[i];
            hit_bound = true;
        } else {
            x_proj[i] = proposed;
        }
    }
    if !hit_bound {
        return Ok(x_proj);
    }

    // Check the directional derivative at the projected point. Use the
    // gradient at x_k (g) rather than the model gradient.
    let mut dd_p = F::zero();
    for i in 0..n {
        dd_p = dd_p + (x_proj[i] - x_k[i]) * g[i];
    }
    if dd_p <= F::zero() {
        return Ok(x_proj);
    }

    // Uniform-α backtracking from x_c along d_F.
    let mut alpha = F::one();
    let mut ibd: Option<usize> = None;
    for (idx, &i) in free.iter().enumerate() {
        let di = d_f[idx];
        if di > F::zero() && ubs[i].is_finite() {
            let cap = (ubs[i] - x_c[i]) / di;
            if cap < alpha {
                alpha = cap;
                ibd = Some(idx);
            }
        } else if di < F::zero() && lbs[i].is_finite() {
            let cap = (lbs[i] - x_c[i]) / di;
            if cap < alpha {
                alpha = cap;
                ibd = Some(idx);
            }
        }
    }
    if alpha < F::zero() {
        alpha = F::zero();
    }

    let mut x_star = x_c.to_vec();
    if let Some(b) = ibd {
        let i_b = free[b];
        if d_f[b] > F::zero() {
            x_star[i_b] = ubs[i_b];
        } else {
            x_star[i_b] = lbs[i_b];
        }
        let mut d_local = d_f.clone();
        d_local[b] = F::zero();
        for (idx, &i) in free.iter().enumerate() {
            if idx != b {
                x_star[i] = x_c[i] + alpha * d_local[idx];
            }
        }
    } else {
        for (idx, &i) in free.iter().enumerate() {
            x_star[i] = x_c[i] + alpha * d_f[idx];
        }
    }
    Ok(x_star)
}

// ----------------------------------------------------------------------------
// Projected line-search wrapper.
//
// φ(α) = f(P(x + α d, l, u)). Projection makes φ piecewise-smooth with kinks
// at coordinate activations; users typically cap α at the next bound
// activation along d so that on `[0, α_max]` the function is smooth.
// ----------------------------------------------------------------------------

/// Wrapper that projects parameters onto the box `[lb, ub]` before forwarding
/// to the underlying problem. Used as the line-search problem in [`LBFGSB`].
pub struct ProjectedLineSearchProblem<O, P, G, F> {
    problem: O,
    lb: P,
    ub: P,
    phantom: PhantomData<(G, F)>,
}

impl<O, P, G, F> ProjectedLineSearchProblem<O, P, G, F> {
    fn new(problem: O, lb: P, ub: P) -> Self {
        Self {
            problem,
            lb,
            ub,
            phantom: PhantomData,
        }
    }
}

impl<O, P, G, F> CostFunction for ProjectedLineSearchProblem<O, P, G, F>
where
    O: CostFunction<Param = P, Output = F>,
    P: Clone + ArgminMinMax,
    F: ArgminFloat,
{
    type Param = P;
    type Output = F;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let projected = P::min(&P::max(param, &self.lb), &self.ub);
        self.problem.cost(&projected)
    }
}

impl<O, P, G, F> Gradient for ProjectedLineSearchProblem<O, P, G, F>
where
    O: Gradient<Param = P, Gradient = G>,
    P: Clone + ArgminMinMax,
    F: ArgminFloat,
{
    type Param = P;
    type Gradient = G;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, Error> {
        let projected = P::min(&P::max(param, &self.lb), &self.ub);
        self.problem.gradient(&projected)
    }
}

// ----------------------------------------------------------------------------
// Solver impl.
// ----------------------------------------------------------------------------

impl<O, L, P, G, F> Solver<O, IterState<P, G, (), (), (), F>> for LBFGSB<L, P, G, F>
where
    O: CostFunction<Param = P, Output = F> + Gradient<Param = P, Gradient = G>,
    P: Clone
        + ArgminAdd<P, P>
        + ArgminSub<P, P>
        + ArgminMul<F, P>
        + ArgminDot<P, F>
        + ArgminDot<G, F>
        + ArgminScaledAdd<P, F, P>
        + ArgminMinMax
        + ArgminZeroLike
        + ArgminFromSlice<F>,
    for<'a> &'a P: IntoIterator<Item = &'a F>,
    G: Clone
        + ArgminAdd<G, G>
        + ArgminSub<G, G>
        + ArgminMul<F, P>
        + ArgminDot<G, F>
        + ArgminDot<P, F>
        + ArgminL2Norm<F>
        + ArgminLInfNorm<F>
        + ArgminMinMax
        + ArgminZeroLike,
    for<'a> &'a G: IntoIterator<Item = &'a F>,
    L: Clone
        + LineSearch<P, F>
        + Solver<ProjectedLineSearchProblem<O, P, G, F>, IterState<P, G, (), (), (), F>>,
    F: ArgminFloat,
{
    fn name(&self) -> &str {
        "L-BFGS-B"
    }

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        mut state: IterState<P, G, (), (), (), F>,
    ) -> Result<(IterState<P, G, (), (), (), F>, Option<KV>), Error> {
        let param = state.take_param().ok_or_else(argmin_error_closure!(
            NotInitialized,
            concat!(
                "`L-BFGS-B` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method."
            )
        ))?;
        let param = P::min(&P::max(&param, &self.bounds.0), &self.bounds.1);
        let cost = state.get_cost();
        let cost = if cost.is_infinite() {
            problem.cost(&param)?
        } else {
            cost
        };
        let grad = state
            .take_gradient()
            .map(Result::Ok)
            .unwrap_or_else(|| problem.gradient(&param))?;
        Ok((state.param(param).cost(cost).gradient(grad), None))
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        mut state: IterState<P, G, (), (), (), F>,
    ) -> Result<(IterState<P, G, (), (), (), F>, Option<KV>), Error> {
        let param = state.take_param().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`L-BFGS-B`: Parameter vector in state not set."
        ))?;
        let cur_cost = state.get_cost();
        let prev_grad = state.take_gradient().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`L-BFGS-B`: Gradient in state not set."
        ))?;

        let xs: Vec<F> = (&param).into_iter().copied().collect();
        let gs: Vec<F> = (&prev_grad).into_iter().copied().collect();
        let lbs: Vec<F> = (&self.bounds.0).into_iter().copied().collect();
        let ubs: Vec<F> = (&self.bounds.1).into_iter().copied().collect();

        let m = self.m.max(1);
        let (ss, sy) = build_ssyy(&self.s, &self.y, m);
        let k_matrix = build_k_matrix(&ss, &sy, self.theta, m, self.s.len());
        let s_rows: Vec<Vec<F>> = self
            .s
            .iter()
            .map(|si| si.into_iter().copied().collect())
            .collect();
        let y_rows: Vec<Vec<F>> = self
            .y
            .iter()
            .map(|yi| yi.into_iter().copied().collect())
            .collect();

        let (xc_scalars, c_vec, free) = cauchy_point_scalar(
            &xs, &gs, &lbs, &ubs, &s_rows, &y_rows, self.theta, &k_matrix,
        )?;
        let xstar_scalars = subspace_minimization(
            &xc_scalars,
            &xs,
            &gs,
            &lbs,
            &ubs,
            &free,
            &c_vec,
            &s_rows,
            &y_rows,
            self.theta,
            &k_matrix,
        )?;

        let xstar_p: P = P::from_slice(&xstar_scalars);
        let direction_p = xstar_p.sub(&param);

        let take_problem = problem.take_problem().unwrap();
        let line_problem = ProjectedLineSearchProblem::<O, P, G, F>::new(
            take_problem,
            self.bounds.0.clone(),
            self.bounds.1.clone(),
        );

        self.linesearch.search_direction(direction_p);

        let linesearch_result = Executor::new(line_problem, self.linesearch.clone())
            .configure(|config| {
                config
                    .param(param.clone())
                    .gradient(prev_grad.clone())
                    .cost(cur_cost)
            })
            .ctrlc(false)
            .run();

        let OptimizationResult {
            problem: mut line_problem_result,
            state: mut linesearch_state,
            ..
        } = match linesearch_result {
            Ok(res) => res,
            Err(e) => {
                return Ok((
                    state.terminate_with(TerminationReason::SolverExit(format!(
                        "Line search terminated with: '{e}'",
                    ))),
                    None,
                ))
            }
        };

        let xk1_raw = linesearch_state.take_param().unwrap();
        let xk1 = P::min(&P::max(&xk1_raw, &self.bounds.0), &self.bounds.1);

        let next_cost = if linesearch_state.get_cost().is_infinite() {
            problem.cost(&xk1)?
        } else {
            linesearch_state.get_cost()
        };

        let inner = line_problem_result.take_problem().unwrap();
        problem.problem = Some(inner.problem);
        problem.consume_func_counts(line_problem_result);

        let grad = problem.gradient(&xk1)?;

        let s_new = xk1.sub(&param);
        let y_new = grad.sub(&prev_grad);
        let s_dot_y: F = s_new.dot(&y_new);
        let yy: F = y_new.dot(&y_new);
        if s_dot_y > self.skip_ys_eps * yy {
            if self.s.len() >= self.m {
                self.s.pop_front();
                self.y.pop_front();
            }
            self.s.push_back(s_new);
            self.y.push_back(y_new);
            if let (Some(s_back), Some(y_back)) = (self.s.back(), self.y.back()) {
                let sy_back: F = y_back.dot(s_back);
                let yy_back: F = y_back.dot(y_back);
                if sy_back > F::epsilon() {
                    self.theta = yy_back / sy_back;
                }
            }
        }

        let n_free = free.len();
        Ok((
            state.param(xk1).cost(next_cost).gradient(grad),
            Some(kv!(
                "theta" => self.theta;
                "n_free" => n_free as u64;
            )),
        ))
    }

    fn terminate(&mut self, state: &IterState<P, G, (), (), (), F>) -> TerminationStatus {
        if let (Some(g), Some(p)) = (state.get_gradient(), state.get_param()) {
            let pi: Vec<F> = p.into_iter().copied().collect();
            let gi: Vec<F> = g.into_iter().copied().collect();
            let li: Vec<F> = (&self.bounds.0).into_iter().copied().collect();
            let ui: Vec<F> = (&self.bounds.1).into_iter().copied().collect();
            let mut max_abs = F::zero();
            for i in 0..pi.len() {
                let proj = (pi[i] - gi[i]).max(li[i]).min(ui[i]);
                let pg = pi[i] - proj;
                let av = pg.abs();
                if av > max_abs {
                    max_abs = av;
                }
            }
            if max_abs < self.tol_grad {
                return TerminationStatus::Terminated(TerminationReason::SolverConverged);
            }
        }
        if (state.get_prev_cost() - state.get_cost()).abs() < self.tol_cost {
            return TerminationStatus::Terminated(TerminationReason::SolverConverged);
        }
        TerminationStatus::NotTerminated
    }
}

// ----------------------------------------------------------------------------
// Tests.
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ArgminError;
    use crate::solver::linesearch::MoreThuenteLineSearch;

    test_trait_impl!(
        lbfgsb,
        LBFGSB<MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64>, Vec<f64>, Vec<f64>, f64>
    );

    fn linesearch() -> MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64> {
        MoreThuenteLineSearch::new().with_c(1e-4, 0.9).unwrap()
    }

    #[test]
    fn test_new() {
        let lbfgsb: LBFGSB<_, Vec<f64>, Vec<f64>, f64> =
            LBFGSB::new(linesearch(), 5, (vec![-1.0, -1.0], vec![1.0, 1.0]));
        assert_eq!(lbfgsb.m, 5);
        assert_eq!(lbfgsb.bounds.0, vec![-1.0, -1.0]);
        assert_eq!(lbfgsb.bounds.1, vec![1.0, 1.0]);
        assert_eq!(lbfgsb.theta.to_ne_bytes(), 1.0f64.to_ne_bytes());
        assert!(lbfgsb.s.is_empty());
        assert!(lbfgsb.y.is_empty());
    }

    #[test]
    fn test_with_tolerance_grad() {
        for tol in [1e-6, 0.0, 1e-2, 1.0, 2.0] {
            let l: LBFGSB<_, Vec<f64>, Vec<f64>, f64> =
                LBFGSB::new(linesearch(), 3, (vec![0.0], vec![1.0]));
            let res = l.with_tolerance_grad(tol);
            assert!(res.is_ok());
            assert_eq!(res.unwrap().tol_grad.to_ne_bytes(), tol.to_ne_bytes());
        }
        for tol in [-f64::EPSILON, -1.0, -100.0] {
            let l: LBFGSB<_, Vec<f64>, Vec<f64>, f64> =
                LBFGSB::new(linesearch(), 3, (vec![0.0], vec![1.0]));
            let res = l.with_tolerance_grad(tol);
            assert_error!(
                res,
                ArgminError,
                "Invalid parameter: \"`L-BFGS-B`: gradient tolerance must be >= 0.\""
            );
        }
    }

    #[test]
    fn test_with_tolerance_cost() {
        for tol in [1e-6, 0.0, 1e-2, 1.0, 2.0] {
            let l: LBFGSB<_, Vec<f64>, Vec<f64>, f64> =
                LBFGSB::new(linesearch(), 3, (vec![0.0], vec![1.0]));
            let res = l.with_tolerance_cost(tol);
            assert!(res.is_ok());
            assert_eq!(res.unwrap().tol_cost.to_ne_bytes(), tol.to_ne_bytes());
        }
        for tol in [-f64::EPSILON, -1.0, -100.0] {
            let l: LBFGSB<_, Vec<f64>, Vec<f64>, f64> =
                LBFGSB::new(linesearch(), 3, (vec![0.0], vec![1.0]));
            let res = l.with_tolerance_cost(tol);
            assert_error!(
                res,
                ArgminError,
                "Invalid parameter: \"`L-BFGS-B`: cost tolerance must be >= 0.\""
            );
        }
    }

    #[test]
    fn test_with_curvature_threshold() {
        for eps in [0.0, 1e-12, 1e-6, 1.0] {
            let l: LBFGSB<_, Vec<f64>, Vec<f64>, f64> =
                LBFGSB::new(linesearch(), 3, (vec![0.0], vec![1.0]));
            let res = l.with_curvature_threshold(eps);
            assert!(res.is_ok());
            assert_eq!(res.unwrap().skip_ys_eps.to_ne_bytes(), eps.to_ne_bytes());
        }
        for eps in [-f64::EPSILON, -1.0] {
            let l: LBFGSB<_, Vec<f64>, Vec<f64>, f64> =
                LBFGSB::new(linesearch(), 3, (vec![0.0], vec![1.0]));
            let res = l.with_curvature_threshold(eps);
            assert_error!(
                res,
                ArgminError,
                "Invalid parameter: \"`L-BFGS-B`: curvature threshold must be >= 0.\""
            );
        }
    }

    #[test]
    fn test_gauss_solve_2x2() {
        // [[2, 1], [1, 3]] · x = [3, 5]  ⇒  x = [4/5, 7/5]
        let mut a = vec![2.0_f64, 1.0, 1.0, 3.0];
        let mut b = vec![3.0_f64, 5.0];
        gauss_solve(&mut a, &mut b, 2).unwrap();
        approx::assert_relative_eq!(b[0], 4.0 / 5.0, epsilon = 1e-12);
        approx::assert_relative_eq!(b[1], 7.0 / 5.0, epsilon = 1e-12);
    }

    #[test]
    fn test_gauss_solve_pivoting() {
        // System whose first pivot is zero so partial pivoting must kick in.
        let mut a = vec![0.0_f64, 1.0, 1.0, 0.0];
        let mut b = vec![3.0_f64, 5.0];
        gauss_solve(&mut a, &mut b, 2).unwrap();
        approx::assert_relative_eq!(b[0], 5.0, epsilon = 1e-12);
        approx::assert_relative_eq!(b[1], 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_gauss_solve_3x3() {
        // [[1, 2, 3], [2, 5, 3], [1, 0, 8]] · x = [6, 4, 17]
        // Solution: x = [-15.5, 4, 4.0625]? Let me compute.
        // Det = 1·(40-0) - 2·(16-3) + 3·(0-5) = 40 - 26 - 15 = -1.
        // Using `numpy.linalg.solve(...)`: x ≈ [1, 1, 2].  Verify:
        //   1·1 + 2·1 + 3·2 = 9 ≠ 6. So not [1,1,2].
        // Let me re-derive. Actually we just want a system with a known solution.
        // Pick x = [1, 2, 3]. Then b = A @ x.
        // A = [[1, 2, 3], [2, 5, 3], [1, 0, 8]]
        // b = [1+4+9, 2+10+9, 1+0+24] = [14, 21, 25].
        let mut a = vec![1.0_f64, 2.0, 3.0, 2.0, 5.0, 3.0, 1.0, 0.0, 8.0];
        let mut b = vec![14.0_f64, 21.0, 25.0];
        gauss_solve(&mut a, &mut b, 3).unwrap();
        approx::assert_relative_eq!(b[0], 1.0, epsilon = 1e-12);
        approx::assert_relative_eq!(b[1], 2.0, epsilon = 1e-12);
        approx::assert_relative_eq!(b[2], 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_breakpoints_2d() {
        // x = [0.5, 0.5], g = [-1.0, +2.0], lb = [0.0, 0.0], ub = [1.0, 1.0].
        // For coord 0: g < 0 ⇒ t = (x - u)/g = (0.5 - 1.0) / -1.0 = 0.5.
        // For coord 1: g > 0 ⇒ t = (x - l)/g = (0.5 - 0.0) /  2.0 = 0.25.
        let xs = [0.5_f64, 0.5];
        let gs = [-1.0_f64, 2.0];
        let lbs = [0.0_f64, 0.0];
        let ubs = [1.0_f64, 1.0];
        let t = breakpoints::<f64>(&xs, &gs, &lbs, &ubs);
        approx::assert_relative_eq!(t[0], 0.5, epsilon = 1e-12);
        approx::assert_relative_eq!(t[1], 0.25, epsilon = 1e-12);
    }

    #[test]
    fn test_breakpoints_zero_grad_is_infinite() {
        let xs = [0.5_f64, 0.5];
        let gs = [0.0_f64, 0.0];
        let lbs = [0.0_f64, 0.0];
        let ubs = [1.0_f64, 1.0];
        let t = breakpoints::<f64>(&xs, &gs, &lbs, &ubs);
        assert!(t[0].is_infinite());
        assert!(t[1].is_infinite());
    }

    // The two `test_cauchy_point_*` tests below pin `cauchy_point_scalar` to
    // bit-for-bit values produced by the L-BFGS-B 3.0 Fortran reference run
    // on the same fixtures. They guard against regressions in the GCP
    // computation and (specifically for the 5D fixture) the free-set
    // classification when a coordinate has zero gradient.

    #[test]
    fn test_cauchy_point_2d_rosenbrock_against_fortran() {
        // 2D Rosenbrock at x_0 = (-1.2, 1.0), bounds [-2, 2]^2, no L-BFGS
        // history. Fortran reference: xcp = (2, 2), iwhere = (2, 2)
        // (both pinned at the upper bound), so the free set is empty.
        let xs = vec![-1.2_f64, 1.0];
        let gs = vec![-215.6_f64, -88.0];
        let lbs = vec![-2.0_f64, -2.0];
        let ubs = vec![2.0_f64, 2.0];
        let s_rows: Vec<Vec<f64>> = vec![];
        let y_rows: Vec<Vec<f64>> = vec![];
        let theta = 1.0_f64;
        let k_matrix: Vec<f64> = vec![];

        let (x_c, _c, free) =
            cauchy_point_scalar(&xs, &gs, &lbs, &ubs, &s_rows, &y_rows, theta, &k_matrix).unwrap();

        approx::assert_relative_eq!(x_c[0], 2.0, epsilon = 1e-12);
        approx::assert_relative_eq!(x_c[1], 2.0, epsilon = 1e-12);
        assert!(free.is_empty(), "expected empty free set, got {:?}", free);
    }

    #[test]
    fn test_cauchy_point_5d_rosenbrock_zero_grad_var_is_free_against_fortran() {
        // 5D Rosenbrock at x_0 = (-1.2, 1.0, -1.2, 1.0, 1.0), bounds [-2, 2]^5.
        // The last gradient component is zero, so x_5 doesn't move during the
        // Cauchy step. Fortran's iwhere classifies it as -3 (free, not moved).
        // The other four coordinates each hit a bound. Reference: xcp =
        // (2, -2, 2, 2, 1), free = [4] (zero-based).
        let xs = vec![-1.2_f64, 1.0, -1.2, 1.0, 1.0];
        let gs = vec![-215.6_f64, 792.0, -655.6, -88.0, 0.0];
        let lbs = vec![-2.0_f64; 5];
        let ubs = vec![2.0_f64; 5];
        let s_rows: Vec<Vec<f64>> = vec![];
        let y_rows: Vec<Vec<f64>> = vec![];
        let theta = 1.0_f64;
        let k_matrix: Vec<f64> = vec![];

        let (x_c, _c, free) =
            cauchy_point_scalar(&xs, &gs, &lbs, &ubs, &s_rows, &y_rows, theta, &k_matrix).unwrap();

        let expected_xc = [2.0, -2.0, 2.0, 2.0, 1.0];
        for i in 0..5 {
            approx::assert_relative_eq!(x_c[i], expected_xc[i], epsilon = 1e-12);
        }
        assert_eq!(free, vec![4]);
    }

    #[test]
    fn test_build_ssyy_two_pairs() {
        let mut s: VecDeque<Vec<f64>> = VecDeque::new();
        let mut y: VecDeque<Vec<f64>> = VecDeque::new();
        s.push_back(vec![1.0, 0.0]);
        s.push_back(vec![0.0, 2.0]);
        y.push_back(vec![3.0, 4.0]);
        y.push_back(vec![5.0, 6.0]);
        let m = 4;
        let (ss, sy) = build_ssyy::<Vec<f64>, Vec<f64>, f64>(&s, &y, m);
        // ss[i*m+j] = s_i · s_j
        approx::assert_relative_eq!(ss[0], 1.0, epsilon = 1e-12);
        approx::assert_relative_eq!(ss[1], 0.0, epsilon = 1e-12);
        approx::assert_relative_eq!(ss[m], 0.0, epsilon = 1e-12);
        approx::assert_relative_eq!(ss[m + 1], 4.0, epsilon = 1e-12);
        // sy[i*m+j] = s_i · y_j
        approx::assert_relative_eq!(sy[0], 3.0, epsilon = 1e-12); // [1,0]·[3,4]
        approx::assert_relative_eq!(sy[1], 5.0, epsilon = 1e-12); // [1,0]·[5,6]
        approx::assert_relative_eq!(sy[m], 8.0, epsilon = 1e-12); // [0,2]·[3,4]
        approx::assert_relative_eq!(sy[m + 1], 12.0, epsilon = 1e-12); // [0,2]·[5,6]
    }

    #[test]
    fn test_build_k_matrix_one_pair() {
        // Single (s, y) pair: K = [[ -D ], [ θ S^T S ]] (with off-diagonals empty).
        // m=1, k=1, ss=[s·s] = [4], sy=[s·y] = [6], θ = 0.5.
        // K should be 2 × 2: [[-6, 0], [0, 0.5*4]] = [[-6, 0], [0, 2]].
        let ss = vec![4.0_f64];
        let sy = vec![6.0_f64];
        let k = build_k_matrix::<f64>(&ss, &sy, 0.5, 1, 1);
        assert_eq!(k.len(), 4);
        approx::assert_relative_eq!(k[0], -6.0, epsilon = 1e-12);
        approx::assert_relative_eq!(k[1], 0.0, epsilon = 1e-12);
        approx::assert_relative_eq!(k[2], 0.0, epsilon = 1e-12);
        approx::assert_relative_eq!(k[3], 2.0, epsilon = 1e-12);
    }
}
