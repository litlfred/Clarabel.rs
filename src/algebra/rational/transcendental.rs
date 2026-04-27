//! [`Transcendental`] impl for [`RationalReal`].
//!
//! - `sqrt`: Newton iteration x ← (x + a/x)/2 with denominator capping.
//! - `recip`: exact `BigRational` reciprocal.
//! - `powi`: exact repeated squaring (no precision loss).
//! - `ln`: argument reduction `ln(a) = k·ln 2 + ln(m)` with m ∈ [1, 2),
//!   then Taylor on `ln((1+y)/(1-y)) = 2·(y + y³/3 + y⁵/5 + …)`.
//! - `exp`: range reduction `exp(x) = 2^k · exp(r)` with `k = floor(x / ln 2)`,
//!   then Taylor on `exp(r) = Σ rⁿ/n!`.
//! - `powf(a, b) = exp(b · ln a)`.
//! - `sin`/`cos`/`atan2`: stubbed `unimplemented!()`; only called by the
//!   analytic 3×3 symmetric eigensolver in `dense3x3/eigen.rs`, which is
//!   used by SDP code paths that the `bigrational` feature excludes via
//!   `compile_error!` in `mod.rs`. They exist as trait methods so the
//!   `Transcendental` impl is complete.
//!
//! All transcendentals round their result to the current thread-local
//! working precision (`precision_bits()`) to keep `BigRational`
//! denominators bounded.

use super::arena;
use super::cap::round_to_pow2_denominator as round_to_precision;
use super::precision::precision_bits;
use super::real::RationalReal;
use crate::algebra::transcendental::Transcendental;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{FromPrimitive, One, Signed, ToPrimitive, Zero};

/// `2^-p` as a [`BigRational`]. Used as the convergence tolerance.
fn tolerance(p: u32) -> BigRational {
    BigRational::new(BigInt::one(), BigInt::one() << (p as usize))
}

/// Two as a [`BigRational`].
fn two() -> BigRational {
    BigRational::from_i32(2).unwrap()
}

// =========================================================
// sqrt: Newton iteration
// =========================================================

fn sqrt_newton(a: &BigRational, p: u32) -> BigRational {
    if a.is_zero() {
        return BigRational::zero();
    }
    if a.is_negative() {
        panic!("RationalReal::sqrt of a negative value");
    }
    let tol = tolerance(p);
    // Initial guess: round-trip through f64 — gives 53 bits, Newton
    // doubles precision per iteration so we converge in ~log2(p/53) steps.
    let init = a
        .to_f64()
        .map(|v| v.sqrt())
        .and_then(BigRational::from_f64)
        .unwrap_or_else(BigRational::one);
    let mut x = init;
    let two = two();
    for _ in 0..200 {
        let next = (&x + a / &x) / &two;
        let diff = (&next - &x).abs();
        x = round_to_precision(&next, p + 8);
        if diff < tol {
            break;
        }
    }
    round_to_precision(&x, p)
}

// =========================================================
// ln: argument reduction + Taylor on ln((1+y)/(1-y))
// =========================================================

/// Cached `ln(2)` at the current thread-local precision. We compute it
/// on demand via a separate pure-rational evaluation.
fn ln2(p: u32) -> BigRational {
    // ln(2) = 2 * (y + y³/3 + y⁵/5 + …) with y = (2-1)/(2+1) = 1/3
    let one = BigRational::one();
    let three = BigRational::from_i32(3).unwrap();
    let y = &one / &three;
    let y2 = &y * &y;
    let mut term = y.clone();
    let mut sum = y.clone();
    let tol = tolerance(p + 4);
    let mut k: u64 = 1;
    loop {
        term = round_to_precision(&(&term * &y2), p + 16);
        k += 2;
        let denom = BigRational::from_u64(k).unwrap();
        let add = &term / &denom;
        sum = round_to_precision(&(&sum + &add), p + 16);
        if add.abs() < tol {
            break;
        }
    }
    round_to_precision(&(&two() * &sum), p)
}

fn ln_rational(a: &BigRational, p: u32) -> BigRational {
    if a <= &BigRational::zero() {
        panic!("RationalReal::ln of a non-positive value");
    }
    // Argument reduction: ln(a) = k·ln 2 + ln(m), m = a / 2^k ∈ [1, 2).
    // Find k = floor(log2(a)).
    let mut k: i64 = 0;
    let mut m = a.clone();
    let two_r = two();
    let one_r = BigRational::one();
    while m >= two_r {
        m = m / &two_r;
        k += 1;
    }
    while m < one_r {
        m = m * &two_r;
        k -= 1;
    }
    // Now m ∈ [1, 2). Set y = (m-1)/(m+1), |y| ≤ 1/3.
    let y = (&m - &one_r) / (&m + &one_r);
    let y2 = &y * &y;
    let mut term = y.clone();
    let mut sum = y.clone();
    let tol = tolerance(p + 4);
    let mut kk: u64 = 1;
    loop {
        term = round_to_precision(&(&term * &y2), p + 16);
        kk += 2;
        let denom = BigRational::from_u64(kk).unwrap();
        let add = &term / &denom;
        sum = round_to_precision(&(&sum + &add), p + 16);
        if add.abs() < tol {
            break;
        }
    }
    let ln_m = &two_r * &sum;
    let result = if k == 0 {
        ln_m
    } else {
        let k_r = BigRational::from_i64(k).unwrap();
        ln_m + k_r * ln2(p + 8)
    };
    round_to_precision(&result, p)
}

// =========================================================
// exp: range reduction + Taylor
// =========================================================

fn exp_rational(x: &BigRational, p: u32) -> BigRational {
    // exp(x) = 2^k · exp(r) where k = round(x / ln 2), r = x - k·ln 2 small.
    let ln_2 = ln2(p + 16);
    let k_rat = round_to_precision(&(x / &ln_2), 0); // round to integer (denom = 1)
    // Convert k_rat to integer k. Since round_to_precision with p=0 produces
    // m / 1 with m the rounded integer.
    let k_int = k_rat.numer().clone();
    let k_i64 = k_int.to_i64().unwrap_or(0);
    let k_back = BigRational::from(BigInt::from(k_i64));
    let r = x - &k_back * &ln_2;
    // Taylor: exp(r) = Σ rⁿ/n!
    let one_r = BigRational::one();
    let mut term = one_r.clone();
    let mut sum = one_r.clone();
    let tol = tolerance(p + 4);
    let mut n: u64 = 1;
    loop {
        let denom = BigRational::from_u64(n).unwrap();
        term = round_to_precision(&(&term * &r / &denom), p + 16);
        sum = round_to_precision(&(&sum + &term), p + 16);
        if term.abs() < tol {
            break;
        }
        n += 1;
        if n > 500 {
            break; // safety
        }
    }
    // Multiply by 2^k.
    let result = if k_i64 >= 0 {
        let scale: BigInt = BigInt::one() << (k_i64 as usize);
        sum * BigRational::from(scale)
    } else {
        let scale: BigInt = BigInt::one() << ((-k_i64) as usize);
        sum / BigRational::from(scale)
    };
    round_to_precision(&result, p)
}

// =========================================================
// powi: exact repeated squaring
// =========================================================

fn powi_exact(base: &BigRational, n: i32) -> BigRational {
    if n == 0 {
        return BigRational::one();
    }
    let mut result = BigRational::one();
    let mut b = if n > 0 { base.clone() } else { BigRational::one() / base };
    let mut e = n.unsigned_abs();
    while e > 0 {
        if e & 1 == 1 {
            result = result * &b;
        }
        e >>= 1;
        if e > 0 {
            b = &b * &b;
        }
    }
    result
}

// =========================================================
// Transcendental impl
// =========================================================

impl Transcendental for RationalReal {
    fn sqrt(self) -> Self {
        let p = precision_bits();
        let v = arena::with(self.0, |a| sqrt_newton(a, p));
        RationalReal::from_bigrational(v)
    }

    fn ln(self) -> Self {
        let p = precision_bits();
        let v = arena::with(self.0, |a| ln_rational(a, p));
        RationalReal::from_bigrational(v)
    }

    fn exp(self) -> Self {
        let p = precision_bits();
        let v = arena::with(self.0, |a| exp_rational(a, p));
        RationalReal::from_bigrational(v)
    }

    fn powf(self, n: Self) -> Self {
        // a^b = exp(b · ln a). Path through ln ⇒ a > 0 required.
        let p = precision_bits();
        let v = arena::with2(self.0, n.0, |a, b| {
            let log_a = ln_rational(a, p + 8);
            let prod = round_to_precision(&(b * &log_a), p + 8);
            exp_rational(&prod, p)
        });
        RationalReal::from_bigrational(v)
    }

    fn powi(self, n: i32) -> Self {
        let v = arena::with(self.0, |a| powi_exact(a, n));
        RationalReal::from_bigrational(v)
    }

    fn recip(self) -> Self {
        let v = arena::with(self.0, |a| BigRational::one() / a);
        RationalReal::from_bigrational(v)
    }

    fn sin(self) -> Self {
        unimplemented!(
            "RationalReal::sin is not implemented; this method is only \
             reached from the analytic 3x3 symmetric eigensolver in \
             dense3x3/eigen.rs, which is used by SDP code paths that \
             the bigrational feature excludes."
        )
    }

    fn cos(self) -> Self {
        unimplemented!(
            "RationalReal::cos is not implemented; see the comment on \
             RationalReal::sin."
        )
    }

    fn atan2(self, _x: Self) -> Self {
        unimplemented!(
            "RationalReal::atan2 is not implemented; see the comment on \
             RationalReal::sin."
        )
    }
}
