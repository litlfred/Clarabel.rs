//! Unit tests for the MpfrFloat backend.

#![allow(unused_imports)]

use super::*;
use crate::algebra::transcendental::{RealConst, RealSentinel, Transcendental};
use num_traits::{FromPrimitive, One, Signed, Zero};

#[test]
fn mpfr_arithmetic_at_default_precision() {
    let a = MpfrFloat::from_f64(0.1).unwrap();
    let b = MpfrFloat::from_f64(0.2).unwrap();
    let s = a + b;
    let s_f = s.to_f64();
    assert!((s_f - 0.3).abs() < 1e-15, "0.1 + 0.2 ≈ 0.3, got {s_f}");
}

#[test]
fn mpfr_default_precision_is_167() {
    set_default_precision(167);
    let z = MpfrFloat::zero();
    assert_eq!(z.prec(), 167);
}

#[test]
fn mpfr_with_precision_scope_guard_restores() {
    set_default_precision(167);
    with_mpfr_precision(300, || {
        let z = MpfrFloat::zero();
        assert_eq!(z.prec(), 300);
    });
    let z = MpfrFloat::zero();
    assert_eq!(z.prec(), 167);
}

#[test]
fn mpfr_sqrt_two_squared_close_to_two() {
    set_default_precision(200);
    let two = MpfrFloat::from_i64(2).unwrap();
    let s = two.clone().sqrt();
    let back = s.clone() * s;
    let diff = (back - two).abs();
    let tol = MpfrFloat::from_f64(1e-50).unwrap();
    assert!(diff < tol, "sqrt(2)² should match 2 within 2^-50ish at 200 bits");
}

#[test]
fn mpfr_exp_ln_round_trip() {
    set_default_precision(200);
    for v in [0.5_f64, 1.0_f64, 2.0_f64, 100.0_f64] {
        let x = MpfrFloat::from_f64(v).unwrap();
        let back = x.clone().ln().exp();
        let err = (back - x).abs();
        let tol = MpfrFloat::from_f64(1e-40).unwrap();
        assert!(err < tol, "exp(ln({v})) round-trip");
    }
}

#[test]
fn mpfr_recip_of_zero_is_infinity() {
    let z = MpfrFloat::zero();
    let r = z.recip();
    // MPFR division-by-zero gives +inf with the standard rounding mode.
    assert!(<MpfrFloat as RealSentinel>::is_infinite(r));
}

#[test]
fn mpfr_pi_close_to_f64_pi() {
    set_default_precision(167);
    let pi = <MpfrFloat as RealConst>::PI();
    let pi_f = pi.to_f64();
    assert!((pi_f - std::f64::consts::PI).abs() < 1e-15);
}

#[test]
fn mpfr_is_floatt() {
    fn assert_floatt<T: crate::algebra::FloatT>() {}
    assert_floatt::<MpfrFloat>();
    let one = MpfrFloat::one();
    assert!(<MpfrFloat as RealSentinel>::is_finite(one));
}

#[cfg(feature = "serde")]
#[test]
fn mpfr_serde_round_trip() {
    set_default_precision(200);
    let r = MpfrFloat::from_f64(0.1).unwrap();
    let s = serde_json::to_string(&r).unwrap();
    let back: MpfrFloat = serde_json::from_str(&s).unwrap();
    let diff = (back - r).abs();
    let tol = MpfrFloat::from_f64(1e-50).unwrap();
    assert!(diff < tol, "serde round-trip preserves value");
}
