//! Unit tests for the `RationalReal` arena backend.

#![allow(unused_imports)]

use super::*;
use crate::algebra::transcendental::{RealConst, RealSentinel, Transcendental};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{FromPrimitive, One, Signed, Zero};

#[test]
fn rational_arena_smoke() {
    reset_arena();
    let _a: RationalReal = 1i64.into();
    let _b: RationalReal = 2i64.into();
    assert!(arena_len() >= 2);
    reset_arena();
    assert_eq!(arena_len(), 0);
}

#[test]
fn rational_basic_arithmetic_is_exact() {
    reset_arena();
    let third: RationalReal =
        RationalReal::from_pair(BigInt::from(1), BigInt::from(3));
    let three_thirds = third + third + third;
    let one = RationalReal::one();
    assert!(
        three_thirds == one,
        "1/3 + 1/3 + 1/3 should be exactly 1, got {:?}",
        three_thirds
    );
    reset_arena();
}

#[test]
fn rational_into_pair_round_trips() {
    reset_arena();
    let r: RationalReal =
        RationalReal::from_pair(BigInt::from(22), BigInt::from(7));
    let (n, d) = r.into_pair();
    assert_eq!(n, BigInt::from(22));
    assert_eq!(d, BigInt::from(7));
    reset_arena();
}

#[test]
fn rational_from_f64_is_exact_in_binary() {
    // 0.1 has no exact binary representation; from_f64 captures the
    // *exact* IEEE bit pattern rather than the decimal "1/10".
    reset_arena();
    let r = RationalReal::from(0.1f64);
    assert!(
        (r.to_f64() - 0.1f64).abs() < f64::EPSILON,
        "round-trip 0.1 -> RationalReal -> f64 within ulp"
    );
    reset_arena();
}

#[test]
fn rational_powi_is_exact() {
    reset_arena();
    let half = RationalReal::from_pair(BigInt::from(1), BigInt::from(2));
    let one_over_eight = half.powi(3);
    let expected = RationalReal::from_pair(BigInt::from(1), BigInt::from(8));
    assert_eq!(one_over_eight, expected);
    reset_arena();
}

#[test]
fn rational_sqrt_two_squared_close_to_two_at_128_bits() {
    reset_arena();
    set_precision_bits(128);
    let two_r = RationalReal::from_i64(2).unwrap();
    let s = two_r.sqrt();
    let back = s * s;
    let diff = (back - two_r).abs();
    let tol = RationalReal::from_pair(BigInt::one(), BigInt::one() << 100);
    assert!(diff < tol, "sqrt(2)² should match 2 to within 2^-100");
    reset_arena();
}

#[test]
fn rational_exp_ln_round_trip() {
    reset_arena();
    set_precision_bits(128);
    for v in [0.5_f64, 1.0_f64, 2.0_f64, 100.0_f64] {
        let x = RationalReal::from(v);
        let back = x.ln().exp();
        let err = (back - x).abs();
        let tol = RationalReal::from_pair(BigInt::one(), BigInt::one() << 80);
        assert!(
            err < tol,
            "exp(ln({v})) should match {v} to within 2^-80, got err = {err:?}"
        );
    }
    reset_arena();
}

#[test]
fn rational_is_floatt() {
    // The compile-time assertion in mod.rs already proves this; here we
    // also verify a few trait method calls work at run time.
    reset_arena();
    let inf = <RationalReal as RealSentinel>::infinity();
    assert!(inf.is_infinite());
    assert!(!inf.is_finite());

    let pi = <RationalReal as RealConst>::PI();
    assert!((pi.to_f64() - std::f64::consts::PI).abs() < 1e-12);
    reset_arena();
}

#[test]
fn rational_max_bits_grows_with_repeated_division() {
    reset_arena();
    let mut x = RationalReal::from_i64(1).unwrap();
    let three = RationalReal::from_i64(3).unwrap();
    let initial_bits = x.max_bits();
    for _ in 0..10 {
        x = x / three;
    }
    let after_bits = x.max_bits();
    assert!(
        after_bits > initial_bits,
        "denominator should grow as 3^10 ≈ 16 bits over 10 divisions"
    );
    reset_arena();
}
