//! Unit tests for the `RationalReal` arena backend.

#![allow(unused_imports)]

use super::*;
use crate::algebra::transcendental::{RealConst, RealSentinel, Transcendental};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{FromPrimitive, One, Signed, Zero};

// Bring `arena::get` into scope for the cache-invalidation test.
use super::arena;

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

#[test]
fn rational_max_arena_bits_caps_growth() {
    // With cap on at 256 bits, repeated 1/3 divisions stay bounded
    // (3^-50 needs only ~80 denom bits, so 256 leaves plenty of margin
    // and the capped value matches the true 3^-50 to ulp in f64).
    reset_arena();
    set_max_arena_bits(Some(256));
    let mut x = RationalReal::from_i64(1).unwrap();
    let three = RationalReal::from_i64(3).unwrap();
    for _ in 0..50 {
        x = x / three;
    }
    let bits = x.max_bits();
    assert!(
        bits <= 256,
        "max_arena_bits(256) should bound denom+numer; got {bits} bits after 50 divisions"
    );
    // Result is approximately 3^-50; check the f64 view.
    let approx = x.to_f64();
    let expected = 3f64.powi(-50);
    let rel_err = (approx - expected).abs() / expected.abs();
    assert!(
        rel_err < 1e-15,
        "capped result {approx} should match 3^-50 = {expected} to ~ulp; rel_err = {rel_err}"
    );
    set_max_arena_bits(None);
    reset_arena();
}

#[test]
fn rational_aggressive_cap_rounds_to_zero_when_value_smaller_than_ulp() {
    // Documents the boundary: at 32-bit cap, anything smaller than
    // ~2^-32 rounds to exactly 0. This is intentional — the cap is
    // a precision floor, not a magnitude floor.
    reset_arena();
    set_max_arena_bits(Some(32));
    let small = RationalReal::from_pair(BigInt::from(1), BigInt::from(1u64) << 50);
    // small = 2^-50, far below 2^-32 ulp at 32-bit precision -> rounds to 0.
    let bumped = small + RationalReal::zero(); // any op triggers cap
    assert_eq!(bumped.to_f64(), 0.0);
    set_max_arena_bits(None);
    reset_arena();
}

#[test]
fn rational_with_max_arena_bits_scope_guard_restores_on_panic() {
    reset_arena();
    let original = max_arena_bits();
    let _ = std::panic::catch_unwind(|| {
        with_max_arena_bits(Some(32), || {
            assert_eq!(max_arena_bits(), Some(32));
            panic!("oops");
        });
    });
    assert_eq!(
        max_arena_bits(),
        original,
        "with_max_arena_bits guard must restore the previous value even on panic"
    );
    reset_arena();
}

#[test]
fn rational_sentinel_cache_invalidates_on_reset_arena() {
    // Regression test for the flaw flagged in Gemini's PR review:
    // if reset_arena() runs and then enough new entries are pushed to
    // reach the old sentinel handles, the cached sentinels would
    // otherwise return stale handles pointing at unrelated values.
    // The generation-counter check now invalidates them on reset.
    reset_arena();
    let inf1 = <RationalReal as RealSentinel>::infinity();
    let inf1_value = arena::get(inf1.0);

    // Force the arena to grow well past the sentinel slots,
    // then reset and re-fetch infinity. The handle must point to
    // a +inf-magnitude value, not whatever happened to land at
    // the old slot.
    for _ in 0..10 {
        let _ = RationalReal::from_i64(42).unwrap();
    }
    reset_arena();
    // Push some unrelated values so the slot 0/1/2 are now occupied
    // by other things.
    let a = RationalReal::from_i64(7).unwrap();
    let b = RationalReal::from_i64(11).unwrap();
    let _ = a + b;
    let _ = a * b;

    let inf2 = <RationalReal as RealSentinel>::infinity();
    assert!(inf2.is_infinite(), "infinity() after reset must still be infinite");
    let inf2_value = arena::get(inf2.0);
    assert_eq!(inf1_value, inf2_value);
    reset_arena();
}

#[test]
fn rational_one_third_plus_one_third_plus_one_third_still_one_in_exact_mode() {
    // Default mode (no cap) preserves the headline guarantee that
    // 1/3 + 1/3 + 1/3 == 1 exactly.
    reset_arena();
    set_max_arena_bits(None); // explicit, even though it's the default
    let third = RationalReal::from_pair(BigInt::from(1), BigInt::from(3));
    let three_thirds = third + third + third;
    assert_eq!(three_thirds, RationalReal::one());
    reset_arena();
}
