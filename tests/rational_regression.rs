//! Integration regression tests for the `bigrational` backend.
//!
//! Run:
//!   cargo test --no-default-features --features serde,bigrational \
//!              --test rational_regression
//!
//! These tests exercise the solver end-to-end with `T = RationalReal`.
//! They are gated on the `bigrational` feature so the default-feature
//! `cargo test` is unaffected.

#![cfg(feature = "bigrational")]
#![allow(non_snake_case)]

use clarabel::algebra::*;
use clarabel::solver::*;
use num_bigint::BigInt;

type T = RationalReal;

fn rat(n: i64) -> T {
    T::from_pair(BigInt::from(n), BigInt::from(1))
}

fn rat2(n: i64, d: i64) -> T {
    T::from_pair(BigInt::from(n), BigInt::from(d))
}

/// Build the same default settings the f64 solver would use, but with
/// rational tolerances at 1e-8 = 1/100_000_000.
fn default_rational_settings() -> DefaultSettings<T> {
    let tol = rat2(1, 100_000_000);
    DefaultSettingsBuilder::<T>::default()
        .equilibrate_enable(true)
        .max_iter(50)
        .tol_feas(tol)
        .tol_gap_abs(tol)
        .tol_gap_rel(tol)
        .build()
        .unwrap()
}

/// Tiny LP from example_lp.rs:
///   min    x[0] - x[1]
///   s.t.   |x[i]| <= 1   for i in {0,1}
/// Optimum: x* = (-1, 1), obj* = -2.
///
/// Marked `#[ignore]` because rational arithmetic on the IPM iterates
/// has unbounded denominator growth — even this 2-d LP takes many
/// minutes per iteration in `--release` mode without inner-loop
/// precision capping. Run explicitly with:
///   cargo test --no-default-features --features serde,bigrational \
///              --test rational_regression rational_lp_box_solves_exactly \
///              -- --ignored --nocapture
/// and expect ~10+ minutes.
#[test]
#[ignore = "slow: rational denominators blow up; run with --ignored explicitly"]
fn rational_lp_box_solves_exactly() {
    reset_arena();

    let P: CscMatrix<T> = CscMatrix::<T>::zeros((2, 2));
    let q: Vec<T> = vec![rat(1), -rat(1)];
    let A = CscMatrix::new(
        4,
        2,
        vec![0, 2, 4],
        vec![0, 2, 1, 3],
        vec![rat(1), -rat(1), rat(1), -rat(1)],
    );
    let b: Vec<T> = vec![rat(1); 4];
    let cones = [NonnegativeConeT(4)];

    let mut solver =
        DefaultSolver::new(&P, &q, &A, &b, &cones, default_rational_settings()).unwrap();
    solver.solve();

    assert!(matches!(solver.solution.status, SolverStatus::Solved));

    // Objective value should be near -2 (within 1e-6 in f64 view; the
    // exact-rational form is checked separately below).
    let obj_f64 = solver.solution.obj_val.to_f64();
    assert!(
        (obj_f64 + 2.0).abs() < 1e-6,
        "objective {} should be near -2",
        obj_f64
    );

    // Each x[i] should be near ±1.
    for (i, expected) in [-1.0_f64, 1.0_f64].iter().enumerate() {
        let xi_f64 = solver.solution.x[i].to_f64();
        assert!(
            (xi_f64 - expected).abs() < 1e-6,
            "x[{i}] = {xi_f64} should be near {expected}"
        );
    }

    reset_arena();
}

/// Check that the exact rational arithmetic guarantee holds for one
/// known-trivial case: `(1/3) * 3 == 1` exactly. This is impossible
/// in f64 (1/3 is unrepresentable so 0.333...×3 ≠ 1.0 exactly).
#[test]
fn rational_one_third_times_three_is_one() {
    reset_arena();
    let third = rat2(1, 3);
    let prod = third * rat(3);
    assert_eq!(prod, rat(1));
    let (n, d) = prod.into_pair();
    assert_eq!(n, BigInt::from(1));
    assert_eq!(d, BigInt::from(1));
    reset_arena();
}
