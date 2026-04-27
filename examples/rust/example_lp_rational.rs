#![allow(non_snake_case)]
//! LP example solved end-to-end with the exact-rational backend.
//!
//! Solves the same 2-D box LP as `example_lp.rs`, but with
//! `T = RationalReal`. The primal iterates and the final solution are
//! exact rationals; only the per-iteration print path rounds (via
//! `LowerExp` -> `f64`) for legibility.
//!
//! Run:
//!   cargo run --no-default-features --features serde,bigrational \
//!             --example lp_rational --release
//!
//! # Runtime warning
//!
//! Even on this 2-d LP, expect **many minutes per IPM iteration** with
//! the current implementation. Basic `+`/`-`/`*`/`/` on
//! `RationalReal` are exact and unbounded; per-iteration rational
//! denominators grow geometrically and each subsequent op gets slower.
//! This is intrinsic to exact rational LP solving — practical
//! exact-rational solvers use iterative refinement (compute in floats,
//! refine to exact rationals only at termination), or specialized
//! rational simplex/ellipsoid methods, neither of which Clarabel does.
//!
//! Despite the speed cost, the iter-print line shows the IPM converging
//! exactly as in the f64 baseline (same `pcost`/`gap`/`pres`/`dres`
//! magnitudes per iteration, modulo the lossy f64-LowerExp rendering),
//! demonstrating the trait wiring is correct end-to-end.
//!
//! For QOU-scale workloads the recommended path is the planned MPFR
//! backend (Phase 8), which gives high-precision floats with bounded
//! denominators. The `bigrational` backend is for small problems where
//! bit-exactness of the iterates matters more than speed.

use clarabel::algebra::*;
use clarabel::solver::*;
use num_bigint::BigInt;

type T = RationalReal;

fn rat_from_int(n: i64) -> T {
    T::from_pair(BigInt::from(n), BigInt::from(1))
}

fn rat_from_pair(n: i64, d: i64) -> T {
    T::from_pair(BigInt::from(n), BigInt::from(d))
}

fn main() {
    // Set the working precision used by transcendentals (sqrt etc).
    // Pure LP doesn't actually consume this — only barrier transcendentals
    // do — but it's the right place to demonstrate the knob.
    set_precision_bits(128);

    // Reset the per-thread arena so the example doesn't inherit any
    // state from a prior invocation.
    reset_arena();

    // P = 0 (LP)
    let P: CscMatrix<T> = CscMatrix::<T>::zeros((2, 2));

    // q = [1, -1]
    let q: Vec<T> = vec![rat_from_int(1), -rat_from_int(1)];

    // A = [I; -I]  (2-d box separated into 4 inequalities)
    // Build A by hand with rational entries.
    let one: T = rat_from_int(1);
    let neg_one: T = -rat_from_int(1);
    let A = CscMatrix::new(
        4,                                     // m
        2,                                     // n
        vec![0, 2, 4],                         // colptr
        vec![0, 2, 1, 3],                      // rowval
        vec![one, neg_one, one, neg_one],      // nzval
    );

    let b: Vec<T> = vec![rat_from_int(1); 4];

    let cones = [NonnegativeConeT(4)];

    // Settings: the float-typed defaults need RationalReal versions.
    // tol_feas / tol_gap_* are T-typed in DefaultSettings, so we must
    // construct them explicitly. Use 1e-8 as a rational: 1/100_000_000.
    let tol = rat_from_pair(1, 100_000_000);

    let settings = DefaultSettingsBuilder::<T>::default()
        .equilibrate_enable(true)
        .max_iter(50)
        .tol_feas(tol)
        .tol_gap_abs(tol)
        .tol_gap_rel(tol)
        .build()
        .unwrap();

    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();

    solver.solve();

    // Extract the primal solution as exact rationals.
    println!();
    println!("=== Exact-rational primal solution ===");
    for (i, xi) in solver.solution.x.iter().enumerate() {
        let (num, den) = xi.into_pair();
        println!("  x[{i}] = {num} / {den}   (≈ {})", xi.to_f64());
    }

    // Objective value, exact.
    let obj = solver.solution.obj_val;
    let (num, den) = obj.into_pair();
    println!("  obj  = {num} / {den}   (≈ {})", obj.to_f64());

    // Diagnostics: arena size + max numerator/denominator bit-length
    // across the primal solution.
    let max_bits = solver
        .solution
        .x
        .iter()
        .map(|xi| xi.max_bits())
        .max()
        .unwrap_or(0);
    println!();
    println!(
        "Arena size: {} entries; max(numer_bits, denom_bits) across primal x: {}",
        arena_len(),
        max_bits
    );

    // Recover memory before exit (would matter for long-running processes).
    reset_arena();
}
