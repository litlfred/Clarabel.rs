//! Thread-local arena that owns the `BigRational` values referenced by
//! [`RationalReal`](super::real::RationalReal) handles.
//!
//! Design: `RationalReal` is a `Copy`-able 4-byte handle (`u32` index).
//! Every arithmetic op pushes a freshly-computed `BigRational` onto the
//! arena and returns a new handle. The arena grows monotonically during
//! a solve and is reset between solves (or explicitly via [`reset`]).
//!
//! See `src/algebra/rational/mod.rs` for the higher-level rationale and
//! the `Send + Sync` invariants.

use num_bigint::BigInt;
use num_rational::BigRational;
use std::cell::RefCell;

thread_local! {
    /// Per-thread storage. `Vec` (not `HashMap`) because handles are dense
    /// `u32` indices so a Vec lookup is O(1) with no hashing.
    static ARENA: RefCell<Vec<BigRational>> = const { RefCell::new(Vec::new()) };
}

/// Push a `BigRational` into the thread-local arena and return its handle.
///
/// Panics on overflow if the arena would exceed `u32::MAX` entries —
/// at that point the arena is ~16 GB minimum and a `reset()` is overdue.
#[inline]
pub(crate) fn push(value: BigRational) -> u32 {
    ARENA.with(|cell| {
        let mut a = cell.borrow_mut();
        let idx = a.len();
        if idx >= u32::MAX as usize {
            panic!(
                "RationalReal arena exhausted ({} entries). Call \
                 clarabel::algebra::rational::reset_arena() between solves.",
                idx
            );
        }
        a.push(value);
        idx as u32
    })
}

/// Read a clone of the value at `idx`.
///
/// `clone` here is one `Arc<...>`-like refcount touch internally to
/// `BigRational` (which is `num_rational::Ratio<BigInt>` — `BigInt` is
/// heap-allocated and gets a deep clone here, but we expect callers to
/// avoid reading the same handle multiple times in hot loops).
#[inline]
pub(crate) fn get(idx: u32) -> BigRational {
    ARENA.with(|cell| cell.borrow()[idx as usize].clone())
}

/// Apply a closure to a borrow of the value at `idx` without cloning.
///
/// Useful for predicates and `Display`/`LowerExp` — anything that only
/// needs to read the value, not own it.
#[inline]
pub(crate) fn with<R>(idx: u32, f: impl FnOnce(&BigRational) -> R) -> R {
    ARENA.with(|cell| f(&cell.borrow()[idx as usize]))
}

/// Apply a closure to two arena borrows at once.
///
/// Used by binary ops (Add/Mul/...) so we can compute the result without
/// two separate `with` calls and the intermediate clone.
#[inline]
pub(crate) fn with2<R>(a: u32, b: u32, f: impl FnOnce(&BigRational, &BigRational) -> R) -> R {
    ARENA.with(|cell| {
        let arena = cell.borrow();
        f(&arena[a as usize], &arena[b as usize])
    })
}

/// Reset the thread-local arena to empty. All outstanding `RationalReal`
/// handles on this thread become invalid; using one after `reset_arena()`
/// is a panic in debug builds (out-of-bounds index) or undefined behaviour
/// in release builds (unrelated `BigRational` returned at the recycled slot).
///
/// Call this between solves to recover memory.
pub fn reset_arena() {
    ARENA.with(|cell| cell.borrow_mut().clear());
}

/// Current arena size, in entries (not bytes). Useful for diagnostics
/// and the per-iteration denominator-bit-length log.
pub fn arena_len() -> usize {
    ARENA.with(|cell| cell.borrow().len())
}

/// Convenience: push the constant zero.
#[inline]
pub(crate) fn push_zero() -> u32 {
    push(BigRational::new(BigInt::from(0), BigInt::from(1)))
}

/// Convenience: push the constant one.
#[inline]
pub(crate) fn push_one() -> u32 {
    push(BigRational::new(BigInt::from(1), BigInt::from(1)))
}
