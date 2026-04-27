//! Thread-local working precision for `RationalReal` transcendentals.
//!
//! `BigRational` arithmetic (`+`, `-`, `*`, `/`) is exact and unbounded;
//! it does not consult this value. The transcendental ops (`sqrt`, `ln`,
//! `exp`, `powf`) are inexact — they evaluate Newton/Taylor iterations
//! to a tolerance derived from this many bits of fractional precision.
//!
//! Default: 128 bits (~38 decimal digits). Set higher for QOU's H_n
//! confinement work where R5_FULL_PLAN.md asks for ≥ 50 dps.

use std::cell::Cell;

const DEFAULT_PRECISION_BITS: u32 = 128;

thread_local! {
    static PRECISION_BITS: Cell<u32> = const { Cell::new(DEFAULT_PRECISION_BITS) };
}

/// Get the current working precision in bits for transcendentals on this
/// thread. Default is 128 bits.
pub fn precision_bits() -> u32 {
    PRECISION_BITS.with(|p| p.get())
}

/// Set the working precision in bits for transcendentals on this thread.
/// Returns the previous value so callers can restore it (use
/// [`with_precision`] for a scope-guarded version).
///
/// Panics if `bits == 0`.
pub fn set_precision_bits(bits: u32) -> u32 {
    assert!(bits > 0, "RationalReal precision must be positive");
    PRECISION_BITS.with(|p| p.replace(bits))
}

/// Run a closure with the working precision temporarily set to `bits`,
/// restoring the previous value on return. Resets unconditionally even on
/// panic (the closure runs in a guard).
pub fn with_precision<R>(bits: u32, f: impl FnOnce() -> R) -> R {
    struct Guard(u32);
    impl Drop for Guard {
        fn drop(&mut self) {
            PRECISION_BITS.with(|p| p.set(self.0));
        }
    }
    let prev = set_precision_bits(bits);
    let _guard = Guard(prev);
    f()
}
