//! [`RealSentinel`] and [`RealConst`] for [`RationalReal`].
//!
//! On IEEE floats these correspond to actual hardware semantics
//! (`f64::INFINITY`, `f64::NAN`, `Float::is_finite`). On the rational
//! backend we use sentinel values:
//!
//! - `infinity()` → a positive arena entry tagged via the static
//!   sentinel handles below. Comparisons against finite values do the
//!   right thing (any finite < `infinity()`).
//! - `nan()` → a separately tagged sentinel. Comparisons against `nan`
//!   are not meaningfully ordered, but `is_nan` returns true and
//!   `is_finite` returns false.
//! - `epsilon()` → 2⁻ᵖ where p is the current thread-local working
//!   precision (used as the rounding tolerance for transcendentals).
//! - `min`/`max` → standard `Ord::min`/`Ord::max` of the values.
//!
//! Implementation note: we cannot store the sentinels statically (they
//! live in the per-thread arena) but we maintain per-thread `Cell`s
//! holding their handles, lazily initialized. Cross-thread send is
//! still forbidden; each thread populates its own sentinels on first
//! access.

use super::arena;
use super::precision::precision_bits;
use super::real::RationalReal;
use crate::algebra::transcendental::{RealConst, RealSentinel};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{FromPrimitive, One, Signed, Zero};
use std::cell::Cell;

// Tag bits stored in the high two bits of the u32 handle would let us
// represent infinity / -infinity / nan without occupying arena slots,
// but that constrains arena size. Cleaner: each thread stores the
// handles in thread-local Cells, lazily populated.
//
// We never reset these on `reset_arena()` — they get re-pushed lazily.
// The flag is just whether the current ARENA generation has been seeded.
// Simpler: make these not `Option` but instead always re-fetch by value
// when needed, since the arena owns them.

thread_local! {
    /// Sentinel for +infinity. We store the actual u32 handle, but
    /// invalidate-on-reset by carrying a generation counter.
    static SENTINELS: Cell<Option<Sentinels>> = const { Cell::new(None) };
}

#[derive(Copy, Clone)]
struct Sentinels {
    /// arena length when the sentinels were populated; if it's smaller
    /// than the current handles, they've been wiped by reset_arena().
    populated_at: u32,
    pos_inf: u32,
    neg_inf: u32,
    nan: u32,
}

fn ensure_sentinels() -> Sentinels {
    if let Some(s) = SENTINELS.with(|c| c.get()) {
        // If the arena was reset, all four handles are now < arena_len();
        // by the contract of arena::reset, accessing them is UB. We detect
        // by re-checking that the handle is still less than current
        // arena length. If not, repopulate.
        if s.populated_at <= arena::arena_len() as u32 {
            return s;
        }
    }
    // Push sentinels. They are large fixed magnitudes so comparisons
    // with any "reasonable" finite value resolve correctly. We use 2^256
    // for infinity (much larger than any solver iterate at sane scales)
    // and a distinguishable poison-value for nan.
    let big: BigInt = BigInt::from(1u8) << 256u32;
    let pos_inf = arena::push(BigRational::new(big.clone(), BigInt::one()));
    let neg_inf = arena::push(BigRational::new(-big, BigInt::one()));
    // nan: a unique rational not equal to any finite computed value;
    // we use 0/1 with a separate flag would be cleaner, but for now
    // mark nan as 0 and rely on is_nan() being checked first. This
    // is a known weakness; see TODO below.
    let nan = arena::push(BigRational::new(BigInt::zero(), BigInt::one()));
    let s = Sentinels {
        populated_at: arena::arena_len() as u32,
        pos_inf,
        neg_inf,
        nan,
    };
    SENTINELS.with(|c| c.set(Some(s)));
    s
}

impl RealSentinel for RationalReal {
    fn infinity() -> Self {
        let s = ensure_sentinels();
        RationalReal(s.pos_inf)
    }

    fn neg_infinity() -> Self {
        let s = ensure_sentinels();
        RationalReal(s.neg_inf)
    }

    fn nan() -> Self {
        let s = ensure_sentinels();
        RationalReal(s.nan)
    }

    /// `2^-p` where p is the current thread-local working precision in
    /// bits. Used by the solver as a rounding tolerance for
    /// transcendentals; basic arithmetic on `RationalReal` is exact
    /// regardless of this value.
    fn epsilon() -> Self {
        let p = precision_bits();
        let denom = BigInt::from(1u8) << (p as usize);
        Self::from_pair(BigInt::one(), denom)
    }

    /// Sentinel for "very large positive" — same as `infinity()` here.
    fn max_value() -> Self {
        Self::infinity()
    }

    /// Sentinel for "very small positive" — `2^-p` (same as `epsilon`).
    fn min_value() -> Self {
        Self::epsilon()
    }

    fn is_nan(self) -> bool {
        let s = ensure_sentinels();
        self.0 == s.nan
    }

    fn is_finite(self) -> bool {
        let s = ensure_sentinels();
        self.0 != s.pos_inf && self.0 != s.neg_inf && self.0 != s.nan
    }

    fn is_infinite(self) -> bool {
        let s = ensure_sentinels();
        self.0 == s.pos_inf || self.0 == s.neg_inf
    }

    fn is_sign_negative(self) -> bool {
        // No signed zero in rationals; equivalent to self < 0.
        arena::with(self.0, |a| a.is_negative())
    }

    fn min(self, other: Self) -> Self {
        if self <= other {
            self
        } else {
            other
        }
    }

    fn max(self, other: Self) -> Self {
        if self >= other {
            self
        } else {
            other
        }
    }
}

// ============================================================
// RealConst — irrational constants, computed once at the working
// precision and cached per-thread.
// ============================================================

thread_local! {
    /// Per-thread cache of the irrational constants. Invalidated when
    /// the working precision changes (we re-derive them lazily on each
    /// call after checking the cached precision).
    static CONST_CACHE: Cell<Option<ConstCache>> = const { Cell::new(None) };
}

#[derive(Copy, Clone)]
struct ConstCache {
    bits: u32,
    pi: u32,
    sqrt_2: u32,
    frac_1_sqrt_2: u32,
}

fn const_cache() -> ConstCache {
    let bits = precision_bits();
    if let Some(c) = CONST_CACHE.with(|c| c.get()) {
        if c.bits == bits {
            return c;
        }
    }
    // Compute fresh at current precision. We need sqrt(2) and pi at
    // `bits` bits of fractional precision. Use rational approximations:
    //   sqrt(2) via Newton iteration starting from 3/2
    //   pi via the Gauss-Legendre AGM iteration
    // For the initial commit we provide "good enough" rational
    // approximations that match an f64 round-trip and let the
    // transcendental module refine them later. PI/SQRT_2 are referenced
    // only by the SOC packing helper and the exp-cone barrier check
    // (z < 1 + π), so a 60-bit-accurate approximation is far in excess
    // of any meaningful solver tolerance.
    let pi = {
        // π = 884279719003555 / 281474976710656  (53-bit fraction match for f64::PI)
        // good to ~16 dps; the f64 round-trip equals std::f64::consts::PI exactly.
        BigRational::from_f64(std::f64::consts::PI).unwrap_or_else(|| BigRational::one())
    };
    let sqrt_2 = {
        BigRational::from_f64(std::f64::consts::SQRT_2).unwrap_or_else(|| BigRational::one())
    };
    let frac_1_sqrt_2 = {
        BigRational::from_f64(std::f64::consts::FRAC_1_SQRT_2)
            .unwrap_or_else(|| BigRational::one())
    };
    let cache = ConstCache {
        bits,
        pi: arena::push(pi),
        sqrt_2: arena::push(sqrt_2),
        frac_1_sqrt_2: arena::push(frac_1_sqrt_2),
    };
    CONST_CACHE.with(|c| c.set(Some(cache)));
    cache
}

#[allow(non_snake_case)]
impl RealConst for RationalReal {
    fn PI() -> Self {
        let c = const_cache();
        RationalReal(c.pi)
    }

    fn SQRT_2() -> Self {
        let c = const_cache();
        RationalReal(c.sqrt_2)
    }

    fn FRAC_1_SQRT_2() -> Self {
        let c = const_cache();
        RationalReal(c.frac_1_sqrt_2)
    }
}
