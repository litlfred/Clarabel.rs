//! [`RationalReal`] — Copy-able arena handle that satisfies `FloatT`.
//!
//! See `super` (mod.rs) for the design rationale and Send + Sync invariant.

use super::arena;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};
use std::cmp::Ordering;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, Sub, SubAssign};

/// Exact-rational scalar handle.
///
/// 4-byte `Copy` index into the thread-local arena that owns the
/// `BigRational` value. See [`super`] (the `rational` module docstring)
/// for the memory model and invariants.
///
/// # Lifecycle
/// Construct via [`RationalReal::from_bigrational`], [`from_pair`], or
/// the `From<f64>`/`From<i64>` impls. Use arithmetic, comparisons, and
/// conversions inline. Extract a final value via
/// [`into_pair`](Self::into_pair) (consuming `(BigInt, BigInt)`),
/// [`to_bigrational`](Self::to_bigrational) (cloning), or
/// [`to_f64`](Self::to_f64) (lossy).
///
/// # Send + Sync
/// `RationalReal` is declared `Send + Sync` via `unsafe impl` because
/// `FloatT` requires it. The runtime invariant: a `RationalReal` may
/// only be dereferenced on the thread that produced it. Sending a
/// `RationalReal` across threads in user code is a logic bug; the
/// destination thread's arena does not contain the indexed value and a
/// debug-mode access will panic out of bounds.
#[derive(Copy, Clone)]
pub struct RationalReal(pub(crate) u32);

// SAFETY: enforced by the per-thread-arena invariant. See module docs.
unsafe impl Send for RationalReal {}
// SAFETY: same.
unsafe impl Sync for RationalReal {}

impl RationalReal {
    /// Construct from an owned `BigRational`. The value is moved into
    /// the thread-local arena.
    pub fn from_bigrational(value: BigRational) -> Self {
        RationalReal(arena::push(value))
    }

    /// Construct from a numerator/denominator pair. Panics if `den == 0`.
    pub fn from_pair(numer: BigInt, denom: BigInt) -> Self {
        Self::from_bigrational(BigRational::new(numer, denom))
    }

    /// Clone the underlying `BigRational` out of the arena.
    pub fn to_bigrational(&self) -> BigRational {
        arena::get(self.0)
    }

    /// Consume into the `(numerator, denominator)` pair (owned).
    pub fn into_pair(self) -> (BigInt, BigInt) {
        let r = arena::get(self.0);
        let (n, d) = r.into_raw();
        (n, d)
    }

    /// Lossy conversion to `f64`. Returns `f64::NAN` if the rational
    /// magnitude exceeds `f64::MAX` or if rounding fails (rare; the
    /// `num-rational` impl returns `None` only in extreme edge cases).
    pub fn to_f64(&self) -> f64 {
        use num_traits::ToPrimitive;
        arena::with(self.0, |r| r.to_f64().unwrap_or(f64::NAN))
    }

    /// Read-only access to the numerator. Returns an owned clone because
    /// the arena borrow doesn't outlive the closure.
    pub fn numer(&self) -> BigInt {
        arena::with(self.0, |r| r.numer().clone())
    }

    /// Read-only access to the denominator.
    pub fn denom(&self) -> BigInt {
        arena::with(self.0, |r| r.denom().clone())
    }

    /// Maximum bit-length across numerator and denominator. Cheap; one
    /// `BigInt::bits()` call each. Useful for the per-iteration
    /// denominator-bit-length log requested by QOU.
    pub fn max_bits(&self) -> u64 {
        arena::with(self.0, |r| r.numer().bits().max(r.denom().bits()))
    }
}

// ============================================================
// Arithmetic — every op pushes a fresh BigRational into the arena
// ============================================================

impl Add for RationalReal {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let v = arena::with2(self.0, rhs.0, |a, b| a + b);
        RationalReal::from_bigrational(v)
    }
}

impl Sub for RationalReal {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let v = arena::with2(self.0, rhs.0, |a, b| a - b);
        RationalReal::from_bigrational(v)
    }
}

impl Mul for RationalReal {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let v = arena::with2(self.0, rhs.0, |a, b| a * b);
        RationalReal::from_bigrational(v)
    }
}

impl Div for RationalReal {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let v = arena::with2(self.0, rhs.0, |a, b| a / b);
        RationalReal::from_bigrational(v)
    }
}

impl Rem for RationalReal {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: Self) -> Self {
        let v = arena::with2(self.0, rhs.0, |a, b| a % b);
        RationalReal::from_bigrational(v)
    }
}

impl Neg for RationalReal {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let v = arena::with(self.0, |a| -a);
        RationalReal::from_bigrational(v)
    }
}

impl AddAssign for RationalReal {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for RationalReal {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for RationalReal {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl DivAssign for RationalReal {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

// ============================================================
// Comparisons — read both operands by reference
// ============================================================

impl PartialEq for RationalReal {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        if self.0 == other.0 {
            return true; // same handle, same value (cheap fast path)
        }
        arena::with2(self.0, other.0, |a, b| a == b)
    }
}

impl Eq for RationalReal {}

impl PartialOrd for RationalReal {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RationalReal {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        if self.0 == other.0 {
            return Ordering::Equal;
        }
        arena::with2(self.0, other.0, |a, b| a.cmp(b))
    }
}

// ============================================================
// num_traits::Zero / One — required by Num
// ============================================================

impl Zero for RationalReal {
    #[inline]
    fn zero() -> Self {
        RationalReal(arena::push_zero())
    }
    #[inline]
    fn is_zero(&self) -> bool {
        arena::with(self.0, |a| a.is_zero())
    }
}

impl One for RationalReal {
    #[inline]
    fn one() -> Self {
        RationalReal(arena::push_one())
    }
}

// ============================================================
// Default — required by CoreFloatT
// ============================================================

impl Default for RationalReal {
    #[inline]
    fn default() -> Self {
        Self::zero()
    }
}
