//! [`RationalReal`] — Copy-able arena handle that satisfies `FloatT`.
//!
//! See `super` (mod.rs) for the design rationale and Send + Sync invariant.

use super::arena;
use super::cap::maybe_cap;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{FromPrimitive, Num, One, Signed, Zero};
use std::cmp::Ordering;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

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
        RationalReal::from_bigrational(maybe_cap(v))
    }
}

impl Sub for RationalReal {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let v = arena::with2(self.0, rhs.0, |a, b| a - b);
        RationalReal::from_bigrational(maybe_cap(v))
    }
}

impl Mul for RationalReal {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let v = arena::with2(self.0, rhs.0, |a, b| a * b);
        RationalReal::from_bigrational(maybe_cap(v))
    }
}

impl Div for RationalReal {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let v = arena::with2(self.0, rhs.0, |a, b| a / b);
        RationalReal::from_bigrational(maybe_cap(v))
    }
}

impl Rem for RationalReal {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: Self) -> Self {
        let v = arena::with2(self.0, rhs.0, |a, b| a % b);
        RationalReal::from_bigrational(maybe_cap(v))
    }
}

impl Neg for RationalReal {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        // Neg never grows numerator/denominator bits, so no cap needed.
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

impl RemAssign for RationalReal {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
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

// ============================================================
// num_traits::Num — required by CoreFloatT (via Num + NumAssign)
// ============================================================

/// Error returned by [`RationalReal::from_str_radix`] when the input
/// cannot be parsed as a rational at the given radix.
#[derive(Debug, Clone)]
pub struct ParseRationalError(String);

impl std::fmt::Display for ParseRationalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RationalReal parse error: {}", self.0)
    }
}

impl std::error::Error for ParseRationalError {}

impl Num for RationalReal {
    type FromStrRadixErr = ParseRationalError;

    /// Parse `"<numer>/<denom>"` or just `"<numer>"`. Both fields go
    /// through `BigInt::from_str_radix`.
    fn from_str_radix(s: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        let (n, d) = match s.split_once('/') {
            Some((ns, ds)) => (ns, ds),
            None => (s, "1"),
        };
        let numer = BigInt::parse_bytes(n.as_bytes(), radix)
            .ok_or_else(|| ParseRationalError(format!("invalid numerator: {n}")))?;
        let denom = BigInt::parse_bytes(d.as_bytes(), radix)
            .ok_or_else(|| ParseRationalError(format!("invalid denominator: {d}")))?;
        if denom.is_zero() {
            return Err(ParseRationalError("zero denominator".into()));
        }
        Ok(Self::from_pair(numer, denom))
    }
}

// ============================================================
// num_traits::Signed — abs / signum / is_positive / is_negative
// ============================================================

impl Signed for RationalReal {
    #[inline]
    fn abs(&self) -> Self {
        let v = arena::with(self.0, |a| a.abs());
        RationalReal::from_bigrational(v)
    }

    #[inline]
    fn abs_sub(&self, other: &Self) -> Self {
        if self <= other {
            Self::zero()
        } else {
            *self - *other
        }
    }

    #[inline]
    fn signum(&self) -> Self {
        let v = arena::with(self.0, |a| a.signum());
        RationalReal::from_bigrational(v)
    }

    #[inline]
    fn is_positive(&self) -> bool {
        arena::with(self.0, |a| a.is_positive())
    }

    #[inline]
    fn is_negative(&self) -> bool {
        arena::with(self.0, |a| a.is_negative())
    }
}

// ============================================================
// num_traits::FromPrimitive — used by AsFloatT (T::from_usize etc)
// ============================================================

impl FromPrimitive for RationalReal {
    fn from_i64(n: i64) -> Option<Self> {
        Some(Self::from_pair(BigInt::from(n), BigInt::from(1)))
    }
    fn from_u64(n: u64) -> Option<Self> {
        Some(Self::from_pair(BigInt::from(n), BigInt::from(1)))
    }
    fn from_i128(n: i128) -> Option<Self> {
        Some(Self::from_pair(BigInt::from(n), BigInt::from(1)))
    }
    fn from_u128(n: u128) -> Option<Self> {
        Some(Self::from_pair(BigInt::from(n), BigInt::from(1)))
    }
    fn from_isize(n: isize) -> Option<Self> {
        Self::from_i64(n as i64)
    }
    fn from_usize(n: usize) -> Option<Self> {
        Self::from_u64(n as u64)
    }
    fn from_i32(n: i32) -> Option<Self> {
        Self::from_i64(n as i64)
    }
    fn from_u32(n: u32) -> Option<Self> {
        Self::from_u64(n as u64)
    }
    /// `f32` is converted exactly into `(mantissa * 2^exp_bias)` form so
    /// that, e.g., `from_f32(0.1)` is the *exact* rational the IEEE
    /// representation 0x3DCCCCCD denotes (not the round-trip back to 1/10).
    fn from_f32(f: f32) -> Option<Self> {
        BigRational::from_f32(f).map(Self::from_bigrational)
    }
    /// `f64` is converted exactly. See [`Self::from_f32`].
    fn from_f64(f: f64) -> Option<Self> {
        BigRational::from_f64(f).map(Self::from_bigrational)
    }
}

// ============================================================
// From conversions for ergonomics
// ============================================================

impl From<BigRational> for RationalReal {
    fn from(v: BigRational) -> Self {
        Self::from_bigrational(v)
    }
}

impl From<i64> for RationalReal {
    fn from(n: i64) -> Self {
        Self::from_pair(BigInt::from(n), BigInt::from(1))
    }
}

impl From<f64> for RationalReal {
    fn from(f: f64) -> Self {
        Self::from_f64(f)
            .unwrap_or_else(|| panic!("cannot convert non-finite f64 ({f}) to RationalReal"))
    }
}
