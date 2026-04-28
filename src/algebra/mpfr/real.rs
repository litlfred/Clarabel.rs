//! [`MpfrFloat`] — newtype around [`rug::Float`] satisfying `FloatT`.
//!
//! Unlike `RationalReal`, MPFR floats live directly in the value (no
//! arena). `rug::Float` is `Send + Sync + Clone` — but **not `Copy`**
//! because MPFR allocates limb storage on the heap. The `CoreFloatT`
//! impl bound is `Clone`, not `Copy`, so this works.

use super::precision::default_precision;
use num_traits::{FromPrimitive, Num, One, Signed, Zero};
use rug::Float as RugFloat;
use std::cmp::Ordering;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// MPFR-precision floating-point scalar.
///
/// Wraps [`rug::Float`]. New values get their precision from
/// [`default_precision`](super::default_precision) (default 167 bits
/// ≈ 50 dps). Binary ops use the max of the two operand precisions.
#[derive(Clone)]
pub struct MpfrFloat(pub(crate) RugFloat);

impl MpfrFloat {
    /// Construct from an owned `rug::Float`. Carries the rug value's
    /// own precision; doesn't reset to the thread default.
    pub fn from_rug(f: RugFloat) -> Self {
        MpfrFloat(f)
    }

    /// Borrow the underlying `rug::Float`.
    pub fn as_rug(&self) -> &RugFloat {
        &self.0
    }

    /// Consume into the underlying `rug::Float`.
    pub fn into_rug(self) -> RugFloat {
        self.0
    }

    /// Construct an `MpfrFloat` with explicit precision and value 0.
    pub fn zero_with_prec(prec: u32) -> Self {
        MpfrFloat(RugFloat::new(prec))
    }

    /// Construct an `MpfrFloat` at the thread's default precision
    /// from any `rug` source value (i64/u64/f64/Rational/...).
    pub fn with_val<V>(value: V) -> Self
    where
        RugFloat: rug::Assign<V>,
    {
        let mut f = RugFloat::new(default_precision());
        rug::Assign::assign(&mut f, value);
        MpfrFloat(f)
    }

    /// Lossy conversion to `f64`.
    pub fn to_f64(&self) -> f64 {
        self.0.to_f64()
    }

    /// Precision of this value, in bits.
    pub fn prec(&self) -> u32 {
        self.0.prec()
    }
}

#[inline]
fn binop_prec(a: &RugFloat, b: &RugFloat) -> u32 {
    a.prec().max(b.prec())
}

// ============================================================
// Arithmetic
// ============================================================

impl Add for MpfrFloat {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let p = binop_prec(&self.0, &rhs.0);
        MpfrFloat(RugFloat::with_val(p, &self.0 + &rhs.0))
    }
}

impl Sub for MpfrFloat {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let p = binop_prec(&self.0, &rhs.0);
        MpfrFloat(RugFloat::with_val(p, &self.0 - &rhs.0))
    }
}

impl Mul for MpfrFloat {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let p = binop_prec(&self.0, &rhs.0);
        MpfrFloat(RugFloat::with_val(p, &self.0 * &rhs.0))
    }
}

impl Div for MpfrFloat {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let p = binop_prec(&self.0, &rhs.0);
        MpfrFloat(RugFloat::with_val(p, &self.0 / &rhs.0))
    }
}

impl Rem for MpfrFloat {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: Self) -> Self {
        let p = binop_prec(&self.0, &rhs.0);
        // rug doesn't expose a free-function remainder constructor; do
        // it via .remainder_round on a clone.
        let mut out = RugFloat::with_val(p, &self.0);
        out.remainder_round(&rhs.0, rug::float::Round::Nearest);
        MpfrFloat(out)
    }
}

impl Neg for MpfrFloat {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        MpfrFloat(-self.0)
    }
}

impl AddAssign for MpfrFloat {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl SubAssign for MpfrFloat {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl MulAssign for MpfrFloat {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl DivAssign for MpfrFloat {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0;
    }
}

impl RemAssign for MpfrFloat {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        let v = self.clone() % rhs;
        *self = v;
    }
}

// ============================================================
// Comparisons
// ============================================================

impl PartialEq for MpfrFloat {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl PartialOrd for MpfrFloat {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

// ============================================================
// num_traits
// ============================================================

impl Zero for MpfrFloat {
    #[inline]
    fn zero() -> Self {
        MpfrFloat(RugFloat::new(default_precision()))
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl One for MpfrFloat {
    #[inline]
    fn one() -> Self {
        MpfrFloat(RugFloat::with_val(default_precision(), 1))
    }
}

impl Default for MpfrFloat {
    #[inline]
    fn default() -> Self {
        Self::zero()
    }
}

#[derive(Debug, Clone)]
pub struct ParseMpfrError(String);

impl std::fmt::Display for ParseMpfrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MpfrFloat parse error: {}", self.0)
    }
}

impl std::error::Error for ParseMpfrError {}

impl Num for MpfrFloat {
    type FromStrRadixErr = ParseMpfrError;
    fn from_str_radix(s: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        RugFloat::parse_radix(s, radix as i32)
            .map(|incomplete| {
                MpfrFloat(RugFloat::with_val(default_precision(), incomplete))
            })
            .map_err(|e| ParseMpfrError(format!("{e}")))
    }
}

impl Signed for MpfrFloat {
    #[inline]
    fn abs(&self) -> Self {
        MpfrFloat(self.0.clone().abs())
    }
    #[inline]
    fn abs_sub(&self, other: &Self) -> Self {
        if self <= other {
            Self::zero()
        } else {
            self.clone() - other.clone()
        }
    }
    #[inline]
    fn signum(&self) -> Self {
        if self.0.is_zero() {
            Self::zero()
        } else if self.0.is_sign_negative() {
            -Self::one()
        } else {
            Self::one()
        }
    }
    #[inline]
    fn is_positive(&self) -> bool {
        !self.0.is_zero() && !self.0.is_sign_negative()
    }
    #[inline]
    fn is_negative(&self) -> bool {
        !self.0.is_zero() && self.0.is_sign_negative()
    }
}

impl FromPrimitive for MpfrFloat {
    fn from_i64(n: i64) -> Option<Self> {
        Some(MpfrFloat(RugFloat::with_val(default_precision(), n)))
    }
    fn from_u64(n: u64) -> Option<Self> {
        Some(MpfrFloat(RugFloat::with_val(default_precision(), n)))
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
    fn from_f32(f: f32) -> Option<Self> {
        Some(MpfrFloat(RugFloat::with_val(default_precision(), f)))
    }
    fn from_f64(f: f64) -> Option<Self> {
        Some(MpfrFloat(RugFloat::with_val(default_precision(), f)))
    }
}

impl From<f64> for MpfrFloat {
    fn from(f: f64) -> Self {
        Self::from_f64(f).expect("f64 -> MpfrFloat is always Some")
    }
}

impl From<i64> for MpfrFloat {
    fn from(n: i64) -> Self {
        Self::from_i64(n).expect("i64 -> MpfrFloat is always Some")
    }
}

impl From<RugFloat> for MpfrFloat {
    fn from(f: RugFloat) -> Self {
        MpfrFloat(f)
    }
}

// ============================================================
// Display / LowerExp / Debug
// ============================================================

impl std::fmt::Debug for MpfrFloat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MpfrFloat({})", self.0)
    }
}

impl std::fmt::Display for MpfrFloat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl std::fmt::LowerExp for MpfrFloat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Forward via f64 for compactness in iter-print logs. The
        // actual MPFR value retains its full working precision; this
        // is only the print path.
        std::fmt::LowerExp::fmt(&self.0.to_f64(), f)
    }
}

// ============================================================
// BitWidthDiagnostic
// ============================================================

impl crate::algebra::transcendental::BitWidthDiagnostic for MpfrFloat {
    /// Returns `(mantissa_prec_bits, 0)`. MPFR floats have fixed
    /// per-value precision so the "denominator bits" axis is zero;
    /// the "numerator" axis surfaces the configured working precision
    /// for diagnostic comparison with the rational backend.
    #[inline]
    fn bit_width(&self) -> (u64, u64) {
        (self.0.prec() as u64, 0)
    }
}
