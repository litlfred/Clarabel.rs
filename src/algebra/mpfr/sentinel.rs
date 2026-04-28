//! `RealSentinel` and `RealConst` for [`MpfrFloat`].
//!
//! Unlike the rational backend, MPFR has native infinity / nan / signed
//! zero, so these all forward directly to `rug::Float` semantics.
//! `RealConst` (PI/SQRT_2/FRAC_1_SQRT_2) uses MPFR's built-in constants.

use super::precision::default_precision;
use super::real::MpfrFloat;
use crate::algebra::transcendental::{RealConst, RealSentinel};
use rug::float::{Constant, Special};
use rug::Float as RugFloat;

impl RealSentinel for MpfrFloat {
    fn infinity() -> Self {
        MpfrFloat(RugFloat::with_val(default_precision(), Special::Infinity))
    }
    fn neg_infinity() -> Self {
        MpfrFloat(RugFloat::with_val(default_precision(), Special::NegInfinity))
    }
    fn nan() -> Self {
        MpfrFloat(RugFloat::with_val(default_precision(), Special::Nan))
    }
    fn epsilon() -> Self {
        // 2^-(prec-1) — one ULP at the working precision.
        // Formed as 1.0 right-shifted by (p-1) bits.
        let p = default_precision();
        let mut out = RugFloat::with_val(p, 1);
        out >>= (p - 1) as i32;
        MpfrFloat(out)
    }
    fn max_value() -> Self {
        MpfrFloat(RugFloat::with_val(default_precision(), f64::MAX))
    }
    fn min_value() -> Self {
        Self::epsilon()
    }
    fn is_nan(self) -> bool {
        self.0.is_nan()
    }
    fn is_finite(self) -> bool {
        self.0.is_finite()
    }
    fn is_infinite(self) -> bool {
        self.0.is_infinite()
    }
    fn is_sign_negative(self) -> bool {
        self.0.is_sign_negative()
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

#[allow(non_snake_case)]
impl RealConst for MpfrFloat {
    fn PI() -> Self {
        MpfrFloat(RugFloat::with_val(default_precision(), Constant::Pi))
    }
    fn SQRT_2() -> Self {
        let two = RugFloat::with_val(default_precision(), 2);
        MpfrFloat(two.sqrt())
    }
    fn FRAC_1_SQRT_2() -> Self {
        let two = RugFloat::with_val(default_precision(), 2);
        let s = two.sqrt();
        let one = RugFloat::with_val(default_precision(), 1);
        MpfrFloat(one / s)
    }
}
