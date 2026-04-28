//! `Transcendental` impl for [`MpfrFloat`].
//!
//! All ops forward to MPFR's native correctly-rounded transcendentals.
//! No precision capping needed (rug's working precision is
//! per-`Float`); each op preserves the operand's precision (or the
//! larger of two for binary ops) like the basic arithmetic does.

use super::real::MpfrFloat;
use crate::algebra::transcendental::Transcendental;
use rug::ops::Pow;
use rug::Float as RugFloat;

impl Transcendental for MpfrFloat {
    fn sqrt(self) -> Self {
        MpfrFloat(self.0.sqrt())
    }
    fn ln(self) -> Self {
        MpfrFloat(self.0.ln())
    }
    fn exp(self) -> Self {
        MpfrFloat(self.0.exp())
    }
    fn powf(self, n: Self) -> Self {
        MpfrFloat(self.0.pow(&n.0))
    }
    fn powi(self, n: i32) -> Self {
        MpfrFloat(self.0.pow(n))
    }
    fn recip(self) -> Self {
        let p = self.0.prec();
        let one = RugFloat::with_val(p, 1);
        MpfrFloat(one / self.0)
    }
    fn sin(self) -> Self {
        MpfrFloat(self.0.sin())
    }
    fn cos(self) -> Self {
        MpfrFloat(self.0.cos())
    }
    fn atan2(self, x: Self) -> Self {
        MpfrFloat(self.0.atan2(&x.0))
    }
}
