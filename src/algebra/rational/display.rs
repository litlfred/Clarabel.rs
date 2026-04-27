//! `Display`/`LowerExp`/`Debug` for [`RationalReal`].
//!
//! - `Debug`: prints `RationalReal(numer/denom)` for diagnostics.
//! - `Display`: prints `numer/denom` (matches `BigRational`'s default).
//! - `LowerExp`: prints decimal scientific notation, lossily routed
//!   through `to_f64`. The solver's iter-print format strings (e.g.
//!   `{:+8.4e}`) only need rough magnitude info, and routing through
//!   `f64` keeps the print path cheap (no decimal expansion of the
//!   bigints). Users who want the exact value should call
//!   [`RationalReal::numer`]/[`RationalReal::denom`] or
//!   [`RationalReal::into_pair`] directly.

use super::arena;
use super::real::RationalReal;
use std::fmt;

impl fmt::Debug for RationalReal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        arena::with(self.0, |r| write!(f, "RationalReal({}/{})", r.numer(), r.denom()))
    }
}

impl fmt::Display for RationalReal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        arena::with(self.0, |r| write!(f, "{}/{}", r.numer(), r.denom()))
    }
}

impl fmt::LowerExp for RationalReal {
    /// Print decimal-scientific via lossy `to_f64`. Format flags
    /// (precision, sign, width) propagate from the caller.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let v = self.to_f64();
        fmt::LowerExp::fmt(&v, f)
    }
}
