//! `serde::Serialize` / `Deserialize` for [`RationalReal`].
//!
//! `RationalReal` is an arena handle, so we cannot serialize the
//! `u32` index directly — the receiving end's arena does not contain
//! that slot. Instead we serialize the underlying `BigRational` value
//! (which `num_rational` already supports via its `serde` feature
//! — bit-exact `(numer, denom)` pair). On deserialize we push the
//! value into the destination thread's arena and return a fresh
//! handle.
//!
//! Format compatibility: the on-the-wire form is exactly
//! `BigRational`'s serde format. JSON looks like `[numer_str, denom_str]`.

use super::arena;
use super::real::RationalReal;
use num_rational::BigRational;
use serde::{de::Deserialize, Deserializer, Serialize, Serializer};

impl Serialize for RationalReal {
    fn serialize<S: Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        // Borrow the arena value and forward to BigRational's Serialize.
        arena::with(self.0, |r| r.serialize(ser))
    }
}

impl<'de> Deserialize<'de> for RationalReal {
    fn deserialize<D: Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        let r = BigRational::deserialize(de)?;
        Ok(RationalReal::from_bigrational(r))
    }
}
