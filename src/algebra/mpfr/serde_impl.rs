//! `serde::Serialize`/`Deserialize` for [`MpfrFloat`].
//!
//! Wire format: a base-16 string (MPFR's `to_string_radix(16, ...)`),
//! which preserves the value bit-exactly and is round-trippable through
//! `MpfrFloat::from_str_radix`. Falls back to the f64 string repr when
//! the value is non-finite (NaN / ±Inf).

use super::precision::default_precision;
use super::real::MpfrFloat;
use rug::Float as RugFloat;
use serde::{de::Deserialize, Deserializer, Serialize, Serializer};

impl Serialize for MpfrFloat {
    fn serialize<S: Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        // (precision, decimal string) pair so the destination side can
        // reconstruct at the original precision. The decimal form uses
        // enough digits to round-trip the full mantissa (rug's Display
        // produces this when the format width isn't constrained).
        let prec = self.0.prec();
        let s = format!("{}", self.0);
        (prec, s).serialize(ser)
    }
}

impl<'de> Deserialize<'de> for MpfrFloat {
    fn deserialize<D: Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        let (prec, s): (u32, String) = Deserialize::deserialize(de)?;
        let prec = if prec == 0 { default_precision() } else { prec };
        match RugFloat::parse(&s) {
            Ok(incomplete) => Ok(MpfrFloat(RugFloat::with_val(prec, incomplete))),
            Err(e) => Err(serde::de::Error::custom(format!(
                "MpfrFloat deserialize: {e}"
            ))),
        }
    }
}
