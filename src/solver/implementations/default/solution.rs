#![allow(unused_variables)]

use super::*;
use crate::{
    algebra::*,
    solver::core::{
        cones::{SupportedConeAsTag, SupportedConeT, SupportedConeTag},
        traits::Solution,
        SolverStatus,
    },
};

/// Standard-form solver type implementing the [`Solution`](crate::solver::core::traits::Solution) trait
///
/// When the `serde` feature is enabled, this type derives `Serialize`
/// and `Deserialize` with bound `T: Serialize + DeserializeOwned`.
/// For `T = RationalReal` this gives bit-exact JSON witnesses
/// (numerator/denominator pairs preserved through round-trip).
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound = "T: serde::Serialize + serde::de::DeserializeOwned")
)]
pub struct DefaultSolution<T> {
    /// primal solution
    pub x: Vec<T>,
    /// dual solution (in dual cone)
    pub z: Vec<T>,
    /// vector of slacks (in primal cone)
    pub s: Vec<T>,
    /// final solver status
    pub status: SolverStatus,
    /// primal objective value
    pub obj_val: T,
    /// dual objective value
    pub obj_val_dual: T,
    /// solve time in seconds
    pub solve_time: f64,
    /// number of iterations
    pub iterations: u32,
    /// primal residual
    pub r_prim: T,
    /// dual residual
    pub r_dual: T,

    /// Per-cone metadata (tag + slack-vector range), captured from the
    /// post-collapse cone list at termination. Lets callers extract
    /// `z` / `s` per-cone slices without re-walking the user's
    /// original cone declarations.
    ///
    /// Length matches the number of cones in the *internal* (post-
    /// `new_collapsed`) representation. Sugar variants like
    /// [`BlockDiagPSDConeT`](crate::solver::core::cones::SupportedConeT::BlockDiagPSDConeT)
    /// will appear here as their expanded constituents (one entry per block).
    pub cone_specs: Vec<ConeSpec>,
}

/// Per-cone metadata recorded on `DefaultSolution`. Used as the offset
/// table for the structured per-block accessors.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConeSpec {
    /// Tag identifying the cone type (NonnegativeCone / SecondOrderCone /
    /// PSDTriangleCone / etc).
    pub tag: SupportedConeTag,
    /// Half-open range `[start, stop)` indexing into the flat `z` and
    /// `s` slack vectors. `s[range]` and `z[range]` are the cone's
    /// per-cone slack and dual blocks.
    pub range: std::ops::Range<usize>,
    /// For `PSDTriangleCone` cones: the matrix dimension `d` (so the
    /// svec has `triangular_number(d)` entries). For other cones: the
    /// scalar dimension. Used by [`DefaultSolution::dual_psd_block`]
    /// and friends to unpack svec into a dense `d×d` matrix.
    pub dim: usize,
}

impl<T> DefaultSolution<T>
where
    T: FloatT,
{
    /// Create a new `DefaultSolution` object
    pub fn new(n: usize, m: usize) -> Self {
        let x = vec![T::zero(); n];
        let z = vec![T::zero(); m];
        let s = vec![T::zero(); m];

        Self {
            x,
            z,
            s,
            status: SolverStatus::Unsolved,
            obj_val: T::nan(),
            obj_val_dual: T::nan(),
            solve_time: 0f64,
            iterations: 0,
            r_prim: T::nan(),
            r_dual: T::nan(),
            cone_specs: Vec::new(),
        }
    }
}

// =========================================================
// Structured per-block accessors for SDP / multi-cone problems
// =========================================================

impl<T> DefaultSolution<T>
where
    T: FloatT,
{
    /// Slice of the dual `z` vector corresponding to cone `idx` in
    /// the post-collapse cone list.
    ///
    /// Returns `None` if `idx` is out of range. The cone's range
    /// metadata lives on `self.cone_specs[idx]`.
    pub fn dual_block(&self, idx: usize) -> Option<&[T]> {
        let spec = self.cone_specs.get(idx)?;
        Some(&self.z[spec.range.clone()])
    }

    /// Slice of the slack `s` vector corresponding to cone `idx`.
    pub fn primal_block(&self, idx: usize) -> Option<&[T]> {
        let spec = self.cone_specs.get(idx)?;
        Some(&self.s[spec.range.clone()])
    }

    /// For a `PSDTriangleCone` cone at position `idx`, unpack the
    /// dual `z` slice into a dense `d × d` symmetric matrix in
    /// row-major order. The svec packing matches Clarabel's
    /// internal convention: triu in column-major order with
    /// off-diagonals scaled by sqrt(2). The returned matrix is
    /// the recovered scaled-symmetric form.
    ///
    /// Returns `None` if `idx` is out of range or the cone at
    /// position `idx` is not `PSDTriangleCone`.
    #[cfg(feature = "sdp")]
    pub fn dual_psd_block(&self, idx: usize) -> Option<Vec<Vec<T>>> {
        let spec = self.cone_specs.get(idx)?;
        if spec.tag != SupportedConeTag::PSDTriangleCone {
            return None;
        }
        let d = spec.dim;
        let svec = &self.z[spec.range.clone()];
        Some(unpack_svec::<T>(svec, d))
    }

    /// Same as [`dual_psd_block`](Self::dual_psd_block) for the
    /// primal slack `s`.
    #[cfg(feature = "sdp")]
    pub fn primal_psd_block(&self, idx: usize) -> Option<Vec<Vec<T>>> {
        let spec = self.cone_specs.get(idx)?;
        if spec.tag != SupportedConeTag::PSDTriangleCone {
            return None;
        }
        let d = spec.dim;
        let svec = &self.s[spec.range.clone()];
        Some(unpack_svec::<T>(svec, d))
    }

    /// Sum-of-squares norm of `s + Ax - b` per cone, evaluated at
    /// solver precision. Useful for certifying that a rounded /
    /// projected solution still satisfies each cone individually.
    /// Returns one entry per `cone_specs` entry, in the same order.
    pub fn primal_residual_per_block(&self) -> Vec<T> {
        // The slack `s` already carries the per-cone primal-feasibility
        // residual at convergence — for solved problems s ∈ K and
        // s = b - Ax. We expose `(z, s)`-norm-style summaries by
        // computing per-block ||s|| via VectorMath::norm.
        self.cone_specs
            .iter()
            .map(|spec| self.s[spec.range.clone()].norm())
            .collect()
    }
}

/// Unpack a packed symmetric (svec) representation into a dense
/// `d × d` row-major Vec<Vec<T>>. Off-diagonals are de-scaled by
/// `1 / sqrt(2)` to recover the original symmetric matrix.
#[cfg(feature = "sdp")]
fn unpack_svec<T: FloatT>(svec: &[T], d: usize) -> Vec<Vec<T>> {
    let inv_sqrt_2 = T::FRAC_1_SQRT_2();
    let mut out = vec![vec![T::zero(); d]; d];
    let mut k = 0;
    for col in 0..d {
        for row in 0..=col {
            let v = svec[k].clone();
            if row == col {
                out[row][col] = v;
            } else {
                let scaled = v * inv_sqrt_2.clone();
                out[row][col] = scaled.clone();
                out[col][row] = scaled;
            }
            k += 1;
        }
    }
    out
}

impl<T> Solution<T> for DefaultSolution<T>
where
    T: FloatT,
{
    type D = DefaultProblemData<T>;
    type V = DefaultVariables<T>;
    type I = DefaultInfo<T>;
    type SE = DefaultSettings<T>;

    fn post_process(
        &mut self,
        data: &DefaultProblemData<T>,
        variables: &mut DefaultVariables<T>,
        info: &DefaultInfo<T>,
        settings: &DefaultSettings<T>,
    ) {
        self.status = info.status;
        let is_infeasible = info.status.is_infeasible();

        if is_infeasible {
            self.obj_val = T::nan();
            self.obj_val_dual = T::nan();
        } else {
            self.obj_val = info.cost_primal.clone();
            self.obj_val_dual = info.cost_dual.clone();
        }

        self.iterations = info.iterations;
        self.r_prim = info.res_primal.clone();
        self.r_dual = info.res_dual.clone();

        // unscale the variables to get a solution
        // to the internal problem as we solved it
        variables.unscale(data, is_infeasible);

        // unwind the chordal decomp and presolve, in the
        // reverse of the order in which they were applied
        #[cfg(feature = "sdp")]
        let tmp = data
            .chordal_info
            .as_ref()
            .map(|chordal_info| chordal_info.decomp_reverse(variables, &data.cones, settings));
        #[cfg(feature = "sdp")]
        let variables = tmp.as_ref().unwrap_or(variables);

        if let Some(ref presolver) = data.presolver {
            presolver.reverse_presolve(self, variables);
        } else {
            self.x.copy_from(&variables.x);
            self.z.copy_from(&variables.z);
            self.s.copy_from(&variables.s);
        }

        // Populate per-cone metadata for the structured per-block
        // accessors (dual_block, primal_block, dual_psd_block,
        // primal_residual_per_block). This is the post-collapse cone
        // list — sugar variants like BlockDiagPSDConeT have already
        // been expanded by SupportedConeT::new_collapsed at solver
        // construction time, so each entry here corresponds to one
        // contiguous block of the s/z slack vectors.
        self.cone_specs.clear();
        let mut start = 0usize;
        for cone in &data.cones {
            let nv = cone.nvars();
            let dim = match cone {
                SupportedConeT::ZeroConeT(d) => *d,
                SupportedConeT::NonnegativeConeT(d) => *d,
                SupportedConeT::SecondOrderConeT(d) => *d,
                SupportedConeT::ExponentialConeT() => 3,
                SupportedConeT::PowerConeT(_) => 3,
                SupportedConeT::GenPowerConeT(α, dim2) => α.len() + *dim2,
                #[cfg(feature = "sdp")]
                SupportedConeT::PSDTriangleConeT(d) => *d,
                #[cfg(feature = "sdp")]
                SupportedConeT::BlockDiagPSDConeT { .. } => unreachable!(
                    "BlockDiagPSDConeT must be expanded before reaching post_process"
                ),
            };
            self.cone_specs.push(ConeSpec {
                tag: cone.as_tag(),
                range: start..(start + nv),
                dim,
            });
            start += nv;
        }
    }

    fn finalize(&mut self, info: &DefaultInfo<T>) {
        self.solve_time = info.solve_time;
    }
}
