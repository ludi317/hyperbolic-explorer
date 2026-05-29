//! Math for the hyperboloid (Minkowski / Lorentz) model of hyperbolic 3-space H³.
//!
//! A point of H³ is represented by a 4-vector `p = (x, y, z, w)` lying on the
//! upper sheet of the hyperboloid defined by the Minkowski quadratic form
//!
//! ```text
//!     <p, p> = x*x + y*y + z*z - w*w = -1,   with w > 0.
//! ```
//!
//! The origin of H³ is `O = (0, 0, 0, 1)`. The spatial part `(x, y, z)` of a
//! point is exactly the direction (and `asinh` of the distance) of the geodesic
//! from the origin toward it, which is what makes the rendering projection so
//! clean (see `assets/shaders/hyper.wgsl`).
//!
//! Isometries of H³ are the linear maps that preserve `<.,.>` — the (orthochronous)
//! Lorentz group `O⁺(3,1)`. We represent them as `Mat4`. Translations are
//! "boosts", rotations are ordinary spatial rotations.
//!
//! Convention used by the game world: the hyperbolic floor (an H² plane) lives in
//! the `y = 0` subspace, so movement happens in the x/z directions and "up"
//! (toward the sky / pillar height) is `+y`.

use bevy::math::{Mat4, Vec4};

/// `cosh(d)` for a point at geodesic distance `d` from the origin equals its
/// `w` coordinate. The full geodesic distance between two points `a`, `b` is
/// `acosh(-<a, b>)`.
#[inline]
pub fn mink_dot(a: Vec4, b: Vec4) -> f32 {
    a.x * b.x + a.y * b.y + a.z * b.z - a.w * b.w
}

/// The origin of hyperbolic space.
#[inline]
pub fn origin() -> Vec4 {
    Vec4::new(0.0, 0.0, 0.0, 1.0)
}

/// Build a `Mat4` from rows (the natural way to read the matrices below).
/// `glam` stores matrices column-major, so we transpose on construction.
#[inline]
fn from_rows(r: [[f32; 4]; 4]) -> Mat4 {
    Mat4::from_cols_array_2d(&r).transpose()
}

/// A point at geodesic distance `r` from the origin, in direction `angle`
/// measured in the x/z floor plane (angle 0 = +x, angle π/2 = +z).
pub fn floor_point(r: f32, angle: f32) -> Vec4 {
    let s = r.sinh();
    Vec4::new(s * angle.cos(), 0.0, s * angle.sin(), r.cosh())
}

// --- Translations (boosts) -------------------------------------------------

/// Translate by hyperbolic distance `d` along the +x geodesic.
pub fn boost_x(d: f32) -> Mat4 {
    let (c, s) = (d.cosh(), d.sinh());
    from_rows([
        [c, 0.0, 0.0, s],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [s, 0.0, 0.0, c],
    ])
}

/// Translate by hyperbolic distance `d` along the +y geodesic ("up").
pub fn boost_y(d: f32) -> Mat4 {
    let (c, s) = (d.cosh(), d.sinh());
    from_rows([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c, 0.0, s],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, s, 0.0, c],
    ])
}

/// Translate by hyperbolic distance `d` along the +z geodesic.
pub fn boost_z(d: f32) -> Mat4 {
    let (c, s) = (d.cosh(), d.sinh());
    from_rows([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, c, s],
        [0.0, 0.0, s, c],
    ])
}

// --- Rotations (ordinary spatial rotations fix the origin) -----------------

/// Rotate about the x axis (pitch) by `a` radians.
pub fn rot_x(a: f32) -> Mat4 {
    let (c, s) = (a.cos(), a.sin());
    from_rows([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c, -s, 0.0],
        [0.0, s, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

/// Rotate about the y axis (yaw, in the x/z floor plane) by `a` radians.
pub fn rot_y(a: f32) -> Mat4 {
    let (c, s) = (a.cos(), a.sin());
    from_rows([
        [c, 0.0, s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

/// Reflection across the geodesic plane `x = 0` (flips the x spatial axis).
pub fn reflect_x() -> Mat4 {
    from_rows([
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

/// Inverse of a Lorentz transform. For any `L` preserving the Minkowski form,
/// `L⁻¹ = J Lᵀ J` where `J = diag(1, 1, 1, -1)`. This is exact and avoids the
/// numerical issues of a general matrix inverse.
pub fn lorentz_inverse(m: Mat4) -> Mat4 {
    let j = Mat4::from_cols_array(&[
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, -1.0,
    ]);
    j * m.transpose() * j
}

/// Re-orthonormalize a Lorentz frame under the Minkowski metric (Gram–Schmidt).
/// Floating-point drift slowly violates `Lᵀ J L = J`; calling this each frame on
/// the player's transform keeps motion stable over long walks.
pub fn renormalize(m: Mat4) -> Mat4 {
    // Columns are the images of the basis vectors; the 4th (timelike) column is
    // the image of the origin and must satisfy <c3, c3> = -1.
    let mut c0 = m.x_axis;
    let mut c1 = m.y_axis;
    let mut c2 = m.z_axis;
    let mut c3 = m.w_axis;

    // Normalize the timelike column first (it should lie on the hyperboloid).
    let n3 = (-mink_dot(c3, c3)).max(1e-9).sqrt();
    c3 /= n3;
    if c3.w < 0.0 {
        c3 = -c3; // stay on the upper sheet (w > 0)
    }

    // Make each spacelike column orthogonal to the timelike one and to the
    // previously fixed spacelike columns, then unit-normalize (<c,c> = +1).
    let ortho = |v: &mut Vec4, basis: &[Vec4]| {
        // Subtract timelike component: note <c3,c3> = -1.
        *v += c3 * mink_dot(*v, c3);
        for b in basis {
            *v -= *b * mink_dot(*v, *b);
        }
        let n = mink_dot(*v, *v).max(1e-9).sqrt();
        *v /= n;
    };
    ortho(&mut c0, &[]);
    ortho(&mut c1, &[c0]);
    ortho(&mut c2, &[c0, c1]);

    Mat4::from_cols(c0, c1, c2, c3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// Every transform we build must preserve the Minkowski form, i.e. map the
    /// hyperboloid to itself: <Lp, Lp> = <p, p> = -1 for the origin.
    #[test]
    fn transforms_are_isometries() {
        let o = origin();
        for m in [
            boost_x(0.7),
            boost_y(-1.3),
            boost_z(0.4),
            rot_x(0.9),
            rot_y(2.1),
            reflect_x(),
            boost_x(0.5) * rot_y(0.6) * boost_z(-0.8),
        ] {
            let p = m * o;
            assert!((mink_dot(p, p) + 1.0).abs() < 1e-4, "not on hyperboloid");
        }
    }

    /// `lorentz_inverse` must actually invert.
    #[test]
    fn inverse_round_trips() {
        let m = boost_x(0.5) * rot_y(0.6) * boost_z(-0.8) * boost_y(0.3);
        let prod = lorentz_inverse(m) * m;
        let err: f32 = (prod - Mat4::IDENTITY)
            .to_cols_array()
            .iter()
            .map(|x| x.abs())
            .sum();
        assert!(err < 1e-3, "inverse error {err}");
    }

    /// Reflecting the origin across a tile edge lands its neighbor's center at
    /// exactly twice the inradius — the defining property used by the tiler.
    #[test]
    fn edge_reflection_distance() {
        let pp = PI / 4.0_f32; // {4,5}
        let pq = PI / 5.0_f32;
        let inradius = (pq.cos() / pp.sin()).acosh();
        let refl = boost_x(inradius) * reflect_x() * boost_x(-inradius);
        let neighbor = refl * origin();
        let dist = neighbor.w.acosh(); // geodesic distance from origin
        assert!((dist - 2.0 * inradius).abs() < 1e-4, "got {dist}");
    }

    /// Renormalizing a slightly-corrupted Lorentz frame restores the metric.
    #[test]
    fn renormalize_restores_metric() {
        let mut m = boost_x(1.0) * rot_y(0.5);
        m.x_axis += Vec4::splat(0.01); // perturb
        let r = renormalize(m);
        let o = r * origin();
        assert!((mink_dot(o, o) + 1.0).abs() < 1e-3);
    }
}
