//! Generates the explorable world: a regular {4,5} tiling of the hyperbolic
//! floor (four-sided tiles, five meeting at every vertex — impossible in flat
//! space) plus pillars rising from it, all baked into mesh geometry whose
//! vertex positions are hyperboloid coordinates.

use crate::hyperbolic as h;
use bevy::math::{Vec3, Vec4};
use bevy::prelude::*;
use bevy::render::mesh::{Indices, MeshVertexAttribute, PrimitiveTopology};
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render::render_resource::VertexFormat;
use std::collections::VecDeque;
use std::f32::consts::PI;

/// Per-vertex color, fed to the shader at location 1.
pub const ATTRIBUTE_COLOR: MeshVertexAttribute =
    MeshVertexAttribute::new("VColor", 0x48_59_50_00, VertexFormat::Float32x4);

/// Schläfli symbol of the tiling: {P, Q} = P-gons, Q around each vertex.
const P: u32 = 4;
const Q: u32 = 5;

/// How far out (in geodesic distance from the origin) to generate tiles.
const MAX_DIST: f32 = 5.2;
/// Safety cap on tile count.
const MAX_TILES: usize = 4000;

/// Height of the pillars, in hyperbolic distance.
const PILLAR_HEIGHT: f32 = 0.9;
/// Half-size (circumradius) of a pillar's square cross-section.
const PILLAR_RADIUS: f32 = 0.14;

/// Inradius (center → edge midpoint) and circumradius (center → vertex) of a
/// regular {P,Q} tile, from the right-triangle relations of the (2,P,Q) triangle.
fn tile_radii() -> (f32, f32) {
    let pp = PI / P as f32;
    let pq = PI / Q as f32;
    let inradius = (pq.cos() / pp.sin()).acosh();
    let circumradius = (pp.cos() / pq.sin()).acosh();
    (inradius, circumradius)
}

/// All the tile transforms (Lorentz matrices mapping the fundamental tile to
/// each tile), discovered by reflecting across tile edges. Each returned value
/// is `(transform, ring)` where `ring` is the BFS depth (used for coloring).
fn generate_tiles() -> Vec<(Mat4, u32)> {
    let (inradius, _) = tile_radii();

    // Reflection across the tile edge whose midpoint is distance `inradius`
    // along +x: translate the edge to the origin plane, flip x, translate back.
    let reflect_edge0 = h::boost_x(inradius) * h::reflect_x() * h::boost_x(-inradius);
    // The four edge reflections, one per side of the square.
    let generators: Vec<Mat4> = (0..P)
        .map(|k| {
            let a = k as f32 * 2.0 * PI / P as f32;
            h::rot_y(a) * reflect_edge0 * h::rot_y(-a)
        })
        .collect();

    let mut tiles = vec![(Mat4::IDENTITY, 0u32)];
    let mut seen: Vec<Vec3> = vec![Vec3::ZERO];
    let mut queue: VecDeque<(Mat4, u32)> = VecDeque::new();
    queue.push_back((Mat4::IDENTITY, 0));

    let is_new = |seen: &Vec<Vec3>, c: Vec3| !seen.iter().any(|s| s.distance_squared(c) < 1e-4);

    while let Some((m, ring)) = queue.pop_front() {
        if tiles.len() >= MAX_TILES {
            break;
        }
        for g in &generators {
            let m2 = m * *g;
            let center = m2 * h::origin();
            // Geodesic distance from the world origin is acosh(w).
            if center.w.acosh() > MAX_DIST {
                continue;
            }
            let c3 = center.truncate();
            if is_new(&seen, c3) {
                seen.push(c3);
                tiles.push((m2, ring + 1));
                queue.push_back((m2, ring + 1));
            }
        }
    }
    tiles
}

/// Accumulates triangles whose vertex positions are hyperboloid spatial coords.
struct MeshBuilder {
    positions: Vec<[f32; 3]>,
    colors: Vec<[f32; 4]>,
    indices: Vec<u32>,
}

impl MeshBuilder {
    fn new() -> Self {
        Self {
            positions: Vec::new(),
            colors: Vec::new(),
            indices: Vec::new(),
        }
    }

    /// Push a convex polygon (hyperboloid points) as a triangle fan, flat-colored.
    fn polygon(&mut self, pts: &[Vec4], color: [f32; 4]) {
        let base = self.positions.len() as u32;
        for p in pts {
            self.positions.push([p.x, p.y, p.z]);
            self.colors.push(color);
        }
        for i in 1..pts.len() as u32 - 1 {
            self.indices
                .extend_from_slice(&[base, base + i, base + i + 1]);
        }
    }

    fn build(self) -> Mesh {
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::RENDER_WORLD,
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, self.positions);
        mesh.insert_attribute(ATTRIBUTE_COLOR, self.colors);
        mesh.insert_indices(Indices::U32(self.indices));
        mesh
    }
}

/// Build the whole world mesh (floor tiles + pillars).
pub fn build_world_mesh() -> Mesh {
    let (_, circumradius) = tile_radii();
    let tiles = generate_tiles();
    info!("generated {} hyperbolic tiles", tiles.len());

    let mut b = MeshBuilder::new();

    // Two-tone checkerboard floor; tile shade keyed on BFS-ring parity.
    let floor_a = [0.09, 0.34, 0.55, 1.0];
    let floor_b = [0.80, 0.74, 0.55, 1.0];

    for (m, ring) in &tiles {
        // Corners of the regular P-gon, at the circumradius.
        let corners: Vec<Vec4> = (0..P)
            .map(|k| {
                let a = PI / P as f32 + k as f32 * 2.0 * PI / P as f32;
                *m * h::floor_point(circumradius, a)
            })
            .collect();

        let base = if ring % 2 == 0 { floor_a } else { floor_b };
        b.polygon(&corners, base);

        // Put a pillar on every other tile so the world has vertical structure
        // and parallax without becoming a forest. Skip the origin tile so the
        // player doesn't spawn inside a pillar.
        if *ring != 0 && ring % 2 == 0 {
            add_pillar(&mut b, *m);
        }
    }

    b.build()
}

/// Add an axis-aligned square pillar standing on tile `m`'s center.
fn add_pillar(b: &mut MeshBuilder, m: Mat4) {
    // Base square corners on the floor (y = 0), in the tile's local frame.
    let base_local: Vec<Vec4> = (0..4)
        .map(|k| {
            let a = PI / 4.0 + k as f32 * PI / 2.0;
            h::floor_point(PILLAR_RADIUS, a)
        })
        .collect();
    let lift = h::boost_y(PILLAR_HEIGHT);

    // World-space base and top rings.
    let base: Vec<Vec4> = base_local.iter().map(|p| m * *p).collect();
    let top: Vec<Vec4> = base_local.iter().map(|p| m * (lift * *p)).collect();

    let side = [0.55, 0.38, 0.30, 1.0];
    let side_dark = [0.40, 0.27, 0.21, 1.0];
    let cap = [0.95, 0.75, 0.45, 1.0];

    // Four side walls.
    for k in 0..4 {
        let n = (k + 1) % 4;
        let quad = [base[k], base[n], top[n], top[k]];
        let c = if k % 2 == 0 { side } else { side_dark };
        b.polygon(&quad, c);
    }
    // Top cap.
    b.polygon(&top, cap);
}
