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

/// Fraction of a tile (toward its center) kept as colored fill; the rest is the
/// dark border "grout" that outlines every tile.
const TILE_INSET: f32 = 0.93;
/// Border / grout color.
const GROUT: [f32; 4] = [0.05, 0.06, 0.09, 1.0];

/// Floor colors, indexed by the per-tile graph-coloring. Five entries is enough
/// for any greedy coloring of this degree-4 graph; in practice only 3–4 are used.
const FLOOR_PALETTE: [[f32; 4]; 5] = [
    [0.13, 0.42, 0.60, 1.0], // blue
    [0.82, 0.74, 0.52, 1.0], // tan
    [0.34, 0.55, 0.40, 1.0], // sage green
    [0.72, 0.44, 0.34, 1.0], // terracotta
    [0.46, 0.38, 0.56, 1.0], // plum
];

/// Inradius (center → edge midpoint) and circumradius (center → vertex) of a
/// regular {P,Q} tile, from the right-triangle relations of the (2,P,Q) triangle.
fn tile_radii() -> (f32, f32) {
    let pp = PI / P as f32;
    let pq = PI / Q as f32;
    let cot = |x: f32| x.cos() / x.sin();
    // Apothem (center → edge midpoint) and circumradius (center → vertex).
    let inradius = (pq.cos() / pp.sin()).acosh();
    let circumradius = (cot(pp) * cot(pq)).acosh();
    (inradius, circumradius)
}

/// The four edge-reflection generators of the {P,Q} tiling. The neighbor of a
/// tile `m` across its k-th edge is the tile `m * generators[k]`.
fn edge_generators() -> Vec<Mat4> {
    let (inradius, _) = tile_radii();
    // Reflection across the tile edge whose midpoint is distance `inradius`
    // along +x: translate the edge to the origin plane, flip x, translate back.
    let reflect_edge0 = h::boost_x(inradius) * h::reflect_x() * h::boost_x(-inradius);
    (0..P)
        .map(|k| {
            let a = k as f32 * 2.0 * PI / P as f32;
            h::rot_y(a) * reflect_edge0 * h::rot_y(-a)
        })
        .collect()
}

/// All tiles, discovered by reflecting across edges, then *properly* colored by
/// greedy graph-coloring over the edge-adjacency graph: each tile gets the
/// lowest color index not used by any already-colored edge-neighbor. Because
/// every vertex is surrounded by an odd (5-) cycle of tiles, the tiling is not
/// 2-colorable; this naturally settles into 3–4 colors. Returns
/// `(transform, ring, color)` per tile (ring is the BFS depth, used for pillars).
fn generate_tiles() -> Vec<(Mat4, u32, usize)> {
    let generators = edge_generators();

    // Breadth-first discovery of tile transforms.
    let mut mats = vec![Mat4::IDENTITY];
    let mut rings = vec![0u32];
    let mut centers = vec![Vec3::ZERO];
    let mut queue: VecDeque<usize> = VecDeque::new();
    queue.push_back(0);

    let find = |centers: &Vec<Vec3>, c: Vec3| {
        centers.iter().position(|s| s.distance_squared(c) < 1e-4)
    };

    while let Some(idx) = queue.pop_front() {
        if mats.len() >= MAX_TILES {
            break;
        }
        let (m, ring) = (mats[idx], rings[idx]);
        for g in &generators {
            let center = m * *g * h::origin();
            // Geodesic distance from the world origin is acosh(w).
            if center.w.acosh() > MAX_DIST {
                continue;
            }
            let c3 = center.truncate();
            if find(&centers, c3).is_none() {
                let new_idx = mats.len();
                mats.push(m * *g);
                rings.push(ring + 1);
                centers.push(c3);
                queue.push_back(new_idx);
            }
        }
    }

    // Edge adjacency, then greedy coloring in BFS order.
    let n = mats.len();
    let mut colors = vec![usize::MAX; n];
    for i in 0..n {
        let mut used = [false; 8];
        for g in &generators {
            let c = (mats[i] * *g * h::origin()).truncate();
            if let Some(j) = find(&centers, c) {
                if j != i && colors[j] != usize::MAX {
                    used[colors[j]] = true;
                }
            }
        }
        let mut c = 0;
        while used[c] {
            c += 1;
        }
        colors[i] = c;
    }

    (0..n).map(|i| (mats[i], rings[i], colors[i])).collect()
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

    /// Draw a polygon with a dark border: a `border`-colored ring between the
    /// `outer` and `inner` rings, then the `inner` polygon in `fill`.
    fn bordered(&mut self, outer: &[Vec4], inner: &[Vec4], fill: [f32; 4], border: [f32; 4]) {
        let n = outer.len();
        for k in 0..n {
            let m = (k + 1) % n;
            self.polygon(&[outer[k], outer[m], inner[m], inner[k]], border);
        }
        self.polygon(inner, fill);
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

    let ncolors = tiles.iter().map(|t| t.2).max().unwrap_or(0) + 1;
    info!("floor uses {} colors (proper graph-coloring)", ncolors);

    for (m, ring, color) in &tiles {
        // Outer corners at the circumradius; inner corners pulled toward the
        // tile center to leave room for the border.
        let corner = |r: f32| -> Vec<Vec4> {
            (0..P)
                .map(|k| {
                    let a = PI / P as f32 + k as f32 * 2.0 * PI / P as f32;
                    *m * h::floor_point(r, a)
                })
                .collect()
        };
        let outer = corner(circumradius);
        let inner = corner(circumradius * TILE_INSET);

        let base = FLOOR_PALETTE[*color % FLOOR_PALETTE.len()];
        b.bordered(&outer, &inner, base, GROUT);

        // Put a pillar on every other tile so the world has vertical structure
        // and parallax without becoming a forest. Skip the origin tile so the
        // player doesn't spawn inside a pillar.
        if *ring != 0 && ring % 2 == 0 {
            add_pillar(&mut b, *m);
        }
    }

    b.build()
}

// ---------------------------------------------------------------------------
// Euclidean counterpart: an ordinary flat {4,4} grid (four squares per vertex),
// for the split-screen comparison. Same tile size, colors, and pillar pattern,
// but drawn with a normal Bevy camera + StandardMaterial so the *only*
// difference between the two views is the curvature of space.
// ---------------------------------------------------------------------------

const GRID_RADIUS: i32 = 14;

fn push_quad(
    pos: &mut Vec<[f32; 3]>,
    nor: &mut Vec<[f32; 3]>,
    col: &mut Vec<[f32; 4]>,
    idx: &mut Vec<u32>,
    c: [Vec3; 4],
    n: Vec3,
    color: [f32; 4],
) {
    let base = pos.len() as u32;
    for v in c {
        pos.push([v.x, v.y, v.z]);
        nor.push([n.x, n.y, n.z]);
        col.push(color);
    }
    idx.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

/// Build the flat Euclidean grid world (floor + pillars) as one normal mesh.
pub fn build_euclidean_mesh() -> Mesh {
    let floor_a = [0.09, 0.34, 0.55, 1.0];
    let floor_b = [0.80, 0.74, 0.55, 1.0];
    let side = [0.55, 0.38, 0.30, 1.0];
    let side_dark = [0.40, 0.27, 0.21, 1.0];
    let cap = [0.95, 0.75, 0.45, 1.0];

    let (mut pos, mut nor, mut col, mut idx) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let hw = PILLAR_RADIUS; // reuse pillar footprint
    let ph = PILLAR_HEIGHT;

    for i in -GRID_RADIUS..=GRID_RADIUS {
        for j in -GRID_RADIUS..=GRID_RADIUS {
            let (x, z) = (i as f32, j as f32);
            let even = (i + j).rem_euclid(2) == 0;

            // Floor tile (unit square, normal up) with a dark border ring.
            let color = if even { floor_a } else { floor_b };
            let o = 0.5; // outer half-size
            let n = 0.5 * TILE_INSET; // inner half-size
            let outer = [
                Vec3::new(x - o, 0.0, z - o),
                Vec3::new(x - o, 0.0, z + o),
                Vec3::new(x + o, 0.0, z + o),
                Vec3::new(x + o, 0.0, z - o),
            ];
            let inner = [
                Vec3::new(x - n, 0.0, z - n),
                Vec3::new(x - n, 0.0, z + n),
                Vec3::new(x + n, 0.0, z + n),
                Vec3::new(x + n, 0.0, z - n),
            ];
            for k in 0..4 {
                let m = (k + 1) % 4;
                push_quad(&mut pos, &mut nor, &mut col, &mut idx,
                    [outer[k], outer[m], inner[m], inner[k]], Vec3::Y, GROUT);
            }
            push_quad(&mut pos, &mut nor, &mut col, &mut idx, inner, Vec3::Y, color);

            // Pillar on even tiles (skip the origin where the player spawns).
            if even && !(i == 0 && j == 0) {
                let (x0, x1, z0, z1) = (x - hw, x + hw, z - hw, z + hw);
                // +x and -x walls, +z and -z walls.
                push_quad(&mut pos, &mut nor, &mut col, &mut idx,
                    [Vec3::new(x1,0.0,z0),Vec3::new(x1,0.0,z1),Vec3::new(x1,ph,z1),Vec3::new(x1,ph,z0)], Vec3::X, side);
                push_quad(&mut pos, &mut nor, &mut col, &mut idx,
                    [Vec3::new(x0,0.0,z1),Vec3::new(x0,0.0,z0),Vec3::new(x0,ph,z0),Vec3::new(x0,ph,z1)], -Vec3::X, side);
                push_quad(&mut pos, &mut nor, &mut col, &mut idx,
                    [Vec3::new(x1,0.0,z1),Vec3::new(x0,0.0,z1),Vec3::new(x0,ph,z1),Vec3::new(x1,ph,z1)], Vec3::Z, side_dark);
                push_quad(&mut pos, &mut nor, &mut col, &mut idx,
                    [Vec3::new(x0,0.0,z0),Vec3::new(x1,0.0,z0),Vec3::new(x1,ph,z0),Vec3::new(x0,ph,z0)], -Vec3::Z, side_dark);
                // Cap.
                push_quad(&mut pos, &mut nor, &mut col, &mut idx,
                    [Vec3::new(x0,ph,z0),Vec3::new(x0,ph,z1),Vec3::new(x1,ph,z1),Vec3::new(x1,ph,z0)], Vec3::Y, cap);
            }
        }
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, pos);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, nor);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, col);
    mesh.insert_indices(Indices::U32(idx));
    mesh
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

#[cfg(test)]
mod tests {
    use super::*;

    fn corners(m: Mat4, r: f32) -> Vec<Vec4> {
        (0..P)
            .map(|k| {
                let a = PI / P as f32 + k as f32 * 2.0 * PI / P as f32;
                m * h::floor_point(r, a)
            })
            .collect()
    }

    /// The floor must actually tessellate: the fundamental tile and each of its
    /// four edge-neighbors must share exactly two corner vertices. This is the
    /// invariant that the circumradius bug violated (tiles were too small and
    /// left gaps), so guard it directly.
    #[test]
    fn tiles_share_edges() {
        let (inradius, circumradius) = tile_radii();
        let reflect_edge0 = h::boost_x(inradius) * h::reflect_x() * h::boost_x(-inradius);
        let origin = corners(Mat4::IDENTITY, circumradius);

        for k in 0..P {
            let a = k as f32 * 2.0 * PI / P as f32;
            let neighbor_tf = h::rot_y(a) * reflect_edge0 * h::rot_y(-a);
            let neighbor = corners(neighbor_tf, circumradius);

            let shared = origin
                .iter()
                .filter(|o| {
                    neighbor
                        .iter()
                        .any(|n| o.truncate().distance(n.truncate()) < 1e-3)
                })
                .count();
            assert_eq!(shared, 2, "edge {k}: expected 2 shared corners, got {shared}");
        }
    }

    /// The greedy coloring must be proper: no two edge-adjacent tiles share a
    /// color. Also confirm it uses 3–4 colors (it cannot be 2, since the tiling
    /// has odd cycles around every vertex).
    #[test]
    fn coloring_is_proper() {
        let tiles = generate_tiles();
        let generators = edge_generators();
        let centers: Vec<(Vec3, usize)> = tiles
            .iter()
            .map(|(m, _, c)| ((*m * h::origin()).truncate(), *c))
            .collect();

        for (m, _, c) in &tiles {
            for g in &generators {
                let nc = (*m * *g * h::origin()).truncate();
                if let Some((_, ncolor)) =
                    centers.iter().find(|(p, _)| p.distance_squared(nc) < 1e-4)
                {
                    assert_ne!(*c, *ncolor, "edge-adjacent tiles share a color");
                }
            }
        }

        let used = tiles.iter().map(|t| t.2).max().unwrap() + 1;
        assert!((3..=4).contains(&used), "expected 3-4 colors, used {used}");
    }
}
