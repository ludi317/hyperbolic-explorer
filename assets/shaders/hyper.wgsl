// Hyperbolic-space renderer for the hyperboloid (Minkowski) model.
//
// Mesh vertex positions hold the *spatial* part (x, y, z) of a world-space
// point on the hyperboloid <p,p> = x²+y²+z²-w² = -1. We reconstruct w, transform
// by the player's Lorentz view matrix, and project. Because the spatial part of
// a point equals the geodesic direction from the camera, the projection is just
// a pinhole divide by the forward axis — and hyperbolic geodesics come out as
// straight screen lines (the Beltrami–Klein property), exactly what a rasterizer
// wants.

@group(2) @binding(0) var<uniform> view: mat4x4<f32>;
@group(2) @binding(1) var<uniform> params: vec4<f32>;   // x=f, y=aspect, z=fog_density
@group(2) @binding(2) var<uniform> fog_color: vec4<f32>;

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) campos: vec3<f32>,   // camera-frame spatial part
    @location(2) cosh_dist: f32,      // camera-frame w = cosh(geodesic distance)
};

@vertex
fn vertex(v: Vertex) -> VsOut {
    let spatial = v.position;
    let w = sqrt(1.0 + dot(spatial, spatial));
    let hp = vec4<f32>(spatial, w);

    // World hyperboloid point -> camera frame (a Lorentz transform).
    let cam = view * hp;

    let f = params.x;
    let aspect = params.y;
    let fwd = -cam.z;                 // distance along the camera's forward (-z) axis

    var out: VsOut;
    // Pinhole projection. Perspective divide is by `fwd`, so screen position is
    // (f/aspect * x/fwd, f * y/fwd). Depth = clip.z/clip.w = 1/cosh(dist), which
    // is 1 at the camera and falls toward 0 with distance — matching Bevy's
    // reverse-Z depth (near = 1).
    out.clip = vec4<f32>(
        (f / aspect) * cam.x,
        f * cam.y,
        fwd / cam.w,
        fwd,
    );
    out.color = v.color;
    out.campos = cam.xyz;
    out.cosh_dist = cam.w;
    return out;
}

@fragment
fn fragment(in: VsOut) -> @location(0) vec4<f32> {
    // Free faceted normal from screen-space derivatives of the camera position.
    let n = normalize(cross(dpdx(in.campos), dpdy(in.campos)));
    let light = normalize(vec3<f32>(0.35, 0.85, 0.40));
    let lambert = clamp(dot(n, light), 0.0, 1.0);
    let shade = 0.5 + 0.5 * lambert;

    // Exponential fog in true hyperbolic distance for strong depth cueing.
    let dist = acosh(max(in.cosh_dist, 1.0));
    let fog = clamp(1.0 - exp(-params.z * dist), 0.0, 1.0);

    let lit = in.color.rgb * shade;
    let rgb = mix(lit, fog_color.rgb, fog);
    return vec4<f32>(rgb, 1.0);
}
