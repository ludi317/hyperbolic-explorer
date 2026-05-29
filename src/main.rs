//! Hyperbolic Explorer — a first-person walk through a {4,5} tiling of
//! hyperbolic 3-space, rendered with the hyperboloid (Minkowski) model.
//!
//! Split-screen mode: the LEFT half is hyperbolic space; the RIGHT half is an
//! ordinary flat Euclidean grid. The same W/A/S/D + mouse inputs drive both
//! worlds at once, so you can watch identical walks diverge — square loops that
//! close in flat space don't close in hyperbolic space, the horizon stays close
//! on the left and far on the right, and five tiles meet at a vertex on the left
//! where four meet on the right.
//!
//! Inspired by HackerPoet's HyperEngine (the non-Euclidean backend of
//! Hyperbolica).
//!
//! Controls: W/A/S/D move, mouse looks, arrow keys also turn/look, Esc quits.

mod hyperbolic;
mod material;
mod world;

use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use bevy::render::camera::{ClearColorConfig, PerspectiveProjection, Projection, Viewport};
use bevy::render::view::{NoFrustumCulling, RenderLayers};
use bevy::ui::TargetCamera;
use bevy::window::{CursorGrabMode, PrimaryWindow};

use hyperbolic as h;
use material::HyperMaterial;

// --- Tunables --------------------------------------------------------------
const EYE_HEIGHT: f32 = 0.6; // distance the eye floats above the floor
const MOVE_SPEED: f32 = 1.1; // distance per second
const TURN_SPEED: f32 = 1.6; // radians per second (keyboard)
const MOUSE_SENS: f32 = 0.0022;
const FOV_Y: f32 = 1.15; // vertical field of view (radians)
const FOG_DENSITY: f32 = 0.14;
const SKY_HYPER: Vec4 = Vec4::new(0.60, 0.73, 0.88, 1.0); // bluish
const SKY_EUCLID: Color = Color::srgb(0.80, 0.82, 0.71); // warm, to tell the halves apart

/// Player state. The hyperbolic and Euclidean worlds are driven by the *same*
/// inputs but evolve as two independent simulations, so they drift apart.
#[derive(Resource)]
struct Player {
    /// Hyperbolic floor frame (a Lorentz transform).
    frame: Mat4,
    /// Euclidean position and heading.
    e_pos: Vec3,
    e_yaw: f32,
    /// Shared look pitch.
    pitch: f32,
}

#[derive(Resource)]
struct WorldMaterial(Handle<HyperMaterial>);

/// The four tiling symmetries used to snap the player back into the center tile
/// so the finite baked patch always surrounds them (endless walking).
#[derive(Resource)]
struct Recenter(Vec<Mat4>);

/// Camera markers.
#[derive(Component)]
struct HyperCam;
#[derive(Component)]
struct EuclidCam;

#[derive(Resource)]
struct ShotState {
    path: String,
    frame: u32,
}

fn main() {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "Hyperbolic Explorer — left: hyperbolic   right: euclidean".into(),
            ..default()
        }),
        ..default()
    }))
    .add_plugins(MaterialPlugin::<HyperMaterial>::default())
    .insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 350.0,
    })
    .insert_resource(Player {
        frame: Mat4::IDENTITY,
        e_pos: Vec3::new(0.0, EYE_HEIGHT, 0.0),
        e_yaw: 0.0,
        pitch: -0.5,
    })
    .insert_resource(Recenter(world::recenter_generators()))
    .add_systems(Startup, (setup, grab_cursor))
    .add_systems(
        Update,
        (
            (player_input, recenter, update_views).chain(),
            update_viewports,
            quit_on_esc,
        ),
    );

    if let Ok(path) = std::env::var("HYPER_SCREENSHOT") {
        app.insert_resource(ShotState { path, frame: 0 })
            .add_systems(Update, screenshot_then_exit);
    }

    app.run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut hyper_materials: ResMut<Assets<HyperMaterial>>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
) {
    // --- Hyperbolic world (render layer 0) ---
    let hyper_mesh = meshes.add(world::build_world_mesh());
    let hyper_material = hyper_materials.add(HyperMaterial {
        view: Mat4::IDENTITY,
        params: Vec4::new(fov_factor(), 1.0, FOG_DENSITY, 0.0),
        fog_color: SKY_HYPER,
    });
    commands.insert_resource(WorldMaterial(hyper_material.clone()));
    commands.spawn((
        Mesh3d(hyper_mesh),
        MeshMaterial3d(hyper_material),
        NoFrustumCulling,
        RenderLayers::layer(0),
    ));

    let left_cam = commands
        .spawn((
            Camera3d::default(),
            Camera {
                order: 0,
                clear_color: ClearColorConfig::Custom(Color::srgb(
                    SKY_HYPER.x,
                    SKY_HYPER.y,
                    SKY_HYPER.z,
                )),
                ..default()
            },
            RenderLayers::layer(0),
            HyperCam,
        ))
        .id();

    // --- Euclidean world (render layer 1) ---
    let euclid_mesh = meshes.add(world::build_euclidean_mesh());
    let euclid_material = std_materials.add(StandardMaterial {
        base_color: Color::WHITE, // modulated by per-vertex colors
        perceptual_roughness: 0.95,
        metallic: 0.0,
        double_sided: true,
        cull_mode: None,
        ..default()
    });
    commands.spawn((
        Mesh3d(euclid_mesh),
        MeshMaterial3d(euclid_material),
        RenderLayers::layer(1),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 7000.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::default().looking_to(Vec3::new(-0.4, -1.0, -0.35), Vec3::Y),
        RenderLayers::layer(1),
    ));

    let right_cam = commands
        .spawn((
            Camera3d::default(),
            Camera {
                order: 1,
                clear_color: ClearColorConfig::Custom(SKY_EUCLID),
                ..default()
            },
            Projection::Perspective(PerspectiveProjection {
                fov: FOV_Y,
                ..default()
            }),
            RenderLayers::layer(1),
            EuclidCam,
        ))
        .id();

    // --- Per-viewport labels ---
    spawn_label(&mut commands, left_cam, "HYPERBOLIC  {4,5}\n5 tiles meet at each vertex");
    spawn_label(&mut commands, right_cam, "EUCLIDEAN  {4,4}\n4 tiles meet at each vertex");
}

fn spawn_label(commands: &mut Commands, camera: Entity, text: &str) {
    commands.spawn((
        Text::new(text),
        TextFont {
            font_size: 16.0,
            ..default()
        },
        TextColor(Color::srgb(0.1, 0.1, 0.1)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(12.0),
            ..default()
        },
        TargetCamera(camera),
    ));
}

fn fov_factor() -> f32 {
    1.0 / (FOV_Y * 0.5).tan()
}

fn grab_cursor(mut windows: Query<&mut Window, With<PrimaryWindow>>) {
    if let Ok(mut window) = windows.get_single_mut() {
        window.cursor_options.grab_mode = CursorGrabMode::Locked;
        window.cursor_options.visible = false;
    }
}

fn player_input(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mouse: Res<AccumulatedMouseMotion>,
    mut player: ResMut<Player>,
) {
    let dt = time.delta_secs();
    let step = MOVE_SPEED * dt;

    // Gather control inputs once, then apply identically to both worlds.
    let mut fwd = 0.0;
    let mut strafe = 0.0;
    if keys.pressed(KeyCode::KeyW) {
        fwd += 1.0;
    }
    if keys.pressed(KeyCode::KeyS) {
        fwd -= 1.0;
    }
    if keys.pressed(KeyCode::KeyD) {
        strafe += 1.0;
    }
    if keys.pressed(KeyCode::KeyA) {
        strafe -= 1.0;
    }

    let mut yaw = 0.0;
    if keys.pressed(KeyCode::ArrowLeft) {
        yaw += TURN_SPEED * dt;
    }
    if keys.pressed(KeyCode::ArrowRight) {
        yaw -= TURN_SPEED * dt;
    }
    yaw -= mouse.delta.x * MOUSE_SENS;

    let mut pitch = player.pitch;
    if keys.pressed(KeyCode::ArrowUp) {
        pitch += TURN_SPEED * dt;
    }
    if keys.pressed(KeyCode::ArrowDown) {
        pitch -= TURN_SPEED * dt;
    }
    pitch -= mouse.delta.y * MOUSE_SENS;
    player.pitch = pitch.clamp(-1.3, 1.3);

    // --- Hyperbolic world: boosts + yaw in the local Lorentz frame ---
    if fwd != 0.0 {
        player.frame *= h::boost_z(-fwd * step);
    }
    if strafe != 0.0 {
        player.frame *= h::boost_x(strafe * step);
    }
    if yaw != 0.0 {
        player.frame *= h::rot_y(yaw);
    }
    player.frame = h::renormalize(player.frame);

    // --- Euclidean world: same inputs, integrated in the flat plane ---
    // Reuse the same rotation convention so the two views agree near the origin.
    player.e_yaw += yaw;
    let fdir = (h::rot_y(player.e_yaw) * Vec4::new(0.0, 0.0, -1.0, 0.0)).truncate();
    let rdir = (h::rot_y(player.e_yaw) * Vec4::new(1.0, 0.0, 0.0, 0.0)).truncate();
    player.e_pos += (fdir * fwd + rdir * strafe) * step;
    player.e_pos.y = EYE_HEIGHT;
}

/// Keep both worlds permanently centered on the player so walking never reaches
/// an edge. Hyperbolic: snap back by tiling symmetries. Euclidean: wrap into the
/// repeating 2×2 cell of the checker/pillar pattern. Both are seamless.
fn recenter(mut player: ResMut<Player>, gens: Res<Recenter>) {
    // Hyperbolic: greedily apply the symmetry that most reduces the player's
    // distance from the origin (w = cosh(distance)), until none helps.
    for _ in 0..1000 {
        let cur = (player.frame * h::origin()).w;
        let mut best = cur;
        let mut best_g: Option<Mat4> = None;
        for g in &gens.0 {
            let w = (*g * player.frame * h::origin()).w;
            if w < best - 1e-4 {
                best = w;
                best_g = Some(*g);
            }
        }
        match best_g {
            Some(g) => player.frame = g * player.frame,
            None => break,
        }
    }

    // Euclidean: the pattern repeats every 2 units, so wrap into [-1, 1]².
    player.e_pos.x -= 2.0 * (player.e_pos.x / 2.0).round();
    player.e_pos.z -= 2.0 * (player.e_pos.z / 2.0).round();
}

fn update_views(
    player: Res<Player>,
    windows: Query<&Window, With<PrimaryWindow>>,
    handle: Res<WorldMaterial>,
    mut hyper_materials: ResMut<Assets<HyperMaterial>>,
    mut euclid_cam: Query<&mut Transform, With<EuclidCam>>,
) {
    // Hyperbolic view matrix: invert the camera's Lorentz pose.
    let cam_world = player.frame * h::boost_y(EYE_HEIGHT) * h::rot_x(player.pitch);
    let view = h::lorentz_inverse(cam_world);

    // Each half of the window is half as wide as it is tall-relative.
    let aspect = windows
        .get_single()
        .map(|w| ((w.width() * 0.5) / w.height()).max(0.01))
        .unwrap_or(0.888);

    if let Some(mat) = hyper_materials.get_mut(&handle.0) {
        mat.view = view;
        mat.params = Vec4::new(fov_factor(), aspect, FOG_DENSITY, 0.0);
    }

    // Euclidean camera: same yaw/pitch, looking down its local -z.
    if let Ok(mut tf) = euclid_cam.get_single_mut() {
        let dir = (h::rot_y(player.e_yaw) * h::rot_x(player.pitch) * Vec4::new(0.0, 0.0, -1.0, 0.0))
            .truncate();
        *tf = Transform::from_translation(player.e_pos).looking_to(dir, Vec3::Y);
    }
}

/// Keep each camera filling its half of the window (in physical pixels).
fn update_viewports(
    windows: Query<&Window, With<PrimaryWindow>>,
    mut left: Query<&mut Camera, (With<HyperCam>, Without<EuclidCam>)>,
    mut right: Query<&mut Camera, (With<EuclidCam>, Without<HyperCam>)>,
) {
    let Ok(window) = windows.get_single() else {
        return;
    };
    let (w, hgt) = (window.physical_width(), window.physical_height());
    if w < 4 || hgt < 2 {
        return;
    }
    let half = w / 2;

    if let Ok(mut cam) = left.get_single_mut() {
        cam.viewport = Some(Viewport {
            physical_position: UVec2::new(0, 0),
            physical_size: UVec2::new(half.saturating_sub(1), hgt),
            ..default()
        });
    }
    if let Ok(mut cam) = right.get_single_mut() {
        cam.viewport = Some(Viewport {
            physical_position: UVec2::new(half + 1, 0),
            physical_size: UVec2::new(w - half - 1, hgt),
            ..default()
        });
    }
}

fn quit_on_esc(keys: Res<ButtonInput<KeyCode>>, mut exit: EventWriter<AppExit>) {
    if keys.just_pressed(KeyCode::Escape) {
        exit.send(AppExit::Success);
    }
}

fn screenshot_then_exit(
    mut commands: Commands,
    mut shot: ResMut<ShotState>,
    mut player: ResMut<Player>,
    mut exit: EventWriter<AppExit>,
) {
    use bevy::render::view::screenshot::{save_to_disk, Screenshot};
    shot.frame += 1;
    // Auto-walk forward ~35 units before capturing, to prove that recentering
    // keeps the floor full far past the baked patch's radius.
    if shot.frame < 90 {
        player.frame *= h::boost_z(-0.4);
        let fwd = (h::rot_y(player.e_yaw) * Vec4::new(0.0, 0.0, -0.4, 0.0)).truncate();
        player.e_pos += fwd;
    }
    if shot.frame == 90 {
        let path = shot.path.clone();
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path));
    }
    if shot.frame >= 140 {
        exit.send(AppExit::Success);
    }
}
