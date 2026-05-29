//! Hyperbolic Explorer — a first-person walk through a {4,5} tiling of
//! hyperbolic 3-space, rendered with the hyperboloid (Minkowski) model.
//!
//! Inspired by HackerPoet's HyperEngine (the non-Euclidean backend of
//! Hyperbolica). Movement is genuine hyperbolic translation: walk away from a
//! pillar and it shrinks far faster than in flat space; circle a vertex and
//! five squares meet where four would in the Euclidean world.
//!
//! Controls: W/A/S/D move, mouse looks, arrow keys also turn/look, Esc quits.

mod hyperbolic;
mod material;
mod world;

use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use bevy::render::view::NoFrustumCulling;
use bevy::window::{CursorGrabMode, PrimaryWindow};

use hyperbolic as h;
use material::HyperMaterial;

// --- Tunables --------------------------------------------------------------
const EYE_HEIGHT: f32 = 0.6; // hyperbolic distance the eye floats above the floor
const MOVE_SPEED: f32 = 1.1; // hyperbolic distance per second
const TURN_SPEED: f32 = 1.6; // radians per second (keyboard)
const MOUSE_SENS: f32 = 0.0022;
const FOV_Y: f32 = 1.15; // vertical field of view (radians)
const FOG_DENSITY: f32 = 0.14;
const SKY: Vec4 = Vec4::new(0.60, 0.73, 0.88, 1.0);

/// Player state, stored as a Lorentz frame on the floor plus a look pitch.
#[derive(Resource)]
struct Player {
    /// Floor frame: a Lorentz transform mapping the origin to the player's
    /// position and heading. Restricted to floor-preserving isometries
    /// (x/z translations and yaw), so the player always stands on `y = 0`.
    frame: Mat4,
    pitch: f32,
}

#[derive(Resource)]
struct WorldMaterial(Handle<HyperMaterial>);

/// Drives the optional headless screenshot used to verify rendering.
#[derive(Resource)]
struct ShotState {
    path: String,
    frame: u32,
}

fn main() {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "Hyperbolic Explorer".into(),
            ..default()
        }),
        ..default()
    }))
    .add_plugins(MaterialPlugin::<HyperMaterial>::default())
    .insert_resource(ClearColor(Color::srgb(SKY.x, SKY.y, SKY.z)))
    .insert_resource(Player {
        frame: Mat4::IDENTITY,
        pitch: -0.5,
    })
    .add_systems(Startup, (setup, grab_cursor))
    .add_systems(Update, (player_input, update_view, quit_on_esc));

    if let Ok(path) = std::env::var("HYPER_SCREENSHOT") {
        app.insert_resource(ShotState { path, frame: 0 })
            .add_systems(Update, screenshot_then_exit);
    }

    app.run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<HyperMaterial>>,
) {
    let mesh = meshes.add(world::build_world_mesh());
    let material = materials.add(HyperMaterial {
        view: Mat4::IDENTITY,
        params: Vec4::new(fov_factor(), 1.0, FOG_DENSITY, 0.0),
        fog_color: SKY,
    });
    commands.insert_resource(WorldMaterial(material.clone()));

    // The world geometry. Frustum culling is disabled because our vertex
    // positions are hyperboloid coordinates, not the on-screen Euclidean ones.
    commands.spawn((Mesh3d(mesh), MeshMaterial3d(material), NoFrustumCulling));

    // Camera. Its own transform/projection are unused by our shader (we project
    // ourselves), but Bevy needs a Camera3d to drive the render graph.
    commands.spawn((Camera3d::default(), Transform::default()));
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

    // Planar translation in the player's local frame (forward is -z).
    if keys.pressed(KeyCode::KeyW) {
        player.frame *= h::boost_z(-step);
    }
    if keys.pressed(KeyCode::KeyS) {
        player.frame *= h::boost_z(step);
    }
    if keys.pressed(KeyCode::KeyA) {
        player.frame *= h::boost_x(-step);
    }
    if keys.pressed(KeyCode::KeyD) {
        player.frame *= h::boost_x(step);
    }

    // Yaw: keyboard arrows plus mouse X.
    let mut yaw = 0.0;
    if keys.pressed(KeyCode::ArrowLeft) {
        yaw += TURN_SPEED * dt;
    }
    if keys.pressed(KeyCode::ArrowRight) {
        yaw -= TURN_SPEED * dt;
    }
    yaw -= mouse.delta.x * MOUSE_SENS;
    if yaw != 0.0 {
        player.frame *= h::rot_y(yaw);
    }

    // Pitch: keyboard arrows plus mouse Y (clamped to avoid flipping over).
    let mut pitch = player.pitch;
    if keys.pressed(KeyCode::ArrowUp) {
        pitch += TURN_SPEED * dt;
    }
    if keys.pressed(KeyCode::ArrowDown) {
        pitch -= TURN_SPEED * dt;
    }
    pitch -= mouse.delta.y * MOUSE_SENS;
    player.pitch = pitch.clamp(-1.3, 1.3);

    // Keep the frame a valid Lorentz transform despite float drift.
    player.frame = h::renormalize(player.frame);
}

fn update_view(
    player: Res<Player>,
    windows: Query<&Window, With<PrimaryWindow>>,
    handle: Res<WorldMaterial>,
    mut materials: ResMut<Assets<HyperMaterial>>,
) {
    // Camera pose = floor frame, lifted to eye height, then pitched.
    let cam_world = player.frame * h::boost_y(EYE_HEIGHT) * h::rot_x(player.pitch);
    let view = h::lorentz_inverse(cam_world);

    let aspect = windows
        .get_single()
        .map(|w| (w.width() / w.height()).max(0.01))
        .unwrap_or(1.777);

    if let Some(mat) = materials.get_mut(&handle.0) {
        mat.view = view;
        mat.params = Vec4::new(fov_factor(), aspect, FOG_DENSITY, 0.0);
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
    mut exit: EventWriter<AppExit>,
) {
    use bevy::render::view::screenshot::{save_to_disk, Screenshot};
    shot.frame += 1;
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
