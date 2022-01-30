use bevy::{
    core_pipeline::{clear_color::ClearColorConfig, core_3d::Camera3dDepthLoadOp},
    prelude::*,
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_startup_system(setup)
        .add_system(movement)
        .add_system(animate_light_direction)
        .run();
}

#[derive(Component)]
struct Movable;

#[derive(Component)]
struct SpinnyLights;

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // sphere
    commands
        .spawn_bundle(PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Icosphere {
                radius: 2.0,
                subdivisions: 10,
                ..Default::default()
            })),
            material: materials.add(StandardMaterial {
                base_color: Color::GRAY,
                ..Default::default()
            }),
            ..Default::default()
        })
        .insert(Movable);
    commands
        .spawn_bundle((
            Transform::default(),
            GlobalTransform::default(),
            SpinnyLights,
        ))
        .with_children(|builder| {
            for (color, a) in [(Color::RED, 0.0), (Color::GREEN, 0.33), (Color::BLUE, 0.66)] {
                let r = 16.0;
                let ang = a * std::f32::consts::TAU;
                let pos = Vec3::new(r * ang.cos(), r * ang.sin(), -0.1);
                builder.spawn_bundle(PointLightBundle {
                    transform: Transform::from_translation(pos),
                    point_light: PointLight {
                        intensity: 1e4, // lumens - roughly a 100W non-halogen incandescent bulb
                        color,
                        range: 19.0,
                        //shadows_enabled: true,
                        ..Default::default()
                    },
                    ..Default::default()
                });
            }
        });

    // directional 'sun' light
    const HALF_SIZE: f32 = 10.0;
    commands.spawn_bundle(DirectionalLightBundle {
        directional_light: DirectionalLight {
            // Configure the projection to better fit the scene
            shadow_projection: OrthographicProjection {
                left: -HALF_SIZE,
                right: HALF_SIZE,
                bottom: -HALF_SIZE,
                top: HALF_SIZE,
                near: -10.0 * HALF_SIZE,
                far: 10.0 * HALF_SIZE,
                ..Default::default()
            },
            shadows_enabled: true,
            illuminance: 2e4,
            ..Default::default()
        },
        transform: Transform::default().looking_at(Vec3::new(-1.0, -1.0, -1.0), Vec3::Y),
        ..Default::default()
    });

    // camera
    commands.spawn_bundle(Camera3dBundle {
        transform: Transform::from_xyz(0.0, 0.0, 16.0).looking_at(Vec3::ZERO, Vec3::Y),
        camera_3d: Camera3d {
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            depth_load_op: Camera3dDepthLoadOp::Clear(0.),
        },
        ..Default::default()
    });
}

fn animate_light_direction(time: Res<Time>, mut query: Query<&mut Transform, With<SpinnyLights>>) {
    for mut transform in query.iter_mut() {
        transform.rotate(Quat::from_rotation_z(time.delta_seconds() * 0.25));
    }
}

fn movement(
    input: Res<Input<KeyCode>>,
    time: Res<Time>,
    mut query: Query<&mut Transform, With<Movable>>,
) {
    for mut transform in query.iter_mut() {
        let mut direction = Vec3::ZERO;
        if input.pressed(KeyCode::Up) {
            direction.y += 1.0;
        }
        if input.pressed(KeyCode::Down) {
            direction.y -= 1.0;
        }
        if input.pressed(KeyCode::Left) {
            direction.x -= 1.0;
        }
        if input.pressed(KeyCode::Right) {
            direction.x += 1.0;
        }

        transform.translation += time.delta_seconds() * 2.0 * direction;
    }
}
