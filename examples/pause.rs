use bevy::prelude::*;
use bevy_h264::{
    decode_video, H264Decoder, H264DecoderLoading, H264DecoderPause, H264Plugin, H264RestartEvent,
    H264UpdateEvent,
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(H264Plugin { fps: Some(120.0) })
        .add_systems(Startup, setup)
        .add_systems(Update, keyboard_commands)
        .run();
}

pub fn keyboard_commands(
    mut commands: Commands,
    mut writer: EventWriter<H264RestartEvent>,
    query: Query<Entity, With<H264Decoder>>,
    input: Res<ButtonInput<KeyCode>>,
) {
    let entity = query.single();
    let mut entity_commands = commands.entity(entity);

    if input.just_pressed(KeyCode::KeyP) {
        // pause
        entity_commands.insert(H264DecoderPause {});
    }
    if input.just_pressed(KeyCode::KeyR) {
        // resume
        entity_commands.remove::<H264DecoderPause>();
    }
    if input.just_pressed(KeyCode::KeyS) {
        // stop
        writer.send(H264RestartEvent(entity));
        entity_commands.insert(H264DecoderPause {});
        // TODO: stopping does not clear the input buffer of the decoder:
        // stopping and resuming will not immediately show frames
        // from the start of the video, but some frames after the stop event
        // probably the SyncSender has to be replaced with a VecDeque that can be cleared.
    }
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    asset_server: Res<AssetServer>,
) {
    let decoder = H264Decoder::new(&mut images, asset_server.load("test.h264"), false);

    println!("p = pause\nr = resume playing\ns = stop");

    commands.spawn(Camera2dBundle::default());
    commands
        .spawn(SpriteBundle {
            texture: decoder.get_render_target(),
            transform: Transform::from_xyz(0.0, 0.0, -1.1),
            ..default()
        })
        .insert(decoder)
        .insert(H264DecoderPause {})
        .insert(H264DecoderLoading {});
}
