mod renderer;
mod simple_arena;

use {
    core::f32::consts::PI,
    gfx_backend_vulkan as back,
    gfx_hal::{self as hal, prelude::*},
    rand::{rngs::ThreadRng, Rng},
};
pub use {
    glam::{Mat4, Quat, Vec3},
    renderer::{vert_coord_consts, DrawInfo, Renderer, TexScissor, VertCoord},
};

fn rand_quat<R: Rng>(rng: &mut R) -> Quat {
    let mut sample_fn = move || rng.gen::<f32>() * PI * 2.;
    Quat::from_rotation_x(sample_fn())
        * Quat::from_rotation_y(sample_fn())
        * Quat::from_rotation_z(sample_fn())
}

const DIMS: hal::window::Extent2D = hal::window::Extent2D { width: 800, height: 800 };

fn main() {
    fly_around()
}

fn rect_example() {
    let event_loop = winit::event_loop::EventLoop::new();
    let wb = winit::window::WindowBuilder::new()
        .with_resizable(false)
        .with_min_inner_size(winit::dpi::Size::Logical(winit::dpi::LogicalSize::new(64.0, 64.0)))
        .with_inner_size(winit::dpi::Size::Physical(winit::dpi::PhysicalSize::new(
            DIMS.width,
            DIMS.height,
        )))
        .with_title("quad".to_string());
    let window = wb.build(&event_loop).unwrap();
    let instance = back::Instance::create("gfx-rs quad", 1).unwrap();
    let surface = unsafe { instance.create_surface(&window).unwrap() };
    let adapter = instance.enumerate_adapters().into_iter().next().unwrap();
    const MAX_TRI_VERTS: u32 = 6;
    const MAX_INSTANCES: u32 = 1000;
    const DRAWS_PER_FRAME: usize = 3;
    let mut renderer = Renderer::new(instance, surface, adapter, MAX_TRI_VERTS, MAX_INSTANCES);

    let img_rgba =
        image::io::Reader::open("./src/data/logo.png").unwrap().decode().unwrap().to_rgba();
    renderer.load_texture(&img_rgba);

    let mut rng = rand::thread_rng();
    let xy_dist = rand::distributions::Uniform::new(-1., 1.);
    let z_dist = rand::distributions::Uniform::new(0., 1.);
    let s_dist = rand::distributions::Uniform::new(0.1, 0.3);
    const SHEET_LAYOUT: [usize; 2] = [11, 5];
    let sheet_x_dist = rand::distributions::Uniform::from(0..SHEET_LAYOUT[0]);
    let sheet_y_dist = rand::distributions::Uniform::from(0..SHEET_LAYOUT[1]);
    let rot_z_dist = rand::distributions::Uniform::new(0., PI * 2.);
    let random_transform = move |rng: &mut ThreadRng| {
        let moved = {
            let [tx, ty, tz] = [rng.sample(&xy_dist), rng.sample(&xy_dist), rng.sample(&z_dist)];
            Mat4::from_translation([tx, ty, tz].into())
        };
        let scale = {
            let [sx, sy] = [rng.sample(&s_dist), rng.sample(&s_dist)];
            Mat4::from_scale([sx, sy, 0.].into())
        };
        let rotate = Mat4::from_rotation_z(rng.sample(&rot_z_dist));
        Mat4::from(moved * scale * rotate)
    };
    let random_tex_scissor = move |rng: &mut ThreadRng| {
        let top_left = [
            rng.sample(&sheet_x_dist) as f32 / SHEET_LAYOUT[0] as f32,
            rng.sample(&sheet_y_dist) as f32 / SHEET_LAYOUT[1] as f32,
        ];
        const TILE_SIZE: [f32; 2] = [
            1. / SHEET_LAYOUT[0] as f32, //
            1. / SHEET_LAYOUT[1] as f32,
        ];
        TexScissor { top_left, size: TILE_SIZE }
    };

    renderer.write_vertex_buffer(0, vert_coord_consts::UNIT_QUAD.iter().copied());
    renderer.write_vertex_buffer(
        0,
        std::iter::repeat_with(|| random_transform(&mut rng)).take(MAX_INSTANCES as usize),
    );
    renderer.write_vertex_buffer(
        0,
        std::iter::repeat_with(|| random_tex_scissor(&mut rng)).take(MAX_INSTANCES as usize),
    );

    // It is important that the closure move captures the Renderer,
    // otherwise it will not be dropped when the event loop exits.

    let view_transforms: Vec<_> = (0..DRAWS_PER_FRAME)
        .map(|_| {
            let rot = rng.sample(&rot_z_dist);
            let [x, y] = [rng.sample(&xy_dist), rng.sample(&xy_dist)];
            Mat4::from_translation([x, y, 0.].into()) * Mat4::from_rotation_z(rot)
        })
        .collect();
    event_loop.run(move |event, _, control_flow| {
        println!("{:?}", event);
        use winit::{
            event::{Event as E, KeyboardInput as Ki, VirtualKeyCode as Vkc, WindowEvent as We},
            event_loop::ControlFlow,
        };
        *control_flow = ControlFlow::Wait;
        match event {
            E::WindowEvent { event: We::CloseRequested, .. } => *control_flow = ControlFlow::Exit,
            E::WindowEvent { event: We::KeyboardInput { input, .. }, .. } => match input {
                Ki { virtual_keycode: Some(Vkc::Escape), .. } => *control_flow = ControlFlow::Exit,
                Ki { virtual_keycode: Some(Vkc::Space), .. } => {}
                _ => {}
            },
            E::WindowEvent { event: We::Resized(_), .. } => unreachable!(),
            E::RedrawEventsCleared => {
                let before = std::time::Instant::now();
                renderer.write_vertex_buffer(
                    0,
                    std::iter::repeat_with(|| random_transform(&mut rng))
                        .take((MAX_INSTANCES / 8) as usize),
                );
                renderer.write_vertex_buffer(
                    0,
                    std::iter::repeat_with(|| random_tex_scissor(&mut rng))
                        .take((MAX_INSTANCES / 8) as usize),
                );
                let draw_info_iter = view_transforms
                    .iter()
                    .map(|trans| DrawInfo::new(trans, 0..MAX_TRI_VERTS, 0..MAX_INSTANCES));
                renderer.render_instances(0, draw_info_iter).unwrap();
                println!("{:?}", before.elapsed());
            }
            _ => {}
        }
    })
}

fn spinning_random_cubes() {
    let event_loop = winit::event_loop::EventLoop::new();
    let wb = winit::window::WindowBuilder::new()
        .with_resizable(false)
        .with_min_inner_size(winit::dpi::Size::Logical(winit::dpi::LogicalSize::new(64.0, 64.0)))
        .with_inner_size(winit::dpi::Size::Physical(winit::dpi::PhysicalSize::new(
            DIMS.width,
            DIMS.height,
        )))
        .with_title("Now in 3D!".to_string());
    let window = wb.build(&event_loop).unwrap();
    let instance = back::Instance::create("gfx-rs 3d", 1).unwrap();
    let surface = unsafe { instance.create_surface(&window).unwrap() };
    let adapter = instance.enumerate_adapters().into_iter().next().unwrap();

    const THIRD: f32 = 1. / 3.;
    let quad_rots = &[
        Mat4::from_rotation_x(0.),       // top
        Mat4::from_rotation_x(PI),       // bottom
        Mat4::from_rotation_x(PI / 2.),  // front
        Mat4::from_rotation_x(PI / -2.), // back
        Mat4::from_rotation_y(PI / 2.),  // left
        Mat4::from_rotation_y(PI / -2.), // right
    ];
    let size = [0.5, THIRD];
    let tex_scissors = (0..3).flat_map(move |i| {
        (0..2).map(move |j| TexScissor { size, top_left: [i as f32 / 2., j as f32 / 3.] })
    });
    let translate_up = Mat4::from_translation([0., 0., -0.5].into());
    let vert_coord_iter =
        quad_rots.iter().map(|&quad_rot| quad_rot * translate_up).zip(tex_scissors).flat_map(
            |(quad_trans, tex_scissor)| {
                vert_coord_consts::UNIT_QUAD.iter().map(move |quad_vert_coord| VertCoord {
                    model_coord: *quad_trans
                        .transform_point3(Vec3::from(quad_vert_coord.model_coord))
                        .as_ref(),
                    tex_coord: tex_scissor * quad_vert_coord.tex_coord,
                })
            },
        );
    let max_tri_verts = vert_coord_iter.clone().count() as u32;
    const MAX_INSTANCES: u32 = 10_000;
    let mut rng = rand::thread_rng();
    let rot_dist = rand::distributions::Uniform::new(0., PI * 2.);
    let translate_dist = rand::distributions::Uniform::new(-2., 2.);
    let scale_dist = rand::distributions::Uniform::new(0.01, 0.06);
    let sprite_index = rand::distributions::Uniform::from(0..3);

    let mut renderer = Renderer::new(instance, surface, adapter, max_tri_verts, MAX_INSTANCES);
    let img_rgba =
        image::io::Reader::open("./src/data/3d.png").unwrap().decode().unwrap().to_rgba();
    renderer.load_texture(&img_rgba);
    renderer.write_vertex_buffer(0, vert_coord_iter);
    renderer.write_vertex_buffer(
        0,
        std::iter::repeat_with(|| {
            Mat4::from_translation(
                [
                    rng.sample(&translate_dist),
                    rng.sample(&translate_dist),
                    rng.sample(&translate_dist),
                ]
                .into(),
            ) * Mat4::from_rotation_x(rng.sample(&rot_dist))
                * Mat4::from_rotation_y(rng.sample(&rot_dist))
                * Mat4::from_rotation_z(rng.sample(&rot_dist))
                * Mat4::from_scale({
                    let n = rng.sample(&scale_dist);
                    [n; 3].into()
                })
        })
        .take(MAX_INSTANCES as usize),
    );
    renderer.write_vertex_buffer(
        0,
        std::iter::repeat_with(|| TexScissor {
            top_left: [rng.sample(&sprite_index) as f32 / 3., 0.],
            size: [THIRD, 1.],
        })
        .take(MAX_INSTANCES as usize),
    );

    let mut t = 0;
    event_loop.run(move |event, _, control_flow| {
        // println!("{:?}", event);
        use winit::{
            event::{Event as E, KeyboardInput as Ki, VirtualKeyCode as Vkc, WindowEvent as We},
            event_loop::ControlFlow,
        };
        *control_flow = ControlFlow::Poll;
        // Until(
        //     std::time::Instant::now() + std::time::Duration::from_millis(16),
        // );
        match event {
            E::WindowEvent { event: We::CloseRequested, .. } => *control_flow = ControlFlow::Exit,
            E::WindowEvent { event: We::KeyboardInput { input, .. }, .. } => match input {
                Ki { virtual_keycode: Some(Vkc::Escape), .. } => *control_flow = ControlFlow::Exit,
                Ki { virtual_keycode: Some(Vkc::Space), .. } => {}
                _ => {}
            },
            E::WindowEvent { event: We::Resized(_), .. } => unreachable!(),
            E::RedrawEventsCleared => {
                let before = std::time::Instant::now();
                let views = [
                    Mat4::from_translation([0., 0., 0.5].into()) // push deeper
                    * Mat4::from_scale([1., 1., 0.2].into()) // squash_axes
                    * Mat4::from_rotation_z(t as f32 / 70_000.)
                    * Mat4::from_rotation_x(t as f32 / 64_000.)
                    * Mat4::from_rotation_y(t as f32 / 55_000.),
                    Mat4::from_translation([0., 0., 0.5].into()) // push deeper
                    * Mat4::from_scale([1., 1., 0.2].into()) // squash_axes
                    * Mat4::from_rotation_z(t as f32 / -43_000.)
                    * Mat4::from_rotation_x(t as f32 / 88_000.)
                    * Mat4::from_rotation_y(t as f32 / 92_000.),
                    Mat4::from_translation([0., 0., 0.5].into()) // push deeper
                    * Mat4::from_scale([1., 1., 0.2].into()) // squash_axes
                    * Mat4::from_rotation_z(t as f32 / 76_000.)
                    * Mat4::from_rotation_x(t as f32 / -81_000.)
                    * Mat4::from_rotation_y(t as f32 / -83_000.),
                    Mat4::from_translation([0., 0., 0.5].into()) // push deeper
                    * Mat4::from_scale([1., 1., 0.2].into()) // squash_axes
                    * Mat4::from_rotation_z(t as f32 / -59_000.)
                    * Mat4::from_rotation_x(t as f32 / 66_000.)
                    * Mat4::from_rotation_y(t as f32 / -83_000.),
                ];
                let passes = views
                    .iter()
                    .map(|view| DrawInfo::new(view, 0..max_tri_verts, 0..MAX_INSTANCES));
                renderer.render_instances(0, passes).unwrap();
                println!("{:?}", before.elapsed());
                t += 1;
            }
            _ => {}
        }
    })
}

fn fly_around() {
    let event_loop = winit::event_loop::EventLoop::new();
    let wb = winit::window::WindowBuilder::new()
        .with_resizable(false)
        .with_min_inner_size(winit::dpi::Size::Logical(winit::dpi::LogicalSize::new(64.0, 64.0)))
        .with_inner_size(winit::dpi::Size::Physical(winit::dpi::PhysicalSize::new(
            DIMS.width,
            DIMS.height,
        )))
        .with_title("Now in 3D!".to_string());
    let window = wb.build(&event_loop).unwrap();
    let instance = back::Instance::create("gfx-rs 3d", 1).unwrap();
    let surface = unsafe { instance.create_surface(&window).unwrap() };
    let adapter = instance.enumerate_adapters().into_iter().next().unwrap();

    const THIRD: f32 = 1. / 3.;
    let quad_rots = &[
        Mat4::from_rotation_x(0.),       // top
        Mat4::from_rotation_x(PI),       // bottom
        Mat4::from_rotation_x(PI / 2.),  // front
        Mat4::from_rotation_x(PI / -2.), // back
        Mat4::from_rotation_y(PI / 2.),  // left
        Mat4::from_rotation_y(PI / -2.), // right
    ];
    let size = [0.5, THIRD];
    let tex_scissors = (0..3).flat_map(move |i| {
        (0..2).map(move |j| TexScissor { size, top_left: [i as f32 / 2., j as f32 / 3.] })
    });
    let translate_up = Mat4::from_translation([0., 0., -0.5].into());
    let vert_coord_iter =
        quad_rots.iter().map(|&quad_rot| quad_rot * translate_up).zip(tex_scissors).flat_map(
            |(quad_trans, tex_scissor)| {
                vert_coord_consts::UNIT_QUAD.iter().map(move |quad_vert_coord| VertCoord {
                    model_coord: *quad_trans
                        .transform_point3(Vec3::from(quad_vert_coord.model_coord))
                        .as_ref(),
                    tex_coord: tex_scissor * quad_vert_coord.tex_coord,
                })
            },
        );
    let max_tri_verts = vert_coord_iter.clone().count() as u32;
    const MAX_INSTANCES: u32 = 30_000;
    let mut rng = rand::thread_rng();
    let dist_sprite_index = rand::distributions::Uniform::from(0..3);
    let dist_trans = rand::distributions::Uniform::new(-20., 20.);
    let dist_scale = rand::distributions::Uniform::new(0.01, 1.1);

    let mut renderer = Renderer::new(instance, surface, adapter, max_tri_verts, MAX_INSTANCES);
    let img_rgba = image::io::Reader::open("data/3d.png").unwrap().decode().unwrap().to_rgba();
    renderer.load_texture(&img_rgba);
    renderer.write_vertex_buffer(0, vert_coord_iter);
    renderer.write_vertex_buffer(
        0,
        std::iter::repeat_with(|| {
            let s = Vec3::new(
                rng.sample(&dist_scale),
                rng.sample(&dist_scale),
                rng.sample(&dist_scale),
            );
            let r = rand_quat(&mut rng);
            let t = Vec3::new(
                rng.sample(&dist_trans),
                rng.sample(&dist_trans),
                rng.sample(&dist_trans),
            );
            Mat4::from_scale_rotation_translation(s, r, t)
        }),
    );
    renderer.write_vertex_buffer(
        0,
        std::iter::repeat_with(|| TexScissor {
            top_left: [rng.sample(&dist_sprite_index) as f32 / 3., 0.],
            size: [THIRD, 1.],
        })
        .take(MAX_INSTANCES as usize),
    );
    let persp = Mat4::perspective_rh(1., 1., 0.5, 11.);

    #[derive(Debug)]
    struct AntagKeys {
        pos: winit::event::ElementState,
        neg: winit::event::ElementState,
    }
    impl AntagKeys {
        fn net_pos(&self) -> Option<bool> {
            use winit::event::ElementState::{Pressed, Released};
            match [self.pos, self.neg] {
                [Pressed, Pressed] | [Released, Released] => None,
                [Pressed, Released] => Some(true),
                [Released, Pressed] => Some(false),
            }
        }
    }
    impl Default for AntagKeys {
        fn default() -> Self {
            Self {
                pos: winit::event::ElementState::Released,
                neg: winit::event::ElementState::Released,
            }
        }
    }
    struct Entity {
        rotation_state: Quat,
        rotation_delta: Quat,
        position: Vec3,
        velocity: Vec3,
        keys: Pressing,
    }
    #[derive(Default, Debug)]
    struct Pressing {
        roll: AntagKeys,
        pitch: AntagKeys,
        yaw: AntagKeys,
        thrust: AntagKeys,
    }
    let mut ent = Entity {
        rotation_state: Quat::identity(),
        rotation_delta: Quat::identity(),
        velocity: Vec3::default(),
        position: Vec3::default(),
        keys: Pressing::default(),
    };

    event_loop.run(move |event, _, control_flow| {
        use winit::{
            event::{Event as E, KeyboardInput as Ki, VirtualKeyCode as Vkc, WindowEvent as We},
            event_loop::ControlFlow,
        };
        trait IsPressed {
            fn is_pressed(self) -> bool;
        }
        impl IsPressed for winit::event::ElementState {
            fn is_pressed(self) -> bool {
                match self {
                    winit::event::ElementState::Pressed => true,
                    winit::event::ElementState::Released => true,
                }
            }
        }
        match event {
            E::NewEvents(_) => {
                *control_flow = ControlFlow::Poll;
            }
            E::WindowEvent { event: We::CloseRequested, .. } => *control_flow = ControlFlow::Exit,
            E::WindowEvent { event: We::KeyboardInput { input, .. }, .. } => match input {
                Ki { virtual_keycode: Some(Vkc::Escape), .. } => *control_flow = ControlFlow::Exit,
                Ki { virtual_keycode: Some(Vkc::Space), state, .. } => ent.keys.pitch.pos = state,
                Ki { virtual_keycode: Some(Vkc::LControl), state, .. } => {
                    ent.keys.pitch.neg = state
                }
                Ki { virtual_keycode: Some(Vkc::Q), state, .. } => ent.keys.roll.neg = state,
                Ki { virtual_keycode: Some(Vkc::E), state, .. } => ent.keys.roll.pos = state,
                Ki { virtual_keycode: Some(Vkc::D), state, .. } => ent.keys.yaw.pos = state,
                Ki { virtual_keycode: Some(Vkc::A), state, .. } => ent.keys.yaw.neg = state,
                Ki { virtual_keycode: Some(Vkc::W), state, .. } => ent.keys.thrust.pos = state,
                Ki { virtual_keycode: Some(Vkc::S), state, .. } => ent.keys.thrust.neg = state,
                _ => {}
            },
            E::WindowEvent { event: We::Resized(_), .. } => unreachable!(),
            E::MainEventsCleared => {
                const ROT_CONST: f32 = 0.0000003;
                const DAMP_MULT: f32 = 0.9997;
                const VEL_CONST: f32 = 0.0000001;

                let rotation_delta_delta = {
                    let mut q = Quat::identity();
                    match ent.keys.pitch.net_pos() {
                        Some(true) => q *= Quat::from_rotation_y(-ROT_CONST),
                        Some(false) => q *= Quat::from_rotation_y(ROT_CONST),
                        None => {}
                    }
                    match ent.keys.yaw.net_pos() {
                        Some(true) => q *= Quat::from_rotation_z(ROT_CONST),
                        Some(false) => q *= Quat::from_rotation_z(-ROT_CONST),
                        None => {}
                    }
                    match ent.keys.roll.net_pos() {
                        Some(true) => q *= Quat::from_rotation_x(-ROT_CONST),
                        Some(false) => q *= Quat::from_rotation_x(ROT_CONST),
                        None => {}
                    }
                    q
                };
                ent.rotation_delta =
                    Quat::identity().lerp(ent.rotation_delta * rotation_delta_delta, DAMP_MULT);
                ent.rotation_state *= ent.rotation_delta;

                match ent.keys.thrust.net_pos() {
                    Some(true) => {
                        ent.velocity += ent.rotation_state.mul_vec3(Vec3::new(VEL_CONST, 0., 0.))
                    }
                    Some(false) => {
                        ent.velocity -= ent.rotation_state.mul_vec3(Vec3::new(VEL_CONST, 0., 0.))
                    }
                    None => {}
                };
                ent.velocity *= DAMP_MULT;
                ent.position += ent.velocity;
            }
            E::RedrawEventsCleared => {
                // let before = std::time::Instant::now();
                let views = [{
                    let look_at = {
                        let eye = ent.position;
                        let center = eye + ent.rotation_state.mul_vec3(Vec3::new(1., 0., 0.));
                        let up = ent.rotation_state.mul_vec3(Vec3::new(0., 0., -1.));
                        Mat4::look_at_rh(eye, center, up)
                    };
                    persp * look_at
                }];
                let passes = views
                    .iter()
                    .map(|view| DrawInfo::new(view, 0..max_tri_verts, 0..MAX_INSTANCES));
                renderer.render_instances(0, passes).unwrap();
                // println!("{:?}", before.elapsed());
            }
            _ => {}
        }
        // println!("{:?}", &ent.keys);
    })
}
