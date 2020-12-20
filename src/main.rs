mod renderer;
mod simple_arena;

use std::time::{Duration, Instant};
use {
    core::f32::consts::PI,
    gfx_backend_dx12 as back,
    gfx_hal::{self as hal, prelude::*},
    rand::{rngs::ThreadRng, Rng},
};
pub use {
    gfx_hal::pso::Face as CullFace,
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
    archery()
}

fn archery() {
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
    const MAX_TRI_VERTS: u32 = vert_coord_consts::UNIT_QUAD.len() as u32;
    const MAX_ARROWS: u32 = 100;
    const MAX_INSTANCES: u32 = 1 + 2 * MAX_ARROWS;
    const FLOOR_TS: TexScissor =
        TexScissor { top_left: [0. / 50., 0. / 6.], size: [1. / 50., 1. / 6.] };
    const ARROW_TS: TexScissor =
        TexScissor { top_left: [0. / 50., 1. / 6.], size: [50. / 50., 5. / 6.] };
    let mut renderer =
        Renderer::new(instance, surface, adapter, MAX_TRI_VERTS, MAX_INSTANCES, CullFace::NONE);

    let img_rgba =
        image::io::Reader::open("./src/data/arrow.png").unwrap().decode().unwrap().to_rgba8();
    renderer.load_texture(&img_rgba);
    struct Yp {
        yaw: f32,
        pitch: f32,
    }
    impl Yp {
        fn quat(&self) -> Quat {
            Quat::from_rotation_ypr(self.yaw, self.pitch, 0.)
        }
    }
    struct Arrow {
        pos: Vec3,
        vel: Vec3,
    }
    let mut arrows: Vec<Arrow> = Vec::with_capacity(MAX_ARROWS as usize);

    // POS Z is DOWN
    // let mut arrows = [
    //     Arrow { pos: Vec3::new(0.3, 2., -0.6), vel: Vec3::new(0.001, 0.0008, -0.007) },
    //     Arrow { pos: Vec3::new(1.3, 1., -0.8), vel: Vec3::new(0.0005, 0.001, -0.009) },
    //     Arrow { pos: Vec3::new(2.3, 2., -2.9), vel: Vec3::new(-0.0001, -0.0018, -0.005) },
    // ];
    const ONE: Vec3 = Vec3 { x: 1., y: 0., z: 0. };
    const UP: Vec3 = Vec3 { x: 0., y: 0., z: 1. };
    impl Arrow {
        fn transforms(&self) -> [Mat4; 2] {
            let q = move |q| {
                Mat4::from_translation(self.pos)
                    * Mat4::from_quat({
                        let axis = ONE.cross(self.vel.normalize());
                        let angle = ONE.angle_between(self.vel);
                        Quat::from_axis_angle(axis, angle)
                    })
                    * Mat4::from_translation(Vec3::new(-0.3, 0., 0.))
                    * Mat4::from_rotation_z(PI)
                    * Mat4::from_rotation_x(q)
                    * Mat4::from_scale(Vec3::new(1., 0.1, 1.))
            };
            [
                q(PI * 0.0), //
                q(PI * 0.5), //
            ]
        }
    }
    let floor_m4 = Mat4::from_scale_rotation_translation(
        Vec3::new(9999., 9999., 1.),
        Quat::identity(),
        Vec3::default(),
    );

    renderer.write_vertex_buffer(0, vert_coord_consts::UNIT_QUAD.iter().copied());
    renderer.write_vertex_buffer(
        0,
        std::iter::once(FLOOR_TS).chain((1..MAX_INSTANCES).map(|_| ARROW_TS)),
    );
    renderer.write_vertex_buffer(0, Some(floor_m4));
    let mut eye = Vec3::new(0., 0., -2.);
    let mut yp = Yp { pitch: PI / 2., yaw: 0. };
    let persp = Mat4::perspective_lh(1., 1., 0.5, 17.);
    let mut next_update_after = Instant::now();
    let start_program_at = Instant::now();
    let [mut updates, mut redraws] = [0; 2];
    const UPS: u64 = 60;
    const UPDATE_DELTA: Duration = Duration::from_micros(1_000_000 / UPS);
    let shadow_squish =
        Mat4::from_translation(Vec3::new(0., 0., -0.01)) * Mat4::from_scale(Vec3::new(1., 1., 0.));
    event_loop.run(move |event, _, control_flow| {
        use winit::{
            event::{
                DeviceEvent as De, Event as E, KeyboardInput as Ki, VirtualKeyCode as Vkc,
                WindowEvent as We,
            },
            event_loop::ControlFlow,
        };
        *control_flow = ControlFlow::Poll;
        match event {
            E::WindowEvent { event: We::CloseRequested, .. } => *control_flow = ControlFlow::Exit,
            E::WindowEvent { event: We::KeyboardInput { input, .. }, .. } => match input {
                Ki { virtual_keycode: Some(Vkc::Escape), .. } => *control_flow = ControlFlow::Exit,
                Ki { virtual_keycode: Some(Vkc::W), .. } => eye[0] += 0.1,
                Ki { virtual_keycode: Some(Vkc::S), .. } => eye[0] -= 0.1,
                Ki { virtual_keycode: Some(Vkc::A), .. } => eye[1] += 0.1,
                Ki { virtual_keycode: Some(Vkc::D), .. } => eye[1] -= 0.1,
                Ki { virtual_keycode: Some(Vkc::LControl), .. } => eye[2] += 0.1,
                Ki { virtual_keycode: Some(Vkc::Space), .. } => eye[2] -= 0.1,
                Ki { virtual_keycode: Some(Vkc::X), state, .. } => {
                    if let winit::event::ElementState::Pressed = state {
                        // add arrows
                        let new_arrow = Arrow { pos: eye, vel: yp.quat() * (ONE * 0.03) };
                        arrows.push(new_arrow);
                        if arrows.len() > MAX_ARROWS as usize {
                            arrows.remove(0);
                        }
                    }
                }
                _ => {}
            },
            E::DeviceEvent { event: De::MouseMotion { delta: (x, y) }, .. } => {
                const MULT: f32 = 0.002;
                yp.yaw = (yp.yaw + x as f32 * MULT) % (PI * 2.);
                yp.pitch = (yp.pitch + y as f32 * MULT).min(PI / 2.).max(-PI / 2.);
                let _ = window.set_cursor_position(winit::dpi::Position::Logical([400.; 2].into()));
            }
            E::WindowEvent { event: We::Resized(_), .. } => unreachable!(),
            E::MainEventsCleared => {
                let mut do_update = || {
                    for a in arrows.iter_mut() {
                        if a.pos[2] <= 0. {
                            a.vel[2] += 0.0001;
                            a.pos += a.vel;
                        }
                    }
                    renderer.write_vertex_buffer(
                        1,
                        arrows.iter().flat_map(|a| {
                            let [t0, t1] = a.transforms();
                            Some(t0).into_iter().chain(Some(t1))
                        }),
                    );
                    updates += 1;
                };
                let now = Instant::now();
                while next_update_after < now {
                    next_update_after += UPDATE_DELTA;
                    do_update();
                }
                let secs_elapsed = start_program_at.elapsed().as_secs_f32();
                println!("{:?}", [updates as f32 / secs_elapsed, redraws as f32 / secs_elapsed]);
                window.request_redraw();
            }
            E::RedrawEventsCleared => {
                let m = Mat4::look_at_lh(eye, eye + yp.quat() * ONE, UP);
                // let m = Mat4::from_quat(-yp.quat()) * Mat4::from_translation(-eye);
                let view = persp * m;
                let view_squashed = view * shadow_squish;
                renderer
                    .render_instances(
                        0,
                        [
                            DrawInfo::new(&view_squashed, 0..MAX_TRI_VERTS, 1..MAX_INSTANCES),
                            DrawInfo::new(&view, 0..MAX_TRI_VERTS, 0..MAX_INSTANCES),
                        ]
                        .iter()
                        .cloned(),
                    )
                    .unwrap();
                redraws += 1;
            }
            _ => {}
        }
        // println!("{:?}", &ent.keys);
    })
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
    let mut renderer =
        Renderer::new(instance, surface, adapter, MAX_TRI_VERTS, MAX_INSTANCES, CullFace::BACK);

    let img_rgba =
        image::io::Reader::open("./src/data/logo.png").unwrap().decode().unwrap().to_rgba8();
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
                let before = Instant::now();
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

    let mut renderer =
        Renderer::new(instance, surface, adapter, max_tri_verts, MAX_INSTANCES, CullFace::BACK);
    let img_rgba =
        image::io::Reader::open("./src/data/3d.png").unwrap().decode().unwrap().to_rgba8();
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
        // Until(
        //     Instant::now() + Duration::from_millis(16),
        // );
        *control_flow = ControlFlow::Poll;
        match event {
            E::WindowEvent { event: We::CloseRequested, .. } => *control_flow = ControlFlow::Exit,
            E::WindowEvent { event: We::KeyboardInput { input, .. }, .. } => match input {
                Ki { virtual_keycode: Some(Vkc::Escape), .. } => *control_flow = ControlFlow::Exit,
                Ki { virtual_keycode: Some(Vkc::Space), .. } => {}
                _ => {}
            },
            E::WindowEvent { event: We::Resized(_), .. } => unreachable!(),
            E::RedrawEventsCleared => {
                let before = Instant::now();
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
    const MAX_INSTANCES: u32 = 20_000;
    let mut rng = rand::thread_rng();
    let dist_sprite_index = rand::distributions::Uniform::from(0..3);
    let dist_trans = rand::distributions::Uniform::new(-170., 170.);
    let dist_scale = rand::distributions::Uniform::new(0.01, 4.1);

    let mut renderer =
        Renderer::new(instance, surface, adapter, max_tri_verts, MAX_INSTANCES, CullFace::BACK);
    let img_rgba =
        image::io::Reader::open("./src/data/3d.png").unwrap().decode().unwrap().to_rgba8();
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
        surge: AntagKeys,
        sway: AntagKeys,
        heave: AntagKeys,
    }
    let mut ent = Entity {
        rotation_state: Quat::identity(),
        rotation_delta: Quat::identity(),
        velocity: Vec3::default(),
        position: Vec3::default(),
        keys: Pressing::default(),
    };
    // renderer.render_instances(0, std::iter::empty()).unwrap();
    let persp = Mat4::perspective_lh(1., 1., 0.8, 70.);
    // let [mut updates, mut frames] = [0; 2];
    // let started_at = Instant::now();
    let mut box_rot = PI;
    let box_rot_delta = 0.0003;
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
        const WAIT_DUR: Duration = Duration::from_millis(16);
        println!("{:?}", &event);
        // *control_flow = ControlFlow::Poll;
        match event {
            E::NewEvents(winit::event::StartCause::Init) => {
                *control_flow = ControlFlow::WaitUntil(Instant::now() + WAIT_DUR);
            }
            E::WindowEvent { event: We::CloseRequested, .. } => *control_flow = ControlFlow::Exit,
            E::WindowEvent { event: We::KeyboardInput { input, .. }, .. } => {
                // println!("IN {:?}", &event);
                match input {
                    Ki { virtual_keycode: Some(Vkc::Escape), .. } => {
                        *control_flow = ControlFlow::Exit
                    }
                    Ki { virtual_keycode: Some(Vkc::LControl), state, .. } => {
                        ent.keys.pitch.pos = state
                    }
                    Ki { virtual_keycode: Some(Vkc::Space), state, .. } => {
                        ent.keys.pitch.neg = state
                    }
                    Ki { virtual_keycode: Some(Vkc::Q), state, .. } => ent.keys.roll.neg = state,
                    Ki { virtual_keycode: Some(Vkc::E), state, .. } => ent.keys.roll.pos = state,
                    Ki { virtual_keycode: Some(Vkc::A), state, .. } => ent.keys.yaw.pos = state,
                    Ki { virtual_keycode: Some(Vkc::D), state, .. } => ent.keys.yaw.neg = state,
                    Ki { virtual_keycode: Some(Vkc::W), state, .. } => ent.keys.surge.pos = state,
                    Ki { virtual_keycode: Some(Vkc::S), state, .. } => ent.keys.surge.neg = state,
                    Ki { virtual_keycode: Some(Vkc::Up), state, .. } => ent.keys.heave.pos = state,
                    Ki { virtual_keycode: Some(Vkc::Down), state, .. } => {
                        ent.keys.heave.neg = state
                    }
                    Ki { virtual_keycode: Some(Vkc::Left), state, .. } => ent.keys.sway.pos = state,
                    Ki { virtual_keycode: Some(Vkc::Right), state, .. } => {
                        ent.keys.sway.neg = state
                    }
                    _ => {}
                }
            }
            E::WindowEvent { event: We::Resized(_), .. } => unreachable!(),
            E::RedrawEventsCleared => {}
            E::MainEventsCleared => {
                *control_flow = ControlFlow::WaitUntil(Instant::now() + WAIT_DUR);
                // updates += 1;
                // if updates % 16 == 0 {
                //     println!("UPS = {:?}", updates as f32 / started_at.elapsed().as_secs_f32());
                // }
                // println!("MAIN {:?}", &event);
                const ROT_CONST: f32 = 0.003;
                const DAMP_MULT: f32 = 0.98;
                const VEL_CONST: f32 = 0.003;
                fn pos_f32(pos: bool) -> f32 {
                    if pos {
                        1.
                    } else {
                        -1.
                    }
                }

                let rotation_delta_delta = {
                    let mut q = Quat::identity();
                    if let Some(pos) = ent.keys.roll.net_pos() {
                        q *= Quat::from_rotation_x(pos_f32(pos) * ROT_CONST);
                    }
                    if let Some(pos) = ent.keys.pitch.net_pos() {
                        q *= Quat::from_rotation_y(pos_f32(pos) * ROT_CONST);
                    }
                    if let Some(pos) = ent.keys.yaw.net_pos() {
                        q *= Quat::from_rotation_z(pos_f32(pos) * ROT_CONST);
                    }
                    q
                };
                ent.rotation_delta =
                    Quat::identity().lerp(ent.rotation_delta * rotation_delta_delta, DAMP_MULT);
                ent.rotation_state *= ent.rotation_delta;

                if let Some(pos) = ent.keys.surge.net_pos() {
                    ent.velocity +=
                        pos_f32(pos) * ent.rotation_state.mul_vec3(Vec3::new(VEL_CONST, 0., 0.));
                };
                if let Some(pos) = ent.keys.heave.net_pos() {
                    ent.velocity +=
                        pos_f32(pos) * ent.rotation_state.mul_vec3(Vec3::new(0., 0., VEL_CONST));
                };
                if let Some(pos) = ent.keys.sway.net_pos() {
                    ent.velocity +=
                        pos_f32(pos) * ent.rotation_state.mul_vec3(Vec3::new(0., VEL_CONST, 0.));
                };
                ent.velocity *= DAMP_MULT;
                ent.position += ent.velocity;
                //     window.request_redraw();
                // }
                // E::RedrawEventsCleared => {
                // frames += 1;
                // if frames % 16 == 0 {
                //     println!("FPS = {:?}", frames as f32 / started_at.elapsed().as_secs_f32());
                // }
                // let before = Instant::now();
                let look_at = {
                    let eye = ent.position;
                    let center = eye + ent.rotation_state.mul_vec3(Vec3::new(1., 0., 0.));
                    let up = ent.rotation_state.mul_vec3(Vec3::new(0., 0., -1.));
                    Mat4::look_at_lh(eye, center, up)
                };
                box_rot += box_rot_delta;
                let views = [
                    {
                        let look_at = {
                            let eye = ent.position;
                            let center = eye + ent.rotation_state.mul_vec3(Vec3::new(1., 0., 0.));
                            let up = ent.rotation_state.mul_vec3(Vec3::new(0., 0., -1.));
                            Mat4::look_at_lh(eye, center, up)
                        };
                        persp * look_at
                    },
                    { persp * look_at * { Mat4::from_rotation_z(2. * box_rot) } },
                    { persp * look_at * { Mat4::from_rotation_x(PI - box_rot) } },
                ];
                let instance_ranges =
                    [0..MAX_INSTANCES, (MAX_INSTANCES / 2)..MAX_INSTANCES, 0..(MAX_INSTANCES / 2)];
                let passes = views.iter().zip(instance_ranges.iter().cloned()).map(
                    |(view, instance_range)| DrawInfo::new(view, 0..max_tri_verts, instance_range),
                );
                renderer.render_instances(0, passes).unwrap();
                // println!("render took {:?}", before.elapsed());
                window.request_redraw();
            }
            _ => {}
        }
        // println!("{:?}", &ent.keys);
    })
}
