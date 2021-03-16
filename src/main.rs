mod renderer;
mod simple_arena;
const DIMS: hal::window::Extent2D = hal::window::Extent2D { width: 800, height: 800 };

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
    image,
    renderer::{vert_coord_consts, DrawInfo, Renderer, TexScissor, VertCoord},
    winit,
};

trait UserSide {
    fn init(&mut self);
    fn handle(&mut self, event: winit::event::WindowEvent);
    fn render(&mut self);
    fn update(&mut self);
}

struct Ctx<B: hal::Backend> {
    pub update_delta: f32, // seconds
    renderer: Renderer<B>,
}

fn rand_quat<R: Rng>(rng: &mut R) -> Quat {
    let mut sample_fn = move || rng.gen::<f32>() * PI * 2.;
    Quat::from_rotation_x(sample_fn())
        * Quat::from_rotation_y(sample_fn())
        * Quat::from_rotation_z(sample_fn())
}

fn main() {
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
    arrows.push(Arrow { pos: Vec3::new(0., 0., -1.), vel: Vec3::default() });
    arrows.push(Arrow { pos: Vec3::new(1., 0., -1.), vel: Vec3::default() });
    arrows.push(Arrow { pos: Vec3::new(0., 1., -1.), vel: Vec3::default() });

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
    let persp = Mat4::perspective_rh(1., 1., 0.5, 17.);
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
                yp.pitch = (yp.pitch + y as f32 * MULT).min(PI).max(0.);
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
                let look_at = Mat4::look_at_rh(eye, eye - (yp.quat() * ONE), UP);
                // let looK_at = Mat4::from_quat(-yp.quat()) * Mat4::from_translation(-eye);
                let view = persp * look_at;
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
