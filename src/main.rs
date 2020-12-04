mod renderer;
mod simple_arena;

use core::f32::consts::PI;
use glam::Vec3;
use {
    gfx_backend_vulkan as back,
    gfx_hal::{self as hal, prelude::*},
};
pub use {
    glam::Mat4,
    renderer::{vert_coord_consts, DrawInfo, Renderer, TexScissor, VertCoord},
};

const DIMS: hal::window::Extent2D = hal::window::Extent2D { width: 800, height: 800 };

fn main() {
    three_dimensions()
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
    use rand::{rngs::ThreadRng, Rng};
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

fn three_dimensions() {
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
    let instance = back::Instance::create("gfx-rs 3d", 1).unwrap();
    let surface = unsafe { instance.create_surface(&window).unwrap() };
    let adapter = instance.enumerate_adapters().into_iter().next().unwrap();
    // use glam::Vec3;
    // const P: f32 = 1.0;
    // const N: f32 = -1.;
    // const NNN: Vec3 = Vec3 { x: N, y: N, z: N };
    // const NNP: Vec3 = Vec3 { x: N, y: N, z: P };
    // const NPN: Vec3 = Vec3 { x: N, y: P, z: N };
    // const NPP: Vec3 = Vec3 { x: N, y: P, z: P };
    // const PNN: Vec3 = Vec3 { x: P, y: N, z: N };
    // const PNP: Vec3 = Vec3 { x: P, y: N, z: P };
    // const PPN: Vec3 = Vec3 { x: P, y: P, z: N };
    // const PPP: Vec3 = Vec3 { x: P, y: P, z: P };
    // const VERT_COORDS: &[VertCoord] = &[
    //     //d
    //     VertCoord,
    // ];

    let quad_rots = &[
        Mat4::from_rotation_x(0.) * Mat4::from_rotation_z(0.), // top
        Mat4::from_rotation_x(PI) * Mat4::from_rotation_z(0.), // bottom
        Mat4::from_rotation_x(PI * -0.5) * Mat4::from_rotation_z(0.), // front
        Mat4::from_rotation_x(PI * 0.5) * Mat4::from_rotation_z(0.), // back
        Mat4::from_rotation_y(PI * -0.5) * Mat4::from_rotation_z(0.), // left
        Mat4::from_rotation_y(PI * 0.5) * Mat4::from_rotation_z(0.), // right
    ];
    const THIRD: f32 = 1. / 3.;
    let size = [0.5, THIRD];
    let tex_scissors = (0..3).flat_map(move |i| {
        (0..2).map(move |j| TexScissor { size, top_left: [i as f32 / 3., j as f32 / 2.] })
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
    for p in vert_coord_iter.clone() {
        println!("-- {:#?}", p);
    }
    let max_tri_verts = (quad_rots.len() * vert_coord_consts::UNIT_QUAD.len()) as u32;
    assert_eq!(max_tri_verts, 6 * 6);
    const MAX_INSTANCES: u32 = 3;
    let mut renderer = Renderer::new(instance, surface, adapter, max_tri_verts, MAX_INSTANCES);
    let img_rgba =
        image::io::Reader::open("./src/data/3d.png").unwrap().decode().unwrap().to_rgba();
    renderer.load_texture(&img_rgba);
    renderer.write_vertex_buffer(6, vert_coord_iter.take(6));
    renderer.write_vertex_buffer(
        0,
        [
            Mat4::identity(),
            Mat4::from_translation([2., 0., 0.].into()),
            Mat4::from_translation([0., 2., 0.].into()),
        ]
        .iter()
        .copied(),
    );
    renderer.write_vertex_buffer(
        0,
        std::iter::repeat(TexScissor { top_left: [0., 0.], size: [THIRD, 1.] }).take(3),
    );

    let mut t = 0;
    event_loop.run(move |event, _, control_flow| {
        println!("{:?}", event);
        use winit::{
            event::{Event as E, KeyboardInput as Ki, VirtualKeyCode as Vkc, WindowEvent as We},
            event_loop::ControlFlow,
        };
        *control_flow = ControlFlow::WaitUntil(
            std::time::Instant::now() + std::time::Duration::from_millis(16),
        );
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
                let view = Mat4::from_translation([0., 0., 0.5].into()) // push deeper
                    * Mat4::from_scale([0.3, 0.3, 0.01].into()) // squash_axes
                    * Mat4::from_rotation_z(t as f32 / 1_000.)
                    * Mat4::from_rotation_x(t as f32 / 200.)
                    * Mat4::from_rotation_y(t as f32 / 70.);
                renderer
                    .render_instances(
                        0,
                        Some(DrawInfo::new(&view, 0..max_tri_verts, 0..MAX_INSTANCES)),
                    )
                    .unwrap();
                println!("{:?}", before.elapsed());
                t += 1;
            }
            _ => {}
        }
    })
}
