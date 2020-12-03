mod renderer;
mod simple_arena;

use {
    gfx_backend_vulkan as back,
    gfx_hal::{self as hal, prelude::*},
};
pub use {
    glam::Mat4,
    renderer::{DrawInfo, Renderer, TexScissor},
};

const DIMS: hal::window::Extent2D = hal::window::Extent2D { width: 800, height: 800 };

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

    // instantiate backend
    let window = wb.build(&event_loop).unwrap();
    window.set_cursor_grab(true).unwrap();
    window.set_cursor_visible(false);
    let instance = back::Instance::create("gfx-rs quad", 1).expect("Failed to create an instance!");
    let surface = unsafe { instance.create_surface(&window).expect("Failed to create a surface!") };
    let adapter = instance.enumerate_adapters().into_iter().next().unwrap();
    const MAX_INSTANCES: u32 = 10_000;
    let mut renderer = Renderer::new(instance, surface, adapter, MAX_INSTANCES);

    let img_rgba =
        image::io::Reader::open("./src/data/logo.png").unwrap().decode().unwrap().to_rgba();
    renderer.add_image(img_rgba).unwrap();

    let mut rng = rand::thread_rng();
    use rand::Rng;
    let xy_dist = rand::distributions::Uniform::new(-1., 1.);
    let z_dist = rand::distributions::Uniform::new(0., 1.);
    let s_dist = rand::distributions::Uniform::new(0.01, 0.03);
    const SHEET_LAYOUT: [usize; 2] = [11, 5];
    let sheet_x_dist = rand::distributions::Uniform::from(0..SHEET_LAYOUT[0]);
    let sheet_y_dist = rand::distributions::Uniform::from(0..SHEET_LAYOUT[1]);
    let rot_z_dist = rand::distributions::Uniform::new(0., PI * 2.);
    renderer
        .write_instance_t_buffer(
            0,
            (0..MAX_INSTANCES).map(|_| {
                let moved = {
                    let [tx, ty, tz] =
                        [rng.sample(&xy_dist), rng.sample(&xy_dist), rng.sample(&z_dist)];
                    Mat4::from_translation([tx, ty, tz].into())
                };
                let scale = {
                    let [sx, sy] = [rng.sample(&s_dist), rng.sample(&s_dist)];
                    Mat4::from_scale([sx, sy, 0.].into())
                };
                let rotate = Mat4::from_rotation_z(rng.sample(&rot_z_dist));
                (moved * scale * rotate).into()
            }),
        )
        .unwrap();
    renderer
        .write_instance_s_buffer(
            0,
            (0..MAX_INSTANCES).map(|_| {
                let top_left = [
                    rng.sample(&sheet_x_dist) as f32 / SHEET_LAYOUT[0] as f32,
                    rng.sample(&sheet_y_dist) as f32 / SHEET_LAYOUT[1] as f32,
                ];
                const TILE_SIZE: [f32; 2] = [
                    1. / SHEET_LAYOUT[0] as f32, //
                    1. / SHEET_LAYOUT[1] as f32,
                ];
                TexScissor { top_left, size: TILE_SIZE }
            }),
        )
        .unwrap();

    // It is important that the closure move captures the Renderer,
    // otherwise it will not be dropped when the event loop exits.
    use std::f32::consts::PI;

    const DRAWS_PER_FRAME: usize = 100;
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
            E::WindowEvent { event: We::CloseRequested, .. }
            | E::WindowEvent {
                event:
                    We::KeyboardInput { input: Ki { virtual_keycode: Some(Vkc::Escape), .. }, .. },
                ..
            } => *control_flow = ControlFlow::Exit,
            E::WindowEvent { event: We::Resized(_), .. } => unreachable!(),
            E::RedrawEventsCleared => {
                let draw_info_iter =
                    view_transforms.iter().map(|trans| DrawInfo::new(trans, 0..MAX_INSTANCES));
                renderer.render_instances(0, draw_info_iter).unwrap();
                let before = std::time::Instant::now();
                println!("{:?}", before.elapsed());
            }
            _ => {}
        }
    })
}
