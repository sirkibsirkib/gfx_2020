mod renderer;
mod simple_arena;

use gfx_backend_vulkan as back;
use gfx_hal::{self as hal, prelude::*};

use glam::Mat4;
pub use renderer::{Rect, Renderer};

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
    const MAX_INSTANCES: u32 = 16;
    let mut renderer = Renderer::new(instance, surface, adapter, MAX_INSTANCES);

    let img_rgba =
        image::io::Reader::open("./src/data/logo.png").unwrap().decode().unwrap().to_rgba();
    renderer.add_image(img_rgba).unwrap();

    let mut instance_t_data = [Mat4::default(); MAX_INSTANCES as usize];
    let mut instance_s_data = [Rect::default(); MAX_INSTANCES as usize];
    for (i, (t, s)) in instance_t_data.iter_mut().zip(instance_s_data.iter_mut()).enumerate() {
        *t = {
            let tx = i as f32 / 10.;
            let ty = i as f32 / 46.;
            let tz = (i % 3) as f32 / 3.;
            let moved = Mat4::from_translation([tx, ty, tz].into());
            let scale = Mat4::from_scale([0.2; 3].into());
            (moved * scale).into()
        };
        const TILE_SIZE: [f32; 2] = [1. / 11., 1. / 5.];
        *s = {
            let top_left = [i as f32 * TILE_SIZE[0], 0. * TILE_SIZE[1]];
            Rect { top_left, size: TILE_SIZE }
        };
    }
    renderer.write_instance_t_buffer(0, &instance_t_data).unwrap();
    renderer.write_instance_s_buffer(0, &instance_s_data).unwrap();

    // It is important that the closure move captures the Renderer,
    // otherwise it will not be dropped when the event loop exits.
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
            E::RedrawEventsCleared => renderer.render(0, 0..MAX_INSTANCES).unwrap(),
            _ => {}
        }
    })
}
