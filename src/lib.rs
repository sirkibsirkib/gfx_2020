mod renderer;
mod simple_arena;

use {
    core::ops::Range,
    gfx_hal::{self as hal, prelude::*, pso::Face, window::Extent2D},
    std::{
        path::Path,
        time::{Duration, Instant},
    },
};
pub use {
    gfx_hal,
    glam::{self, Mat4, Quat, Vec2, Vec3},
    image,
    renderer::Renderer,
    winit,
};
pub mod vert_coord_consts {
    use super::{Vec2, Vec3, VertCoord};
    const N: f32 = -0.5; // consider changing s.t. up is +y for later (more standard)
    const P: f32 = 0.5;
    const TL: VertCoord =
        VertCoord { model_coord: Vec3 { x: N, y: N, z: 0. }, tex_coord: Vec2 { x: 0., y: 0. } };
    const TR: VertCoord =
        VertCoord { model_coord: Vec3 { x: P, y: N, z: 0. }, tex_coord: Vec2 { x: 1., y: 0. } };
    const BR: VertCoord =
        VertCoord { model_coord: Vec3 { x: P, y: P, z: 0. }, tex_coord: Vec2 { x: 1., y: 1. } };
    const BL: VertCoord =
        VertCoord { model_coord: Vec3 { x: N, y: P, z: 0. }, tex_coord: Vec2 { x: 0., y: 1. } };
    pub const UNIT_QUAD: [VertCoord; 6] = [BR, TR, TL, TL, BL, BR];
}
///////////////////////////////////

#[derive(Debug, Copy, Clone)]
pub struct HaltLoop;

pub type TexId = usize;
pub type ProceedWith<T> = Result<T, HaltLoop>;
pub type Proceed = ProceedWith<()>;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VertCoord {
    pub tex_coord: Vec2,
    pub model_coord: Vec3,
}
#[derive(Debug, Clone)]
pub struct DrawInfo {
    pub view_transform: Mat4,
    pub vertex_range: Range<u32>,
    pub instance_range: Range<u32>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TexScissor {
    pub top_left: Vec2,
    pub size: Vec2,
}

#[derive(Debug, Clone)]
pub struct RendererConfig<'a> {
    pub init: RendererInitConfig<'a>,
    pub max_buffer_args: MaxBufferArgs,
    pub init_update_delta: Duration,
}
#[derive(Debug, Clone)]
pub struct RendererInitConfig<'a> {
    pub window_dims: Extent2D,
    pub window_title: &'a str,
    pub cull_face: Face,
}
#[derive(Debug, Clone)]
pub struct MaxBufferArgs {
    pub max_tri_verts: u32,
    pub max_instances: u32,
}

#[allow(unused_variables)]
pub trait DrivesMainLoop {
    fn handle_event<B: hal::Backend>(
        &mut self,
        renderer: &mut Renderer<B>,
        event: winit::event::Event<()>,
    ) -> Proceed {
        use winit::event::{
            Event as Ev, KeyboardInput as Ki, VirtualKeyCode as Vkc, WindowEvent as We,
        };
        match event {
            Ev::WindowEvent { event: We::CloseRequested, .. }
            | Ev::WindowEvent {
                event:
                    We::KeyboardInput { input: Ki { virtual_keycode: Some(Vkc::Escape), .. }, .. },
                ..
            } => Err(HaltLoop),
            _ => Ok(()),
        }
    }

    fn update<B: hal::Backend>(&mut self, renderer: &mut Renderer<B>) -> Proceed {
        Ok(())
    }

    fn render<B: hal::Backend>(
        &mut self,
        renderer: &mut Renderer<B>,
    ) -> ProceedWith<(TexId, &[DrawInfo])>;
}

pub fn main_loop<B, D, I>(config: &RendererConfig, state_init: I)
where
    B: hal::Backend,
    D: DrivesMainLoop + 'static,
    I: FnOnce(&mut Renderer<B>) -> ProceedWith<&'static mut D>,
{
    let event_loop = winit::event_loop::EventLoop::new();
    let wb = winit::window::WindowBuilder::new()
        .with_resizable(false)
        .with_min_inner_size(winit::dpi::Size::Logical(winit::dpi::LogicalSize::new(64.0, 64.0)))
        .with_inner_size(winit::dpi::Size::Physical(winit::dpi::PhysicalSize::new(
            config.init.window_dims.width,
            config.init.window_dims.height,
        )))
        .with_title(config.init.window_title.to_string());
    let window = wb.build(&event_loop).unwrap();
    let instance = B::Instance::create(config.init.window_title, 1).unwrap();
    let surface = unsafe { instance.create_surface(&window).unwrap() };
    let adapter = instance.enumerate_adapters().into_iter().next().unwrap();
    let mut renderer = Renderer::<B>::new(instance, surface, adapter, &config);
    let state = match state_init(&mut renderer) {
        Ok(state) => state,
        Err(HaltLoop) => return,
    };
    let mut next_update_after = Instant::now();
    event_loop.run(move |event, _, control_flow| {
        use winit::{
            event::{Event as E, WindowEvent as We},
            event_loop::ControlFlow,
        };
        *control_flow = ControlFlow::Poll;
        match event {
            E::WindowEvent { event: We::Resized(_), .. } => unreachable!(),
            E::MainEventsCleared => {
                let now = Instant::now();
                while next_update_after < now {
                    next_update_after += renderer.update_delta;
                    if let Err(HaltLoop) = state.update(&mut renderer) {
                        *control_flow = ControlFlow::Exit;
                        return;
                    }
                }
                window.request_redraw();
            }
            E::RedrawEventsCleared => {
                match state.render(&mut renderer) {
                    Ok((tex_id, draw_info_slice)) => {
                        renderer.render_instances(tex_id, draw_info_slice.iter()).unwrap()
                    }
                    Err(HaltLoop) => {
                        *control_flow = ControlFlow::Exit;
                        return;
                    }
                };
            }
            event => {
                if let Err(HaltLoop) = state.handle_event(&mut renderer, event) {
                    *control_flow = ControlFlow::Exit;
                    return;
                }
            }
        }
    })
}

//////////////////////////////////////////////
impl Default for RendererConfig<'_> {
    fn default() -> Self {
        Self {
            init: Default::default(),
            max_buffer_args: Default::default(),
            init_update_delta: Duration::from_millis(16),
        }
    }
}
impl Default for MaxBufferArgs {
    fn default() -> Self {
        Self { max_tri_verts: 256, max_instances: 2048 }
    }
}
impl Default for RendererInitConfig<'_> {
    fn default() -> Self {
        Self {
            window_dims: Extent2D { width: 600, height: 600 },
            window_title: "My program using gfx_2020",
            cull_face: Face::BACK,
        }
    }
}
pub fn load_texture_from_path(
    path: impl AsRef<Path>,
) -> Result<image::ImageBuffer<image::Rgba<u8>, Vec<u8>>, String> {
    Ok(image::io::Reader::open(path)
        .map_err(|e| format!("{}", e))?
        .decode()
        .map_err(|e| format!("{}", e))?
        .to_rgba8())
}
