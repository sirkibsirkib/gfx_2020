mod renderer;
mod simple_arena;

use {
    gfx_hal::{self as hal, prelude::*, pso::Face, window::Extent2D},
    std::time::{Duration, Instant},
};
pub use {
    glam::{Mat4, Quat, Vec3},
    image,
    renderer::{vert_coord_consts, DrawInfo, Renderer, TexScissor, VertCoord},
    winit,
};
///////////////////////////////////
pub type TexId = usize;

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
    pub max_instances: u32,
    pub max_tri_verts: u32,
}

#[allow(unused_variables)]
pub trait UserSide<B: hal::Backend> {
    fn init(&mut self, renderer: &mut Renderer<B>) -> Result<(), ()>;
    fn handle_event(
        &mut self,
        renderer: &mut Renderer<B>,
        event: winit::event::Event<()>,
    ) -> Result<(), ()> {
        Ok(())
    }
    fn update(&mut self, renderer: &mut Renderer<B>) -> Result<(), ()> {
        Ok(())
    }
    fn render(&mut self, renderer: &mut Renderer<B>) -> Result<&[DrawInfo], ()> {
        Ok(&[])
    }
}

//////////////////////////////////////////////

pub fn main_loop<B: hal::Backend, U: UserSide<B>>(
    user_side: &'static mut U,
    config: &RendererConfig,
) -> Result<(), ()> {
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
    user_side.init(&mut renderer)?;
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
                    if let Err(()) = user_side.update(&mut renderer) {
                        *control_flow = ControlFlow::Exit;
                        return;
                    }
                }
                window.request_redraw();
            }
            E::RedrawEventsCleared => {
                match user_side.render(&mut renderer) {
                    Ok(draw_info_slice) => {
                        renderer.render_instances(0, draw_info_slice.iter().cloned()).unwrap()
                    }
                    Err(()) => {
                        *control_flow = ControlFlow::Exit;
                        return;
                    }
                };
            }
            event => {
                if let Err(()) = user_side.handle_event(&mut renderer, event) {
                    *control_flow = ControlFlow::Exit;
                    return;
                }
            }
        }
    })
}
