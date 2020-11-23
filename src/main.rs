use gfx_backend_vulkan as back;
use gfx_hal::{
    self as hal, command,
    format::{self as f, AsFormat, ChannelType, Rgba8Srgb as ColorFormat, Swizzle},
    image as i, memory as m,
    pass::{self, Subpass},
    prelude::*,
    pso::{self, PipelineStage, ShaderStageFlags, VertexInputRate},
    queue::{QueueGroup, Submission},
    window,
};

use std::{
    borrow::Borrow,
    io::Cursor,
    iter,
    mem::{self, ManuallyDrop},
    ptr,
};

const DIMS: window::Extent2D = window::Extent2D {
    width: 1024,
    height: 768,
};

const ENTRY_NAME: &str = "main";

#[derive(Debug, Clone, Copy)]
#[allow(non_snake_case)]
struct Vertex {
    a_Pos: [f32; 2], // on screen
    a_Uv: [f32; 2],  // in texture
}

const PIC_X: f32 = 0.5;
const PIC_Y: f32 = 0.33;
const QUAD: [Vertex; 6] = [
    Vertex {
        a_Pos: [-PIC_X, PIC_Y], // A
        a_Uv: [0.0, 1.0],
    },
    Vertex {
        a_Pos: [PIC_X, PIC_Y], // B
        a_Uv: [1.0, 1.0],
    },
    Vertex {
        a_Pos: [PIC_X, -PIC_Y], // C
        a_Uv: [1.0, 0.0],
    },
    Vertex {
        a_Pos: [-PIC_X, PIC_Y], // A
        a_Uv: [0.0, 1.0],
    },
    Vertex {
        a_Pos: [PIC_X, -PIC_Y], // C
        a_Uv: [1.0, 0.0],
    },
    Vertex {
        a_Pos: [-PIC_X, -PIC_Y], // D
        a_Uv: [0.0, 0.0],
    },
];

fn main() {
    let event_loop = winit::event_loop::EventLoop::new();

    let wb = winit::window::WindowBuilder::new()
        .with_min_inner_size(winit::dpi::Size::Logical(winit::dpi::LogicalSize::new(
            64.0, 64.0,
        )))
        .with_inner_size(winit::dpi::Size::Physical(winit::dpi::PhysicalSize::new(
            DIMS.width,
            DIMS.height,
        )))
        .with_title("quad".to_string());

    // instantiate backend
    let window = wb.build(&event_loop).unwrap();
    let instance = back::Instance::create("gfx-rs quad", 1).expect("Failed to create an instance!");
    let surface = unsafe {
        instance
            .create_surface(&window)
            .expect("Failed to create a surface!")
    };

    for adapter in instance.enumerate_adapters().iter() {
        // DEBUG
        println!("{:?}", adapter.info);
    }

    let adapter = instance.enumerate_adapters().into_iter().next().unwrap();
    let mut renderer = Renderer::new(instance, surface, adapter);

    renderer.render();

    // It is important that the closure move captures the Renderer,
    // otherwise it will not be dropped when the event loop exits.
    event_loop.run(move |event, _, control_flow| {
        *control_flow = winit::event_loop::ControlFlow::Wait;

        match event {
            winit::event::Event::WindowEvent { event, .. } => match event {
                winit::event::WindowEvent::CloseRequested => {
                    *control_flow = winit::event_loop::ControlFlow::Exit
                }
                winit::event::WindowEvent::KeyboardInput {
                    input:
                        winit::event::KeyboardInput {
                            virtual_keycode: Some(winit::event::VirtualKeyCode::Escape),
                            ..
                        },
                    ..
                } => *control_flow = winit::event_loop::ControlFlow::Exit,
                winit::event::WindowEvent::Resized(dims) => {
                    println!("resized to {:?}", dims);
                    renderer.dimensions = window::Extent2D {
                        width: dims.width,
                        height: dims.height,
                    };
                    renderer.recreate_swapchain();
                }
                _ => {}
            },
            winit::event::Event::RedrawEventsCleared => renderer.render(),
            _ => {}
        }
    });
}

#[macro_use]
extern crate debug_stub_derive;

#[derive(Debug)]
struct Renderer<B: hal::Backend> {
    device: B::Device,
    queue_group: QueueGroup<B>,
    adapter: hal::adapter::Adapter<B>,
    format: hal::format::Format,
    dimensions: window::Extent2D,
    viewport: pso::Viewport,
    desc_set: B::DescriptorSet,
    inner: ManuallyDrop<RendererInner<B>>,
}

/// Things that must be manually dropped, because they correspond to Gfx resources
#[derive(DebugStub)]
struct RendererInner<B: hal::Backend> {
    #[debug_stub = "Instance"]
    instance: B::Instance,
    render_pass: B::RenderPass,
    pipeline: B::GraphicsPipeline,
    pipeline_layout: B::PipelineLayout,
    desc_pool: B::DescriptorPool,
    surface: B::Surface,
    submission_complete_semaphore: B::Semaphore,
    submission_complete_fence: B::Fence,
    cmd_pool: B::CommandPool,
    cmd_buffer: B::CommandBuffer,
    set_layout: B::DescriptorSetLayout,
    vertex_buffer: B::Buffer,
    image_upload_buffer: B::Buffer,
    image_logo: B::Image,
    image_srv: B::ImageView,
    buffer_memory: B::Memory,
    image_memory: B::Memory,
    image_upload_memory: B::Memory,
    sampler: B::Sampler,
}

impl<B> Renderer<B>
where
    B: hal::Backend,
{
    fn new(
        instance: B::Instance,
        mut surface: B::Surface,
        adapter: hal::adapter::Adapter<B>,
    ) -> Self {
        let memory_types = adapter.physical_device.memory_properties().memory_types;
        let limits = adapter.physical_device.limits();

        // Build a new device and associated command queues
        let family = adapter
            .queue_families
            .iter()
            .find(|family| {
                surface.supports_queue_family(family) && family.queue_type().supports_graphics()
            })
            .unwrap();
        for (i, fam) in adapter.queue_families.iter().enumerate() {
            println!("fam {}: {:#?}", i, fam);
        }
        dbg!(&memory_types, &limits, &family);
        let (mut queue_group, device) = {
            let mut gpu = unsafe {
                adapter
                    .physical_device
                    .open(&[(family, &[1.0])], hal::Features::empty())
                    .unwrap()
            };
            (gpu.queue_groups.pop().unwrap(), gpu.device)
        };

        // PSO: list of stages TODO
        // command queue: list of pools: list of commands(tasks?):
        // 1 task is an enum: one variant can refer to a PSO
        // ----
        // creating/destroying PSOs is expensive
        // appending commands/tasks is cheap

        // Setup renderpass and pipeline
        let set_layout = unsafe {
            // arugments that are passed to fragment function
            device.create_descriptor_set_layout(
                &[
                    pso::DescriptorSetLayoutBinding {
                        binding: 0,
                        // SOME IMAGE
                        // see quad.frag. "arguments" to shader
                        ty: pso::DescriptorType::Image {
                            ty: pso::ImageDescriptorType::Sampled {
                                with_sampler: false,
                            },
                        },
                        count: 1,
                        stage_flags: ShaderStageFlags::FRAGMENT, // need during frament
                        immutable_samplers: false,
                    },
                    pso::DescriptorSetLayoutBinding {
                        // a reference to the sampler
                        binding: 1,
                        ty: pso::DescriptorType::Sampler,
                        count: 1,
                        stage_flags: ShaderStageFlags::FRAGMENT, // need during frament
                        immutable_samplers: false,
                    },
                ],
                &[], // immutable samplers? lel
            )
        }
        .expect("Can't create descriptor set layout");

        // Descriptors
        let mut desc_pool = unsafe {
            device.create_descriptor_pool(
                1, // sets
                &[
                    pso::DescriptorRangeDesc {
                        ty: pso::DescriptorType::Image {
                            ty: pso::ImageDescriptorType::Sampled {
                                with_sampler: false,
                            },
                        },
                        count: 1,
                    },
                    pso::DescriptorRangeDesc {
                        ty: pso::DescriptorType::Sampler,
                        count: 1,
                    },
                ],
                pso::DescriptorPoolCreateFlags::empty(),
            )
        }
        .expect("Can't create descriptor pool");
        let desc_set = unsafe { desc_pool.allocate_set(&set_layout) }.unwrap();

        // Buffer allocations
        println!("Memory types: {:#?}", memory_types);
        let non_coherent_alignment = limits.non_coherent_atom_size as u64;

        let buffer_stride = mem::size_of::<Vertex>() as u64;
        let buffer_len = QUAD.len() as u64 * buffer_stride;
        assert_ne!(buffer_len, 0);
        let padded_buffer_len = ((buffer_len + non_coherent_alignment - 1)
            / non_coherent_alignment)
            * non_coherent_alignment;

        let mut vertex_buffer =
            unsafe { device.create_buffer(padded_buffer_len, hal::buffer::Usage::VERTEX) }.unwrap();

        let buffer_req = unsafe { device.get_buffer_requirements(&vertex_buffer) };

        let upload_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, mem_type)| {
                // type_mask is a bit field where each bit represents a memory type. If the bit is set
                // to 1 it means we can use that type for our buffer. So this code finds the first
                // memory type that has a `1` (or, is allowed), and is visible to the CPU.
                buffer_req.type_mask & (1 << id) != 0
                    && mem_type.properties.contains(m::Properties::CPU_VISIBLE)
            })
            .unwrap()
            .into();

        // TODO: check transitions: read/write mapping and vertex buffer read
        let buffer_memory = unsafe {
            let memory = device
                .allocate_memory(upload_type, buffer_req.size)
                .unwrap();
            device
                .bind_buffer_memory(&memory, 0, &mut vertex_buffer)
                .unwrap();
            let mapping = device.map_memory(&memory, m::Segment::ALL).unwrap();
            ptr::copy_nonoverlapping(QUAD.as_ptr() as *const u8, mapping, buffer_len as usize);
            device
                .flush_mapped_memory_ranges(iter::once((&memory, m::Segment::ALL)))
                .unwrap();
            device.unmap_memory(&memory);
            memory
        };

        // Image
        let img = image::io::Reader::open("./src/data/logo.png")
            .unwrap()
            .decode()
            .unwrap()
            .to_rgba();
        // let img = image::load(
        //     Cursor::new(include_bytes!("data/logo.png")),
        //     image::ImageFormat::Png,
        // )
        // .unwrap()
        // .to_rgba();
        let (width, height) = img.dimensions();
        let kind = i::Kind::D2(width as i::Size, height as i::Size, 1, 1);
        let row_alignment_mask = limits.optimal_buffer_copy_pitch_alignment as u32 - 1;
        let image_stride = 4usize;
        let row_pitch = (width * image_stride as u32 + row_alignment_mask) & !row_alignment_mask;
        let upload_size = (height * row_pitch) as u64;
        let padded_upload_size = ((upload_size + non_coherent_alignment - 1)
            / non_coherent_alignment)
            * non_coherent_alignment;

        let mut image_upload_buffer =
            unsafe { device.create_buffer(padded_upload_size, hal::buffer::Usage::TRANSFER_SRC) }
                .unwrap();
        let image_mem_reqs = unsafe { device.get_buffer_requirements(&image_upload_buffer) };

        // copy image data into staging buffer
        let image_upload_memory = unsafe {
            let memory = device
                .allocate_memory(upload_type, image_mem_reqs.size)
                .unwrap();
            device
                .bind_buffer_memory(&memory, 0, &mut image_upload_buffer)
                .unwrap();
            let mapping = device.map_memory(&memory, m::Segment::ALL).unwrap();
            for y in 0..height as usize {
                let row = &(*img)[y * (width as usize) * image_stride
                    ..(y + 1) * (width as usize) * image_stride];
                ptr::copy_nonoverlapping(
                    row.as_ptr(),
                    mapping.offset(y as isize * row_pitch as isize),
                    width as usize * image_stride,
                );
            }
            device
                .flush_mapped_memory_ranges(iter::once((&memory, m::Segment::ALL)))
                .unwrap();
            device.unmap_memory(&memory);
            memory
        };

        let mut image_logo = unsafe {
            device.create_image(
                kind,
                1,
                ColorFormat::SELF,
                i::Tiling::Optimal,
                i::Usage::TRANSFER_DST | i::Usage::SAMPLED,
                i::ViewCapabilities::empty(),
            )
        }
        .unwrap();
        let image_req = unsafe { device.get_image_requirements(&image_logo) };

        let device_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, memory_type)| {
                image_req.type_mask & (1 << id) != 0
                    && memory_type.properties.contains(m::Properties::DEVICE_LOCAL)
            })
            .unwrap()
            .into();
        let image_memory = unsafe { device.allocate_memory(device_type, image_req.size) }.unwrap();

        unsafe { device.bind_image_memory(&image_memory, 0, &mut image_logo) }.unwrap();
        let image_srv = unsafe {
            device.create_image_view(
                &image_logo,
                i::ViewKind::D2,
                ColorFormat::SELF,
                Swizzle::NO,
                i::SubresourceRange {
                    aspects: f::Aspects::COLOR,
                    ..Default::default()
                },
            )
        }
        .unwrap();

        let sampler = unsafe {
            device.create_sampler(&i::SamplerDesc::new(i::Filter::Linear, i::WrapMode::Clamp))
        }
        .expect("Can't create sampler");

        unsafe {
            device.write_descriptor_sets(iter::once(pso::DescriptorSetWrite {
                set: &desc_set,
                binding: 0,
                array_offset: 0,
                descriptors: vec![
                    pso::Descriptor::Image(&image_srv, i::Layout::ShaderReadOnlyOptimal),
                    pso::Descriptor::Sampler(&sampler),
                ],
            }));
        }

        // copy buffer to texture
        let mut cmd_pool = unsafe {
            device.create_command_pool(
                queue_group.family,
                hal::pool::CommandPoolCreateFlags::empty(),
            )
        }
        .expect("Can't create command pool");
        let mut copy_fence = device.create_fence(false).expect("Could not create fence");
        unsafe {
            let mut cmd_buffer = cmd_pool.allocate_one(command::Level::Primary);
            cmd_buffer.begin_primary(command::CommandBufferFlags::ONE_TIME_SUBMIT);

            let image_barrier = m::Barrier::Image {
                states: (i::Access::empty(), i::Layout::Undefined)
                    ..(i::Access::TRANSFER_WRITE, i::Layout::TransferDstOptimal),
                target: &image_logo,
                families: None,
                range: i::SubresourceRange {
                    aspects: f::Aspects::COLOR,
                    ..Default::default()
                },
            };

            cmd_buffer.pipeline_barrier(
                PipelineStage::TOP_OF_PIPE..PipelineStage::TRANSFER,
                m::Dependencies::empty(),
                &[image_barrier],
            );

            cmd_buffer.copy_buffer_to_image(
                &image_upload_buffer,
                &image_logo,
                i::Layout::TransferDstOptimal,
                &[command::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_width: row_pitch / (image_stride as u32),
                    buffer_height: height as u32,
                    image_layers: i::SubresourceLayers {
                        aspects: f::Aspects::COLOR,
                        level: 0,
                        layers: 0..1,
                    },
                    image_offset: i::Offset { x: 0, y: 0, z: 0 },
                    image_extent: i::Extent {
                        width,
                        height,
                        depth: 1,
                    },
                }],
            );

            let image_barrier = m::Barrier::Image {
                states: (i::Access::TRANSFER_WRITE, i::Layout::TransferDstOptimal)
                    ..(i::Access::SHADER_READ, i::Layout::ShaderReadOnlyOptimal),
                target: &image_logo,
                families: None,
                range: i::SubresourceRange {
                    aspects: f::Aspects::COLOR,
                    ..Default::default()
                },
            };
            cmd_buffer.pipeline_barrier(
                PipelineStage::TRANSFER..PipelineStage::FRAGMENT_SHADER,
                m::Dependencies::empty(),
                &[image_barrier],
            );

            cmd_buffer.finish();

            queue_group.queues[0]
                .submit_without_semaphores(Some(&cmd_buffer), Some(&mut copy_fence));

            device
                .wait_for_fence(&copy_fence, !0)
                .expect("Can't wait for fence");
        }

        unsafe {
            device.destroy_fence(copy_fence);
        }

        let caps = surface.capabilities(&adapter.physical_device);
        let formats = surface.supported_formats(&adapter.physical_device);
        println!("formats: {:?}", formats);
        let format = formats.map_or(f::Format::Rgba8Srgb, |formats| {
            formats
                .iter()
                .find(|format| format.base_format().1 == ChannelType::Srgb)
                .map(|format| *format)
                .unwrap_or(formats[0])
        });

        let swap_config = window::SwapchainConfig::from_caps(&caps, format, DIMS);
        println!("{:?}", swap_config);
        let extent = swap_config.extent;
        unsafe {
            surface
                .configure_swapchain(&device, swap_config)
                .expect("Can't configure swapchain");
        };

        let render_pass = {
            // output of frag shader
            let attachment = pass::Attachment {
                format: Some(format),
                samples: 1,
                ops: pass::AttachmentOps::new(
                    pass::AttachmentLoadOp::Clear,
                    pass::AttachmentStoreOp::Store,
                ),
                stencil_ops: pass::AttachmentOps::DONT_CARE,
                layouts: i::Layout::Undefined..i::Layout::Present,
            };
            let subpass = pass::SubpassDesc {
                colors: &[(0, i::Layout::ColorAttachmentOptimal)],
                depth_stencil: None,
                inputs: &[],
                resolves: &[],
                preserves: &[],
            };
            unsafe { device.create_render_pass(&[attachment], &[subpass], &[]) }
                .expect("Can't create render pass")
        };

        let submission_complete_semaphore = device
            .create_semaphore()
            .expect("Could not create semaphore");
        let submission_complete_fence = device.create_fence(true).expect("Could not create fence");
        let cmd_buffer = unsafe { cmd_pool.allocate_one(command::Level::Primary) };

        let pipeline_layout =
            unsafe { device.create_pipeline_layout(iter::once(&set_layout), &[]) }
                .expect("Can't create pipeline layout");
        let pipeline = {
            let vs_module = {
                let spirv =
                    gfx_auxil::read_spirv(Cursor::new(&include_bytes!("data/quad.vert.spv")[..]))
                        .unwrap();
                unsafe { device.create_shader_module(&spirv) }.unwrap()
            };
            let fs_module = {
                let spirv =
                    gfx_auxil::read_spirv(Cursor::new(&include_bytes!("./data/quad.frag.spv")[..]))
                        .unwrap();
                unsafe { device.create_shader_module(&spirv) }.unwrap()
            };

            let pipeline = {
                let (vs_entry, fs_entry) = (
                    pso::EntryPoint {
                        entry: ENTRY_NAME,
                        module: &vs_module,
                        specialization: hal::spec_const_list![0.8f32],
                    },
                    pso::EntryPoint {
                        entry: ENTRY_NAME,
                        module: &fs_module,
                        specialization: pso::Specialization::default(),
                    },
                );

                let subpass = Subpass {
                    index: 0,
                    main_pass: &render_pass,
                };

                let vertex_buffers = vec![pso::VertexBufferDesc {
                    binding: 0,
                    stride: mem::size_of::<Vertex>() as u32,
                    rate: VertexInputRate::Vertex,
                }];

                let attributes = [
                    pso::AttributeDesc {
                        location: 0,
                        binding: 0,
                        element: pso::Element {
                            format: f::Format::Rg32Sfloat,
                            offset: 0,
                        },
                    },
                    pso::AttributeDesc {
                        location: 1,
                        binding: 0,
                        element: pso::Element {
                            format: f::Format::Rg32Sfloat,
                            offset: 8,
                        },
                    },
                ];

                let mut pipeline_desc = pso::GraphicsPipelineDesc::new(
                    pso::PrimitiveAssemblerDesc::Vertex {
                        buffers: &vertex_buffers,
                        attributes: &attributes,
                        input_assembler: pso::InputAssemblerDesc {
                            primitive: pso::Primitive::TriangleList,
                            with_adjacency: false,
                            restart_index: None,
                        },
                        vertex: vs_entry,
                        geometry: None,
                        tessellation: None,
                    },
                    pso::Rasterizer::FILL,
                    Some(fs_entry),
                    &pipeline_layout,
                    subpass,
                );

                pipeline_desc.blender.targets.push(pso::ColorBlendDesc {
                    mask: pso::ColorMask::ALL,
                    blend: Some(pso::BlendState::ALPHA),
                });

                unsafe { device.create_graphics_pipeline(&pipeline_desc, None) }
            };

            unsafe {
                device.destroy_shader_module(vs_module);
            }
            unsafe {
                device.destroy_shader_module(fs_module);
            }

            pipeline.unwrap()
        };

        // Rendering setup
        let viewport = pso::Viewport {
            rect: pso::Rect {
                x: 0,
                y: 0,
                w: extent.width as _,
                h: extent.height as _,
            },
            depth: 0.0..1.0,
        };

        let me = Renderer {
            device,
            queue_group,
            adapter,
            format,
            dimensions: DIMS,
            viewport,
            desc_set,
            inner: ManuallyDrop::new(RendererInner {
                instance,
                surface,
                render_pass,
                desc_pool,
                set_layout,
                vertex_buffer,
                image_upload_buffer,
                pipeline,
                pipeline_layout,
                image_logo,
                image_srv,
                buffer_memory,
                image_memory,
                image_upload_memory,
                sampler,
                submission_complete_semaphore,
                submission_complete_fence,
                cmd_pool,
                cmd_buffer,
            }),
        };
        println!("{:#?}", &me);
        me
    }

    fn recreate_swapchain(&mut self) {
        let inner = &mut *self.inner;
        let caps = inner.surface.capabilities(&self.adapter.physical_device);
        let swap_config = window::SwapchainConfig::from_caps(&caps, self.format, self.dimensions);
        println!("SWAP CONFIG {:?}", swap_config);
        let extent = swap_config.extent.to_extent();

        unsafe {
            inner
                .surface
                .configure_swapchain(&self.device, swap_config)
                .expect("Can't create swapchain");
        }

        self.viewport.rect.w = extent.width as _;
        self.viewport.rect.h = extent.height as _;
    }

    fn render(&mut self) {
        let inner = &mut *self.inner;
        let surface_image = unsafe {
            match inner.surface.acquire_image(!0) {
                Ok((image, _)) => image,
                Err(_) => {
                    self.recreate_swapchain();
                    return;
                }
            }
        };

        let framebuffer = unsafe {
            self.device
                .create_framebuffer(
                    &inner.render_pass,
                    iter::once(surface_image.borrow()),
                    i::Extent {
                        width: self.dimensions.width,
                        height: self.dimensions.height,
                        depth: 1,
                    },
                )
                .unwrap()
        };

        // Wait for the fence of the previous submission of this frame and reset it; ensures we are
        // submitting only up to maximum number of frames_in_flight if we are submitting faster than
        // the gpu can keep up with. This would also guarantee that any resources which need to be
        // updated with a CPU->GPU data copy are not in use by the GPU, so we can perform those updates.
        // In this case there are none to be done, however.
        unsafe {
            let fence = &inner.submission_complete_fence;
            self.device
                .wait_for_fence(fence, !0)
                .expect("Failed to wait for fence");
            self.device
                .reset_fence(fence)
                .expect("Failed to reset fence");
            inner.cmd_pool.reset(false);
        }

        // Rendering
        unsafe {
            inner
                .cmd_buffer
                .begin_primary(command::CommandBufferFlags::ONE_TIME_SUBMIT);

            inner.cmd_buffer.set_viewports(0, &[self.viewport.clone()]); // normalized device -> screen coords
            inner.cmd_buffer.set_scissors(0, &[self.viewport.rect]); // TODO mess with this and see if it crops
            inner.cmd_buffer.bind_graphics_pipeline(&inner.pipeline);
            inner.cmd_buffer.bind_vertex_buffers(
                0,
                iter::once((&inner.vertex_buffer, hal::buffer::SubRange::WHOLE)),
            );
            inner.cmd_buffer.bind_graphics_descriptor_sets(
                &inner.pipeline_layout,
                0,
                iter::once(&self.desc_set),
                &[],
            );

            inner.cmd_buffer.begin_render_pass(
                &inner.render_pass,
                &framebuffer,
                self.viewport.rect,
                &[command::ClearValue {
                    color: command::ClearColor {
                        float32: [0.2, 0.8, 0.8, 1.0],
                    },
                }],
                command::SubpassContents::Inline,
            );
            inner.cmd_buffer.draw(0..6, 0..1);
            inner.cmd_buffer.end_render_pass();
            inner.cmd_buffer.finish();

            let submission = Submission {
                command_buffers: iter::once(&inner.cmd_buffer),
                wait_semaphores: None,
                signal_semaphores: iter::once(&inner.submission_complete_semaphore),
            };
            self.queue_group.queues[0].submit(submission, Some(&inner.submission_complete_fence));

            // present frame
            let result = self.queue_group.queues[0].present(
                &mut inner.surface,
                surface_image,
                Some(&inner.submission_complete_semaphore),
            );

            self.device.destroy_framebuffer(framebuffer);

            if result.is_err() {
                self.recreate_swapchain();
            }
        }
    }
}

impl<B> Drop for Renderer<B>
where
    B: hal::Backend,
{
    fn drop(&mut self) {
        self.device.wait_idle().unwrap();
        unsafe {
            let mut inner = ManuallyDrop::take(&mut self.inner);
            self.device.destroy_descriptor_pool(inner.desc_pool);
            self.device.destroy_descriptor_set_layout(inner.set_layout);

            self.device.destroy_buffer(inner.vertex_buffer);
            self.device.destroy_buffer(inner.image_upload_buffer);
            self.device.destroy_image(inner.image_logo);
            self.device.destroy_image_view(inner.image_srv);
            self.device.destroy_sampler(inner.sampler);
            self.device.destroy_command_pool(inner.cmd_pool);
            self.device
                .destroy_semaphore(inner.submission_complete_semaphore);

            self.device.destroy_fence(inner.submission_complete_fence);

            self.device.destroy_render_pass(inner.render_pass);
            inner.surface.unconfigure_swapchain(&self.device);
            self.device.free_memory(inner.buffer_memory);
            self.device.free_memory(inner.image_memory);
            self.device.free_memory(inner.image_upload_memory);
            self.device.destroy_graphics_pipeline(inner.pipeline);
            self.device.destroy_pipeline_layout(inner.pipeline_layout);
            inner.instance.destroy_surface(inner.surface);
        }
        println!("DROPPED!");
    }
}
