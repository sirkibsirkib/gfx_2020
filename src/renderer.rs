use {
    super::*,
    crate::simple_arena::SimpleArena,
    gfx_hal::{
        self as hal, command,
        format::{self as f, AsFormat, ChannelType, Rgba8Srgb as ColorFormat, Swizzle},
        image as i, memory as m,
        pass::{self, Subpass},
        prelude::*,
        pso::{self, PipelineStage, ShaderStageFlags, VertexInputRate},
        queue::{QueueGroup, Submission},
    },
    std::{
        borrow::Borrow,
        io::Cursor,
        iter,
        marker::PhantomData,
        mem::{self, ManuallyDrop},
        ops::Range,
    },
};

pub trait HasVertexBufferFor<B: hal::Backend, T: Copy> {
    fn get_vertex_buffer_cap(&self) -> u32;
    fn get_vertex_buffer_bundle(&self) -> &VertexBufferBundle<T, B>;
}
impl<B: hal::Backend> HasVertexBufferFor<B, VertCoord> for Renderer<B> {
    fn get_vertex_buffer_cap(&self) -> u32 {
        self.max_tri_verts
    }
    fn get_vertex_buffer_bundle(&self) -> &VertexBufferBundle<VertCoord, B> {
        &self.inner.vertex_buffer_bundles.vc
    }
}
impl<B: hal::Backend> HasVertexBufferFor<B, Mat4> for Renderer<B> {
    fn get_vertex_buffer_cap(&self) -> u32 {
        self.max_instances
    }
    fn get_vertex_buffer_bundle(&self) -> &VertexBufferBundle<Mat4, B> {
        &self.inner.vertex_buffer_bundles.m4
    }
}
impl<B: hal::Backend> HasVertexBufferFor<B, TexScissor> for Renderer<B> {
    fn get_vertex_buffer_cap(&self) -> u32 {
        self.max_instances
    }
    fn get_vertex_buffer_bundle(&self) -> &VertexBufferBundle<TexScissor, B> {
        &self.inner.vertex_buffer_bundles.ts
    }
}

#[derive(Debug, Clone)]
pub struct DrawInfo<'a> {
    pub view_transform: &'a Mat4,
    pub vertex_range: Range<u32>,
    pub instance_range: Range<u32>,
}
impl<'a> DrawInfo<'a> {
    pub fn new(
        view_transform: &'a Mat4,
        vertex_range: Range<u32>,
        instance_range: Range<u32>,
    ) -> Self {
        Self { view_transform, vertex_range, instance_range }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum RenderErr {
    UnknownTextureIndex,
}

trait AsU32Slice {
    fn as_u32_slice(&self) -> &[u32];
}
impl AsU32Slice for Mat4 {
    #[inline]
    fn as_u32_slice(&self) -> &[u32] {
        let f32_slice: &[f32] = self.as_ref();
        unsafe { mem::transmute(f32_slice) }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VertCoord {
    pub tex_coord: [f32; 2],
    pub model_coord: [f32; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TexScissor {
    pub top_left: [f32; 2],
    pub size: [f32; 2],
}
impl core::ops::Mul<Self> for TexScissor {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        use glam::Vec2;
        let size = *(Vec2::from(self.size) * Vec2::from(rhs.size)).as_ref();
        Self { top_left: self * rhs.top_left, size }
    }
}
impl core::ops::Mul<[f32; 2]> for TexScissor {
    type Output = [f32; 2];
    fn mul(self, rhs: [f32; 2]) -> [f32; 2] {
        use glam::Vec2;
        let tl = Vec2::from(self.top_left);
        let sz = Vec2::from(self.size);
        let pt = Vec2::from(rhs);
        *(tl + sz * pt).as_ref()
    }
}

impl Default for TexScissor {
    fn default() -> Self {
        Self { top_left: [0.; 2], size: [1.; 2] }
    }
}

pub mod vert_coord_consts {
    use super::VertCoord;
    const N: f32 = -0.5; // consider changing s.t. up is +y for later (more standard)
    const P: f32 = 0.5;
    const TL: VertCoord = VertCoord { model_coord: [N, N, 0.], tex_coord: [0., 0.] };
    const TR: VertCoord = VertCoord { model_coord: [P, N, 0.], tex_coord: [1., 0.] };
    const BR: VertCoord = VertCoord { model_coord: [P, P, 0.], tex_coord: [1., 1.] };
    const BL: VertCoord = VertCoord { model_coord: [N, P, 0.], tex_coord: [0., 1.] };
    pub const UNIT_QUAD: [VertCoord; 6] = [BR, TR, TL, TL, BL, BR];
}

trait DeviceDestroy<T> {
    unsafe fn device_destroy(&mut self, t: T);
}

#[derive(Debug)]
struct PerFif<B: hal::Backend> {
    cmd_buffer: B::CommandBuffer,
    fence: B::Fence,
    semaphore: B::Semaphore,
    depth_image_bundle: ImageBundle<B>,
}

#[derive(Debug)]
pub struct VertexBufferBundles<B: hal::Backend> {
    vc: VertexBufferBundle<VertCoord, B>,
    m4: VertexBufferBundle<Mat4, B>,
    ts: VertexBufferBundle<TexScissor, B>,
}

#[derive(Debug)]
pub struct VertexBufferBundle<T, B: hal::Backend> {
    buffer: B::Buffer,
    memory: B::Memory,
    buffered_type_phantom: PhantomData<T>,
    mapping_ptr: *mut T,
}

#[derive(Debug)]
struct ImageBundle<B: hal::Backend> {
    image: B::Image,
    memory: B::Memory,
    view: B::ImageView,
}

#[derive(Debug)]
pub struct Renderer<B: hal::Backend> {
    device: B::Device,
    queue_group: QueueGroup<B>,
    adapter: hal::adapter::Adapter<B>,
    format: hal::format::Format,
    viewport: pso::Viewport,
    desc_set: B::DescriptorSet,
    inner: ManuallyDrop<RendererInner<B>>,
    max_instances: u32,
    max_tri_verts: u32,
}

/// Things that must be manually dropped, because they correspond to Gfx resources
#[derive(debug_stub_derive::DebugStub)]
struct RendererInner<B: hal::Backend> {
    #[debug_stub = "Instance"]
    instance: B::Instance,
    render_pass: B::RenderPass,
    pipeline: B::GraphicsPipeline,
    pipeline_layout: B::PipelineLayout,
    desc_pool: B::DescriptorPool,
    surface: B::Surface,
    cmd_pool: B::CommandPool,
    set_layout: B::DescriptorSetLayout,
    vertex_buffer_bundles: VertexBufferBundles<B>,
    sampler: B::Sampler,
    tex_arena: SimpleArena<ImageBundle<B>>,
    per_fif: Vec<PerFif<B>>,
    next_fif_index: usize,
}

//////////////////////////////////////////////////////

const fn padded_len(n: usize, non_coherent_atom_size: usize) -> usize {
    ((n + non_coherent_atom_size - 1) / non_coherent_atom_size) * non_coherent_atom_size
}

fn mem_type_for_buffer(
    memory_types: &Vec<hal::adapter::MemoryType>,
    buffer_req: &m::Requirements,
    properties: m::Properties,
) -> Option<hal::MemoryTypeId> {
    let t = memory_types.iter().enumerate().position(|(id, mem_type)| {
        buffer_req.type_mask & (1 << id) != 0 && mem_type.properties.contains(properties)
    })?;
    Some(t.into())
}

impl<B: hal::Backend> DeviceDestroy<ImageBundle<B>> for B::Device {
    unsafe fn device_destroy(&mut self, ib: ImageBundle<B>) {
        let ImageBundle { image, memory, view } = ib;
        self.destroy_image(image);
        self.free_memory(memory);
        self.destroy_image_view(view);
    }
}
impl<B: hal::Backend> DeviceDestroy<VertexBufferBundles<B>> for B::Device {
    unsafe fn device_destroy(&mut self, vb: VertexBufferBundles<B>) {
        let VertexBufferBundles { vc, m4, ts } = vb;
        self.device_destroy(vc);
        self.device_destroy(m4);
        self.device_destroy(ts);
    }
}
impl<T, B: hal::Backend> DeviceDestroy<VertexBufferBundle<T, B>> for B::Device {
    unsafe fn device_destroy(&mut self, vbb: VertexBufferBundle<T, B>) {
        let VertexBufferBundle { buffer, memory, buffered_type_phantom: _, mapping_ptr: _ } = vbb;
        self.unmap_memory(&memory);
        self.destroy_buffer(buffer);
        self.free_memory(memory);
    }
}

impl<T: Copy, B: hal::Backend> VertexBufferBundle<T, B> {
    pub fn new(
        device: &B::Device,
        limits: &hal::Limits,
        memory_types: &Vec<hal::adapter::MemoryType>,
        capacity: u32,
    ) -> Self {
        let stride = mem::size_of::<T>();
        let padded_len =
            padded_len(capacity as usize * stride, limits.non_coherent_atom_size) as u64;
        let mut buffer =
            unsafe { device.create_buffer(padded_len, hal::buffer::Usage::VERTEX) }.unwrap();
        let buffer_req = unsafe { device.get_buffer_requirements(&buffer) };
        let upload_type =
            mem_type_for_buffer(memory_types, &buffer_req, m::Properties::CPU_VISIBLE).unwrap();
        let memory = unsafe { device.allocate_memory(upload_type, buffer_req.size) }.unwrap();
        unsafe { device.bind_buffer_memory(&memory, 0, &mut buffer) }.unwrap();
        let mapping_ptr = unsafe { device.map_memory(&memory, m::Segment::ALL).unwrap() as *mut T };
        VertexBufferBundle { buffer, memory, buffered_type_phantom: PhantomData, mapping_ptr }
    }

    unsafe fn write_buffer(
        &self,
        device: &B::Device,
        start_offset: usize,
        bounds_checked_iter: impl Iterator<Item = T>,
    ) -> usize {
        let mut count_written = 0;
        for data in bounds_checked_iter {
            let dest = self.mapping_ptr.add(start_offset + count_written);
            dest.write(data);
            count_written += 1;
        }
        let stride = mem::size_of::<T>() as u64;
        device
            .flush_mapped_memory_ranges(iter::once((
                &self.memory,
                m::Segment {
                    offset: start_offset as u64 * stride,
                    size: Some(count_written as u64 * stride),
                },
            )))
            .unwrap();
        count_written
    }
}

impl<B: hal::Backend> Renderer<B> {
    pub fn new(
        instance: B::Instance,
        mut surface: B::Surface,
        adapter: hal::adapter::Adapter<B>,
        max_tri_verts: u32,
        max_instances: u32,
        cull_face: pso::Face,
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
        // for (i, fam) in adapter.queue_families.iter().enumerate() {
        //     println!("fam {}: {:#?}", i, fam);
        // }
        // dbg!(&memory_types, &limits, &family);
        let (queue_group, device) = {
            let mut gpu = unsafe {
                adapter.physical_device.open(&[(family, &[1.0])], hal::Features::empty()).unwrap()
            };
            (gpu.queue_groups.pop().unwrap(), gpu.device)
        };
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
                            ty: pso::ImageDescriptorType::Sampled { with_sampler: false },
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
                            ty: pso::ImageDescriptorType::Sampled { with_sampler: false },
                        },
                        count: 1,
                    },
                    pso::DescriptorRangeDesc { ty: pso::DescriptorType::Sampler, count: 1 },
                ],
                pso::DescriptorPoolCreateFlags::empty(),
            )
        }
        .expect("Can't create descriptor pool");
        let desc_set = unsafe { desc_pool.allocate_set(&set_layout) }.unwrap();
        let sampler = unsafe {
            device.create_sampler(&i::SamplerDesc::new(i::Filter::Nearest, i::WrapMode::Clamp))
        }
        .expect("Can't create sampler");

        ///////////////////////////////////////////////////////
        // ALLOCATE AND INIT VERTEX BUFFER
        let vertex_buffer_bundles = VertexBufferBundles {
            vc: VertexBufferBundle::new(&device, &limits, &memory_types, max_tri_verts),
            m4: VertexBufferBundle::new(&device, &limits, &memory_types, max_instances),
            ts: VertexBufferBundle::new(&device, &limits, &memory_types, max_instances),
        };

        let formats = surface.supported_formats(&adapter.physical_device);
        let format = formats.map_or(f::Format::Rgba8Srgb, |formats| {
            formats
                .iter()
                .find(|format| format.base_format().1 == ChannelType::Srgb)
                .copied()
                .unwrap_or(formats[0])
        });

        let caps = surface.capabilities(&adapter.physical_device);
        let swap_config = hal::window::SwapchainConfig::from_caps(&caps, format, DIMS);
        let frames_in_flight = {
            use hal::window::PresentMode as Pm;
            match swap_config.present_mode {
                Pm::MAILBOX => 3,
                Pm::FIFO => 2,
                Pm::IMMEDIATE | Pm::RELAXED | _ => todo!(),
            }
        };
        let extent = swap_config.extent;
        unsafe {
            surface.configure_swapchain(&device, swap_config).expect("Can't configure swapchain");
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
            let depth_attachment = pass::Attachment {
                format: Some(f::Format::D32Sfloat),
                samples: 1,
                ops: pass::AttachmentOps {
                    load: pass::AttachmentLoadOp::Clear,
                    store: pass::AttachmentStoreOp::Store,
                },
                stencil_ops: pass::AttachmentOps::DONT_CARE, // PRESERVE ?
                layouts: i::Layout::Undefined..i::Layout::DepthStencilAttachmentOptimal,
            };
            let subpass = pass::SubpassDesc {
                colors: &[(0, i::Layout::ColorAttachmentOptimal)],
                depth_stencil: Some(&(1, i::Layout::DepthStencilAttachmentOptimal)),
                inputs: &[],
                resolves: &[],
                preserves: &[],
            };
            let in_dependency = pass::SubpassDependency {
                passes: None..Some(0),
                stages: PipelineStage::COLOR_ATTACHMENT_OUTPUT
                    ..PipelineStage::COLOR_ATTACHMENT_OUTPUT | PipelineStage::EARLY_FRAGMENT_TESTS,
                accesses: i::Access::empty()
                    ..(i::Access::COLOR_ATTACHMENT_READ
                        | i::Access::COLOR_ATTACHMENT_WRITE
                        | i::Access::DEPTH_STENCIL_ATTACHMENT_READ
                        | i::Access::DEPTH_STENCIL_ATTACHMENT_WRITE),
                flags: m::Dependencies::empty(),
            };
            let out_dependency = pass::SubpassDependency {
                passes: Some(0)..None,
                stages: PipelineStage::COLOR_ATTACHMENT_OUTPUT | PipelineStage::EARLY_FRAGMENT_TESTS
                    ..PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                accesses: (i::Access::COLOR_ATTACHMENT_READ
                    | i::Access::COLOR_ATTACHMENT_WRITE
                    | i::Access::DEPTH_STENCIL_ATTACHMENT_READ
                    | i::Access::DEPTH_STENCIL_ATTACHMENT_WRITE)
                    ..i::Access::empty(),
                flags: m::Dependencies::empty(),
            };
            unsafe {
                device.create_render_pass(
                    &[attachment, depth_attachment],
                    &[subpass],
                    &[in_dependency, out_dependency],
                )
            }
            .expect("Can't create render pass")
        };

        let pipeline_layout = {
            let push_constant_bytes = mem::size_of::<Mat4>() as u32;
            unsafe {
                device.create_pipeline_layout(
                    iter::once(&set_layout),
                    &[(ShaderStageFlags::VERTEX, 0..push_constant_bytes)],
                )
            }
        }
        .expect("Can't create pipeline layout");

        let pipeline = {
            let vs_module = {
                let spirv =
                    // gfx_auxil::read_spirv(std::fs::File::open("./src/data/quad.vert.spv").unwrap())
                        gfx_auxil::read_spirv(Cursor::new(&include_bytes!("data/quad.vert.spv")[..]))
                        .unwrap();
                unsafe { device.create_shader_module(&spirv) }.unwrap()
            };
            let fs_module = {
                let spirv =
                    // gfx_auxil::read_spirv(std::fs::File::open("./src/data/quad.frag.spv").unwrap())
                        gfx_auxil::read_spirv(Cursor::new(&include_bytes!("./data/quad.frag.spv")[..]))
                        .unwrap();
                unsafe { device.create_shader_module(&spirv) }.unwrap()
            };
            let pipeline = {
                const ENTRY_NAME: &str = "main";
                let vs_entry = pso::EntryPoint {
                    entry: ENTRY_NAME,
                    module: &vs_module,
                    specialization: pso::Specialization::default(),
                };
                let fs_entry = pso::EntryPoint {
                    entry: ENTRY_NAME,
                    module: &fs_module,
                    specialization: pso::Specialization::default(),
                };
                let buffers = &[
                    pso::VertexBufferDesc {
                        binding: 0,
                        stride: mem::size_of::<VertCoord>() as u32,
                        rate: VertexInputRate::Vertex,
                    },
                    pso::VertexBufferDesc {
                        binding: 1,
                        stride: mem::size_of::<Mat4>() as u32,
                        rate: VertexInputRate::Instance(1),
                    },
                    pso::VertexBufferDesc {
                        binding: 2,
                        stride: mem::size_of::<TexScissor>() as u32,
                        rate: VertexInputRate::Instance(1),
                    },
                ];
                let attributes = &{
                    let mut attributes = vec![];
                    attributes.push(pso::AttributeDesc {
                        location: 0,
                        binding: 0,
                        element: pso::Element { format: f::Format::Rg32Sfloat, offset: 0 },
                    });
                    attributes.push(pso::AttributeDesc {
                        location: 1,
                        binding: 0,
                        element: pso::Element {
                            format: f::Format::Rgb32Sfloat,
                            offset: mem::size_of::<[f32; 2]>() as u32,
                        },
                    });
                    for i in 0..4 {
                        attributes.push(pso::AttributeDesc {
                            location: i + 2,
                            binding: 1,
                            element: pso::Element {
                                format: f::Format::Rgba32Sfloat,
                                offset: i * mem::size_of::<[f32; 4]>() as u32,
                            },
                        });
                    }
                    for i in 0..2 {
                        attributes.push(pso::AttributeDesc {
                            location: i + 2 + 4,
                            binding: 2,
                            element: pso::Element {
                                format: f::Format::Rg32Sfloat,
                                offset: i * mem::size_of::<[f32; 2]>() as u32,
                            },
                        });
                    }
                    attributes
                };
                let mut pipeline_desc = pso::GraphicsPipelineDesc::new(
                    pso::PrimitiveAssemblerDesc::Vertex {
                        buffers,
                        attributes,
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
                    Subpass { index: 0, main_pass: &render_pass },
                );
                pipeline_desc.rasterizer.cull_face = cull_face;
                pipeline_desc.depth_stencil = pso::DepthStencilDesc {
                    depth: Some(pso::DepthTest { fun: pso::Comparison::LessEqual, write: true }),
                    depth_bounds: false,
                    stencil: None,
                };
                pipeline_desc.blender.targets.push(pso::ColorBlendDesc {
                    mask: pso::ColorMask::ALL,
                    blend: Some(pso::BlendState::ALPHA),
                });
                unsafe { device.create_graphics_pipeline(&pipeline_desc, None) }.unwrap()
            };
            unsafe {
                device.destroy_shader_module(vs_module);
                device.destroy_shader_module(fs_module);
            }
            pipeline
        };

        // Rendering setup
        let viewport = pso::Viewport {
            rect: pso::Rect { x: 0, y: 0, w: extent.width as _, h: extent.height as _ },
            depth: 0.0..1.0,
        };

        let mut cmd_pool = unsafe {
            device
                .create_command_pool(queue_group.family, hal::pool::CommandPoolCreateFlags::empty())
        }
        .expect("Can't create command pool");
        let new_per_fif = || {
            let fence = device.create_fence(true).expect("Could not create fence");
            let semaphore = device.create_semaphore().expect("Could not create semaphore");
            let cmd_buffer = unsafe { cmd_pool.allocate_one(command::Level::Primary) };
            let depth_image_bundle = {
                let mut image = unsafe {
                    device.create_image(
                        i::Kind::D2(viewport.rect.w as i::Size, viewport.rect.h as i::Size, 1, 1),
                        1,
                        f::Format::D32Sfloat,
                        i::Tiling::Optimal,
                        i::Usage::DEPTH_STENCIL_ATTACHMENT,
                        i::ViewCapabilities::empty(),
                    )
                }
                .unwrap();
                let requirements = unsafe { device.get_image_requirements(&image) };
                let image_type = memory_types
                    .iter()
                    .enumerate()
                    .position(|(id, mem_type)| {
                        // type_mask is a bit field where each bit represents a memory type. If the bit is set
                        // to 1 it means we can use that type for our buffer. So this code finds the first
                        // memory type that has a `1` (or, is allowed), and is visible to the CPU.
                        requirements.type_mask & (1 << id) != 0
                            && mem_type.properties.contains(m::Properties::DEVICE_LOCAL)
                    })
                    .unwrap()
                    .into();
                let memory =
                    unsafe { device.allocate_memory(image_type, requirements.size) }.unwrap();
                unsafe { device.bind_image_memory(&memory, 0, &mut image) }.unwrap();
                let view = unsafe {
                    device.create_image_view(
                        &image,
                        gfx_hal::image::ViewKind::D2,
                        f::Format::D32Sfloat,
                        gfx_hal::format::Swizzle::NO,
                        i::SubresourceRange { aspects: f::Aspects::DEPTH, ..Default::default() },
                    )
                }
                .unwrap();
                ImageBundle { image, memory, view }
            };
            PerFif { semaphore, fence, cmd_buffer, depth_image_bundle }
        };
        let per_fif = iter::repeat_with(new_per_fif).take(frames_in_flight).collect();
        Renderer {
            max_instances,
            max_tri_verts,
            device,
            queue_group,
            adapter,
            format,
            viewport,
            desc_set,
            inner: ManuallyDrop::new(RendererInner {
                cmd_pool,
                instance,
                surface,
                render_pass,
                desc_pool,
                set_layout,
                vertex_buffer_bundles,
                pipeline,
                pipeline_layout,
                sampler,
                tex_arena: SimpleArena::default(),
                per_fif,
                next_fif_index: 0,
            }),
        }
    }

    fn await_prev_fence(&self) {
        let fif_index =
            self.inner.next_fif_index.checked_sub(1).unwrap_or(self.inner.per_fif.len() - 1);
        let per_fif = &self.inner.per_fif[fif_index];
        unsafe { self.device.wait_for_fence(&per_fif.fence, !0).expect("Can't wait for fence") };
    }

    pub fn write_vertex_buffer<T>(
        &mut self,
        start: usize,
        data: impl IntoIterator<Item = T>,
    ) -> usize
    where
        Self: HasVertexBufferFor<B, T>,
        T: Copy,
    {
        let cap = HasVertexBufferFor::<B, T>::get_vertex_buffer_cap(self) as usize;
        // println!("size {:?} has cap {:?}", mem::size_of::<T>(), cap);
        if let Some(max_size) = (cap).checked_sub(start) {
            self.await_prev_fence();
            unsafe {
                self.get_vertex_buffer_bundle().write_buffer(
                    &self.device,
                    start,
                    data.into_iter().take(max_size),
                )
            }
        } else {
            0
        }
    }

    // Attempt to unload the given Rgba texture image from the GPU, given the image's index.
    pub fn unload_texture(&mut self, index: usize) -> Result<(), ()> {
        let image_bundle = self.inner.tex_arena.remove(index).ok_or(())?;
        unsafe { self.device.device_destroy(image_bundle) };
        Ok(())
    }

    // Load the given texture image to the GPU, and return its index.
    pub fn load_texture(
        &mut self,
        img_rgba: &image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
    ) -> usize {
        let memory_types = self.adapter.physical_device.memory_properties().memory_types;
        let limits = self.adapter.physical_device.limits();

        let mut fence = self.device.create_fence(false).expect("Could not create fence");
        let (width, height) = img_rgba.dimensions();
        let image_stride = 4usize;
        let row_pitch = {
            let row_alignment_mask = limits.optimal_buffer_copy_pitch_alignment as u32 - 1;
            (width * image_stride as u32 + row_alignment_mask) & !row_alignment_mask
        };

        let mut image_upload_buffer = {
            let upload_size = (height * row_pitch) as usize;
            unsafe {
                self.device.create_buffer(
                    padded_len(upload_size, limits.non_coherent_atom_size) as u64,
                    hal::buffer::Usage::TRANSFER_SRC,
                )
            }
            .unwrap()
        };
        let image_mem_reqs = unsafe { self.device.get_buffer_requirements(&image_upload_buffer) };

        // copy image data into staging buffer
        let image_upload_memory = unsafe {
            let upload_type =
                mem_type_for_buffer(&memory_types, &image_mem_reqs, m::Properties::CPU_VISIBLE)
                    .unwrap();
            let memory = self.device.allocate_memory(upload_type, image_mem_reqs.size).unwrap();
            self.device.bind_buffer_memory(&memory, 0, &mut image_upload_buffer).unwrap();
            let mapping = self.device.map_memory(&memory, m::Segment::ALL).unwrap();
            for y in 0..height as usize {
                let row = &(**img_rgba)[y * (width as usize) * image_stride
                    ..(y + 1) * (width as usize) * image_stride];
                std::ptr::copy_nonoverlapping(
                    row.as_ptr(),
                    mapping.offset(y as isize * row_pitch as isize),
                    width as usize * image_stride,
                );
            }
            self.device.flush_mapped_memory_ranges(iter::once((&memory, m::Segment::ALL))).unwrap();
            self.device.unmap_memory(&memory);
            memory
        };
        let mut image = unsafe {
            self.device.create_image(
                i::Kind::D2(width as i::Size, height as i::Size, 1, 1),
                1,
                ColorFormat::SELF,
                i::Tiling::Optimal,
                i::Usage::TRANSFER_DST | i::Usage::SAMPLED,
                i::ViewCapabilities::empty(),
            )
        }
        .unwrap();
        let memory = {
            let image_req = unsafe { self.device.get_image_requirements(&image) };
            let upload_type =
                mem_type_for_buffer(&memory_types, &image_req, m::Properties::DEVICE_LOCAL)
                    .unwrap();
            unsafe { self.device.allocate_memory(upload_type, image_req.size) }.unwrap()
        };

        unsafe { self.device.bind_image_memory(&memory, 0, &mut image) }.unwrap();
        let view = unsafe {
            self.device.create_image_view(
                &image,
                i::ViewKind::D2,
                ColorFormat::SELF,
                Swizzle::NO,
                i::SubresourceRange { aspects: f::Aspects::COLOR, ..Default::default() },
            )
        }
        .unwrap();
        unsafe {
            self.device.write_descriptor_sets(iter::once(pso::DescriptorSetWrite {
                set: &self.desc_set,
                binding: 0,
                array_offset: 0,
                descriptors: [
                    pso::Descriptor::Image(&view, i::Layout::ShaderReadOnlyOptimal),
                    pso::Descriptor::Sampler(&self.inner.sampler),
                ]
                .iter(),
            }));
            let mut cmd_buffer = self.inner.cmd_pool.allocate_one(command::Level::Primary);
            cmd_buffer.begin_primary(command::CommandBufferFlags::ONE_TIME_SUBMIT);

            let image_barrier = m::Barrier::Image {
                states: (i::Access::empty(), i::Layout::Undefined)
                    ..(i::Access::TRANSFER_WRITE, i::Layout::TransferDstOptimal),
                target: &image,
                families: None,
                range: i::SubresourceRange { aspects: f::Aspects::COLOR, ..Default::default() },
            };

            cmd_buffer.pipeline_barrier(
                PipelineStage::TOP_OF_PIPE..PipelineStage::TRANSFER,
                m::Dependencies::empty(),
                &[image_barrier],
            );

            cmd_buffer.copy_buffer_to_image(
                &image_upload_buffer,
                &image,
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
                    image_extent: i::Extent { width, height, depth: 1 },
                }],
            );
            let image_barrier = m::Barrier::Image {
                states: (i::Access::TRANSFER_WRITE, i::Layout::TransferDstOptimal)
                    ..(i::Access::SHADER_READ, i::Layout::ShaderReadOnlyOptimal),
                target: &image,
                families: None,
                range: i::SubresourceRange { aspects: f::Aspects::COLOR, ..Default::default() },
            };
            cmd_buffer.pipeline_barrier(
                PipelineStage::TRANSFER..PipelineStage::FRAGMENT_SHADER,
                m::Dependencies::empty(),
                &[image_barrier],
            );
            cmd_buffer.finish();
            self.queue_group.queues[0]
                .submit_without_semaphores(Some(&cmd_buffer), Some(&mut fence));
            self.device.wait_for_fence(&fence, !0).expect("Can't wait for fence");
            self.inner.cmd_pool.free(iter::once(cmd_buffer));
            self.device.destroy_buffer(image_upload_buffer);
            drop(img_rgba);
        }
        unsafe { self.device.free_memory(image_upload_memory) };
        self.inner.tex_arena.add(ImageBundle { image, memory, view })
    }

    pub fn render_instances<'a>(
        &mut self,
        texture_index: usize,
        draw_info_iter: impl IntoIterator<Item = DrawInfo<'a>>,
    ) -> Result<(), RenderErr> {
        let inner = &mut *self.inner;
        let tex_image_bundle: &ImageBundle<B> =
            inner.tex_arena.get(texture_index).ok_or(RenderErr::UnknownTextureIndex)?;
        let per_fif = inner.per_fif.get_mut(inner.next_fif_index).expect("next FIF out of range");
        let surface_image =
            if let Ok((surface_image, _)) = unsafe { inner.surface.acquire_image(1000) } {
                surface_image
            } else {
                // println!("FAILURE");
                return Ok(());
            };
        // println!("SUCCEE");
        unsafe {
            self.device.wait_for_fence(&per_fif.fence, !0).expect("Failed to wait for fence");
            self.device.reset_fence(&per_fif.fence).expect("Failed to reset fence");
            per_fif.cmd_buffer.reset(false);
        }

        let framebuffer = unsafe {
            let views = &[surface_image.borrow(), per_fif.depth_image_bundle.view.borrow()];
            self.device
                .create_framebuffer(
                    &inner.render_pass,
                    views.iter().copied(),
                    i::Extent { width: DIMS.width, height: DIMS.height, depth: 1 },
                )
                .unwrap()
        };

        unsafe {
            self.device.write_descriptor_sets(iter::once(pso::DescriptorSetWrite {
                set: &self.desc_set,
                binding: 0,
                array_offset: 0,
                descriptors: [
                    pso::Descriptor::Image(
                        &tex_image_bundle.view,
                        i::Layout::ShaderReadOnlyOptimal,
                    ),
                    pso::Descriptor::Sampler(&inner.sampler),
                ]
                .iter(),
            }))
        };

        // Rendering
        unsafe {
            per_fif.cmd_buffer.begin_primary(command::CommandBufferFlags::ONE_TIME_SUBMIT);
            per_fif.cmd_buffer.set_viewports(0, &[self.viewport.clone()]); // normalized device -> screen coords
            per_fif.cmd_buffer.set_scissors(0, &[self.viewport.rect]); // TODO mess with this and see if it crops
            per_fif.cmd_buffer.bind_graphics_pipeline(&inner.pipeline);
            let vert_bindings = [
                &inner.vertex_buffer_bundles.vc.buffer,
                &inner.vertex_buffer_bundles.m4.buffer,
                &inner.vertex_buffer_bundles.ts.buffer,
            ];
            let vert_binding_iter =
                vert_bindings.iter().map(|&b| (b, hal::buffer::SubRange::WHOLE));
            per_fif.cmd_buffer.bind_vertex_buffers(0, vert_binding_iter);
            per_fif.cmd_buffer.bind_graphics_descriptor_sets(
                &inner.pipeline_layout,
                0,
                iter::once(&self.desc_set),
                &[],
            );

            per_fif.cmd_buffer.begin_render_pass(
                &inner.render_pass,
                &framebuffer,
                self.viewport.rect,
                &[
                    command::ClearValue {
                        color: command::ClearColor { float32: [0., 0., 0., 1.] },
                    },
                    command::ClearValue {
                        depth_stencil: command::ClearDepthStencil { depth: 1., stencil: 0 },
                    },
                ],
                command::SubpassContents::Inline,
            );
            for DrawInfo { view_transform, mut vertex_range, mut instance_range } in draw_info_iter
            {
                vertex_range.end = vertex_range.end.min(self.max_tri_verts);
                instance_range.end = instance_range.end.min(self.max_instances);
                per_fif.cmd_buffer.push_graphics_constants(
                    &inner.pipeline_layout,
                    ShaderStageFlags::VERTEX,
                    0,
                    view_transform.as_u32_slice(),
                );
                per_fif.cmd_buffer.draw(vertex_range, instance_range);
            }
            per_fif.cmd_buffer.end_render_pass();
            per_fif.cmd_buffer.finish();
            let submission = Submission {
                command_buffers: iter::once(&per_fif.cmd_buffer),
                wait_semaphores: iter::empty(),
                signal_semaphores: iter::once(&per_fif.semaphore),
            };
            self.queue_group.queues[0].submit(submission, Some(&per_fif.fence));
            let _result = self.queue_group.queues[0].present(
                &mut inner.surface,
                surface_image,
                Some(&per_fif.semaphore),
            );
            self.device.destroy_framebuffer(framebuffer);
        }
        inner.next_fif_index += 1;
        if inner.next_fif_index >= inner.per_fif.len() {
            inner.next_fif_index = 0;
        }
        Ok(())
    }
}

impl<B: hal::Backend> Drop for Renderer<B> {
    fn drop(&mut self) {
        self.device.wait_idle().unwrap();
        unsafe {
            let mut inner = ManuallyDrop::take(&mut self.inner);
            self.device.destroy_descriptor_pool(inner.desc_pool);
            self.device.destroy_descriptor_set_layout(inner.set_layout);
            self.device.device_destroy(inner.vertex_buffer_bundles);
            for tex_image_bundle in inner.tex_arena.into_iter() {
                self.device.device_destroy(tex_image_bundle);
            }
            self.device.destroy_sampler(inner.sampler);
            for PerFif { semaphore, fence, cmd_buffer, depth_image_bundle } in inner.per_fif {
                self.device.destroy_semaphore(semaphore);
                self.device.destroy_fence(fence);
                self.inner.cmd_pool.free(iter::once(cmd_buffer));
                self.device.device_destroy(depth_image_bundle);
            }
            self.device.destroy_command_pool(inner.cmd_pool);
            self.device.destroy_render_pass(inner.render_pass);
            inner.surface.unconfigure_swapchain(&self.device);
            self.device.destroy_graphics_pipeline(inner.pipeline);
            self.device.destroy_pipeline_layout(inner.pipeline_layout);
            inner.instance.destroy_surface(inner.surface);
        }
    }
}
