use gfx_backend_vulkan as backend;

use gfx_hal::prelude::*;

fn main2() {
    // Create an instance
    let instance = backend::Instance::create("My App", 1).expect("ZOOP");
    let adapter = instance
        .enumerate_adapters()
        .into_iter()
        .find(|a| {
            a.queue_families
                .iter()
                .any(|family| family.queue_type().supports_compute())
        })
        .expect("Failed to find a GPU with compute support!");
    let memory_properties = adapter.physical_device.memory_properties();
    let family = adapter
        .queue_families
        .iter()
        .find(|family| family.queue_type().supports_compute())
        .unwrap();
    let mut gpu = unsafe {
        adapter
            .physical_device
            .open(&[(family, &[1.0])], gfx_hal::Features::empty())
            .unwrap()
    };
    let device = &gpu.device;
    let queue_group = gpu.queue_groups.first_mut().unwrap();
}

fn main() {
    const DIMS: gfx_hal::window::Extent2D = gfx_hal::window::Extent2D {
        width: 1024,
        height: 768,
    };

    let event_loop = winit::event_loop::EventLoop::new();

    let wb = winit::window::WindowBuilder::new()
        .with_min_inner_size(winit::dpi::Size::Logical(winit::dpi::LogicalSize::new(
            64.0, 64.0,
        )))
        .with_inner_size(winit::dpi::Size::Physical(winit::dpi::PhysicalSize::new(
            DIMS.width,
            DIMS.height,
        )))
        .with_title("mesh shading".to_string());

    // instantiate backend
    let (_window, instance, mut adapters, surface) = {
        let window = wb.build(&event_loop).unwrap();
        let instance = backend::Instance::create("gfx-rs mesh shading", 1)
            .expect("Failed to create an instance!");
        let adapters = instance.enumerate_adapters();
        let surface = unsafe {
            instance
                .create_surface(&window)
                .expect("Failed to create a surface!")
        };
        // Return `window` so it is not dropped: dropping it invalidates `surface`.
        (window, Some(instance), adapters, surface)
    };

    let adapter = adapters.remove(0);

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
    println!("family = {:#?}", family);
    let mut gpu = unsafe {
        adapter
            .physical_device
            .open(&[(family, &[1.0])], gfx_hal::Features::MESH_SHADER)
            .unwrap()
    };
    // let queue_group = gpu.queue_groups.pop().unwrap();
    // let device = gpu.device;

    // let command_pool = unsafe {
    //     device.create_command_pool(
    //         queue_group.family,
    //         gfx_hal::pool::CommandPoolCreateFlags::empty(),
    //     )
    // }
    // .expect("Can't create command pool");

    // // Setup renderpass and pipeline
    // let set_layout = std::mem::ManuallyDrop::new(
    //     unsafe {
    //         device.create_descriptor_set_layout(
    //             &[gfx_hal::pso::DescriptorSetLayoutBinding {
    //                 binding: 0,
    //                 ty: gfx_hal::pso::DescriptorType::Buffer {
    //                     ty: gfx_hal::pso::BufferDescriptorType::Storage { read_only: true },
    //                     format: gfx_hal::pso::BufferDescriptorFormat::Structured {
    //                         dynamic_offset: false,
    //                     },
    //                 },
    //                 count: 1,
    //                 stage_flags: gfx_hal::pso::ShaderStageFlags::MESH,
    //                 immutable_samplers: false,
    //             }],
    //             &[],
    //         )
    //     }
    //     .expect("Can't create descriptor set layout"),
    // );
}
