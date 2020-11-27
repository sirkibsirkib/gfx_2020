use core::ops::Range;

struct Gfx {}

struct InstanceData {
    trans: Mat4,
    tex_scissor: Scissor,
}
struct Mat4([[f32; 4]; 4]);
struct Scissor {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

impl Gfx {
    pub fn new(max_instances: u32) -> Self {
        todo!()
    }

    pub fn write_indices(&mut self, start_idx: u32, slice: &[InstanceData]) -> Result<(), ()> {
        todo!()
    }

    pub fn load_image(&mut self, image: ()) -> Result<u32, ()> {
        todo!()
    }
    pub fn unload_image(&mut self, index: u32) -> Result<(), ()> {
        todo!()
    }

    pub fn draw(
        &mut self,
        global_trans: Mat4,
        instance_data: Range<u32>,
        image_index: u32,
    ) -> Result<(), ()> {
        todo!()
    }

    pub fn clear_frame(&mut self, color: [u8; 4]) {
        todo!()
    }

    pub fn set_depth_buffer(&mut self, value: f32) {
        todo!()
    }

    pub fn next_frame(&mut self) {
        todo!()
    }
}
