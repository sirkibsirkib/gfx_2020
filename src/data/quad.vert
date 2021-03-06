#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(push_constant) uniform PushConstants {
    vec4 trans_0;
    vec4 trans_1;
    vec4 trans_2;
    vec4 trans_3;
    // vec4 color_mult_vertin;
} pc;

// vertex input
layout(location = 0) in vec2 tex_coord;
layout(location = 1) in vec3 model_coord;

// instance input A: transform
layout(location = 2) in vec4 inst_0;
layout(location = 3) in vec4 inst_1;
layout(location = 4) in vec4 inst_2;
layout(location = 5) in vec4 inst_3;

// instance input B: texture scissor
layout(location = 6) in vec2 tex_scissor_top_left;
layout(location = 7) in vec2 tex_scissor_size;

// vertex output
layout(location = 0) out vec2 v_uv;
// layout(location = 1) out vec4 color_mult_fragin;
out gl_PerVertex {
    vec4 gl_Position;
};


void main() {
    // color_mult_fragin = pc.color_mult_vertin;
    v_uv = tex_scissor_top_left + (tex_coord * tex_scissor_size);
    vec4 coord = vec4(model_coord, 1.0);
    mat4 inst = mat4(inst_0, inst_1, inst_2 ,inst_3);
    mat4 view = mat4(pc.trans_0, pc.trans_1, pc.trans_2, pc.trans_3);
    gl_Position = view * (inst * coord);
}

