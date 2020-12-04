#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(push_constant) uniform PushConstants {
    vec4 trans_0;
    vec4 trans_1;
    vec4 trans_2;
    vec4 trans_3;
} pc;

layout(location = 0) in vec3 model_coord;
layout(location = 1) in vec2 tex_coord;

layout(location = 2) in vec4 inst_0;
layout(location = 3) in vec4 inst_1;
layout(location = 4) in vec4 inst_2;
layout(location = 5) in vec4 inst_3;

layout(location = 6) in vec2 tex_scissor_top_left;
layout(location = 7) in vec2 tex_scissor_size;

layout(location = 0) out vec2 v_uv;
out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    v_uv = tex_scissor_top_left + (tex_coord * tex_scissor_size);
    vec4 coord = vec4(model_coord, 1.0);
    mat4 inst = mat4(inst_0, inst_1, inst_2 ,inst_3);
    mat4 view = mat4(pc.trans_0, pc.trans_1, pc.trans_2, pc.trans_3);
    gl_Position = view * inst * coord;
}

