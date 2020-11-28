#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(push_constant) uniform PushConstants {
    vec4 trans_0;
    vec4 trans_1;
    vec4 trans_2;
    vec4 trans_3;
} pc;

layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_uv;

layout(location = 2) in vec4 model_0;
layout(location = 3) in vec4 model_1;
layout(location = 4) in vec4 model_2;
layout(location = 5) in vec4 model_3;

layout(location = 0) out vec2 v_uv;
out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    v_uv = a_uv;
    vec4 pos = vec4(a_pos, 0.0, 1.0);
    mat4 model = mat4(model_0, model_1, model_2 ,model_3);
    mat4 view = mat4(pc.trans_0, pc.trans_1, pc.trans_2, pc.trans_3);
    gl_Position = pos * model * view;
}

