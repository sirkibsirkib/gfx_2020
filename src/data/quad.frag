#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 v_uv;
// layout(location = 1) in vec4 color_mult_fragin;

layout(location = 0) out vec4 target0;

layout(set = 0, binding = 0) uniform texture2D u_texture;
layout(set = 0, binding = 1) uniform sampler u_sampler;

void main() {
    target0 = texture(sampler2D(u_texture, u_sampler), v_uv);
    // target0 *= color_mult_fragin;
    if(target0.w < 0.0001) {
    	discard;
    }
    target0.xyz = target0.xyz * (1.0 - gl_FragCoord.z);
}

