#version 450

layout(location = 0) out vec4 out_color;

layout(location = 0) in vec2 uv_coord;
layout(location = 1) in vec4 color;

layout(set = 0, binding = 0) uniform Uniforms {
    vec2 screen_size;
};

layout(set = 0, binding = 1) uniform sampler nearest;
layout(set = 1, binding = 0) uniform texture2D alpha_texture;

void main() {
    float alpha = texture(sampler2D(alpha_texture, nearest), uv_coord).r;
    out_color = color * alpha;
    // out_color = vec4(screen_size / vec2(2400, 1400), 1 / pixels_per_point, 1);
}

