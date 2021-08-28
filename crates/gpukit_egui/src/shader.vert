#version 450

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 uv_coord;
layout(location = 2) in uint color;

layout(set = 0, binding = 0) uniform Uniforms {
    vec2 screen_size;
};

layout(location = 0) out vec2 frag_uv_coord;
layout(location = 1) out vec4 frag_color;

vec3 linear_from_srgb(vec3 srgb) {
    bvec3 cutoff = greaterThanEqual(srgb, vec3(0.04045));
    vec3 small = srgb / 12.92;
    vec3 large = pow((srgb + 0.055) / 1.055, vec3(2.4));
    return mix(small, large, cutoff);
}

void main() {
    vec2 normalized_position = position / screen_size;
    gl_Position = vec4(normalized_position.x * 2 - 1, 1 - 2 * normalized_position.y, 0, 1);

    vec4 srgba_color = vec4(
        ((color >> 0) & 0xff), 
        ((color >> 8) & 0xff), 
        ((color >> 16) & 0xff), 
        ((color >> 24) & 0xff)
    ) / 255.0;

    frag_color = vec4(
        linear_from_srgb(srgba_color.rgb),
        srgba_color.a
    );

    frag_uv_coord = uv_coord;
}

