#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

layout(binding = 0) uniform Uniforms {
    mat4 camera_transform;
} uniforms;

layout(location = 0) out vec3 frag_position;
layout(location = 1) out vec3 frag_normal;

void main() {
    frag_position = position;
    frag_normal = normal;
    gl_Position = uniforms.camera_transform * vec4(position, 1);
}
