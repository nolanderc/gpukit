#version 450

layout(location = 0) out vec4 out_color;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

void main() {
    const vec3 light_pos = vec3(1, 1, 2);

    vec3 dir_toward_light = light_pos - position;

    float attenuation = 1.0 / pow(distance(light_pos, position), 2);
    float brightness = attenuation * max(0.1, dot(normal, dir_toward_light));

    out_color = vec4(
            brightness * abs(normal),
            1
        );
}
