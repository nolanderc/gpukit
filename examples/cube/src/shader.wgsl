
struct VertexOutput {
    [[builtin(position)]] out_position: vec4<f32>;
    [[location(0)]] position: vec3<f32>;
    [[location(1)]] normal: vec3<f32>;
};

struct LightData {
    position: vec3<f32>;
    brightness: f32;
    specular: f32;
};

[[block]]
struct FrameUniforms {
    camera_transform: mat4x4<f32>;
    camera_position: vec3<f32>;
    light: LightData;
};

[[group(0), binding(0)]]
var frame: FrameUniforms;

[[stage(vertex)]]
fn vs_main(
    [[location(0)]] position: vec3<f32>,
    [[location(1)]] normal: vec3<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.out_position = frame.camera_transform * vec4<f32>(position, 1.0);
    out.position = position;
    out.normal = normal;
    return out;
}

let PI: f32 = 3.14159265;

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    let normal = normalize(in.normal);

    let dir_toward_light = frame.light.position - in.position;
    let light_distance = length(dir_toward_light);
    let attenuation = 1.0 / (light_distance * light_distance);
    let incidence = max(0.0, dot(normal, normalize(dir_toward_light)));
    let diffuse = incidence * attenuation;

    let dir_toward_camera = normalize(frame.camera_position - in.position);
    let halfway_dir = normalize(dir_toward_light / light_distance + dir_toward_camera);

    let specularity = 1.0e6 / pow(10.0, 6.0 * frame.light.specular);
    let energy_conservation = (specularity + 8.0) / (8.0 * PI);
    let specular = diffuse * energy_conservation * pow(dot(normal, halfway_dir), specularity);

    let lightness = frame.light.brightness * (diffuse + specular);

    return vec4<f32>(vec3<f32>(lightness), 1.0);
}
