use anyhow::{anyhow, Context as _};

use gpukit::Bindings;
use std::sync::Arc;

use gpukit::{wgpu, winit};
use gpukit_egui::egui;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
}

#[derive(Bindings)]
struct FrameBindings<'a> {
    #[uniform(binding = 0)]
    uniforms: &'a gpukit::Buffer<FrameUniforms>,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct FrameUniforms {
    camera_transform: glam::Mat4,
    camera_position: glam::Vec3,
    _padding0: f32,
    light: LightData,
    _padding1: [f32; 3],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LightData {
    position: glam::Vec3,
    brightness: f32,
    specular: f32,
}

impl Default for LightData {
    fn default() -> Self {
        LightData {
            position: glam::vec3(1.0, 1.0, 1.0),
            brightness: 1.0,
            specular: 0.7,
        }
    }
}

struct State {
    settings: Settings,

    viewport: egui::Rect,
    camera: Camera,

    frame_uniforms: gpukit::UniformBuffer<FrameUniforms>,
}

struct Settings {
    clear_color: egui::color::Hsva,
}

impl State {
    pub fn new(context: &gpukit::Context, window: &winit::window::Window) -> Self {
        let size = window.inner_size();

        State {
            settings: Settings {
                clear_color: egui::color::Hsva::from_srgb([50, 50, 50]),
            },
            viewport: egui::Rect::from_min_size(
                egui::pos2(0.0, 0.0),
                egui::vec2(size.width as f32, size.height as f32),
            ),
            camera: Camera {
                position: [0.0, 0.0, 4.0].into(),
                focus: glam::vec3(0.0, 0.0, 0.0),
                up: glam::Vec3::Y,
                fov: 70f32.to_radians(),
                near: 0.1,
                far: 50.0,
            },
            frame_uniforms: gpukit::UniformBuffer::new(context, FrameUniforms::default()),
        }
    }
}

struct Camera {
    position: glam::Vec3,
    focus: glam::Vec3,
    up: glam::Vec3,
    fov: f32,
    near: f32,
    far: f32,
}

impl Camera {
    pub fn projection_matrix(&self, size: [f32; 2]) -> glam::Mat4 {
        let [width, height] = size;
        glam::Mat4::perspective_rh(self.fov, width / height, self.near, self.far)
    }

    pub fn view_matrix(&self) -> glam::Mat4 {
        glam::Mat4::look_at_rh(self.position, self.focus, self.up)
    }

    pub fn transform(&self, size: [f32; 2]) -> glam::Mat4 {
        self.projection_matrix(size) * self.view_matrix()
    }

    fn right(&self) -> glam::Vec3 {
        let dir = (self.focus - self.position).normalize();
        self.up.cross(dir)
    }
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::FmtSubscriber::builder()
        .pretty()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_inner_size(winit::dpi::LogicalSize::new(1200, 700))
        .build(&event_loop)
        .context("failed to build window")?;
    let window = Arc::new(window);

    let (context, mut surface) = pollster::block_on(gpukit::init(&window))?;
    let mut state = State::new(&context, &window);

    let mut depth_texture = create_depth_texture(&context, surface.size());

    let frame_layout = FrameBindings::layout(&context.device);
    let frame_bindings = Bindings::create_bind_group(
        &FrameBindings {
            uniforms: state.frame_uniforms.buffer(),
        },
        &context.device,
        &frame_layout,
    );

    let (vertices, indices) = load_model("examples/cube/models/monkey.glb")?;

    let vertex_buffer = context
        .build_buffer::<Vertex>()
        .with_usage(wgpu::BufferUsages::VERTEX)
        .init_with_data(&vertices);

    let index_buffer = context
        .build_buffer::<u32>()
        .with_usage(wgpu::BufferUsages::INDEX)
        .init_with_data(&indices);

    let shader = context
        .build_shader("cube shader")
        .from_wgsl(include_str!("shader.wgsl"))?;

    let pipeline = context.create_render_pipeline(gpukit::RenderPipelineDescriptor {
        label: Some("cube render pipeline"),

        vertex: shader.entry("vs_main"),
        fragment: shader.entry("fs_main"),

        color_targets: &[surface.color_target(None)],
        bind_group_layouts: &[&frame_layout],
        vertex_buffers: &[gpukit::vertex_buffer_layout! [
            // Position
            0 => Float32x3,
            // Normal
            1 => Float32x3,
        ]],
        depth_stencil: Some(wgpu::DepthStencilState {
            format: depth_texture.format(),
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
    })?;

    let mut gui = gpukit_egui::Gui::new(gpukit_egui::GuiDescriptor {
        context: context.clone(),
        window: window.clone(),
        target_format: surface.texture_format(),
    })?;

    event_loop.run(move |event, _target, flow| match event {
        winit::event::Event::WindowEvent { event, .. } => {
            use winit::event::WindowEvent;

            gui.handle_window_event(&event);

            match event {
                WindowEvent::CloseRequested => *flow = winit::event_loop::ControlFlow::Exit,
                WindowEvent::Resized(size)
                | WindowEvent::ScaleFactorChanged {
                    new_inner_size: &mut size,
                    ..
                } => {
                    surface.resize(size);
                    depth_texture = create_depth_texture(&context, size);
                }
                _ => {}
            }
        }
        winit::event::Event::MainEventsCleared => {
            let result = || -> anyhow::Result<()> {
                gui.update(&mut state, draw_gui)?;

                state.frame_uniforms.camera_position = state.camera.position;
                state.frame_uniforms.camera_transform =
                    state.camera.transform(state.viewport.size().into());
                state.frame_uniforms.update(&context);

                let frame = surface.get_current_frame()?;
                let frame_view = frame.output.texture.create_view(&Default::default());
                let depth_view = depth_texture.create_view();

                let mut encoder = context.create_encoder();

                let window_size = window.inner_size();
                let pixels_per_point = window.scale_factor() as f32;

                if window_size.width > 0 && window_size.height > 0 {
                    use gpukit::RenderPassExt;

                    let create_color_attachment = |clear: bool| {
                        let load = if clear {
                            wgpu::LoadOp::Clear({
                                let [r, g, b] = state.settings.clear_color.to_rgb();
                                gpukit::util::color(r as f64, g as f64, b as f64, 1.0)
                            })
                        } else {
                            wgpu::LoadOp::Load
                        };

                        wgpu::RenderPassColorAttachment {
                            view: &frame_view,
                            resolve_target: None,
                            ops: wgpu::Operations { load, store: true },
                        }
                    };

                    if state.viewport.width() > 0.0 && state.viewport.height() > 0.0 {
                        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: None,
                            color_attachments: &[create_color_attachment(true)],
                            depth_stencil_attachment: Some(
                                wgpu::RenderPassDepthStencilAttachment {
                                    view: &depth_view,
                                    depth_ops: Some(wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(1.0),
                                        store: true,
                                    }),
                                    stencil_ops: None,
                                },
                            ),
                        });

                        rpass.set_viewport(
                            state.viewport.left() * pixels_per_point,
                            state.viewport.top() * pixels_per_point,
                            state.viewport.width() * pixels_per_point,
                            state.viewport.height() * pixels_per_point,
                            0.0,
                            1.0,
                        );

                        rpass.set_pipeline(&pipeline);
                        rpass.set_index_buffer_ext(index_buffer.slice(..));
                        rpass.set_vertex_buffer(0, *vertex_buffer.slice(..));
                        rpass.set_bind_group(0, &frame_bindings, &[]);
                        rpass.draw_indexed(0..index_buffer.len() as u32, 0, 0..1);
                    }

                    {
                        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: None,
                            color_attachments: &[create_color_attachment(false)],
                            depth_stencil_attachment: None,
                        });

                        gui.render(&mut rpass);
                    }
                }

                context.submit(encoder);

                Ok(())
            }();

            if let Err(error) = result {
                tracing::error!(?error, "failed to render frame");
                *flow = winit::event_loop::ControlFlow::Exit;
            }
        }
        _ => {}
    })
}

fn create_depth_texture(
    context: &Arc<gpukit::Context>,
    size: winit::dpi::PhysicalSize<u32>,
) -> gpukit::Texture<gpukit::format::Depth32Float> {
    context
        .build_texture()
        .with_usage(wgpu::TextureUsages::RENDER_ATTACHMENT)
        .init_with_size(size.into())
}

fn load_model(path: impl AsRef<std::path::Path>) -> anyhow::Result<(Vec<Vertex>, Vec<u32>)> {
    let (gltf, buffers, _) = gltf::import(path)?;
    let mut meshes = Vec::new();

    for scene in gltf.scenes() {
        extract_meshes(&mut scene.nodes(), &mut meshes);
    }

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for mesh in meshes {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| buffers.get(buffer.index()).map(|data| &**data));

            let positions = reader
                .read_positions()
                .ok_or_else(|| anyhow!("mesh did not contain positions"))?;
            let normals = reader
                .read_normals()
                .ok_or_else(|| anyhow!("mesh did not contain normals"))?;

            let index_offset = vertices.len() as u32;
            if let Some(raw_indices) = reader.read_indices() {
                match raw_indices {
                    gltf::mesh::util::ReadIndices::U8(raw_indices) => {
                        indices.extend(raw_indices.map(|index| index as u32 + index_offset))
                    }
                    gltf::mesh::util::ReadIndices::U16(raw_indices) => {
                        indices.extend(raw_indices.map(|index| index as u32 + index_offset))
                    }
                    gltf::mesh::util::ReadIndices::U32(raw_indices) => {
                        indices.extend(raw_indices.map(|index| index as u32 + index_offset))
                    }
                }
            } else {
                indices.extend(index_offset..index_offset + positions.len() as u32);
            }

            let vertex_data = positions
                .zip(normals)
                .map(|(position, normal)| Vertex { position, normal });
            vertices.extend(vertex_data);
        }
    }

    Ok((vertices, indices))
}

fn extract_meshes<'node>(
    nodes: &mut dyn Iterator<Item = gltf::Node<'node>>,
    meshes: &mut Vec<gltf::Mesh<'node>>,
) {
    for node in nodes {
        extract_meshes(&mut node.children(), meshes);

        if let Some(mesh) = node.mesh() {
            meshes.push(mesh);
        }
    }
}

fn draw_gui(ctx: egui::CtxRef, state: &mut State) -> anyhow::Result<()> {
    egui::SidePanel::left("settings_panel")
        .resizable(true)
        .min_width(250.0)
        .show(&ctx, |ui| {
            egui::ScrollArea::auto_sized().show(ui, |ui| {
                ui.vertical_centered(|ui| {
                    ui.heading("settings");
                });
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("clear color");
                    egui::color_picker::color_edit_button_hsva(
                        ui,
                        &mut state.settings.clear_color,
                        egui::color_picker::Alpha::Opaque,
                    );
                });

                ui.collapsing("camera", |ui| {
                    let camera = &mut state.camera;
                    egui::Grid::new("properties").show(ui, |ui| {
                        ui.label("fov");
                        if angle_slider(ui, &mut camera.fov, 1.0..=179.0).changed() {
                            let angle = camera.fov / 2.0;
                            let distance = 2.0 * angle.tan().recip();
                            camera.position = distance * camera.position.normalize();
                        }
                        ui.end_row();

                        ui.label("near");
                        ui.add(
                            egui::Slider::new(&mut camera.near, 0.1..=camera.far)
                                .max_decimals(2)
                                .logarithmic(true)
                                .clamp_to_range(false),
                        );
                        ui.end_row();

                        ui.label("far");
                        ui.add(
                            egui::Slider::new(&mut camera.far, camera.near..=100.0)
                                .max_decimals(2)
                                .logarithmic(true)
                                .clamp_to_range(false),
                        );
                        ui.end_row();
                    })
                });

                ui.collapsing("light", |ui| {
                    let light = &mut state.frame_uniforms.light;
                    egui::Grid::new("properties").show(ui, |ui| {
                        ui.label("position");
                        vec3_slider(ui, &mut light.position, -5.0..=5.0);
                        ui.end_row();

                        ui.label("brightness");
                        ui.add(egui::Slider::new(&mut light.brightness, 0.0..=5.0));
                        ui.end_row();

                        ui.label("specular");
                        ui.add(egui::Slider::new(&mut light.specular, 0.0..=1.0));
                        ui.end_row();
                    });
                });
            });
        });

    egui::CentralPanel::default()
        .frame(egui::Frame {
            margin: egui::vec2(0.0, 0.0),
            corner_radius: 0.0,
            ..Default::default()
        })
        .show(&ctx, |ui| {
            state.viewport = ui.max_rect_finite();
            let response = ui.allocate_rect(state.viewport, egui::Sense::click_and_drag());

            let look_delta = state.camera.focus - state.camera.position;
            let mut distance = look_delta.length();
            let look_dir = look_delta / distance;

            let drag = response.drag_delta();

            let mut rotation = glam::Quat::IDENTITY;
            if response.dragged_by(egui::PointerButton::Primary) {
                let dx = 5.0 * drag.x / state.viewport.width();
                let dy = 5.0 * drag.y / state.viewport.height();
                rotation *= glam::Quat::from_rotation_y(-dx);
                rotation *= glam::Quat::from_axis_angle(state.camera.right(), dy);
            }

            const ZOOM_SPEED: f32 = 2e-3;
            if response.dragged_by(egui::PointerButton::Secondary) {
                distance *= (1.0 + ZOOM_SPEED).powf(drag.y);
            }
            if response.hovered() {
                let scroll = ui.input().scroll_delta;
                distance *= (1.0 + ZOOM_SPEED).powf(-scroll.y);
            }

            state.camera.position =
                state.camera.focus - distance * (rotation * look_dir).normalize();
        });

    Ok(())
}

fn vec3_slider(
    ui: &mut egui::Ui,
    vec: &mut glam::Vec3,
    ranges: impl Into<MultiRange<f32, 3>>,
) -> egui::Response {
    let MultiRange([x_range, y_range, z_range]) = ranges.into();
    let [x, y, z] = vec.as_mut();
    multi_slider(
        ui,
        [
            ("X", egui::Slider::new(x, x_range)),
            ("Y", egui::Slider::new(y, y_range)),
            ("Z", egui::Slider::new(z, z_range)),
        ],
    )
}

struct MultiRange<T, const N: usize>([std::ops::RangeInclusive<T>; N]);

impl<T, const N: usize> From<[std::ops::RangeInclusive<T>; N]> for MultiRange<T, N> {
    fn from(ranges: [std::ops::RangeInclusive<T>; N]) -> Self {
        MultiRange(ranges)
    }
}

impl<T, const N: usize> From<std::ops::RangeInclusive<T>> for MultiRange<T, N>
where
    T: Clone,
{
    fn from(range: std::ops::RangeInclusive<T>) -> Self {
        MultiRange(array_init_clone(range))
    }
}

fn array_init_clone<T: Clone, const N: usize>(value: T) -> [T; N] {
    use std::mem::MaybeUninit;
    let mut array: [MaybeUninit<T>; N] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
    for slot in &mut array {
        *slot = MaybeUninit::new(value.clone());
    }
    unsafe { (&array as *const _ as *const [T; N]).read() }
}

fn multi_slider<const N: usize>(
    ui: &mut egui::Ui,
    values: [(&str, egui::Slider); N],
) -> egui::Response {
    ui.vertical(|ui| {
        let mut sizes = [egui::vec2(0.0, 0.0); N];
        for (size, (label, _)) in sizes.iter_mut().zip(&values) {
            let layout = egui::Label::new(label).layout(ui);
            *size = layout.size;
        }

        let label_size = sizes.iter().fold(egui::vec2(0.0, 0.0), |acc, size| {
            egui::vec2(f32::max(acc.x, size.x), f32::max(acc.y, size.y))
        });

        let mut responses: Option<egui::Response> = None;
        for (label, slider) in values {
            ui.horizontal(|ui| {
                ui.add_sized(label_size, egui::Label::new(label));
                let response = ui.add(slider);

                if let Some(responses) = responses.as_mut() {
                    *responses = responses.union(response)
                } else {
                    responses = Some(response);
                }
            });
        }

        responses.unwrap()
    })
    .inner
}

fn angle_slider(
    ui: &mut egui::Ui,
    radians: &mut f32,
    degree_range: std::ops::RangeInclusive<f32>,
) -> egui::Response {
    let mut degrees = radians.to_degrees();

    let response = ui.add(
        egui::Slider::new(&mut degrees, degree_range)
            .clamp_to_range(true)
            .show_value(true)
            .suffix("˚"),
    );

    #[allow(clippy::float_cmp)]
    if degrees != radians.to_degrees() {
        *radians = degrees.to_radians();
    }

    response
}
