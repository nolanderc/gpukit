use anyhow::{anyhow, Context as _};

use std::sync::Arc;

use gpukit::Bindings;
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
    material: MaterialData,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LightData {
    position: glam::Vec3,
    brightness: f32,
}

impl Default for LightData {
    fn default() -> Self {
        LightData {
            position: glam::vec3(1.0, 1.0, 1.0),
            brightness: 1.0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MaterialData {
    diffuse: glam::Vec3,
    _padding0: f32,
    specular: glam::Vec3,
    roughness: f32,
}

impl Default for MaterialData {
    fn default() -> Self {
        MaterialData {
            diffuse: glam::vec3(0.5, 0.0, 0.0),
            _padding0: Default::default(),
            specular: glam::vec3(0.5, 0.5, 0.5),
            roughness: 0.7,
        }
    }
}

struct State {
    settings: Settings,

    viewport: egui::Rect,
    camera: Camera,

    frame_uniforms: gpukit::UniformBuffer<FrameUniforms>,

    load_model: Option<std::path::PathBuf>,
}

struct Settings {
    clear_color: egui::color::Hsva,
}

impl State {
    pub fn new(context: &gpukit::Context, window: &winit::window::Window) -> anyhow::Result<Self> {
        let size = window.inner_size();

        Ok(State {
            settings: Settings {
                clear_color: egui::color::Hsva::from_srgb([0, 0, 0]),
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

            load_model: None,
        })
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
        dir.cross(self.up)
    }

    fn view_up(&self) -> glam::Vec3 {
        let dir = (self.focus - self.position).normalize();
        dir.cross(self.up).cross(dir)
    }
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::FmtSubscriber::builder()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_inner_size(winit::dpi::LogicalSize::new(1500, 900))
        .build(&event_loop)
        .context("failed to build window")?;
    let window = Arc::new(window);

    let (context, mut surface) = pollster::block_on(gpukit::init(&window))?;
    let mut state = State::new(&context, &window)?;

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

    let mut vertex_buffer = context
        .build_buffer::<Vertex>()
        .with_usage(wgpu::BufferUsages::VERTEX)
        .init_with_data(&vertices);

    let mut index_buffer = context
        .build_buffer::<u32>()
        .with_usage(wgpu::BufferUsages::INDEX)
        .init_with_data(&indices);

    let shader = context
        .build_shader("cube shader")
        .init_from_wgsl(include_str!("shader.wgsl"))?;

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
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
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

                if let Some(path) = state.load_model.take() {
                    match load_model(&path) {
                        Err(error) => tracing::error!(path = %path.display(), "failed to load model: {}", error),
                        Ok((vertices, indices)) => {
                            vertex_buffer = context
                                .build_buffer::<Vertex>()
                                .with_usage(wgpu::BufferUsages::VERTEX)
                                .init_with_data(&vertices);

                            index_buffer = context
                                .build_buffer::<u32>()
                                .with_usage(wgpu::BufferUsages::INDEX)
                                .init_with_data(&indices);
                        }
                    }
                }

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
    let path = path.as_ref();
    let mut gltf = gltf::Gltf::open(path)?;

    let blob = gltf.blob.take();

    let mut meshes = Vec::new();
    for scene in gltf.scenes() {
        extract_meshes(&mut scene.nodes(), &mut meshes);
    }

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for mesh in meshes {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| match buffer.source() {
                gltf::buffer::Source::Bin => blob.as_deref(),
                gltf::buffer::Source::Uri(uri) => {
                    tracing::warn!(%uri, "ignoring buffer");
                    None
                }
            });

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
    egui::SidePanel::left("settings")
        .resizable(true)
        .default_width(250.0)
        .min_width(200.0)
        .show(&ctx, |ui| show_settings(ui, state));

    egui::TopBottomPanel::bottom("file_explorer")
        .resizable(true)
        .default_height(250.0)
        .min_height(100.0)
        .show(&ctx, |ui| {
            ui.set_min_height(ui.available_height());
            show_file_explorer(ui, state);
        });

    egui::CentralPanel::default()
        .frame(egui::Frame {
            margin: egui::vec2(0.0, 0.0),
            corner_radius: 0.0,
            ..Default::default()
        })
        .show(&ctx, |ui| show_viewport(ui, state));

    Ok(())
}

fn show_file_explorer(ui: &mut egui::Ui, state: &mut State) {
    use gpukit_egui::file_explorer;
    file_explorer::FileExplorer::new("explorer")
        .initial_path(".")
        .show(ui, |event| match event {
            file_explorer::Event::OpenFile(path) => state.load_model = Some(path.to_owned()),
        });
}

fn show_settings(ui: &mut egui::Ui, state: &mut State) {
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
            egui::Grid::new("properties").num_columns(2).show(ui, |ui| {
                ui.label("fov");
                angle_slider(ui, &mut camera.fov, 1.0..=179.0);
                ui.end_row();

                ui.label("near");
                ui.add(DragValue::new(&mut camera.near, 0.0..camera.far).logarithmic(true));
                ui.end_row();

                ui.label("far");
                ui.add(DragValue::new(&mut camera.far, camera.near..).logarithmic(true));
                ui.end_row();
            })
        });

        ui.collapsing("light", |ui| {
            let light = &mut state.frame_uniforms.light;
            egui::Grid::new("properties").num_columns(2).show(ui, |ui| {
                ui.label("position");
                vec3_slider(ui, &mut light.position, make_inclusive(..));
                ui.end_row();

                ui.label("brightness");
                ui.add(DragValue::new(&mut light.brightness, 0.0..).logarithmic(true));
                ui.end_row();
            });
        });

        ui.collapsing("material", |ui| {
            let material = &mut state.frame_uniforms.material;
            egui::Grid::new("properties").num_columns(2).show(ui, |ui| {
                ui.label("diffuse");
                vec3_slider(
                    ui,
                    &mut material.diffuse,
                    [
                        make_inclusive(0.0..1.0 - material.specular.x),
                        make_inclusive(0.0..1.0 - material.specular.y),
                        make_inclusive(0.0..1.0 - material.specular.z),
                    ],
                );
                ui.end_row();

                ui.label("specular");
                vec3_slider(
                    ui,
                    &mut material.specular,
                    [
                        make_inclusive(0.0..1.0 - material.diffuse.x),
                        make_inclusive(0.0..1.0 - material.diffuse.y),
                        make_inclusive(0.0..1.0 - material.diffuse.z),
                    ],
                );
                ui.end_row();

                ui.label("roughness");
                ui.add(InlineSlider::new(&mut material.roughness, 0.0..=1.0));
                ui.end_row();
            });
        });
    });
}

fn show_viewport(ui: &mut egui::Ui, state: &mut State) {
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
        rotation *= glam::Quat::from_axis_angle(state.camera.right(), -dy);
    }

    const ZOOM_SPEED: f32 = 2e-3;
    if response.dragged_by(egui::PointerButton::Secondary) {
        distance *= (1.0 + ZOOM_SPEED).powf(drag.y);
    }
    if response.hovered() {
        let scroll = ui.input().scroll_delta;
        distance *= (1.0 + ZOOM_SPEED).powf(-scroll.y);
    }

    if response.dragged_by(egui::PointerButton::Middle) {
        state.camera.focus -= distance * drag.x / state.viewport.width() * state.camera.right();
        state.camera.focus += distance * drag.y / state.viewport.height() * state.camera.view_up();
    }

    state.camera.position = state.camera.focus - distance * (rotation * look_dir).normalize();
}

struct InlineSlider<'a> {
    get_set: Box<dyn FnMut(Option<f64>) -> f64 + 'a>,
    bounds: std::ops::RangeInclusive<f64>,
}

impl<'a> InlineSlider<'a> {
    pub fn new<T>(value: &'a mut T, range: std::ops::RangeInclusive<T>) -> Self
    where
        T: egui::math::Numeric,
    {
        InlineSlider {
            get_set: Box::new(move |new| {
                if let Some(new) = new {
                    *value = T::from_f64(new);
                }
                value.to_f64()
            }),
            bounds: range.start().to_f64()..=range.end().to_f64(),
        }
    }
}

impl egui::Widget for InlineSlider<'_> {
    fn ui(mut self, ui: &mut egui::Ui) -> egui::Response {
        let size = ui.available_size();

        let value = (self.get_set)(None);
        let range = self.bounds.end() - self.bounds.start();
        let values_per_point = range / size.x as f64;
        let decimals = (-values_per_point.log10()).ceil().max(0.0) as usize;
        let percentage = value / range;

        let text = format!("{:.decimals$}", value, decimals = decimals);
        let text_style = ui
            .style()
            .override_text_style
            .unwrap_or(egui::TextStyle::Button);
        let galley = ui.fonts().layout_no_wrap(text_style, text);

        let (rect, mut response) = ui.allocate_at_least(size, egui::Sense::click_and_drag());
        let visuals = ui.style().interact(&response);
        ui.painter()
            .rect_filled(rect, visuals.corner_radius, visuals.bg_fill);
        {
            let mut painter = ui.painter().clone();
            let clip_width = (1.0 - percentage) as f32 * rect.width();
            let old_clip = painter.clip_rect();
            painter.set_clip_rect(egui::Rect::from_min_max(
                egui::pos2(rect.max.x - clip_width, old_clip.min.y),
                egui::pos2(rect.max.x, old_clip.max.y),
            ));
            let color = egui::Color32::from_black_alpha(100);
            painter.rect_filled(rect, visuals.corner_radius, color);
        }
        ui.painter()
            .rect_stroke(rect, visuals.corner_radius, visuals.bg_stroke);

        let padding = ui.spacing().button_padding;
        let text_pos = ui
            .layout()
            .align_size_within_rect(galley.size, rect.shrink2(padding))
            .min;
        let text_color = ui
            .visuals()
            .override_text_color
            .unwrap_or_else(|| visuals.text_color());
        ui.painter().galley(text_pos, galley, text_color);

        if let Some(pointer) = response.interact_pointer_pos() {
            let distance = pointer.x - response.rect.left();
            let new_value = self.bounds.start() + distance as f64 * values_per_point;
            let new_value = new_value.clamp(*self.bounds.start(), *self.bounds.end());

            (self.get_set)(Some(new_value));
            response.mark_changed();
        }

        response
    }
}

fn make_inclusive<T>(range: impl std::ops::RangeBounds<T>) -> std::ops::RangeInclusive<T>
where
    T: egui::math::Numeric,
{
    let start = match range.start_bound() {
        std::ops::Bound::Included(value) => value.to_f64(),
        std::ops::Bound::Excluded(value) => value.to_f64() + T::INTEGRAL as u64 as f64,
        std::ops::Bound::Unbounded => T::MIN.to_f64(),
    };

    let end = match range.end_bound() {
        std::ops::Bound::Included(value) => value.to_f64(),
        std::ops::Bound::Excluded(value) => value.to_f64() - T::INTEGRAL as u64 as f64,
        std::ops::Bound::Unbounded => T::MAX.to_f64(),
    };

    T::from_f64(start)..=T::from_f64(end)
}

struct DragValue<'a> {
    drag: egui::DragValue<'a>,
    logarithmic: bool,
    value: f64,
}

impl<'a> DragValue<'a> {
    pub fn new<T>(value: &'a mut T, range: impl std::ops::RangeBounds<T>) -> Self
    where
        T: egui::math::Numeric,
    {
        let old_value = value.to_f64();
        DragValue {
            drag: egui::DragValue::new(value).clamp_range(make_inclusive(range)),
            logarithmic: false,
            value: old_value,
        }
    }

    pub fn logarithmic(mut self, logarithmic: bool) -> Self {
        self.logarithmic = logarithmic;
        self
    }

    pub fn suffix(mut self, suffix: impl ToString) -> Self {
        self.drag = self.drag.suffix(suffix);
        self
    }

    pub fn speed(mut self, speed: impl egui::math::Numeric) -> Self {
        self.drag = self.drag.speed(speed.to_f64());
        self
    }
}

impl<'a> egui::Widget for DragValue<'a> {
    fn ui(mut self, ui: &mut egui::Ui) -> egui::Response {
        if self.logarithmic {
            self.drag = self.drag.speed(1e-6 + self.value.abs() * 1e-2);
        };
        ui.add_sized(ui.available_size(), self.drag)
    }
}

enum MultiRange<T, const N: usize> {
    Single(std::ops::RangeInclusive<T>),
    Multi([std::ops::RangeInclusive<T>; N]),
}

impl<T, const N: usize> From<std::ops::RangeInclusive<T>> for MultiRange<T, N> {
    fn from(range: std::ops::RangeInclusive<T>) -> Self {
        MultiRange::Single(range)
    }
}

impl<T, const N: usize> From<[std::ops::RangeInclusive<T>; N]> for MultiRange<T, N> {
    fn from(ranges: [std::ops::RangeInclusive<T>; N]) -> Self {
        MultiRange::Multi(ranges)
    }
}

fn vec3_slider(
    ui: &mut egui::Ui,
    vec: &mut glam::Vec3,
    ranges: impl Into<MultiRange<f32, 3>>,
) -> egui::Response {
    let [x, y, z] = vec.as_mut();

    let [x_range, y_range, z_range] = match ranges.into() {
        MultiRange::Single(range) => [range.clone(), range.clone(), range],
        MultiRange::Multi(ranges) => ranges,
    };

    multi_slider(
        ui,
        [
            ("X", DragValue::new(x, x_range).speed(1e-2)),
            ("Y", DragValue::new(y, y_range).speed(1e-2)),
            ("Z", DragValue::new(z, z_range).speed(1e-2)),
        ],
    )
}

fn multi_slider<const N: usize>(
    ui: &mut egui::Ui,
    values: [(&str, DragValue); N],
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

    let response = ui.add(DragValue::new(&mut degrees, degree_range).suffix("˚"));

    #[allow(clippy::float_cmp)]
    if degrees != radians.to_degrees() {
        *radians = degrees.to_radians();
    }

    response
}
