use gpukit::wgpu;
use std::sync::Arc;

pub struct Renderer {
    context: Arc<gpukit::Context>,

    pipeline: wgpu::RenderPipeline,
    vertex_buffer: gpukit::Buffer<Vertex>,
    index_buffer: gpukit::Buffer<u32>,

    bind_group: gpukit::BindGroup,
    screen_uniforms: UniformBuffer<ScreenUniforms>,

    texture_bind_group: gpukit::BindGroup,
    texture_version: Option<u64>,
    texture: gpukit::Texture<gpukit::format::R8Unorm>,
}

#[derive(gpukit::Bindings)]
struct Bindings<'a> {
    #[uniform(binding = 0)]
    screen_uniforms: &'a gpukit::Buffer<ScreenUniforms>,
    #[sampler(binding = 1, filtering)]
    sampler: &'a wgpu::Sampler,
}

#[derive(gpukit::Bindings)]
struct TextureBindings<'a> {
    #[texture(binding = 0)]
    texture: &'a gpukit::TextureView<gpukit::format::R8Unorm>,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ScreenUniforms {
    width_in_points: f32,
    height_in_points: f32,
    pixels_per_point: f32,
    _padding: u32,
}

struct UniformBuffer<T: bytemuck::Pod> {
    buffer: gpukit::Buffer<T>,
    value: T,
}

impl<T: bytemuck::Pod> std::ops::Deref for UniformBuffer<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T: bytemuck::Pod> std::ops::DerefMut for UniformBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

impl<T: bytemuck::Pod> UniformBuffer<T> {
    pub fn new(context: &gpukit::Context, value: T) -> Self {
        let buffer = context
            .build_buffer()
            .with_usage(wgpu::BufferUsages::UNIFORM)
            .init_with_data(std::slice::from_ref(&value));
        UniformBuffer { buffer, value }
    }

    fn update(&self, context: &gpukit::Context) {
        self.buffer
            .update(context, std::slice::from_ref(&self.value));
    }
}

#[repr(transparent)]
#[derive(Debug, Copy, Clone)]
struct Vertex(egui::paint::Vertex);

// SAFETY: `egui::paint::Vertex` is `#[repr(C)]`.
unsafe impl bytemuck::Pod for Vertex {}
unsafe impl bytemuck::Zeroable for Vertex {}

impl Vertex {
    // SAFETY: `egui::paint::Vertex` is `#[repr(C)]`.
    fn cast_slice(vertices: &[egui::paint::Vertex]) -> &[Vertex] {
        let ptr = vertices.as_ptr() as *const Vertex;
        let len = vertices.len();
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}

pub struct RendererDescriptor {
    pub context: Arc<gpukit::Context>,
    pub target_format: wgpu::TextureFormat,

    // Size of the screen: [width, height]
    pub size: [u32; 2],
    // Number of pixels per point
    pub pixels_per_point: f32,
}

impl Renderer {
    pub fn new(desc: RendererDescriptor) -> anyhow::Result<Renderer> {
        let RendererDescriptor {
            context,
            target_format,
            size,
            pixels_per_point,
        } = desc;

        let vertex_buffer = Self::create_vertex_buffer(&context, 0);
        let index_buffer = Self::create_index_buffer(&context, 0);

        let uniforms = UniformBuffer::new(
            &context,
            ScreenUniforms {
                width_in_points: size[0] as f32 / pixels_per_point,
                height_in_points: size[1] as f32 / pixels_per_point,
                pixels_per_point,
                _padding: 0,
            },
        );

        let sampler = context.device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let bind_group = gpukit::BindGroup::new(
            &context,
            &Bindings {
                screen_uniforms: &uniforms.buffer,
                sampler: &sampler,
            },
        );

        let texture = context
            .build_texture()
            .with_usage(wgpu::TextureUsages::TEXTURE_BINDING)
            .init_with_data([1, 1], &[0]);

        let texture_bind_group = gpukit::BindGroup::new(
            &context,
            &TextureBindings {
                texture: &texture.create_view(),
            },
        );

        let vertex = context
            .build_shader("gukit_egui vertex shader")
            .init_from_glsl(include_str!("shader.vert"), gpukit::ShaderStage::Vertex)?;

        let fragment = context
            .build_shader("gukit_egui fragment shader")
            .init_from_glsl(include_str!("shader.frag"), gpukit::ShaderStage::Fragment)?;

        let pipeline = context.create_render_pipeline(gpukit::RenderPipelineDescriptor {
            label: Some("gpukit_egui renderer"),
            vertex: vertex.entry("main"),
            fragment: fragment.entry("main"),

            vertex_buffers: &[gpukit::vertex_buffer_layout![
                // Position
                0 => Float32x2,
                // Texture Coordinates
                1 => Float32x2,
                // Color
                2 => Uint32,
            ]],
            bind_group_layouts: &[&bind_group.layout, &texture_bind_group.layout],
            color_targets: &[wgpu::ColorTargetState {
                format: target_format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::OneMinusDstAlpha,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::all(),
            }],
            depth_stencil: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
        })?;

        Ok(Renderer {
            context,
            pipeline,
            vertex_buffer,
            index_buffer,

            bind_group,
            screen_uniforms: uniforms,

            texture_bind_group,
            texture_version: None,
            texture,
        })
    }

    fn create_vertex_buffer(context: &gpukit::Context, len: usize) -> gpukit::Buffer<Vertex> {
        context
            .build_buffer()
            .with_usage(wgpu::BufferUsages::VERTEX)
            .init_with_capacity(len)
    }

    fn create_index_buffer(context: &gpukit::Context, len: usize) -> gpukit::Buffer<u32> {
        context
            .build_buffer()
            .with_usage(wgpu::BufferUsages::INDEX)
            .init_with_capacity(len)
    }

    pub fn set_size(&mut self, size: [u32; 2], scale_factor: f32) {
        self.screen_uniforms.width_in_points = size[0] as f32 / scale_factor;
        self.screen_uniforms.height_in_points = size[1] as f32 / scale_factor;
        self.screen_uniforms.pixels_per_point = scale_factor;
        self.screen_uniforms.update(&self.context)
    }

    pub fn render<'encoder>(
        &'encoder mut self,
        rpass: &mut wgpu::RenderPass<'encoder>,
        meshes: &[egui::ClippedMesh],
        texture: &egui::Texture,
    ) {
        use gpukit::RenderPassExt;

        let offsets = self.update_buffers(meshes);
        self.update_texture(texture);

        rpass.set_pipeline(&self.pipeline);
        rpass.set_vertex_buffer(0, *self.vertex_buffer.slice(..));
        rpass.set_index_buffer_ext(self.index_buffer.slice(..));
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.set_bind_group(1, &self.texture_bind_group, &[]);

        for (egui::ClippedMesh(rect, mesh), offset) in meshes.iter().zip(offsets) {
            if Self::set_scissor_region(rpass, &self.screen_uniforms, *rect) {
                let index_range = offset.index..offset.index + mesh.indices.len() as u32;
                rpass.draw_indexed(index_range, offset.vertex as i32, 0..1);
            }
        }
    }

    fn set_scissor_region(
        rpass: &mut wgpu::RenderPass,
        screen: &ScreenUniforms,
        rect: egui::Rect,
    ) -> bool {
        let left = rect.left() * screen.pixels_per_point;
        let right = rect.right() * screen.pixels_per_point;
        let top = rect.top() * screen.pixels_per_point;
        let bottom = rect.bottom() * screen.pixels_per_point;

        let screen_width = screen.width_in_points * screen.pixels_per_point;
        let screen_height = screen.height_in_points * screen.pixels_per_point;

        let left = left.clamp(0.0, screen_width);
        let top = top.clamp(0.0, screen_height);
        let right = right.clamp(left, screen_width);
        let bottom = bottom.clamp(top, screen_height);

        let left = left.round() as u32;
        let top = top.round() as u32;
        let right = right.round() as u32;
        let bottom = bottom.round() as u32;

        let width = right - left;
        let height = bottom - top;

        if width == 0 || height == 0 {
            false
        } else {
            rpass.set_scissor_rect(left, top, width, height);
            true
        }
    }

    fn update_buffers(&mut self, meshes: &[egui::ClippedMesh]) -> Vec<BufferOffset> {
        let mut offsets = Vec::with_capacity(meshes.len());

        // Find out how many vertices/indices we need to render
        let mut vertex_count = 0;
        let mut index_count = 0;
        for egui::ClippedMesh(_, mesh) in meshes {
            offsets.push(BufferOffset {
                vertex: vertex_count,
                index: index_count,
            });
            vertex_count += align_to_power_of_two(mesh.vertices.len() as u32, BUFFER_ALIGNMENT);
            index_count += align_to_power_of_two(mesh.indices.len() as u32, BUFFER_ALIGNMENT);
        }

        // Allocate space for the vertices/indices
        if vertex_count as usize > self.vertex_buffer.len() {
            self.vertex_buffer = Self::create_vertex_buffer(&self.context, vertex_count as usize);
        }
        if index_count as usize > self.index_buffer.len() {
            self.index_buffer = Self::create_index_buffer(&self.context, index_count as usize);
        }

        // Write vertices/indices to their respective buffers
        for (egui::ClippedMesh(_, mesh), offset) in meshes.iter().zip(&offsets) {
            let vertex_slice = Vertex::cast_slice(&mesh.vertices);
            self.vertex_buffer
                .write(&self.context, offset.vertex as usize, vertex_slice);
            self.index_buffer
                .write(&self.context, offset.index as usize, &mesh.indices);
        }

        offsets
    }

    fn update_texture(&mut self, texture: &egui::Texture) {
        if self.texture_version != Some(texture.version) {
            self.texture_version = Some(texture.version);
            self.texture = self
                .context
                .build_texture()
                .with_usage(wgpu::TextureUsages::TEXTURE_BINDING)
                .init_with_data(
                    [texture.width as u32, texture.height as u32],
                    &texture.pixels,
                );
            self.texture_bind_group.update(
                &self.context,
                &TextureBindings {
                    texture: &self.texture.create_view(),
                },
            );
        }
    }
}

struct BufferOffset {
    vertex: u32,
    index: u32,
}

const BUFFER_ALIGNMENT: u32 = wgpu::COPY_BUFFER_ALIGNMENT.next_power_of_two() as u32;

const fn align_to_power_of_two(x: u32, power: u32) -> u32 {
    (x + (power - 1)) & !(power - 1)
}
