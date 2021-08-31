#[macro_use]
extern crate anyhow;
#[macro_use]
extern crate tracing;

mod bind_group;
mod buffer;
mod macros;
mod shader;
mod texture;

pub use wgpu;
pub use winit;

pub use self::bind_group::*;
pub use self::buffer::*;
pub use self::shader::*;
pub use self::texture::*;

#[cfg(feature = "derive")]
pub use gpukit_derive::Bindings;

use anyhow::Context as _;
use std::sync::Arc;

pub async fn init(window: &winit::window::Window) -> anyhow::Result<(Arc<Context>, Surface)> {
    let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);
    let raw_surface = unsafe { instance.create_surface(window) };

    let context = Context::new(instance, &raw_surface).await?;
    let surface = Surface::new(context.clone(), window, Some(raw_surface));

    Ok((context, surface))
}

pub struct Context {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl Context {
    pub async fn new(
        instance: wgpu::Instance,
        compatible_surface: &wgpu::Surface,
    ) -> anyhow::Result<Arc<Context>> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(compatible_surface),
            })
            .await
            .ok_or_else(|| anyhow!("failed to find compatible GPU adapter"))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::default(),
                    limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .context("failed to find compatible GPU device")?;

        let context = Arc::new(Context {
            instance,
            adapter,
            device,
            queue,
        });

        Ok(context)
    }

    pub fn build_buffer<T>(&self) -> BufferBuilder<T>
    where
        T: bytemuck::Pod,
    {
        BufferBuilder::new(&self.device)
    }

    pub fn build_texture<Format, Dimension>(&self) -> TextureBuilder<Format, Dimension>
    where
        Format: texture::format::TextureFormat,
        Dimension: texture::dimension::TextureDimension,
    {
        TextureBuilder::new(self)
    }

    pub fn build_shader<'a>(&'a self, label: &'a str) -> ShaderBuilder<'a> {
        ShaderBuilder::new(self, label)
    }

    pub fn create_render_pipeline(
        &self,
        desc: RenderPipelineDescriptor,
    ) -> anyhow::Result<wgpu::RenderPipeline> {
        let vertex = wgpu::VertexState {
            module: desc.vertex.module,
            entry_point: desc.vertex.entry_point,
            buffers: desc.vertex_buffers,
        };

        let fragment = Some(wgpu::FragmentState {
            module: desc.fragment.module,
            entry_point: desc.fragment.entry_point,
            targets: desc.color_targets,
        });

        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: desc.bind_group_layouts,
                push_constant_ranges: &[],
            });

        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: desc.label,
                layout: Some(&layout),
                vertex,
                primitive: desc.primitive,
                depth_stencil: desc.depth_stencil,
                multisample: wgpu::MultisampleState::default(),
                fragment,
            });

        Ok(pipeline)
    }

    pub fn create_encoder(&self) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None })
    }

    pub fn submit(&self, encoder: wgpu::CommandEncoder) {
        let commands = encoder.finish();
        self.queue.submit(Some(commands));
    }
}

pub struct Surface {
    context: Arc<Context>,
    surface: wgpu::Surface,
    config: wgpu::SurfaceConfiguration,
}

impl Surface {
    pub fn new(
        context: Arc<Context>,
        window: &winit::window::Window,
        raw_surface: Option<wgpu::Surface>,
    ) -> Surface {
        let raw_surface =
            raw_surface.unwrap_or_else(|| unsafe { context.instance.create_surface(window) });

        let format = raw_surface
            .get_preferred_format(&context.adapter)
            .unwrap_or(wgpu::TextureFormat::Bgra8UnormSrgb);

        let size = window.inner_size();

        let mut surface = Surface {
            context,
            surface: raw_surface,
            config: wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: size.width,
                height: size.height,
                present_mode: wgpu::PresentMode::Mailbox,
            },
        };

        surface.configure();

        surface
    }

    pub fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        self.config.width = size.width;
        self.config.height = size.height;
        self.configure();
    }

    pub fn recreate(&mut self, window: &winit::window::Window) {
        self.surface = unsafe { self.context.instance.create_surface(window) };
        self.resize(window.inner_size());
    }

    pub fn texture_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    pub fn color_target(&self, blend: Option<wgpu::BlendState>) -> wgpu::ColorTargetState {
        wgpu::ColorTargetState {
            format: self.texture_format(),
            blend,
            write_mask: wgpu::ColorWrites::all(),
        }
    }

    pub fn size(&self) -> winit::dpi::PhysicalSize<u32> {
        [self.config.width, self.config.height].into()
    }

    fn configure(&mut self) {
        self.surface.configure(&self.context.device, &self.config);
    }

    pub fn get_current_frame(&mut self) -> anyhow::Result<wgpu::SurfaceFrame> {
        for _ in 0..3 {
            match self.surface.get_current_frame() {
                Ok(frame) => return Ok(frame),
                Err(error) => match error {
                    wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost => {
                        warn!("need to recreate swapchain: {}", error);
                        self.configure();
                        continue;
                    }
                    _ => return Err(anyhow!("failed to get the next frame: {}", error)),
                },
            }
        }

        Err(anyhow!("failed to get the next frame: ran out of attempts"))
    }
}

pub struct RenderPipelineDescriptor<'a> {
    pub label: Option<&'a str>,
    pub vertex: ShaderEntry<'a>,
    pub fragment: ShaderEntry<'a>,
    pub vertex_buffers: &'a [wgpu::VertexBufferLayout<'a>],
    pub bind_group_layouts: &'a [&'a wgpu::BindGroupLayout],
    pub color_targets: &'a [wgpu::ColorTargetState],
    pub depth_stencil: Option<wgpu::DepthStencilState>,
    pub primitive: wgpu::PrimitiveState,
}

pub trait RenderPassExt<'encoder> {
    fn set_index_buffer_ext<T>(&mut self, buffer: BufferSlice<'encoder, T>)
    where
        T: IndexElement;
}

impl<'encoder, R> RenderPassExt<'encoder> for R
where
    R: wgpu::util::RenderEncoder<'encoder>,
{
    fn set_index_buffer_ext<T>(&mut self, buffer: BufferSlice<'encoder, T>)
    where
        T: IndexElement,
    {
        self.set_index_buffer(*buffer, T::INDEX_FORMAT)
    }
}

pub trait IndexElement: bytemuck::Pod {
    const INDEX_FORMAT: wgpu::IndexFormat;
}

impl IndexElement for u32 {
    const INDEX_FORMAT: wgpu::IndexFormat = wgpu::IndexFormat::Uint32;
}

impl IndexElement for u16 {
    const INDEX_FORMAT: wgpu::IndexFormat = wgpu::IndexFormat::Uint16;
}

pub mod util {
    pub fn color(r: f64, g: f64, b: f64, a: f64) -> wgpu::Color {
        wgpu::Color { r, g, b, a }
    }
}
