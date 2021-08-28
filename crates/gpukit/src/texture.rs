pub mod dimension;
pub mod format;

pub struct Texture<Format = format::Rgba32Float, Dimension = dimension::D2>
where
    Format: format::TextureFormat,
    Dimension: dimension::TextureDimension,
{
    raw: wgpu::Texture,
    _phantom: std::marker::PhantomData<(Format, Dimension)>,
}

impl<Format, Dimension> Texture<Format, Dimension>
where
    Format: format::TextureFormat,
    Dimension: dimension::TextureDimension,
{
    fn new(raw: wgpu::Texture) -> Self {
        Texture {
            raw,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn create_view(&self) -> TextureView<Format, Dimension> {
        let raw = self.raw.create_view(&wgpu::TextureViewDescriptor {
            label: None,
            format: Some(Format::TEXTURE_FORMAT),
            dimension: Some(Dimension::TEXTURE_VIEW_DIMENSION),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });

        TextureView::new(raw)
    }

    pub const FORMAT: wgpu::TextureFormat = Format::TEXTURE_FORMAT;

    pub fn format(&self) -> wgpu::TextureFormat {
        Self::FORMAT
    }
}

impl<Format, Dimension> std::ops::Deref for Texture<Format, Dimension>
where
    Format: format::TextureFormat,
    Dimension: dimension::TextureDimension,
{
    type Target = wgpu::Texture;

    fn deref(&self) -> &Self::Target {
        &self.raw
    }
}

pub struct TextureView<Format = format::Rgba32Float, Dimension = dimension::D2>
where
    Format: format::TextureFormat,
    Dimension: dimension::TextureDimension,
{
    raw: wgpu::TextureView,
    _phantom: std::marker::PhantomData<(Format, Dimension)>,
}

impl<Format, Dimension> TextureView<Format, Dimension>
where
    Format: format::TextureFormat,
    Dimension: dimension::TextureDimension,
{
    fn new(raw: wgpu::TextureView) -> Self {
        TextureView {
            raw,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<Format, Dimension> std::ops::Deref for TextureView<Format, Dimension>
where
    Format: format::TextureFormat,
    Dimension: dimension::TextureDimension,
{
    type Target = wgpu::TextureView;

    fn deref(&self) -> &Self::Target {
        &self.raw
    }
}

pub struct TextureBuilder<'a, Format = format::Rgba32Float, Dimension = dimension::D2>
where
    Format: format::TextureFormat,
    Dimension: dimension::TextureDimension,
{
    context: &'a crate::Context,
    usages: wgpu::TextureUsages,
    _phantom: std::marker::PhantomData<(Format, Dimension)>,
}

impl<'a, Format, Dimension> TextureBuilder<'a, Format, Dimension>
where
    Format: format::TextureFormat,
    Dimension: dimension::TextureDimension,
{
    pub(super) fn new(context: &'a crate::Context) -> Self {
        TextureBuilder {
            context,
            usages: wgpu::TextureUsages::empty(),
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn with_usage(&mut self, usages: wgpu::TextureUsages) -> &mut Self {
        self.usages |= usages;
        self
    }

    fn get_descriptor(&self, size: Dimension::Size) -> wgpu::TextureDescriptor {
        use dimension::IntoExtent3D;

        wgpu::TextureDescriptor {
            label: None,
            size: size.into_extent_3d(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: Dimension::TEXTURE_VIEW_DIMENSION.compatible_texture_dimension(),
            format: Format::TEXTURE_FORMAT,
            usage: self.usages,
        }
    }

    pub fn init_with_size(&mut self, size: Dimension::Size) -> Texture<Format, Dimension> {
        let raw = self
            .context
            .device
            .create_texture(&self.get_descriptor(size));

        Texture::new(raw)
    }

    pub fn init_with_data(
        &mut self,
        size: Dimension::Size,
        data: &[Format::Pixel],
    ) -> Texture<Format, Dimension> {
        use wgpu::util::DeviceExt;

        let raw = self.context.device.create_texture_with_data(
            &self.context.queue,
            &self.get_descriptor(size),
            bytemuck::cast_slice(data),
        );

        Texture::new(raw)
    }
}

pub trait TextureResource {
    fn texture_view_binding(&self) -> &wgpu::TextureView;
}

impl<T: TextureResource> TextureResource for &T {
    fn texture_view_binding(&self) -> &wgpu::TextureView {
        T::texture_view_binding(*self)
    }
}

impl TextureResource for wgpu::TextureView {
    fn texture_view_binding(&self) -> &wgpu::TextureView {
        self
    }
}

impl<Format, Dimension> TextureResource for TextureView<Format, Dimension>
where
    Format: format::TextureFormat,
    Dimension: dimension::TextureDimension,
{
    fn texture_view_binding(&self) -> &wgpu::TextureView {
        &*self
    }
}
