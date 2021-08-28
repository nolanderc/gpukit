pub trait TextureFormat {
    const TEXTURE_FORMAT: wgpu::TextureFormat;
    type Pixel: bytemuck::Pod;
}

macro_rules! texture_format {
    ($ident:ident: $pixel:ty) => {
        pub struct $ident;

        impl TextureFormat for $ident {
            const TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::$ident;
            type Pixel = $pixel;
        }
    };
}

texture_format!(R8Unorm: u8);
texture_format!(Rg8Unorm: u8);

texture_format!(Bgra8Unorm: [u8; 4]);
texture_format!(Rgba8Unorm: [u8; 4]);

texture_format!(Bgra8UnormSrgb: [u8; 4]);
texture_format!(Rgba8UnormSrgb: [u8; 4]);

texture_format!(R32Float: [f32; 1]);
texture_format!(Rg32Float: [f32; 2]);
texture_format!(Rgba32Float: [f32; 4]);

texture_format!(Depth32Float: f32);
