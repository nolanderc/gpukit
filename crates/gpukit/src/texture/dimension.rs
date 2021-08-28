pub trait TextureDimension {
    const TEXTURE_VIEW_DIMENSION: wgpu::TextureViewDimension;

    /// Number of dimensions of the underlying storage required to represent this texture
    const SIZE: usize;

    type Size: IntoExtent3D;
}

pub trait IntoExtent3D {
    fn into_extent_3d(self) -> wgpu::Extent3d;
}

impl IntoExtent3D for [u32; 1] {
    fn into_extent_3d(self) -> wgpu::Extent3d {
        wgpu::Extent3d {
            width: self[0],
            height: 1,
            depth_or_array_layers: 1,
        }
    }
}

impl IntoExtent3D for [u32; 2] {
    fn into_extent_3d(self) -> wgpu::Extent3d {
        wgpu::Extent3d {
            width: self[0],
            height: self[1],
            depth_or_array_layers: 1,
        }
    }
}

impl IntoExtent3D for [u32; 3] {
    fn into_extent_3d(self) -> wgpu::Extent3d {
        wgpu::Extent3d {
            width: self[0],
            height: self[1],
            depth_or_array_layers: self[2],
        }
    }
}

macro_rules! texture_dimension {
    ($ident:ident, $size:expr) => {
        pub struct $ident;

        impl TextureDimension for $ident {
            const TEXTURE_VIEW_DIMENSION: wgpu::TextureViewDimension =
                wgpu::TextureViewDimension::$ident;
            const SIZE: usize = $size;

            type Size = [u32; $size];
        }
    };
}

texture_dimension!(D1, 1);
texture_dimension!(D2, 2);
texture_dimension!(D3, 3);
texture_dimension!(D2Array, 3);
texture_dimension!(Cube, 2);
texture_dimension!(CubeArray, 3);
