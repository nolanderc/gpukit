#[macro_export]
macro_rules! vertex_buffer_layout {
    [
        $($binding:expr => $format:ident),+ $(,)?
    ] => {
        wgpu::VertexBufferLayout {
            array_stride: {
                0u64 $( + wgpu::VertexFormat::$format.size())+
            },
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array!($($binding => $format),+)
        }
    }
}

#[macro_export]
macro_rules! instance_buffer_layout {
    [
        $($binding:expr => $format:ident),+ $(,)?
    ] => {
        wgpu::VertexBufferLayout {
            array_stride: {
                0u64 $( + wgpu::VertexFormat::$format.size())+
            },
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array!($($binding => $format),+)
        }
    }
}
