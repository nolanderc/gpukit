pub trait Bindings: Sized {
    const LABEL: Option<&'static str>;
    const LAYOUT_ENTRIES: &'static [wgpu::BindGroupLayoutEntry];

    fn layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Self::LABEL,
            entries: Self::LAYOUT_ENTRIES,
        })
    }

    fn create_bind_group(
        &self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
    ) -> wgpu::BindGroup;
}

pub struct BindGroup {
    pub layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
}

impl std::ops::Deref for BindGroup {
    type Target = wgpu::BindGroup;

    fn deref(&self) -> &Self::Target {
        &self.bind_group
    }
}

impl std::ops::DerefMut for BindGroup {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.bind_group
    }
}

impl BindGroup {
    pub fn new<T: Bindings>(context: &crate::Context, bindings: &T) -> BindGroup {
        let layout = T::layout(&context.device);
        let bind_group = bindings.create_bind_group(&context.device, &layout);
        BindGroup { layout, bind_group }
    }

    pub fn update<T: Bindings>(&mut self, context: &crate::Context, bindings: &T) {
        self.bind_group = bindings.create_bind_group(&context.device, &self.layout);
    }
}
