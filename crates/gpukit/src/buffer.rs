use std::num::NonZeroU64;

pub struct Buffer<T: bytemuck::Pod> {
    raw: wgpu::Buffer,
    len: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: bytemuck::Pod> Buffer<T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn slice(&self, range: impl std::ops::RangeBounds<usize>) -> BufferSlice<T> {
        BufferSlice {
            raw: self.raw_slice(range),
            _phantom: std::marker::PhantomData,
        }
    }

    fn raw_slice(&self, range: impl std::ops::RangeBounds<usize>) -> wgpu::BufferSlice {
        fn map_bound<T, U>(
            bound: std::ops::Bound<T>,
            mut f: impl FnMut(T) -> U,
        ) -> std::ops::Bound<U> {
            use std::ops::Bound::{Excluded, Included, Unbounded};
            match bound {
                Included(a) => Included(f(a)),
                Excluded(b) => Excluded(f(b)),
                Unbounded => Unbounded,
            }
        }

        let index_to_address = |index| (index * std::mem::size_of::<T>()) as u64;

        let start = map_bound(range.start_bound(), index_to_address);
        let end = map_bound(range.end_bound(), index_to_address);

        self.raw.slice((start, end))
    }

    pub fn update(&self, context: &crate::Context, data: &[T]) {
        assert_eq!(self.len, data.len());
        context
            .queue
            .write_buffer(&self.raw, 0, bytemuck::cast_slice(data));
    }

    pub fn write(&self, context: &crate::Context, offset: usize, data: &[T]) {
        assert!(offset + data.len() <= self.len);
        context.queue.write_buffer(
            &self.raw,
            (offset * std::mem::size_of::<T>()) as u64,
            bytemuck::cast_slice(data),
        )
    }
}

impl<T: bytemuck::Pod> std::ops::Deref for Buffer<T> {
    type Target = wgpu::Buffer;

    fn deref(&self) -> &Self::Target {
        &self.raw
    }
}

pub struct BufferBuilder<'device, 'label, T: bytemuck::Pod> {
    device: &'device wgpu::Device,
    label: Option<&'label str>,
    usage: wgpu::BufferUsages,
    _phantom: std::marker::PhantomData<T>,
}

impl<'device, 'label, T: bytemuck::Pod> BufferBuilder<'device, 'label, T> {
    pub fn new(device: &'device wgpu::Device) -> Self {
        BufferBuilder {
            device,
            label: None,
            usage: wgpu::BufferUsages::COPY_DST,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn with_label(&mut self, label: impl Into<Option<&'label str>>) -> &mut Self {
        self.label = label.into();
        self
    }

    pub fn with_usage(&mut self, usage: wgpu::BufferUsages) -> &mut Self {
        self.usage |= usage;
        self
    }

    pub fn init_with_capacity(&mut self, len: usize) -> Buffer<T> {
        let raw = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: self.label,
            size: (len * std::mem::size_of::<T>()) as u64,
            usage: self.usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Buffer {
            raw,
            len,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn init_with_data(&mut self, data: &[T]) -> Buffer<T> {
        use wgpu::util::DeviceExt;

        let label;
        let label = match self.label {
            Some(label) => Some(label),
            None => {
                label = format!("Buffer<{}>", std::any::type_name::<T>());
                Some(label.as_str())
            }
        };

        let raw = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label,
                usage: self.usage | wgpu::BufferUsages::COPY_DST,
                contents: bytemuck::cast_slice(data),
            });

        Buffer {
            raw,
            len: data.len(),
            _phantom: std::marker::PhantomData,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct BufferSlice<'buffer, T: bytemuck::Pod> {
    raw: wgpu::BufferSlice<'buffer>,
    _phantom: std::marker::PhantomData<T>,
}

impl<'buffer, T: bytemuck::Pod> std::ops::Deref for BufferSlice<'buffer, T> {
    type Target = wgpu::BufferSlice<'buffer>;

    fn deref(&self) -> &Self::Target {
        &self.raw
    }
}

pub trait BufferResource {
    const ELEMENT_SIZE: Option<NonZeroU64>;

    fn buffer_binding(&self) -> wgpu::BufferBinding;
}

impl<T: bytemuck::Pod> BufferResource for Buffer<T> {
    const ELEMENT_SIZE: Option<NonZeroU64> = NonZeroU64::new(std::mem::size_of::<T>() as u64);

    fn buffer_binding(&self) -> wgpu::BufferBinding {
        self.raw.as_entire_buffer_binding()
    }
}

impl<T: BufferResource> BufferResource for &T {
    const ELEMENT_SIZE: Option<NonZeroU64> = T::ELEMENT_SIZE;

    fn buffer_binding(&self) -> wgpu::BufferBinding {
        T::buffer_binding(*self)
    }
}

pub struct UniformBuffer<T: bytemuck::Pod> {
    buffer: Buffer<T>,
    data: T,
}

impl<T> std::ops::Deref for UniformBuffer<T>
where
    T: bytemuck::Pod,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T> std::ops::DerefMut for UniformBuffer<T>
where
    T: bytemuck::Pod,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<T> UniformBuffer<T>
where
    T: bytemuck::Pod,
{
    pub fn new(context: &crate::Context, data: T) -> Self {
        let buffer = context
            .build_buffer()
            .with_usage(wgpu::BufferUsages::UNIFORM)
            .init_with_data(std::slice::from_ref(&data));
        UniformBuffer { buffer, data }
    }

    pub fn update(&self, context: &crate::Context) {
        self.buffer
            .update(context, std::slice::from_ref(&self.data));
    }

    pub fn buffer(&self) -> &Buffer<T> {
        &self.buffer
    }
}
