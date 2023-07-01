use wgpu::util::DeviceExt;
use wgpu::{
    util::BufferInitDescriptor, BindGroupDescriptor, BindGroupEntry, BindingResource, Buffer,
    BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, ShaderModuleDescriptor, ShaderSource, TextureViewDescriptor,
};

use super::buffer::WgImageBuffer;
use super::context::WgContext;
use super::utils::compute_work_group_count;

const GAUSSIAN_BLUR_SHADER: &str = include_str!("shaders/gaussian_blur.wgsl");

pub struct GaussianBlur<'a> {
    vertical_pass_image: WgImageBuffer,
    pub output_image: WgImageBuffer,
    context: &'a WgContext,
    pipeline: ComputePipeline,
    settings: Buffer,
    kernel: Buffer,
    vertical: Buffer,
    horizontal: Buffer,
}

struct Kernel {
    sum: f32,
    values: Vec<f32>,
}

impl Kernel {
    fn new(values: Vec<f32>) -> Self {
        let sum = values.iter().sum();
        Self { sum, values }
    }

    fn packed_data(&self) -> Vec<f32> {
        let mut data = vec![0.0; self.values.len() + 1];
        data[0] = self.sum;
        data[1..].copy_from_slice(&self.values);
        data
    }

    fn size(&self) -> usize {
        self.values.len()
    }
}

fn create_kernel(sigma: f32) -> Kernel {
    let kernel_size = 2 * (sigma * 3.0).ceil() as u32 + 1;
    let mut values = vec![0.0; kernel_size as usize];
    let kernel_radius = (kernel_size as usize - 1) / 2;
    for index in 0..=kernel_radius {
        let x = index as f32;
        let normpdf = 0.39894 * (-0.5 * x * x / (sigma * sigma)).exp() / sigma;
        values[kernel_radius + index] = normpdf;
        values[kernel_radius - index] = normpdf;
    }
    Kernel::new(values)
}

impl<'a> GaussianBlur<'a> {
    pub fn new(context: &'a WgContext, width: u32, height: u32, sigma: f32) -> Self {
        let kernel = create_kernel(sigma);
        let kernel_size = kernel.size() as u32;
        let vertical_pass_image = WgImageBuffer::from_size(context, width, height);
        let horizontal_pass_image = WgImageBuffer::from_size(context, width, height);
        let shader = context.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("gaussian blur shader"),
            source: ShaderSource::Wgsl(GAUSSIAN_BLUR_SHADER.into()),
        });
        let pipeline = context
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("gaussian blur pipeline"),
                layout: None,
                module: &shader,
                entry_point: "main",
            });
        let settings = context.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Image info"),
            contents: bytemuck::cast_slice(&[kernel_size]),
            usage: BufferUsages::UNIFORM,
        });
        let kernel = context.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&kernel.packed_data()[..]),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let vertical = context.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Orientation"),
            contents: bytemuck::cast_slice::<u32, u8>(&[1]),
            usage: BufferUsages::UNIFORM,
        });
        let horizontal = context.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Orientation"),
            contents: bytemuck::cast_slice::<u32, u8>(&[0]),
            usage: BufferUsages::UNIFORM,
        });
        GaussianBlur {
            vertical_pass_image,
            output_image: horizontal_pass_image,
            context,
            pipeline,
            settings,
            kernel,
            vertical,
            horizontal,
        }
    }
    pub fn run(&mut self, input_image: &WgImageBuffer) {
        let compute_constants = self.context.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Compute constants"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.settings.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: self.kernel.as_entire_binding(),
                },
            ],
        });
        let vertical_bind_group = self.context.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Texture bind group"),
            layout: &self.pipeline.get_bind_group_layout(1),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(
                        &input_image
                            .texture
                            .create_view(&TextureViewDescriptor::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(
                        &self
                            .vertical_pass_image
                            .texture
                            .create_view(&TextureViewDescriptor::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: self.vertical.as_entire_binding(),
                },
            ],
        });
        let horizontal_bind_group = self.context.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Texture bind group"),
            layout: &self.pipeline.get_bind_group_layout(1),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(
                        &self
                            .vertical_pass_image
                            .texture
                            .create_view(&TextureViewDescriptor::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(
                        &self
                            .output_image
                            .texture
                            .create_view(&TextureViewDescriptor::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: self.horizontal.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .context
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut compute_pass =
                encoder.begin_compute_pass(&ComputePassDescriptor { label: None });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &compute_constants, &[]);
            compute_pass.set_bind_group(1, &vertical_bind_group, &[]);
            let (dispatch_with, dispatch_height) = compute_work_group_count(
                (
                    input_image.texture_extent.width,
                    input_image.texture_extent.height,
                ),
                (128, 1),
            );
            compute_pass.dispatch_workgroups(dispatch_with, dispatch_height, 1);
            compute_pass.set_bind_group(1, &horizontal_bind_group, &[]);
            let (dispatch_height, dispatch_with) = compute_work_group_count(
                (
                    input_image.texture_extent.width,
                    input_image.texture_extent.height,
                ),
                (1, 128),
            );
            compute_pass.dispatch_workgroups(dispatch_with, dispatch_height, 1);
        }

        self.context.queue.submit(Some(encoder.finish()));
    }
}
