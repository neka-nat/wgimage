use wgpu::util::DeviceExt;
use wgpu::{
    util::BufferInitDescriptor, BindGroupDescriptor, BindGroupEntry, BindingResource, Buffer,
    BufferUsages, ComputePipeline, ComputePipelineDescriptor, ShaderModuleDescriptor, ShaderSource,
    TextureViewDescriptor,
};

use super::buffer::WgImageBuffer;
use super::context::WgContext;
use super::utils::compute_work_group_count;

const THRESHOLD_SHADER: &str = include_str!("shaders/threshold.wgsl");

pub struct Threshold<'a> {
    pub output_image: WgImageBuffer,
    context: &'a WgContext,
    pipeline: ComputePipeline,
    settings: Buffer,
}

impl<'a> Threshold<'a> {
    pub fn new(context: &'a WgContext, width: u32, height: u32, threshold: u32) -> Self {
        let threshold: f32 = threshold as f32 / 255.0;
        let output_image = WgImageBuffer::from_size(context, width, height);
        let shader = context.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("threshold shader"),
            source: ShaderSource::Wgsl(THRESHOLD_SHADER.into()),
        });
        let settings = context.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Image info"),
            contents: bytemuck::cast_slice(&[threshold]),
            usage: BufferUsages::UNIFORM,
        });

        let pipeline = context
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("threshold pipeline"),
                layout: None,
                module: &shader,
                entry_point: "main",
            });

        Threshold {
            output_image,
            context,
            pipeline,
            settings,
        }
    }
    pub fn run(&mut self, input_image: &WgImageBuffer) {
        let compute_constants = self.context.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Compute constants"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[BindGroupEntry {
                binding: 0,
                resource: self.settings.as_entire_binding(),
            }],
        });
        let image_bind_group = self.context.device.create_bind_group(&BindGroupDescriptor {
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
                            .output_image
                            .texture
                            .create_view(&TextureViewDescriptor::default()),
                    ),
                },
            ],
        });
        let mut encoder = self
            .context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let (dispatch_width, dispatch_height) = compute_work_group_count(
                (
                    input_image.texture_extent.width,
                    input_image.texture_extent.height,
                ),
                (16, 16),
            );
            let mut compute_pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &compute_constants, &[]);
            compute_pass.set_bind_group(1, &image_bind_group, &[]);
            compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
        }
        self.context.queue.submit(Some(encoder.finish()));
    }
}
