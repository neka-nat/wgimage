use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindingResource, ComputePipeline,
    ComputePipelineDescriptor, ShaderModuleDescriptor, ShaderSource, TextureViewDescriptor,
};

use super::buffer::WgImageBuffer;
use super::context::WgContext;
use super::utils::compute_work_group_count;

const GRAYSCALE_SHADER: &str = include_str!("shaders/grayscale.wgsl");

pub struct GrayScale<'a> {
    pub output_image: WgImageBuffer,
    context: &'a WgContext,
    pipeline: ComputePipeline,
}

impl<'a> GrayScale<'a> {
    pub fn new(context: &'a WgContext, width: u32, height: u32) -> Self {
        let output_image = WgImageBuffer::from_size(context, width, height);
        let shader = context.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("grayscale shader"),
            source: ShaderSource::Wgsl(GRAYSCALE_SHADER.into()),
        });
        let pipeline = context
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("grayscale pipeline"),
                layout: None,
                module: &shader,
                entry_point: "main",
            });
        GrayScale {
            output_image,
            context,
            pipeline,
        }
    }
    pub fn run(&mut self, input_image: &WgImageBuffer) {
        let bind_group = self.context.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Texture bind group"),
            layout: &self.pipeline.get_bind_group_layout(0),
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
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
        }
        self.context.queue.submit(Some(encoder.finish()));
    }
}
