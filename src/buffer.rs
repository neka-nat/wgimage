use super::context::WgContext;
use super::utils::padded_bytes_per_row;
use bytemuck;
use image;
use wgpu::{
    BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Extent3d, ImageCopyTexture, Texture,
    TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
};

pub struct WgImageBuffer {
    pub texture: Texture,
    pub texture_extent: Extent3d,
}

impl WgImageBuffer {
    fn from_host_image_with_additional_flag(
        context: &WgContext,
        image: image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
        additional_flag: TextureUsages,
    ) -> Self {
        let (w, h) = image.dimensions();
        let texture_extent = Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        };
        let texture = context.device.create_texture(&TextureDescriptor {
            size: texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | additional_flag,
            label: None,
            view_formats: &[],
        });
        context.queue.write_texture(
            texture.as_image_copy(),
            &image,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * texture_extent.width),
                rows_per_image: Some(texture_extent.height),
            },
            texture_extent,
        );

        WgImageBuffer {
            texture,
            texture_extent,
        }
    }
    pub fn from_host_image_readonly(
        context: &WgContext,
        image: image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
    ) -> Self {
        Self::from_host_image_with_additional_flag(context, image, TextureUsages::empty())
    }
    pub fn from_host_image(
        context: &WgContext,
        image: image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
    ) -> Self {
        Self::from_host_image_with_additional_flag(
            context,
            image,
            TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
        )
    }
    pub fn from_size(context: &WgContext, width: u32, height: u32) -> WgImageBuffer {
        let texture_extent = Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = context.device.create_texture(&TextureDescriptor {
            size: texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::STORAGE_BINDING
                | TextureUsages::COPY_SRC,
            label: None,
            view_formats: &[],
        });
        WgImageBuffer {
            texture,
            texture_extent,
        }
    }
    pub fn to_host_image(
        &self,
        context: &WgContext,
    ) -> Option<image::ImageBuffer<image::Rgba<u8>, Vec<u8>>> {
        let mut encoder = context
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });

        let padded_bytes_per_row = padded_bytes_per_row(self.texture_extent.width);
        let unpadded_bytes_per_row = self.texture_extent.width as usize * 4;

        let output_buffer_size = padded_bytes_per_row as u64
            * self.texture_extent.height as u64
            * std::mem::size_of::<u8>() as u64;
        let output_buffer = context.device.create_buffer(&BufferDescriptor {
            label: None,
            size: output_buffer_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_texture_to_buffer(
            ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::ImageCopyBuffer {
                buffer: &output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row as u32),
                    rows_per_image: Some(self.texture_extent.height),
                },
            },
            self.texture_extent,
        );
        context.queue.submit(Some(encoder.finish()));

        let buffer_slice = output_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});

        context.device.poll(wgpu::Maintain::Wait);

        let padded_data = buffer_slice.get_mapped_range();

        let mut flat_pixels =
            vec![0; unpadded_bytes_per_row * (self.texture_extent.height as usize)];
        for (padded, flat_pixels) in padded_data
            .chunks_exact(padded_bytes_per_row)
            .zip(flat_pixels.chunks_exact_mut(padded_bytes_per_row))
        {
            flat_pixels.copy_from_slice(bytemuck::cast_slice(&padded[..unpadded_bytes_per_row]));
        }

        image::ImageBuffer::<image::Rgba<u8>, Vec<u8>>::from_raw(
            self.texture_extent.width,
            self.texture_extent.height,
            flat_pixels,
        )
    }
}
