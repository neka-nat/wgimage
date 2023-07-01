use futures::executor::block_on;
use wgimage::*;

fn main() {
    let context = WgContext::new();
    let context = block_on(context);
    let image = image::open("examples/lenna.png").unwrap().to_rgba8();
    let width = image.width();
    let height = image.height();
    let image_buffer = WgImageBuffer::from_host_image(&context, image);
    let mut gaussian_blur = GaussianBlur::new(&context, width, height, 5.0);
    gaussian_blur.run(&image_buffer);
    let grayscale_image = gaussian_blur.output_image.to_host_image(&context);
    grayscale_image
        .unwrap()
        .save("examples/lenna_gaussian_blur.png")
        .unwrap();
}
