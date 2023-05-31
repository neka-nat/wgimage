use futures::executor::block_on;
use wgimage::*;

fn main() {
    let context = WgContext::new();
    let context = block_on(context);
    let image = image::open("examples/lenna.png").unwrap().to_rgba8();
    let width = image.width();
    let height = image.height();
    let image_buffer = WgImageBuffer::from_host_image(&context, image);
    let mut grayscale = GrayScale::new(&context, width, height);
    grayscale.run(&image_buffer);
    let grayscale_image = grayscale.output_image.to_host_image(&context);
    grayscale_image
        .unwrap()
        .save("examples/lenna_grayscale.png")
        .unwrap();
}
