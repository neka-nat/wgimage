struct Settings {
    threshold : f32,
};

@group(0) @binding(0) var<uniform> settings : Settings;
@group(1) @binding(0) var input_texture : texture_2d<f32>;
@group(1) @binding(1) var output_texture : texture_storage_2d<rgba8unorm, write>;

@compute
@workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_invocation_id : vec3<u32>) {
    let dimensions = textureDimensions(input_texture);
    let coords = vec2<i32>(global_invocation_id.xy);
    if(coords.x >= i32(dimensions.x) || coords.y >= i32(dimensions.y)) {
        return;
    }

    let color = textureLoad(input_texture, coords.xy, 0);
    let threshold_r = select(1.0, 0.0, color.r > settings.threshold);
    let threshold_g = select(1.0, 0.0, color.g > settings.threshold);
    let threshold_b = select(1.0, 0.0, color.b > settings.threshold);

    textureStore(output_texture, coords.xy, vec4<f32>(threshold_r, threshold_g, threshold_b, color.a));
}
