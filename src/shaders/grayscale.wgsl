@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var output_texture : texture_storage_2d<rgba8unorm, write>;

@compute
@workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_invocation_id : vec3<u32>) {
    let dimensions = textureDimensions(input_texture);
    let coords = vec2<i32>(global_invocation_id.xy);

    let color = textureLoad(input_texture, coords.xy, 0);
    let gray = dot(vec3<f32>(0.299, 0.578, 0.114), color.rgb);

    textureStore(output_texture, coords.xy, vec4<f32>(gray, gray, gray, color.a));
}