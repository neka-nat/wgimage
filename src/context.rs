use wgpu::{Device, Instance, InstanceDescriptor, PowerPreference, Queue};

pub struct WgContext {
    pub device: Device,
    pub queue: Queue,
}

impl WgContext {
    pub async fn new() -> Self {
        let instance = Instance::new(InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptionsBase {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&Default::default(), None)
            .await
            .unwrap();

        Self { device, queue }
    }
}
