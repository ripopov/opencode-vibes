use bytemuck::{Pod, Zeroable};
use crossbeam_channel::Sender;
use eframe::{egui, wgpu};
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

#[cfg(not(target_arch = "wasm32"))]
use crossbeam_channel::{bounded, Receiver, RecvTimeoutError};
#[cfg(not(target_arch = "wasm32"))]
use image::RgbaImage;
#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::spawn_local;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

const IMAGE_WIDTH: usize = 640;
const IMAGE_HEIGHT: usize = 640;
const MAX_BOUNCES: u32 = 10;
const FAST_BOUNCES: u32 = 1;
const SNAPSHOT_INTERVAL: Duration = Duration::from_secs(3);
const SNAPSHOT_FILE: &str = "snapshot.png";
const CAMERA_MOVE_SPEED: f32 = 1.8;
const CAMERA_LOOK_SENSITIVITY: f32 = 0.0028;
const CAMERA_IDLE_TO_PATH: Duration = Duration::from_secs(1);

#[cfg(target_arch = "wasm32")]
std::thread_local! {
    static WEB_RUNNER: RefCell<Option<eframe::WebRunner>> = const { RefCell::new(None) };
}

const PRIMITIVE_XY: u32 = 0;
const PRIMITIVE_XZ: u32 = 1;
const PRIMITIVE_YZ: u32 = 2;
const PRIMITIVE_SPHERE: u32 = 3;

const MATERIAL_DIFFUSE: u32 = 0;
const MATERIAL_METAL: u32 = 1;
const MATERIAL_GLOSSY: u32 = 2;
const MATERIAL_EMISSIVE: u32 = 3;

const RENDER_MODE_PATH: u32 = 0;
const RENDER_MODE_FAST: u32 = 1;

#[derive(Clone, Copy, Debug, Default)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    fn length_squared(self) -> f32 {
        self.dot(self)
    }

    fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    fn normalized(self) -> Self {
        let len = self.length();
        if len > 0.0 {
            self / len
        } else {
            Self::new(0.0, 0.0, 0.0)
        }
    }
}

impl Add for Vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub for Vec3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Mul<f32> for Vec3 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Mul<Vec3> for f32 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Self::Output {
        rhs * self
    }
}

impl Div<f32> for Vec3 {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl Neg for Vec3 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y, -self.z)
    }
}

#[derive(Clone, Copy)]
enum Material {
    Diffuse {
        albedo: Vec3,
    },
    Metal {
        albedo: Vec3,
        fuzz: f32,
    },
    Glossy {
        albedo: Vec3,
        roughness: f32,
        reflectivity: f32,
    },
    Emissive {
        color: Vec3,
    },
}

#[derive(Clone, Copy)]
struct RectXY {
    x0: f32,
    x1: f32,
    y0: f32,
    y1: f32,
    z: f32,
    normal_z: f32,
    material: Material,
}

#[derive(Clone, Copy)]
struct RectXZ {
    x0: f32,
    x1: f32,
    z0: f32,
    z1: f32,
    y: f32,
    normal_y: f32,
    material: Material,
}

#[derive(Clone, Copy)]
struct RectYZ {
    y0: f32,
    y1: f32,
    z0: f32,
    z1: f32,
    x: f32,
    normal_x: f32,
    material: Material,
}

#[derive(Clone, Copy)]
struct Sphere {
    center: Vec3,
    radius: f32,
    material: Material,
}

#[derive(Clone, Copy)]
enum Primitive {
    XY(RectXY),
    XZ(RectXZ),
    YZ(RectYZ),
    Sphere(Sphere),
}

struct Scene {
    objects: Vec<Primitive>,
}

#[derive(Clone, Copy)]
struct Camera {
    origin: Vec3,
    lower_left: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
}

impl Camera {
    fn new(look_from: Vec3, look_at: Vec3, up: Vec3, fov_degrees: f32, aspect: f32) -> Self {
        let theta = fov_degrees.to_radians();
        let half_height = (theta * 0.5).tan();
        let viewport_height = 2.0 * half_height;
        let viewport_width = aspect * viewport_height;

        let w = (look_from - look_at).normalized();
        let u = up.cross(w).normalized();
        let v = w.cross(u);

        let horizontal = u * viewport_width;
        let vertical = v * viewport_height;
        let lower_left = look_from - horizontal * 0.5 - vertical * 0.5 - w;

        Self {
            origin: look_from,
            lower_left,
            horizontal,
            vertical,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RenderMode {
    Path,
    Fast,
}

impl RenderMode {
    const fn as_gpu(self) -> u32 {
        match self {
            Self::Path => RENDER_MODE_PATH,
            Self::Fast => RENDER_MODE_FAST,
        }
    }

    const fn bounces(self) -> u32 {
        match self {
            Self::Path => MAX_BOUNCES,
            Self::Fast => FAST_BOUNCES,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Path => "Path tracing",
            Self::Fast => "Fast raytracer",
        }
    }
}

struct FpsCamera {
    position: Vec3,
    yaw: f32,
    pitch: f32,
    fov_degrees: f32,
    aspect: f32,
}

impl FpsCamera {
    fn from_look_at(look_from: Vec3, look_at: Vec3, fov_degrees: f32, aspect: f32) -> Self {
        let forward = (look_at - look_from).normalized();
        let yaw = forward.x.atan2(forward.z);
        let pitch = forward.y.clamp(-0.999, 0.999).asin();

        Self {
            position: look_from,
            yaw,
            pitch,
            fov_degrees,
            aspect,
        }
    }

    fn forward(&self) -> Vec3 {
        let cos_pitch = self.pitch.cos();
        Vec3::new(
            self.yaw.sin() * cos_pitch,
            self.pitch.sin(),
            self.yaw.cos() * cos_pitch,
        )
        .normalized()
    }

    fn to_camera(&self) -> Camera {
        let forward = self.forward();
        Camera::new(
            self.position,
            self.position + forward,
            Vec3::new(0.0, 1.0, 0.0),
            self.fov_degrees,
            self.aspect,
        )
    }

    fn update_from_input(&mut self, ctx: &egui::Context, dt_seconds: f32) -> bool {
        let (
            pointer_delta,
            right_mouse_down,
            boost,
            move_forward,
            move_back,
            move_left,
            move_right,
        ) = ctx.input(|input| {
            (
                input.pointer.delta(),
                input.pointer.button_down(egui::PointerButton::Secondary),
                input.modifiers.shift,
                input.key_down(egui::Key::W),
                input.key_down(egui::Key::S),
                input.key_down(egui::Key::A),
                input.key_down(egui::Key::D),
            )
        });

        let mut changed = false;
        if right_mouse_down
            && (pointer_delta.x.abs() > f32::EPSILON || pointer_delta.y.abs() > f32::EPSILON)
        {
            self.yaw += pointer_delta.x * CAMERA_LOOK_SENSITIVITY;
            self.pitch =
                (self.pitch - pointer_delta.y * CAMERA_LOOK_SENSITIVITY).clamp(-1.52, 1.52);
            changed = true;
        }

        let mut horizontal_forward = self.forward();
        horizontal_forward.y = 0.0;
        if horizontal_forward.length_squared() > 1e-6 {
            horizontal_forward = horizontal_forward.normalized();
        } else {
            horizontal_forward = Vec3::new(0.0, 0.0, 1.0);
        }

        let mut right = Vec3::new(horizontal_forward.z, 0.0, -horizontal_forward.x);
        if right.length_squared() > 1e-6 {
            right = right.normalized();
        } else {
            right = Vec3::new(1.0, 0.0, 0.0);
        }

        let mut movement = Vec3::new(0.0, 0.0, 0.0);
        if move_forward {
            movement = movement + horizontal_forward;
        }
        if move_back {
            movement = movement - horizontal_forward;
        }
        if move_right {
            movement = movement - right;
        }
        if move_left {
            movement = movement + right;
        }

        if movement.length_squared() > 0.0 {
            let speed = CAMERA_MOVE_SPEED * if boost { 1.8 } else { 1.0 };
            self.position = self.position + movement.normalized() * (speed * dt_seconds.max(0.0));
            changed = true;
        }

        changed
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuUniforms {
    dims: [u32; 4],
    frame: [u32; 4],
    origin: [f32; 4],
    lower_left: [f32; 4],
    horizontal: [f32; 4],
    vertical: [f32; 4],
}

impl GpuUniforms {
    fn new(
        width: u32,
        height: u32,
        primitive_count: u32,
        sample_index: u32,
        seed: u32,
        render_mode: RenderMode,
        max_bounces: u32,
        camera: Camera,
    ) -> Self {
        Self {
            dims: [width, height, primitive_count, max_bounces],
            frame: [sample_index, seed, render_mode.as_gpu(), 0],
            origin: [camera.origin.x, camera.origin.y, camera.origin.z, 0.0],
            lower_left: [
                camera.lower_left.x,
                camera.lower_left.y,
                camera.lower_left.z,
                0.0,
            ],
            horizontal: [
                camera.horizontal.x,
                camera.horizontal.y,
                camera.horizontal.z,
                0.0,
            ],
            vertical: [camera.vertical.x, camera.vertical.y, camera.vertical.z, 0.0],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuPrimitive {
    header: [u32; 4],
    p0: [f32; 4],
    p1: [f32; 4],
    color: [f32; 4],
    params: [f32; 4],
}

struct GpuPathTracer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    width: u32,
    height: u32,
    primitive_count: u32,
    camera: Camera,
    uniform_buffer: wgpu::Buffer,
    _primitive_buffer: wgpu::Buffer,
    accumulation_buffer: wgpu::Buffer,
    output_texture: wgpu::Texture,
    _output_view: wgpu::TextureView,
    texture_id: egui::TextureId,
    readback_buffer: wgpu::Buffer,
    padded_bytes_per_row: usize,
    bind_group: wgpu::BindGroup,
    compute_pipeline: wgpu::ComputePipeline,
    sample_index: u32,
    preview_frame_index: u32,
    seed: u32,
    render_mode: RenderMode,
    accumulation_dirty: bool,
}

impl GpuPathTracer {
    fn new(
        render_state: &eframe::egui_wgpu::RenderState,
        width: u32,
        height: u32,
        scene: &Scene,
        camera: Camera,
    ) -> Result<Self, String> {
        if width == 0 || height == 0 {
            return Err("Image dimensions must be non-zero".to_owned());
        }

        let device = render_state.device.clone();
        let queue = render_state.queue.clone();

        let gpu_primitives: Vec<GpuPrimitive> = scene
            .objects
            .iter()
            .copied()
            .map(primitive_to_gpu)
            .collect();

        let primitive_count = u32::try_from(gpu_primitives.len())
            .map_err(|_| "Scene has too many primitives for GPU uniforms".to_owned())?;
        if primitive_count == 0 {
            return Err("Scene has no primitives to render".to_owned());
        }

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pathtracer-uniforms"),
            size: std::mem::size_of::<GpuUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let primitive_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pathtracer-primitives"),
            size: (gpu_primitives.len() * std::mem::size_of::<GpuPrimitive>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&primitive_buffer, 0, bytemuck::cast_slice(&gpu_primitives));

        let pixel_count = width as u64 * height as u64;
        let accumulation_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pathtracer-accumulation"),
            size: pixel_count * std::mem::size_of::<[f32; 4]>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pathtracer-output"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let texture_id = {
            let mut renderer = render_state.renderer.write();
            renderer.register_native_texture(&device, &output_view, wgpu::FilterMode::Linear)
        };

        let padded_bytes_per_row = padded_bytes_per_row(width);
        let readback_size = padded_bytes_per_row as u64 * height as u64;
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pathtracer-readback"),
            size: readback_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pathtracer-bind-group-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pathtracer-bind-group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: primitive_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: accumulation_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pathtracer-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("pathtracer.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pathtracer-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pathtracer-compute"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            width,
            height,
            primitive_count,
            camera,
            uniform_buffer,
            _primitive_buffer: primitive_buffer,
            accumulation_buffer,
            output_texture,
            _output_view: output_view,
            texture_id,
            readback_buffer,
            padded_bytes_per_row,
            bind_group,
            compute_pipeline,
            sample_index: 0,
            preview_frame_index: 0,
            seed: 0x1234_5678,
            render_mode: RenderMode::Path,
            accumulation_dirty: true,
        })
    }

    fn render_sample(&mut self) {
        if self.render_mode == RenderMode::Path && self.accumulation_dirty {
            self.clear_accumulation();
            self.sample_index = 0;
        }

        let frame_index = match self.render_mode {
            RenderMode::Path => self.sample_index,
            RenderMode::Fast => self.preview_frame_index,
        };

        let uniforms = GpuUniforms::new(
            self.width,
            self.height,
            self.primitive_count,
            frame_index,
            self.seed,
            self.render_mode,
            self.render_mode.bounces(),
            self.camera,
        );
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pathtracer-encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pathtracer-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.compute_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
        }

        self.queue.submit(Some(encoder.finish()));

        match self.render_mode {
            RenderMode::Path => {
                self.sample_index = self.sample_index.saturating_add(1);
            }
            RenderMode::Fast => {
                self.sample_index = 0;
                self.preview_frame_index = self.preview_frame_index.saturating_add(1);
            }
        }

        self.seed = self
            .seed
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
    }

    fn set_camera(&mut self, camera: Camera) {
        self.camera = camera;
        self.sample_index = 0;
        self.accumulation_dirty = true;
    }

    fn set_render_mode(&mut self, render_mode: RenderMode) {
        if self.render_mode == render_mode {
            return;
        }

        self.render_mode = render_mode;
        if render_mode == RenderMode::Path {
            self.sample_index = 0;
            self.accumulation_dirty = true;
        }
    }

    fn clear_accumulation(&mut self) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pathtracer-clear-accumulation"),
            });
        encoder.clear_buffer(&self.accumulation_buffer, 0, None);
        self.queue.submit(Some(encoder.finish()));
        self.accumulation_dirty = false;
    }

    fn readback_output(&self, output: &mut [u8]) -> Result<(), String> {
        let expected_len = self.width as usize * self.height as usize * 4;
        if output.len() != expected_len {
            return Err(format!(
                "RGBA output length mismatch: expected {expected_len}, got {}",
                output.len()
            ));
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pathtracer-readback-encoder"),
            });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.output_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_bytes_per_row as u32),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(Some(encoder.finish()));

        let slice = self.readback_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        let map_result = receiver
            .recv_timeout(Duration::from_secs(2))
            .map_err(|_| "Timed out waiting for GPU readback".to_owned())?;
        map_result.map_err(|err| format!("GPU readback map failed: {err}"))?;

        let data = slice.get_mapped_range();
        let row_bytes = self.width as usize * 4;
        for y in 0..self.height as usize {
            let src_start = y * self.padded_bytes_per_row;
            let dst_start = y * row_bytes;
            output[dst_start..dst_start + row_bytes]
                .copy_from_slice(&data[src_start..src_start + row_bytes]);
        }
        drop(data);
        self.readback_buffer.unmap();

        Ok(())
    }

    fn texture_id(&self) -> egui::TextureId {
        self.texture_id
    }

    fn sample_count(&self) -> u32 {
        self.sample_index
    }

    fn total_samples(&self) -> u64 {
        self.sample_index as u64 * self.width as u64 * self.height as u64
    }
}

#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
struct SnapshotFrame {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

#[cfg(not(target_arch = "wasm32"))]
fn start_snapshot_worker(
    stop_flag: &Arc<AtomicBool>,
) -> (Option<Sender<SnapshotFrame>>, Option<JoinHandle<()>>) {
    let (snapshot_sender, snapshot_receiver) = bounded::<SnapshotFrame>(1);
    let thread_stop = Arc::clone(stop_flag);
    let output_path = PathBuf::from(SNAPSHOT_FILE);
    let snapshot_thread = std::thread::Builder::new()
        .name("snapshot-writer".to_owned())
        .spawn(move || snapshot_writer_loop(snapshot_receiver, thread_stop, output_path))
        .ok();

    (Some(snapshot_sender), snapshot_thread)
}

#[cfg(target_arch = "wasm32")]
fn start_snapshot_worker(
    _stop_flag: &Arc<AtomicBool>,
) -> (Option<Sender<SnapshotFrame>>, Option<JoinHandle<()>>) {
    (None, None)
}

struct PathTracerApp {
    gpu: Option<GpuPathTracer>,
    gpu_texture_id: Option<egui::TextureId>,
    init_error: Option<String>,
    fps_camera: FpsCamera,
    render_mode: RenderMode,
    rgba_buffer: Vec<u8>,
    stop_flag: Arc<AtomicBool>,
    snapshot_sender: Option<Sender<SnapshotFrame>>,
    snapshot_thread: Option<JoinHandle<()>>,
    last_snapshot: Instant,
    last_camera_motion: Instant,
    last_frame_time: Instant,
    dump_interval: Duration,
    started_at: Instant,
}

impl PathTracerApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let (snapshot_sender, snapshot_thread) = start_snapshot_worker(&stop_flag);

        let camera_origin = Vec3::new(0.5, 0.5, -2.2);
        let camera_target = Vec3::new(0.5, 0.5, 0.5);
        let fps_camera = FpsCamera::from_look_at(
            camera_origin,
            camera_target,
            40.0,
            IMAGE_WIDTH as f32 / IMAGE_HEIGHT as f32,
        );
        let camera = fps_camera.to_camera();
        let scene = build_cornell_box_scene(camera_origin);

        let (gpu, init_error) = match cc.wgpu_render_state.as_ref() {
            Some(render_state) => {
                match GpuPathTracer::new(
                    render_state,
                    IMAGE_WIDTH as u32,
                    IMAGE_HEIGHT as u32,
                    &scene,
                    camera,
                ) {
                    Ok(gpu) => (Some(gpu), None),
                    Err(err) => (
                        None,
                        Some(format!("Failed to initialize GPU renderer: {err}")),
                    ),
                }
            }
            None => (
                None,
                Some(
                    "WGPU render state is unavailable. Start eframe with Renderer::Wgpu."
                        .to_owned(),
                ),
            ),
        };

        let gpu_texture_id = gpu.as_ref().map(GpuPathTracer::texture_id);
        let now = Instant::now();
        let initial_camera_motion = now.checked_sub(CAMERA_IDLE_TO_PATH).unwrap_or(now);

        Self {
            gpu,
            gpu_texture_id,
            init_error,
            fps_camera,
            render_mode: RenderMode::Path,
            rgba_buffer: vec![0_u8; IMAGE_WIDTH * IMAGE_HEIGHT * 4],
            stop_flag,
            snapshot_sender,
            snapshot_thread,
            last_snapshot: now,
            last_camera_motion: initial_camera_motion,
            last_frame_time: now,
            dump_interval: SNAPSHOT_INTERVAL,
            started_at: now,
        }
    }

    fn update_camera_and_mode(&mut self, ctx: &egui::Context) {
        let now = Instant::now();
        let dt_seconds = now
            .saturating_duration_since(self.last_frame_time)
            .as_secs_f32()
            .clamp(0.0, 0.1);
        self.last_frame_time = now;

        let camera_changed = self.fps_camera.update_from_input(ctx, dt_seconds);
        if camera_changed {
            self.last_camera_motion = now;
            let camera = self.fps_camera.to_camera();
            if let Some(gpu) = &mut self.gpu {
                gpu.set_camera(camera);
            }
        }

        let target_mode =
            if now.saturating_duration_since(self.last_camera_motion) >= CAMERA_IDLE_TO_PATH {
                RenderMode::Path
            } else {
                RenderMode::Fast
            };

        if target_mode != self.render_mode {
            self.render_mode = target_mode;
            if let Some(gpu) = &mut self.gpu {
                gpu.set_render_mode(target_mode);
            }
        }
    }

    fn maybe_queue_snapshot(&mut self) {
        if self.snapshot_sender.is_none() {
            return;
        }

        if self.last_snapshot.elapsed() < self.dump_interval {
            return;
        }

        if let Some(gpu) = self.gpu.as_ref() {
            if let Err(err) = gpu.readback_output(&mut self.rgba_buffer) {
                self.init_error = Some(format!("Snapshot readback failed: {err}"));
                self.last_snapshot = Instant::now();
                return;
            }
        } else {
            self.last_snapshot = Instant::now();
            return;
        }

        if let Some(sender) = &self.snapshot_sender {
            let frame = SnapshotFrame {
                width: IMAGE_WIDTH as u32,
                height: IMAGE_HEIGHT as u32,
                pixels: self.rgba_buffer.clone(),
            };
            let _ = sender.try_send(frame);
        }

        self.last_snapshot = Instant::now();
    }
}

impl eframe::App for PathTracerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.update_camera_and_mode(ctx);

        if let Some(gpu) = &mut self.gpu {
            gpu.render_sample();
            self.maybe_queue_snapshot();
        }

        let elapsed = self.started_at.elapsed().as_secs_f32();
        let sample_count = self.gpu.as_ref().map_or(0, GpuPathTracer::sample_count);
        let total_samples = self.gpu.as_ref().map_or(0, GpuPathTracer::total_samples);
        let sample_label = if self.render_mode == RenderMode::Path {
            format!("Samples per pixel: {sample_count}")
        } else {
            "Samples per pixel: realtime".to_owned()
        };
        let total_label = if self.render_mode == RenderMode::Path {
            format!("Total samples: {total_samples}")
        } else {
            "Total samples: paused".to_owned()
        };
        let snapshot_label = if self.snapshot_sender.is_some() {
            format!("Snapshot: {SNAPSHOT_FILE}")
        } else {
            "Snapshot: disabled".to_owned()
        };

        egui::TopBottomPanel::top("status").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(format!("Resolution: {}x{}", IMAGE_WIDTH, IMAGE_HEIGHT));
                ui.separator();
                ui.label("Backend: WGPU/WGSL");
                ui.separator();
                ui.label(format!("Mode: {}", self.render_mode.label()));
                ui.separator();
                ui.label(sample_label.as_str());
                ui.separator();
                ui.label(total_label.as_str());
                ui.separator();
                ui.label(format!("Elapsed: {:.1}s", elapsed));
                ui.separator();
                ui.label(snapshot_label.as_str());
            });
            ui.label("Controls: WASD move, hold right mouse + drag to look");

            if let Some(err) = &self.init_error {
                ui.colored_label(egui::Color32::from_rgb(220, 80, 80), err);
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(texture_id) = self.gpu_texture_id {
                let image_size = egui::vec2(IMAGE_WIDTH as f32, IMAGE_HEIGHT as f32);
                let avail = ui.available_size();
                let scale = (avail.x / image_size.x)
                    .min(avail.y / image_size.y)
                    .max(0.01);
                let desired = image_size * scale;

                ui.centered_and_justified(|ui| {
                    ui.add(egui::Image::new((texture_id, image_size)).fit_to_exact_size(desired));
                });
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label("Waiting for GPU output...");
                });
            }
        });

        ctx.request_repaint_after(Duration::from_millis(16));
    }
}

impl Drop for PathTracerApp {
    fn drop(&mut self) {
        self.stop_flag.store(true, Ordering::SeqCst);
        self.snapshot_sender.take();

        if let Some(handle) = self.snapshot_thread.take() {
            let _ = handle.join();
        }
    }
}

fn material_to_gpu(material: Material) -> (u32, [f32; 4], [f32; 4]) {
    match material {
        Material::Diffuse { albedo } => (
            MATERIAL_DIFFUSE,
            [albedo.x, albedo.y, albedo.z, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ),
        Material::Metal { albedo, fuzz } => (
            MATERIAL_METAL,
            [albedo.x, albedo.y, albedo.z, 0.0],
            [fuzz, 0.0, 0.0, 0.0],
        ),
        Material::Glossy {
            albedo,
            roughness,
            reflectivity,
        } => (
            MATERIAL_GLOSSY,
            [albedo.x, albedo.y, albedo.z, 0.0],
            [roughness, reflectivity, 0.0, 0.0],
        ),
        Material::Emissive { color } => (
            MATERIAL_EMISSIVE,
            [color.x, color.y, color.z, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ),
    }
}

fn primitive_to_gpu(primitive: Primitive) -> GpuPrimitive {
    match primitive {
        Primitive::XY(rect) => {
            let (material_kind, color, params) = material_to_gpu(rect.material);
            GpuPrimitive {
                header: [PRIMITIVE_XY, material_kind, 0, 0],
                p0: [rect.x0, rect.x1, rect.y0, rect.y1],
                p1: [rect.z, rect.normal_z, 0.0, 0.0],
                color,
                params,
            }
        }
        Primitive::XZ(rect) => {
            let (material_kind, color, params) = material_to_gpu(rect.material);
            GpuPrimitive {
                header: [PRIMITIVE_XZ, material_kind, 0, 0],
                p0: [rect.x0, rect.x1, rect.z0, rect.z1],
                p1: [rect.y, rect.normal_y, 0.0, 0.0],
                color,
                params,
            }
        }
        Primitive::YZ(rect) => {
            let (material_kind, color, params) = material_to_gpu(rect.material);
            GpuPrimitive {
                header: [PRIMITIVE_YZ, material_kind, 0, 0],
                p0: [rect.y0, rect.y1, rect.z0, rect.z1],
                p1: [rect.x, rect.normal_x, 0.0, 0.0],
                color,
                params,
            }
        }
        Primitive::Sphere(sphere) => {
            let (material_kind, color, params) = material_to_gpu(sphere.material);
            GpuPrimitive {
                header: [PRIMITIVE_SPHERE, material_kind, 0, 0],
                p0: [
                    sphere.center.x,
                    sphere.center.y,
                    sphere.center.z,
                    sphere.radius,
                ],
                p1: [0.0, 0.0, 0.0, 0.0],
                color,
                params,
            }
        }
    }
}

fn padded_bytes_per_row(width: u32) -> usize {
    let row_bytes = width as usize * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
    (row_bytes + align - 1) / align * align
}

fn build_cornell_box_scene(camera_origin: Vec3) -> Scene {
    let mut objects = Vec::new();

    let white = Material::Diffuse {
        albedo: Vec3::new(0.42, 0.42, 0.42),
    };
    let green = Material::Diffuse {
        albedo: Vec3::new(0.09, 0.28, 0.14),
    };
    let metallic_wall = Material::Metal {
        albedo: Vec3::new(0.58, 0.6, 0.64),
        fuzz: 0.05,
    };
    let glossy_box_a = Material::Glossy {
        albedo: Vec3::new(0.86, 0.76, 0.68),
        roughness: 0.18,
        reflectivity: 0.22,
    };
    let glossy_box_b = Material::Glossy {
        albedo: Vec3::new(0.68, 0.78, 0.87),
        roughness: 0.22,
        reflectivity: 0.2,
    };
    let fill_light = Material::Emissive {
        color: Vec3::new(1.8, 1.75, 1.7),
    };
    let disco_magenta = Material::Emissive {
        color: Vec3::new(5.8, 1.0, 4.7),
    };
    let disco_cyan = Material::Emissive {
        color: Vec3::new(1.0, 4.9, 5.6),
    };
    let disco_lime = Material::Emissive {
        color: Vec3::new(3.1, 5.0, 1.1),
    };
    let disco_amber = Material::Emissive {
        color: Vec3::new(5.4, 3.3, 0.9),
    };
    let disco_ball = Material::Metal {
        albedo: Vec3::new(0.94, 0.95, 0.97),
        fuzz: 0.01,
    };
    let psyduck_yellow = Material::Glossy {
        albedo: Vec3::new(0.94, 0.84, 0.19),
        roughness: 0.2,
        reflectivity: 0.12,
    };
    let psyduck_beak = Material::Glossy {
        albedo: Vec3::new(0.94, 0.79, 0.49),
        roughness: 0.25,
        reflectivity: 0.08,
    };
    let psyduck_eye_white = Material::Diffuse {
        albedo: Vec3::new(0.97, 0.97, 0.97),
    };
    let pokemon_detail = Material::Diffuse {
        albedo: Vec3::new(0.05, 0.04, 0.04),
    };
    let pikachu_yellow = Material::Glossy {
        albedo: Vec3::new(0.97, 0.86, 0.13),
        roughness: 0.19,
        reflectivity: 0.11,
    };
    let pikachu_cheek = Material::Glossy {
        albedo: Vec3::new(0.9, 0.22, 0.2),
        roughness: 0.25,
        reflectivity: 0.08,
    };
    let pikachu_tail = Material::Diffuse {
        albedo: Vec3::new(0.52, 0.3, 0.15),
    };
    let slowpoke_pink = Material::Glossy {
        albedo: Vec3::new(0.9, 0.63, 0.72),
        roughness: 0.3,
        reflectivity: 0.08,
    };
    let slowpoke_muzzle = Material::Diffuse {
        albedo: Vec3::new(0.97, 0.89, 0.86),
    };
    let slowpoke_eye_white = Material::Diffuse {
        albedo: Vec3::new(0.97, 0.97, 0.97),
    };
    let slowpoke_detail = Material::Diffuse {
        albedo: Vec3::new(0.05, 0.04, 0.04),
    };
    let slowpoke_tail_tip = Material::Glossy {
        albedo: Vec3::new(0.94, 0.93, 0.93),
        roughness: 0.22,
        reflectivity: 0.06,
    };

    objects.push(Primitive::YZ(RectYZ {
        y0: 0.0,
        y1: 1.0,
        z0: 0.0,
        z1: 1.0,
        x: 0.0,
        normal_x: 1.0,
        material: metallic_wall,
    }));

    objects.push(Primitive::YZ(RectYZ {
        y0: 0.0,
        y1: 1.0,
        z0: 0.0,
        z1: 1.0,
        x: 1.0,
        normal_x: -1.0,
        material: green,
    }));

    objects.push(Primitive::XZ(RectXZ {
        x0: 0.0,
        x1: 1.0,
        z0: 0.0,
        z1: 1.0,
        y: 0.0,
        normal_y: 1.0,
        material: white,
    }));

    objects.push(Primitive::XZ(RectXZ {
        x0: 0.0,
        x1: 1.0,
        z0: 0.0,
        z1: 1.0,
        y: 1.0,
        normal_y: -1.0,
        material: white,
    }));

    objects.push(Primitive::XY(RectXY {
        x0: 0.0,
        x1: 1.0,
        y0: 0.0,
        y1: 1.0,
        z: 1.0,
        normal_z: -1.0,
        material: white,
    }));

    objects.push(Primitive::XZ(RectXZ {
        x0: 0.36,
        x1: 0.64,
        z0: 0.32,
        z1: 0.68,
        y: 0.999,
        normal_y: -1.0,
        material: fill_light,
    }));

    objects.push(Primitive::XZ(RectXZ {
        x0: 0.09,
        x1: 0.23,
        z0: 0.12,
        z1: 0.28,
        y: 0.998,
        normal_y: -1.0,
        material: disco_magenta,
    }));
    objects.push(Primitive::XZ(RectXZ {
        x0: 0.77,
        x1: 0.91,
        z0: 0.12,
        z1: 0.28,
        y: 0.998,
        normal_y: -1.0,
        material: disco_cyan,
    }));
    objects.push(Primitive::XZ(RectXZ {
        x0: 0.77,
        x1: 0.91,
        z0: 0.72,
        z1: 0.88,
        y: 0.998,
        normal_y: -1.0,
        material: disco_lime,
    }));
    objects.push(Primitive::XZ(RectXZ {
        x0: 0.09,
        x1: 0.23,
        z0: 0.72,
        z1: 0.88,
        y: 0.998,
        normal_y: -1.0,
        material: disco_amber,
    }));

    add_sphere(&mut objects, Vec3::new(0.5, 0.82, 0.52), 0.072, disco_ball);
    add_sphere(
        &mut objects,
        Vec3::new(0.31, 0.92, 0.34),
        0.029,
        disco_magenta,
    );
    add_sphere(&mut objects, Vec3::new(0.69, 0.92, 0.32), 0.029, disco_cyan);
    add_sphere(&mut objects, Vec3::new(0.72, 0.91, 0.70), 0.029, disco_lime);
    add_sphere(
        &mut objects,
        Vec3::new(0.28, 0.91, 0.72),
        0.029,
        disco_amber,
    );

    add_axis_aligned_box(
        &mut objects,
        Vec3::new(0.16, 0.0, 0.56),
        Vec3::new(0.42, 0.43, 0.84),
        glossy_box_a,
    );
    add_axis_aligned_box(
        &mut objects,
        Vec3::new(0.58, 0.0, 0.24),
        Vec3::new(0.86, 0.69, 0.58),
        glossy_box_b,
    );

    add_psyduck(
        &mut objects,
        Vec3::new(0.72, 0.69, 0.41),
        camera_origin,
        psyduck_yellow,
        psyduck_beak,
        psyduck_eye_white,
        pokemon_detail,
    );

    add_pikachu(
        &mut objects,
        Vec3::new(0.30, 0.43, 0.72),
        camera_origin,
        pikachu_yellow,
        pokemon_detail,
        pikachu_cheek,
        pikachu_tail,
    );

    add_slowpoke(
        &mut objects,
        Vec3::new(0.30, 0.0, 0.20),
        camera_origin,
        slowpoke_pink,
        slowpoke_muzzle,
        slowpoke_eye_white,
        slowpoke_detail,
        slowpoke_tail_tip,
    );

    Scene { objects }
}

fn add_axis_aligned_box(objects: &mut Vec<Primitive>, min: Vec3, max: Vec3, material: Material) {
    objects.push(Primitive::XY(RectXY {
        x0: min.x,
        x1: max.x,
        y0: min.y,
        y1: max.y,
        z: max.z,
        normal_z: 1.0,
        material,
    }));
    objects.push(Primitive::XY(RectXY {
        x0: min.x,
        x1: max.x,
        y0: min.y,
        y1: max.y,
        z: min.z,
        normal_z: -1.0,
        material,
    }));
    objects.push(Primitive::XZ(RectXZ {
        x0: min.x,
        x1: max.x,
        z0: min.z,
        z1: max.z,
        y: max.y,
        normal_y: 1.0,
        material,
    }));
    objects.push(Primitive::XZ(RectXZ {
        x0: min.x,
        x1: max.x,
        z0: min.z,
        z1: max.z,
        y: min.y,
        normal_y: -1.0,
        material,
    }));
    objects.push(Primitive::YZ(RectYZ {
        y0: min.y,
        y1: max.y,
        z0: min.z,
        z1: max.z,
        x: max.x,
        normal_x: 1.0,
        material,
    }));
    objects.push(Primitive::YZ(RectYZ {
        y0: min.y,
        y1: max.y,
        z0: min.z,
        z1: max.z,
        x: min.x,
        normal_x: -1.0,
        material,
    }));
}

fn add_sphere(objects: &mut Vec<Primitive>, center: Vec3, radius: f32, material: Material) {
    objects.push(Primitive::Sphere(Sphere {
        center,
        radius,
        material,
    }));
}

fn add_psyduck(
    objects: &mut Vec<Primitive>,
    perch: Vec3,
    camera_origin: Vec3,
    yellow: Material,
    beak: Material,
    eye_white: Material,
    detail: Material,
) {
    let body_radius = 0.06;
    let body_center = Vec3::new(perch.x, perch.y + body_radius + 0.0015, perch.z);

    let to_camera = (camera_origin - body_center).normalized();
    let forward = Vec3::new(to_camera.x, to_camera.y * 0.25, to_camera.z).normalized();

    let world_up = Vec3::new(0.0, 1.0, 0.0);
    let mut right = forward.cross(world_up);
    if right.length_squared() < 1e-6 {
        right = Vec3::new(1.0, 0.0, 0.0);
    }
    right = right.normalized();
    let up = right.cross(forward).normalized();

    add_sphere(objects, body_center, body_radius, yellow);
    add_sphere(
        objects,
        body_center - forward * 0.041 + up * 0.008,
        0.046,
        yellow,
    );

    let head_center = body_center + forward * 0.03 + up * 0.073;
    add_sphere(objects, head_center, 0.046, yellow);
    add_sphere(
        objects,
        head_center + forward * 0.018 + up * 0.006,
        0.034,
        yellow,
    );

    let beak_center = head_center + forward * 0.048 - up * 0.012;
    add_sphere(objects, beak_center, 0.023, beak);
    add_sphere(
        objects,
        beak_center + forward * 0.013 - up * 0.006,
        0.015,
        beak,
    );

    add_sphere(
        objects,
        body_center + forward * 0.026 - right * 0.055 + up * 0.006,
        0.018,
        yellow,
    );
    add_sphere(
        objects,
        body_center + forward * 0.026 + right * 0.055 + up * 0.006,
        0.018,
        yellow,
    );

    let foot_y = perch.y + 0.017;
    let foot_drop = body_center.y - foot_y;
    add_sphere(
        objects,
        body_center + forward * 0.03 - right * 0.03 - world_up * foot_drop,
        0.017,
        yellow,
    );
    add_sphere(
        objects,
        body_center + forward * 0.03 + right * 0.03 - world_up * foot_drop,
        0.017,
        yellow,
    );

    let eye_base = head_center + forward * 0.024 + up * 0.016;
    add_sphere(objects, eye_base - right * 0.016, 0.0085, eye_white);
    add_sphere(objects, eye_base + right * 0.016, 0.0085, eye_white);
    add_sphere(
        objects,
        eye_base - right * 0.016 + forward * 0.004,
        0.0038,
        detail,
    );
    add_sphere(
        objects,
        eye_base + right * 0.016 + forward * 0.004,
        0.0038,
        detail,
    );

    let hair_root = head_center + up * 0.046 - forward * 0.006;
    add_sphere(
        objects,
        hair_root - right * 0.01 + up * 0.018,
        0.006,
        detail,
    );
    add_sphere(objects, hair_root + up * 0.024, 0.0065, detail);
    add_sphere(
        objects,
        hair_root + right * 0.01 + up * 0.018,
        0.006,
        detail,
    );
}

fn add_pikachu(
    objects: &mut Vec<Primitive>,
    perch: Vec3,
    camera_origin: Vec3,
    yellow: Material,
    detail: Material,
    cheek: Material,
    tail: Material,
) {
    let body_radius = 0.066;
    let body_center = Vec3::new(perch.x, perch.y + body_radius + 0.0015, perch.z);

    let to_camera = (camera_origin - body_center).normalized();
    let forward = Vec3::new(to_camera.x, to_camera.y * 0.18, to_camera.z).normalized();

    let world_up = Vec3::new(0.0, 1.0, 0.0);
    let mut right = forward.cross(world_up);
    if right.length_squared() < 1e-6 {
        right = Vec3::new(1.0, 0.0, 0.0);
    }
    right = right.normalized();
    let up = right.cross(forward).normalized();

    add_sphere(objects, body_center, body_radius, yellow);
    add_sphere(
        objects,
        body_center + forward * 0.022 + up * 0.01,
        0.055,
        yellow,
    );

    let head_center = body_center + forward * 0.045 + up * 0.075;
    add_sphere(objects, head_center, 0.054, yellow);
    add_sphere(
        objects,
        head_center + forward * 0.02 + up * 0.008,
        0.042,
        yellow,
    );

    let ear_base = head_center + up * 0.046;
    let left_ear = ear_base - right * 0.032;
    add_sphere(
        objects,
        left_ear + up * 0.024 - forward * 0.002,
        0.018,
        yellow,
    );
    add_sphere(
        objects,
        left_ear + up * 0.049 - forward * 0.004,
        0.015,
        yellow,
    );
    add_sphere(
        objects,
        left_ear + up * 0.071 - forward * 0.006,
        0.01,
        detail,
    );

    let right_ear = ear_base + right * 0.032;
    add_sphere(
        objects,
        right_ear + up * 0.024 - forward * 0.002,
        0.018,
        yellow,
    );
    add_sphere(
        objects,
        right_ear + up * 0.049 - forward * 0.004,
        0.015,
        yellow,
    );
    add_sphere(
        objects,
        right_ear + up * 0.071 - forward * 0.006,
        0.01,
        detail,
    );

    add_sphere(
        objects,
        body_center + forward * 0.032 - right * 0.056 + up * 0.008,
        0.019,
        yellow,
    );
    add_sphere(
        objects,
        body_center + forward * 0.032 + right * 0.056 + up * 0.008,
        0.019,
        yellow,
    );

    let foot_y = perch.y + 0.018;
    let foot_drop = body_center.y - foot_y;
    add_sphere(
        objects,
        body_center + forward * 0.038 - right * 0.034 - world_up * foot_drop,
        0.02,
        yellow,
    );
    add_sphere(
        objects,
        body_center + forward * 0.038 + right * 0.034 - world_up * foot_drop,
        0.02,
        yellow,
    );

    let cheek_base = head_center + forward * 0.04 - up * 0.004;
    add_sphere(objects, cheek_base - right * 0.026, 0.011, cheek);
    add_sphere(objects, cheek_base + right * 0.026, 0.011, cheek);

    let eye_base = head_center + forward * 0.038 + up * 0.013;
    add_sphere(objects, eye_base - right * 0.017, 0.0072, detail);
    add_sphere(objects, eye_base + right * 0.017, 0.0072, detail);
    add_sphere(
        objects,
        head_center + forward * 0.051 - up * 0.005,
        0.0042,
        detail,
    );

    let tail_base = body_center - forward * 0.055 + up * 0.025 + right * 0.01;
    add_sphere(objects, tail_base, 0.016, tail);
    add_sphere(
        objects,
        tail_base - forward * 0.038 + up * 0.028 + right * 0.016,
        0.014,
        tail,
    );
    add_sphere(
        objects,
        tail_base - forward * 0.064 + up * 0.055 + right * 0.03,
        0.012,
        tail,
    );
    add_sphere(
        objects,
        tail_base - forward * 0.08 + up * 0.079 + right * 0.044,
        0.01,
        yellow,
    );

    add_sphere(
        objects,
        body_center - forward * 0.024 - right * 0.021 + up * 0.018,
        0.01,
        tail,
    );
    add_sphere(
        objects,
        body_center - forward * 0.024 + right * 0.021 + up * 0.018,
        0.01,
        tail,
    );
}

#[allow(clippy::too_many_arguments)]
fn add_slowpoke(
    objects: &mut Vec<Primitive>,
    floor_anchor: Vec3,
    camera_origin: Vec3,
    pink: Material,
    muzzle: Material,
    eye_white: Material,
    detail: Material,
    tail_tip: Material,
) {
    let world_up = Vec3::new(0.0, 1.0, 0.0);
    let scale = 1.65;
    let body_radius = 0.105 * scale;
    let body_center = Vec3::new(
        floor_anchor.x,
        floor_anchor.y + body_radius + 0.0015,
        floor_anchor.z,
    );

    let to_camera = (camera_origin - body_center).normalized();
    let face_dir = Vec3::new(to_camera.x, to_camera.y * 0.2, to_camera.z).normalized();

    let mut side_hint = world_up.cross(face_dir);
    if side_hint.length_squared() < 1e-6 {
        side_hint = Vec3::new(1.0, 0.0, 0.0);
    }
    side_hint = side_hint.normalized();

    let forward = (face_dir * 0.9 + side_hint * 0.435).normalized();

    let mut right = forward.cross(world_up);
    if right.length_squared() < 1e-6 {
        right = Vec3::new(1.0, 0.0, 0.0);
    }
    right = right.normalized();
    let up = right.cross(forward).normalized();

    add_sphere(objects, body_center, body_radius, pink);
    add_sphere(
        objects,
        body_center - forward * (0.012 * scale) + up * (0.006 * scale),
        0.092 * scale,
        pink,
    );

    let leg_radius = 0.033 * scale;
    let leg_y = floor_anchor.y + leg_radius + 0.001;
    let leg_drop = body_center.y - leg_y;

    add_sphere(
        objects,
        body_center + forward * (0.046 * scale) - right * (0.056 * scale) - world_up * leg_drop,
        leg_radius,
        pink,
    );
    add_sphere(
        objects,
        body_center + forward * (0.046 * scale) + right * (0.056 * scale) - world_up * leg_drop,
        leg_radius,
        pink,
    );
    add_sphere(
        objects,
        body_center - forward * (0.039 * scale) - right * (0.058 * scale) - world_up * leg_drop,
        leg_radius,
        pink,
    );
    add_sphere(
        objects,
        body_center - forward * (0.039 * scale) + right * (0.058 * scale) - world_up * leg_drop,
        leg_radius,
        pink,
    );

    let head_center = body_center + forward * (0.095 * scale) + up * (0.01 * scale);
    add_sphere(objects, head_center, 0.074 * scale, pink);

    let snout_center = head_center + forward * (0.067 * scale) - up * (0.008 * scale);
    add_sphere(objects, snout_center, 0.042 * scale, muzzle);
    add_sphere(
        objects,
        snout_center + forward * (0.03 * scale) - up * (0.002 * scale),
        0.009 * scale,
        detail,
    );

    let eye_base = head_center + forward * (0.052 * scale) + up * (0.024 * scale);
    let eye_offset = right * (0.026 * scale);
    let pupil_shift = forward * (0.008 * scale) - up * (0.001 * scale);
    add_sphere(objects, eye_base - eye_offset, 0.012 * scale, eye_white);
    add_sphere(objects, eye_base + eye_offset, 0.012 * scale, eye_white);
    add_sphere(
        objects,
        eye_base - eye_offset + pupil_shift,
        0.0055 * scale,
        detail,
    );
    add_sphere(
        objects,
        eye_base + eye_offset + pupil_shift,
        0.0055 * scale,
        detail,
    );

    let ear_base = head_center + up * (0.048 * scale);
    add_sphere(
        objects,
        ear_base - right * (0.03 * scale) + forward * (0.004 * scale),
        0.019 * scale,
        pink,
    );
    add_sphere(
        objects,
        ear_base + right * (0.03 * scale) + forward * (0.004 * scale),
        0.019 * scale,
        pink,
    );

    let tail_base =
        body_center - forward * (0.102 * scale) + up * (0.028 * scale) + right * (0.01 * scale);
    add_sphere(objects, tail_base, 0.038 * scale, pink);
    add_sphere(
        objects,
        tail_base - forward * (0.038 * scale) + up * (0.034 * scale) + right * (0.012 * scale),
        0.032 * scale,
        pink,
    );
    add_sphere(
        objects,
        tail_base - forward * (0.066 * scale) + up * (0.067 * scale) + right * (0.024 * scale),
        0.026 * scale,
        pink,
    );
    add_sphere(
        objects,
        tail_base - forward * (0.084 * scale) + up * (0.099 * scale) + right * (0.036 * scale),
        0.021 * scale,
        tail_tip,
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn snapshot_writer_loop(
    receiver: Receiver<SnapshotFrame>,
    stop_flag: Arc<AtomicBool>,
    output_path: PathBuf,
) {
    while !stop_flag.load(Ordering::Relaxed) || !receiver.is_empty() {
        match receiver.recv_timeout(Duration::from_millis(250)) {
            Ok(mut frame) => {
                while let Ok(newer) = receiver.try_recv() {
                    frame = newer;
                }

                if let Some(image) = RgbaImage::from_raw(frame.width, frame.height, frame.pixels) {
                    let _ = image.save(&output_path);
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default()
            .with_title("Cornell Box Path Tracer")
            .with_inner_size([980.0, 860.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Cornell Box Path Tracer",
        options,
        Box::new(|cc| Ok(Box::new(PathTracerApp::new(cc)))),
    )
}

#[cfg(target_arch = "wasm32")]
fn set_loading_text(text: &str) {
    if let Some(document) = web_sys::window().and_then(|window| window.document()) {
        if let Some(loading) = document.get_element_by_id("loading") {
            loading.set_text_content(Some(text));
            let _ = loading.set_attribute("style", "display:flex");
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn hide_loading_overlay() {
    if let Some(document) = web_sys::window().and_then(|window| window.document()) {
        if let Some(loading) = document.get_element_by_id("loading") {
            let _ = loading.set_attribute("style", "display:none");
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn js_value_to_string(value: wasm_bindgen::JsValue) -> String {
    value.as_string().unwrap_or_else(|| format!("{value:?}"))
}

#[cfg(target_arch = "wasm32")]
fn format_web_start_error(raw: String) -> String {
    if raw.contains("no suitable adapter found") {
        return "WebGPU adapter was not found. Enable hardware acceleration and WebGPU in Chrome, then reload."
            .to_owned();
    }

    format!("Failed to start web renderer: {raw}")
}

#[cfg(target_arch = "wasm32")]
fn main() {
    console_error_panic_hook::set_once();

    let runner = WEB_RUNNER.with(|runner| {
        let mut runner = runner.borrow_mut();
        if runner.is_none() {
            *runner = Some(eframe::WebRunner::new());
        }
        runner
            .as_ref()
            .expect("web runner must exist after initialization")
            .clone()
    });

    spawn_local(async move {
        let start_result: Result<(), String> = async {
            let window = web_sys::window().ok_or_else(|| "window is unavailable".to_owned())?;
            let document = window
                .document()
                .ok_or_else(|| "document is unavailable".to_owned())?;
            let canvas = document
                .get_element_by_id("the_canvas_id")
                .ok_or_else(|| "missing canvas with id 'the_canvas_id'".to_owned())?
                .dyn_into::<web_sys::HtmlCanvasElement>()
                .map_err(|_| "element with id 'the_canvas_id' is not a canvas".to_owned())?;

            let web_options = eframe::WebOptions::default();

            hide_loading_overlay();

            runner
                .start(
                    canvas,
                    web_options,
                    Box::new(|cc| Ok(Box::new(PathTracerApp::new(cc)))),
                )
                .await
                .map_err(js_value_to_string)
        }
        .await;

        match start_result {
            Ok(()) => hide_loading_overlay(),
            Err(err) => set_loading_text(format_web_start_error(err).as_str()),
        }
    });
}
