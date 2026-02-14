use crossbeam_channel::{bounded, Receiver, RecvTimeoutError, Sender};
use eframe::egui;
use image::RgbaImage;
use parking_lot::Mutex;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, Neg, Sub};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const IMAGE_WIDTH: usize = 640;
const IMAGE_HEIGHT: usize = 640;
const MAX_BOUNCES: usize = 10;
const SNAPSHOT_INTERVAL: Duration = Duration::from_secs(3);
const SNAPSHOT_FILE: &str = "snapshot.png";

#[derive(Clone, Copy, Debug, Default)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    const ZERO: Self = Self::splat(0.0);

    const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    const fn splat(v: f32) -> Self {
        Self { x: v, y: v, z: v }
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
            Self::ZERO
        }
    }

    fn mul_elem(self, rhs: Self) -> Self {
        Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }

    fn max_component(self) -> f32 {
        self.x.max(self.y).max(self.z)
    }
}

impl Add for Vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
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

impl DivAssign<f32> for Vec3 {
    fn div_assign(&mut self, rhs: f32) {
        self.x /= rhs;
        self.y /= rhs;
        self.z /= rhs;
    }
}

impl Neg for Vec3 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y, -self.z)
    }
}

#[derive(Clone, Copy)]
struct Ray {
    origin: Vec3,
    direction: Vec3,
}

impl Ray {
    fn new(origin: Vec3, direction: Vec3) -> Self {
        Self {
            origin,
            direction: direction.normalized(),
        }
    }

    fn at(self, t: f32) -> Vec3 {
        self.origin + self.direction * t
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

impl Material {
    fn emitted(self) -> Vec3 {
        match self {
            Self::Emissive { color } => color,
            _ => Vec3::ZERO,
        }
    }

    fn scatter(self, ray_in: &Ray, hit: &HitRecord, rng: &mut SmallRng) -> Option<(Ray, Vec3)> {
        match self {
            Self::Diffuse { albedo } => {
                let direction = cosine_hemisphere(hit.normal, rng);
                Some((Ray::new(hit.point + hit.normal * 1e-4, direction), albedo))
            }
            Self::Metal { albedo, fuzz } => {
                let reflected = reflect(ray_in.direction, hit.normal);
                let direction = (reflected + random_in_unit_sphere(rng) * fuzz).normalized();
                if direction.dot(hit.normal) > 0.0 {
                    Some((Ray::new(hit.point + hit.normal * 1e-4, direction), albedo))
                } else {
                    None
                }
            }
            Self::Glossy {
                albedo,
                roughness,
                reflectivity,
            } => {
                if rng.gen::<f32>() < reflectivity {
                    let reflected = reflect(ray_in.direction, hit.normal);
                    let direction =
                        (reflected + random_in_unit_sphere(rng) * roughness).normalized();
                    if direction.dot(hit.normal) > 0.0 {
                        Some((Ray::new(hit.point + hit.normal * 1e-4, direction), albedo))
                    } else {
                        let diffuse = cosine_hemisphere(hit.normal, rng);
                        Some((Ray::new(hit.point + hit.normal * 1e-4, diffuse), albedo))
                    }
                } else {
                    let diffuse = cosine_hemisphere(hit.normal, rng);
                    Some((Ray::new(hit.point + hit.normal * 1e-4, diffuse), albedo))
                }
            }
            Self::Emissive { .. } => None,
        }
    }
}

#[derive(Clone, Copy)]
struct HitRecord {
    point: Vec3,
    normal: Vec3,
    t: f32,
    material: Material,
}

impl HitRecord {
    fn new(ray: &Ray, t: f32, point: Vec3, outward_normal: Vec3, material: Material) -> Self {
        let front_face = ray.direction.dot(outward_normal) < 0.0;
        let normal = if front_face {
            outward_normal
        } else {
            -outward_normal
        };

        Self {
            point,
            normal,
            t,
            material,
        }
    }
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

impl Primitive {
    fn hit(self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        match self {
            Self::XY(rect) => {
                if ray.direction.z.abs() < 1e-6 {
                    return None;
                }
                let t = (rect.z - ray.origin.z) / ray.direction.z;
                if t <= t_min || t >= t_max {
                    return None;
                }
                let x = ray.origin.x + t * ray.direction.x;
                let y = ray.origin.y + t * ray.direction.y;
                if x < rect.x0 || x > rect.x1 || y < rect.y0 || y > rect.y1 {
                    return None;
                }
                let point = ray.at(t);
                Some(HitRecord::new(
                    ray,
                    t,
                    point,
                    Vec3::new(0.0, 0.0, rect.normal_z),
                    rect.material,
                ))
            }
            Self::XZ(rect) => {
                if ray.direction.y.abs() < 1e-6 {
                    return None;
                }
                let t = (rect.y - ray.origin.y) / ray.direction.y;
                if t <= t_min || t >= t_max {
                    return None;
                }
                let x = ray.origin.x + t * ray.direction.x;
                let z = ray.origin.z + t * ray.direction.z;
                if x < rect.x0 || x > rect.x1 || z < rect.z0 || z > rect.z1 {
                    return None;
                }
                let point = ray.at(t);
                Some(HitRecord::new(
                    ray,
                    t,
                    point,
                    Vec3::new(0.0, rect.normal_y, 0.0),
                    rect.material,
                ))
            }
            Self::YZ(rect) => {
                if ray.direction.x.abs() < 1e-6 {
                    return None;
                }
                let t = (rect.x - ray.origin.x) / ray.direction.x;
                if t <= t_min || t >= t_max {
                    return None;
                }
                let y = ray.origin.y + t * ray.direction.y;
                let z = ray.origin.z + t * ray.direction.z;
                if y < rect.y0 || y > rect.y1 || z < rect.z0 || z > rect.z1 {
                    return None;
                }
                let point = ray.at(t);
                Some(HitRecord::new(
                    ray,
                    t,
                    point,
                    Vec3::new(rect.normal_x, 0.0, 0.0),
                    rect.material,
                ))
            }
            Self::Sphere(sphere) => {
                let oc = ray.origin - sphere.center;
                let a = ray.direction.length_squared();
                let half_b = oc.dot(ray.direction);
                let c = oc.length_squared() - sphere.radius * sphere.radius;
                let discriminant = half_b * half_b - a * c;
                if discriminant < 0.0 {
                    return None;
                }

                let sqrtd = discriminant.sqrt();
                let mut t = (-half_b - sqrtd) / a;
                if t <= t_min || t >= t_max {
                    t = (-half_b + sqrtd) / a;
                    if t <= t_min || t >= t_max {
                        return None;
                    }
                }

                let point = ray.at(t);
                let outward_normal = (point - sphere.center) / sphere.radius;
                Some(HitRecord::new(
                    ray,
                    t,
                    point,
                    outward_normal,
                    sphere.material,
                ))
            }
        }
    }
}

struct Scene {
    objects: Vec<Primitive>,
}

impl Scene {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let mut closest = t_max;
        let mut hit = None;

        for object in &self.objects {
            if let Some(candidate) = object.hit(ray, t_min, closest) {
                closest = candidate.t;
                hit = Some(candidate);
            }
        }

        hit
    }
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

    fn get_ray(self, s: f32, t: f32) -> Ray {
        let dir = self.lower_left + self.horizontal * s + self.vertical * t - self.origin;
        Ray::new(self.origin, dir)
    }
}

#[derive(Clone, Copy)]
struct TileSpec {
    id: usize,
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
}

impl TileSpec {
    fn width(self) -> usize {
        self.x1.saturating_sub(self.x0)
    }

    fn height(self) -> usize {
        self.y1.saturating_sub(self.y0)
    }

    fn pixel_count(self) -> usize {
        self.width() * self.height()
    }
}

struct TileData {
    accum: Vec<Vec3>,
    spp: u32,
}

struct SharedTile {
    spec: TileSpec,
    data: Mutex<TileData>,
}

struct SharedRenderer {
    width: usize,
    height: usize,
    tiles: Vec<SharedTile>,
    total_samples: AtomicU64,
    started_at: Instant,
}

struct SnapshotFrame {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

struct PathTracerApp {
    renderer: Arc<SharedRenderer>,
    texture: Option<egui::TextureHandle>,
    rgba_buffer: Vec<u8>,
    stop_flag: Arc<AtomicBool>,
    worker_threads: Vec<JoinHandle<()>>,
    snapshot_sender: Option<Sender<SnapshotFrame>>,
    snapshot_thread: Option<JoinHandle<()>>,
    last_snapshot: Instant,
    dump_interval: Duration,
    core_count: usize,
}

impl PathTracerApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let core_count = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        let tile_specs = build_vertical_tiles(IMAGE_WIDTH, IMAGE_HEIGHT, core_count);
        let tiles = tile_specs
            .iter()
            .copied()
            .map(|spec| SharedTile {
                spec,
                data: Mutex::new(TileData {
                    accum: vec![Vec3::ZERO; spec.pixel_count()],
                    spp: 0,
                }),
            })
            .collect();

        let renderer = Arc::new(SharedRenderer {
            width: IMAGE_WIDTH,
            height: IMAGE_HEIGHT,
            tiles,
            total_samples: AtomicU64::new(0),
            started_at: Instant::now(),
        });

        let stop_flag = Arc::new(AtomicBool::new(false));

        let camera_origin = Vec3::new(0.5, 0.5, -2.2);
        let camera_target = Vec3::new(0.5, 0.5, 0.5);

        let scene = Arc::new(build_cornell_box_scene(camera_origin));
        let camera = Camera::new(
            camera_origin,
            camera_target,
            Vec3::new(0.0, 1.0, 0.0),
            40.0,
            IMAGE_WIDTH as f32 / IMAGE_HEIGHT as f32,
        );

        let mut worker_threads = Vec::with_capacity(tile_specs.len());
        for spec in tile_specs {
            let thread_renderer = Arc::clone(&renderer);
            let thread_scene = Arc::clone(&scene);
            let thread_stop = Arc::clone(&stop_flag);
            let name = format!("render-tile-{}", spec.id);

            if let Ok(handle) = thread::Builder::new().name(name).spawn(move || {
                render_tile_loop(spec, thread_renderer, thread_scene, camera, thread_stop)
            }) {
                worker_threads.push(handle);
            }
        }

        let (snapshot_sender, snapshot_receiver) = bounded::<SnapshotFrame>(1);
        let snapshot_thread = {
            let thread_stop = Arc::clone(&stop_flag);
            let output_path = PathBuf::from(SNAPSHOT_FILE);
            thread::Builder::new()
                .name("snapshot-writer".to_owned())
                .spawn(move || snapshot_writer_loop(snapshot_receiver, thread_stop, output_path))
                .ok()
        };

        Self {
            renderer,
            texture: None,
            rgba_buffer: vec![0_u8; IMAGE_WIDTH * IMAGE_HEIGHT * 4],
            stop_flag,
            worker_threads,
            snapshot_sender: Some(snapshot_sender),
            snapshot_thread,
            last_snapshot: Instant::now(),
            dump_interval: SNAPSHOT_INTERVAL,
            core_count,
        }
    }

    fn refresh_texture(&mut self, ctx: &egui::Context) {
        compose_rgba(&self.renderer, &mut self.rgba_buffer);

        let color_image = egui::ColorImage::from_rgba_unmultiplied(
            [self.renderer.width, self.renderer.height],
            &self.rgba_buffer,
        );

        if let Some(texture) = &mut self.texture {
            texture.set(color_image, egui::TextureOptions::LINEAR);
        } else {
            self.texture =
                Some(ctx.load_texture("pathtracer", color_image, egui::TextureOptions::LINEAR));
        }
    }

    fn maybe_queue_snapshot(&mut self) {
        if self.last_snapshot.elapsed() < self.dump_interval {
            return;
        }

        if let Some(sender) = &self.snapshot_sender {
            let frame = SnapshotFrame {
                width: self.renderer.width as u32,
                height: self.renderer.height as u32,
                pixels: self.rgba_buffer.clone(),
            };
            let _ = sender.try_send(frame);
        }

        self.last_snapshot = Instant::now();
    }
}

impl eframe::App for PathTracerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.refresh_texture(ctx);
        self.maybe_queue_snapshot();

        let elapsed = self.renderer.started_at.elapsed().as_secs_f32();
        let total_samples = self.renderer.total_samples.load(Ordering::Relaxed);
        let avg_spp = total_samples as f32 / (self.renderer.width * self.renderer.height) as f32;
        let min_spp = min_tile_spp(&self.renderer);

        egui::TopBottomPanel::top("status").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(format!(
                    "Resolution: {}x{}",
                    self.renderer.width, self.renderer.height
                ));
                ui.separator();
                ui.label(format!("Threads/Tiles: {}", self.core_count));
                ui.separator();
                ui.label(format!("Avg spp: {:.2}", avg_spp));
                ui.separator();
                ui.label(format!("Min tile spp: {}", min_spp));
                ui.separator();
                ui.label(format!("Elapsed: {:.1}s", elapsed));
                ui.separator();
                ui.label(format!("Snapshot: {}", SNAPSHOT_FILE));
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(texture) = &self.texture {
                let image_size =
                    egui::vec2(self.renderer.width as f32, self.renderer.height as f32);
                let avail = ui.available_size();
                let scale = (avail.x / image_size.x)
                    .min(avail.y / image_size.y)
                    .max(0.01);
                let desired = image_size * scale;

                ui.centered_and_justified(|ui| {
                    ui.add(egui::Image::new(texture).fit_to_exact_size(desired));
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

        for handle in self.worker_threads.drain(..) {
            let _ = handle.join();
        }

        if let Some(handle) = self.snapshot_thread.take() {
            let _ = handle.join();
        }
    }
}

fn build_vertical_tiles(width: usize, height: usize, count: usize) -> Vec<TileSpec> {
    let tile_count = count.max(1);
    let mut tiles = Vec::with_capacity(tile_count);

    for id in 0..tile_count {
        let x0 = id * width / tile_count;
        let x1 = (id + 1) * width / tile_count;
        tiles.push(TileSpec {
            id,
            x0,
            x1,
            y0: 0,
            y1: height,
        });
    }

    tiles
}

fn build_cornell_box_scene(camera_origin: Vec3) -> Scene {
    let mut objects = Vec::new();

    let white = Material::Diffuse {
        albedo: Vec3::new(0.78, 0.78, 0.78),
    };
    let green = Material::Diffuse {
        albedo: Vec3::new(0.18, 0.62, 0.22),
    };
    let metallic_wall = Material::Metal {
        albedo: Vec3::new(0.9, 0.91, 0.93),
        fuzz: 0.02,
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
    let light = Material::Emissive {
        color: Vec3::splat(15.0),
    };
    let duck_yellow = Material::Glossy {
        albedo: Vec3::new(0.92, 0.82, 0.16),
        roughness: 0.2,
        reflectivity: 0.14,
    };
    let duck_orange = Material::Glossy {
        albedo: Vec3::new(0.96, 0.5, 0.1),
        roughness: 0.24,
        reflectivity: 0.08,
    };
    let duck_eye = Material::Diffuse {
        albedo: Vec3::splat(0.03),
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
        albedo: Vec3::new(0.05, 0.03, 0.03),
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
        x0: 0.31,
        x1: 0.69,
        z0: 0.27,
        z1: 0.73,
        y: 0.999,
        normal_y: -1.0,
        material: light,
    }));

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

    add_rubber_duck(
        &mut objects,
        Vec3::new(0.72, 0.69, 0.41),
        camera_origin,
        duck_yellow,
        duck_orange,
        duck_eye,
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

fn add_rubber_duck(
    objects: &mut Vec<Primitive>,
    perch: Vec3,
    camera_origin: Vec3,
    yellow: Material,
    orange: Material,
    eye: Material,
) {
    let body_radius = 0.058;
    let body_center = Vec3::new(perch.x, perch.y + body_radius + 0.0015, perch.z);

    let to_camera = (camera_origin - body_center).normalized();
    let forward = Vec3::new(to_camera.x, to_camera.y * 0.35, to_camera.z).normalized();

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
        body_center + forward * 0.026 + up * 0.01,
        0.048,
        yellow,
    );
    add_sphere(
        objects,
        body_center - forward * 0.052 + up * 0.014,
        0.021,
        yellow,
    );
    add_sphere(
        objects,
        body_center - right * 0.045 + up * 0.009,
        0.021,
        yellow,
    );
    add_sphere(
        objects,
        body_center + right * 0.045 + up * 0.009,
        0.021,
        yellow,
    );

    let head_center = body_center + forward * 0.054 + up * 0.062;
    add_sphere(objects, head_center, 0.034, yellow);
    add_sphere(
        objects,
        head_center + forward * 0.014 + up * 0.001,
        0.023,
        yellow,
    );

    let beak_center = head_center + forward * 0.038 - up * 0.005;
    add_sphere(objects, beak_center, 0.013, orange);
    add_sphere(
        objects,
        beak_center + forward * 0.007 - up * 0.01,
        0.0105,
        orange,
    );

    let eye_base = head_center + forward * 0.026 + up * 0.01;
    add_sphere(objects, eye_base - right * 0.012, 0.0045, eye);
    add_sphere(objects, eye_base + right * 0.012, 0.0045, eye);
}

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

fn render_tile_loop(
    spec: TileSpec,
    renderer: Arc<SharedRenderer>,
    scene: Arc<Scene>,
    camera: Camera,
    stop_flag: Arc<AtomicBool>,
) {
    let tile_width = spec.width();
    let tile_height = spec.height();

    if tile_width == 0 || tile_height == 0 {
        while !stop_flag.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_millis(50));
        }
        return;
    }

    let mut rng = SmallRng::from_entropy();
    let mut pass_buffer = vec![Vec3::ZERO; tile_width * tile_height];

    'render: while !stop_flag.load(Ordering::Relaxed) {
        for local_y in 0..tile_height {
            let y = spec.y0 + local_y;
            for local_x in 0..tile_width {
                if stop_flag.load(Ordering::Relaxed) {
                    break 'render;
                }

                let x = spec.x0 + local_x;
                let idx = local_y * tile_width + local_x;
                pass_buffer[idx] = sample_pixel(
                    x,
                    y,
                    renderer.width,
                    renderer.height,
                    camera,
                    scene.as_ref(),
                    &mut rng,
                );
            }
        }

        let mut tile = renderer.tiles[spec.id].data.lock();
        for (accum, sample) in tile.accum.iter_mut().zip(pass_buffer.iter()) {
            *accum += *sample;
        }
        tile.spp = tile.spp.saturating_add(1);
        drop(tile);

        renderer
            .total_samples
            .fetch_add((tile_width * tile_height) as u64, Ordering::Relaxed);
    }
}

fn sample_pixel(
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    camera: Camera,
    scene: &Scene,
    rng: &mut SmallRng,
) -> Vec3 {
    let w = (width.saturating_sub(1)).max(1) as f32;
    let h = (height.saturating_sub(1)).max(1) as f32;

    let u = (x as f32 + rng.gen::<f32>()) / w;
    let v = ((height - 1 - y) as f32 + rng.gen::<f32>()) / h;

    let ray = camera.get_ray(u, v);
    trace_ray(ray, scene, rng)
}

fn trace_ray(mut ray: Ray, scene: &Scene, rng: &mut SmallRng) -> Vec3 {
    let mut throughput = Vec3::splat(1.0);
    let mut radiance = Vec3::ZERO;

    for bounce in 0..MAX_BOUNCES {
        if let Some(hit) = scene.hit(&ray, 0.001, f32::INFINITY) {
            radiance += throughput.mul_elem(hit.material.emitted());

            if let Some((scattered, attenuation)) = hit.material.scatter(&ray, &hit, rng) {
                throughput = throughput.mul_elem(attenuation);

                if bounce >= 3 {
                    let survive_probability = throughput.max_component().clamp(0.05, 0.95);
                    if rng.gen::<f32>() > survive_probability {
                        break;
                    }
                    throughput /= survive_probability;
                }

                ray = scattered;
            } else {
                break;
            }
        } else {
            break;
        }
    }

    radiance
}

fn reflect(v: Vec3, n: Vec3) -> Vec3 {
    v - n * (2.0 * v.dot(n))
}

fn random_in_unit_sphere(rng: &mut SmallRng) -> Vec3 {
    loop {
        let p = Vec3::new(
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        );
        if p.length_squared() < 1.0 {
            return p;
        }
    }
}

fn cosine_hemisphere(normal: Vec3, rng: &mut SmallRng) -> Vec3 {
    let r1 = rng.gen::<f32>();
    let r2 = rng.gen::<f32>();
    let phi = 2.0 * std::f32::consts::PI * r1;

    let x = phi.cos() * r2.sqrt();
    let y = phi.sin() * r2.sqrt();
    let z = (1.0 - r2).sqrt();

    let w = normal.normalized();
    let a = if w.x.abs() > 0.9 {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };
    let v = w.cross(a).normalized();
    let u = v.cross(w);

    (u * x + v * y + w * z).normalized()
}

fn compose_rgba(renderer: &SharedRenderer, output: &mut [u8]) {
    for tile in &renderer.tiles {
        let Some(data) = tile.data.try_lock() else {
            continue;
        };

        let spp = data.spp;
        let inv_spp = if spp > 0 { 1.0 / spp as f32 } else { 0.0 };
        let tile_width = tile.spec.width();
        let tile_height = tile.spec.height();

        for local_y in 0..tile_height {
            let y = tile.spec.y0 + local_y;
            for local_x in 0..tile_width {
                let x = tile.spec.x0 + local_x;
                let idx = local_y * tile_width + local_x;

                let linear = data.accum[idx] * inv_spp;
                let rgba = linear_to_rgba(linear);
                let out_idx = (y * renderer.width + x) * 4;
                output[out_idx] = rgba[0];
                output[out_idx + 1] = rgba[1];
                output[out_idx + 2] = rgba[2];
                output[out_idx + 3] = 255;
            }
        }
    }
}

fn min_tile_spp(renderer: &SharedRenderer) -> u32 {
    renderer
        .tiles
        .iter()
        .filter_map(|tile| tile.data.try_lock().map(|guard| guard.spp))
        .min()
        .unwrap_or(0)
}

fn linear_to_rgba(color: Vec3) -> [u8; 4] {
    let mapped = Vec3::new(
        color.x.max(0.0) / (1.0 + color.x.max(0.0)),
        color.y.max(0.0) / (1.0 + color.y.max(0.0)),
        color.z.max(0.0) / (1.0 + color.z.max(0.0)),
    );

    let gamma = Vec3::new(
        mapped.x.powf(1.0 / 2.2),
        mapped.y.powf(1.0 / 2.2),
        mapped.z.powf(1.0 / 2.2),
    );

    [
        (gamma.x.clamp(0.0, 0.999) * 256.0) as u8,
        (gamma.y.clamp(0.0, 0.999) * 256.0) as u8,
        (gamma.z.clamp(0.0, 0.999) * 256.0) as u8,
        255,
    ]
}

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

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
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
