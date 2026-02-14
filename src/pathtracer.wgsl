struct Uniforms {
    dims: vec4<u32>,
    frame: vec4<u32>,
    origin: vec4<f32>,
    lower_left: vec4<f32>,
    horizontal: vec4<f32>,
    vertical: vec4<f32>,
};

struct Primitive {
    header: vec4<u32>,
    p0: vec4<f32>,
    p1: vec4<f32>,
    color: vec4<f32>,
    params: vec4<f32>,
};

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
};

struct Hit {
    t: f32,
    point: vec3<f32>,
    normal: vec3<f32>,
    prim_index: u32,
};

struct Scatter {
    valid: bool,
    ray: Ray,
    attenuation: vec3<f32>,
};

const PI: f32 = 3.14159265358979323846;

const PRIMITIVE_XY: u32 = 0u;
const PRIMITIVE_XZ: u32 = 1u;
const PRIMITIVE_YZ: u32 = 2u;
const PRIMITIVE_SPHERE: u32 = 3u;

const MATERIAL_DIFFUSE: u32 = 0u;
const MATERIAL_METAL: u32 = 1u;
const MATERIAL_GLOSSY: u32 = 2u;
const MATERIAL_EMISSIVE: u32 = 3u;

const RENDER_MODE_PATH: u32 = 0u;
const RENDER_MODE_FAST: u32 = 1u;

const FOG_BASE_DENSITY: f32 = 0.38;
const DISPLAY_EXPOSURE: f32 = 0.55;

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var<storage, read> primitives: array<Primitive>;

@group(0) @binding(2)
var<storage, read_write> accumulation: array<vec4<f32>>;

@group(0) @binding(3)
var output_tex: texture_storage_2d<rgba8unorm, write>;

fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
    let len2 = dot(v, v);
    if len2 > 0.0 {
        return v * inverseSqrt(len2);
    }
    return vec3<f32>(0.0);
}

fn face_forward(normal: vec3<f32>, direction: vec3<f32>) -> vec3<f32> {
    if dot(direction, normal) < 0.0 {
        return normal;
    }
    return -normal;
}

fn reflect(v: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    return v - n * (2.0 * dot(v, n));
}

fn max_component(v: vec3<f32>) -> f32 {
    return max(max(v.x, v.y), v.z);
}

fn hash31(p: vec3<f32>) -> f32 {
    let n = dot(p, vec3<f32>(127.1, 311.7, 74.7));
    return fract(sin(n) * 43758.5453123);
}

fn night_sky(direction: vec3<f32>) -> vec3<f32> {
    let dir = safe_normalize(direction);
    var sky = vec3<f32>(0.0);

    let moon_dir = safe_normalize(vec3<f32>(-0.18, 0.24, 0.95));
    let moon_alignment = clamp(dot(dir, moon_dir), 0.0, 1.0);
    let moon_glow = smoothstep(0.992, 0.9995, moon_alignment);
    let moon_disc = smoothstep(0.9996, 0.99992, moon_alignment);
    let moon_color = vec3<f32>(0.88, 0.9, 1.0);
    sky = sky + moon_color * (moon_glow * 0.22 + moon_disc * 3.4);

    let star_grid = floor((dir + vec3<f32>(1.0)) * 820.0);
    let star_seed = hash31(star_grid);
    let star_mask = step(0.9972, star_seed) * smoothstep(-0.18, 0.08, dir.y);
    let star_variation = pow(hash31(star_grid + vec3<f32>(19.0, 47.0, 73.0)), 6.0);
    let star_intensity = star_mask * (0.2 + star_variation * 1.4);
    let star_tint = vec3<f32>(0.72, 0.79, 1.0)
        + vec3<f32>(0.55, 0.36, 0.15) * hash31(star_grid + vec3<f32>(2.0, 11.0, 37.0));

    sky = sky + star_tint * star_intensity;
    return sky;
}

fn init_rng(pixel: vec2<u32>, sample_index: u32, seed: u32) -> u32 {
    var state = pixel.x * 1973u + pixel.y * 9277u + sample_index * 26699u + seed * 31847u + 1u;
    if state == 0u {
        state = 1u;
    }
    return state;
}

fn rand_f32(state: ptr<function, u32>) -> f32 {
    var x = *state;
    x = x ^ (x << 13u);
    x = x ^ (x >> 17u);
    x = x ^ (x << 5u);
    *state = x;
    return f32(x) * (1.0 / 4294967296.0);
}

fn random_in_unit_sphere(state: ptr<function, u32>) -> vec3<f32> {
    var p = vec3<f32>(0.0);
    loop {
        p = vec3<f32>(
            rand_f32(state) * 2.0 - 1.0,
            rand_f32(state) * 2.0 - 1.0,
            rand_f32(state) * 2.0 - 1.0,
        );
        if dot(p, p) < 1.0 {
            break;
        }
    }
    return p;
}

fn cosine_hemisphere(normal: vec3<f32>, state: ptr<function, u32>) -> vec3<f32> {
    let r1 = rand_f32(state);
    let r2 = rand_f32(state);
    let phi = 2.0 * PI * r1;

    let x = cos(phi) * sqrt(r2);
    let y = sin(phi) * sqrt(r2);
    let z = sqrt(1.0 - r2);

    let w = safe_normalize(normal);
    let helper = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(w.x) > 0.9);
    let v = safe_normalize(cross(w, helper));
    let u = cross(v, w);

    return safe_normalize(u * x + v * y + w * z);
}

fn miss_hit() -> Hit {
    return Hit(-1.0, vec3<f32>(0.0), vec3<f32>(0.0), 0u);
}

fn is_hit(hit: Hit) -> bool {
    return hit.t > 0.0;
}

fn intersect_primitive(ray: Ray, prim: Primitive, t_min: f32, t_max: f32) -> Hit {
    let kind = prim.header.x;

    if kind == PRIMITIVE_XY {
        if abs(ray.direction.z) < 1e-6 {
            return miss_hit();
        }

        let z = prim.p1.x;
        let t = (z - ray.origin.z) / ray.direction.z;
        if t <= t_min || t >= t_max {
            return miss_hit();
        }

        let x = ray.origin.x + t * ray.direction.x;
        let y = ray.origin.y + t * ray.direction.y;
        if x < prim.p0.x || x > prim.p0.y || y < prim.p0.z || y > prim.p0.w {
            return miss_hit();
        }

        let point = ray.origin + ray.direction * t;
        let outward = vec3<f32>(0.0, 0.0, prim.p1.y);
        return Hit(t, point, face_forward(outward, ray.direction), 0u);
    }

    if kind == PRIMITIVE_XZ {
        if abs(ray.direction.y) < 1e-6 {
            return miss_hit();
        }

        let y_plane = prim.p1.x;
        let t = (y_plane - ray.origin.y) / ray.direction.y;
        if t <= t_min || t >= t_max {
            return miss_hit();
        }

        let x = ray.origin.x + t * ray.direction.x;
        let z = ray.origin.z + t * ray.direction.z;
        if x < prim.p0.x || x > prim.p0.y || z < prim.p0.z || z > prim.p0.w {
            return miss_hit();
        }

        let point = ray.origin + ray.direction * t;
        let outward = vec3<f32>(0.0, prim.p1.y, 0.0);
        return Hit(t, point, face_forward(outward, ray.direction), 0u);
    }

    if kind == PRIMITIVE_YZ {
        if abs(ray.direction.x) < 1e-6 {
            return miss_hit();
        }

        let x_plane = prim.p1.x;
        let t = (x_plane - ray.origin.x) / ray.direction.x;
        if t <= t_min || t >= t_max {
            return miss_hit();
        }

        let y = ray.origin.y + t * ray.direction.y;
        let z = ray.origin.z + t * ray.direction.z;
        if y < prim.p0.x || y > prim.p0.y || z < prim.p0.z || z > prim.p0.w {
            return miss_hit();
        }

        let point = ray.origin + ray.direction * t;
        let outward = vec3<f32>(prim.p1.y, 0.0, 0.0);
        return Hit(t, point, face_forward(outward, ray.direction), 0u);
    }

    if kind == PRIMITIVE_SPHERE {
        let center = prim.p0.xyz;
        let radius = prim.p0.w;

        let oc = ray.origin - center;
        let a = dot(ray.direction, ray.direction);
        let half_b = dot(oc, ray.direction);
        let c = dot(oc, oc) - radius * radius;
        let discriminant = half_b * half_b - a * c;
        if discriminant < 0.0 {
            return miss_hit();
        }

        let sqrtd = sqrt(discriminant);
        var t = (-half_b - sqrtd) / a;
        if t <= t_min || t >= t_max {
            t = (-half_b + sqrtd) / a;
            if t <= t_min || t >= t_max {
                return miss_hit();
            }
        }

        let point = ray.origin + ray.direction * t;
        let outward = (point - center) / radius;
        return Hit(t, point, face_forward(outward, ray.direction), 0u);
    }

    return miss_hit();
}

fn scene_hit(ray: Ray, t_min: f32, t_max: f32) -> Hit {
    var closest = t_max;
    var result = miss_hit();
    let primitive_count = uniforms.dims.z;

    var i = 0u;
    loop {
        if i >= primitive_count {
            break;
        }

        let candidate = intersect_primitive(ray, primitives[i], t_min, closest);
        if is_hit(candidate) {
            closest = candidate.t;
            result = candidate;
            result.prim_index = i;
        }

        i = i + 1u;
    }

    return result;
}

fn invalid_scatter() -> Scatter {
    return Scatter(false, Ray(vec3<f32>(0.0), vec3<f32>(0.0)), vec3<f32>(0.0));
}

fn emitted(prim: Primitive) -> vec3<f32> {
    if prim.header.y == MATERIAL_EMISSIVE {
        return prim.color.xyz;
    }
    return vec3<f32>(0.0);
}

fn scatter(ray_in: Ray, hit: Hit, prim: Primitive, rng_state: ptr<function, u32>) -> Scatter {
    let material_kind = prim.header.y;
    let albedo = prim.color.xyz;

    if material_kind == MATERIAL_DIFFUSE {
        let direction = cosine_hemisphere(hit.normal, rng_state);
        let origin = hit.point + hit.normal * 1e-4;
        return Scatter(true, Ray(origin, direction), albedo);
    }

    if material_kind == MATERIAL_METAL {
        let fuzz = prim.params.x;
        let reflected = reflect(ray_in.direction, hit.normal);
        let direction = safe_normalize(reflected + random_in_unit_sphere(rng_state) * fuzz);
        if dot(direction, hit.normal) > 0.0 {
            let origin = hit.point + hit.normal * 1e-4;
            return Scatter(true, Ray(origin, direction), albedo);
        }
        return invalid_scatter();
    }

    if material_kind == MATERIAL_GLOSSY {
        let roughness = prim.params.x;
        let reflectivity = prim.params.y;

        if rand_f32(rng_state) < reflectivity {
            let reflected = reflect(ray_in.direction, hit.normal);
            let direction = safe_normalize(reflected + random_in_unit_sphere(rng_state) * roughness);
            if dot(direction, hit.normal) > 0.0 {
                let origin = hit.point + hit.normal * 1e-4;
                return Scatter(true, Ray(origin, direction), albedo);
            }
        }

        let diffuse_direction = cosine_hemisphere(hit.normal, rng_state);
        let origin = hit.point + hit.normal * 1e-4;
        return Scatter(true, Ray(origin, diffuse_direction), albedo);
    }

    return invalid_scatter();
}

fn fog_density(point: vec3<f32>) -> f32 {
    let height = clamp(point.y, 0.0, 1.0);
    let height_factor = 1.2 - 0.55 * height;
    let swirl = 0.76 + 0.24 * sin(point.x * 17.0 + point.z * 11.0);
    return FOG_BASE_DENSITY * height_factor * swirl;
}

fn fog_light(point: vec3<f32>, light_pos: vec3<f32>, tint: vec3<f32>) -> vec3<f32> {
    let to_light = light_pos - point;
    let dist2 = dot(to_light, to_light);
    return tint / (1.0 + dist2 * 140.0);
}

fn fog_scatter_color(point: vec3<f32>) -> vec3<f32> {
    var disco = vec3<f32>(0.0);
    disco = disco + fog_light(point, vec3<f32>(0.31, 0.92, 0.34), vec3<f32>(0.95, 0.24, 0.84));
    disco = disco + fog_light(point, vec3<f32>(0.69, 0.92, 0.32), vec3<f32>(0.23, 0.88, 0.96));
    disco = disco + fog_light(point, vec3<f32>(0.72, 0.91, 0.70), vec3<f32>(0.55, 0.92, 0.26));
    disco = disco + fog_light(point, vec3<f32>(0.28, 0.91, 0.72), vec3<f32>(0.96, 0.68, 0.18));

    return disco * 0.34;
}

fn integrate_fog(ray: Ray, segment_distance: f32) -> vec4<f32> {
    let distance = max(segment_distance, 0.0);
    let midpoint = ray.origin + ray.direction * (0.5 * distance);
    let density = fog_density(midpoint);
    let transmittance = exp(-density * distance);
    let scatter = fog_scatter_color(midpoint) * (1.0 - transmittance);
    return vec4<f32>(scatter, transmittance);
}

fn party_light(point: vec3<f32>, normal: vec3<f32>, light_pos: vec3<f32>, tint: vec3<f32>) -> vec3<f32> {
    let to_light = light_pos - point;
    let dist2 = dot(to_light, to_light);
    let light_dir = safe_normalize(to_light);
    let diffuse = max(dot(normal, light_dir), 0.0);
    return tint * diffuse / (1.0 + dist2 * 110.0);
}

fn fast_surface_shading(ray: Ray, hit: Hit, prim: Primitive) -> vec3<f32> {
    let albedo = prim.color.xyz;
    var lighting = vec3<f32>(0.0);

    lighting = lighting + party_light(hit.point, hit.normal, vec3<f32>(0.31, 0.92, 0.34), vec3<f32>(0.95, 0.24, 0.84));
    lighting = lighting + party_light(hit.point, hit.normal, vec3<f32>(0.69, 0.92, 0.32), vec3<f32>(0.23, 0.88, 0.96));
    lighting = lighting + party_light(hit.point, hit.normal, vec3<f32>(0.72, 0.91, 0.70), vec3<f32>(0.55, 0.92, 0.26));
    lighting = lighting + party_light(hit.point, hit.normal, vec3<f32>(0.28, 0.91, 0.72), vec3<f32>(0.96, 0.68, 0.18));
    lighting = lighting + party_light(hit.point, hit.normal, vec3<f32>(0.50, 0.999, 0.50), vec3<f32>(0.36, 0.34, 0.33));

    var shaded = albedo * lighting * 2.2;

    let view_dir = -ray.direction;
    let rim = pow(clamp(1.0 - max(dot(hit.normal, view_dir), 0.0), 0.0, 1.0), 2.0);
    shaded = shaded + albedo * rim * 0.08;

    if prim.header.y == MATERIAL_METAL || prim.header.y == MATERIAL_GLOSSY {
        let highlight_dir = safe_normalize(view_dir + vec3<f32>(0.18, 0.95, 0.23));
        let spec_power = select(22.0, 48.0, prim.header.y == MATERIAL_METAL);
        let spec_strength = select(0.14, 0.3, prim.header.y == MATERIAL_METAL);
        let spec = pow(max(dot(hit.normal, highlight_dir), 0.0), spec_power) * spec_strength;
        shaded = shaded + vec3<f32>(spec);
    }

    return shaded;
}

fn trace_fast(ray: Ray) -> vec3<f32> {
    let hit = scene_hit(ray, 0.001, 1e20);
    if !is_hit(hit) {
        return night_sky(ray.direction);
    }

    let fog = integrate_fog(ray, hit.t);
    var radiance = fog.xyz;

    let prim = primitives[hit.prim_index];
    radiance = radiance + fog.w * emitted(prim);

    if prim.header.y != MATERIAL_EMISSIVE {
        radiance = radiance + fog.w * fast_surface_shading(ray, hit, prim);
    }

    return radiance;
}

fn trace_ray(initial_ray: Ray, rng_state: ptr<function, u32>) -> vec3<f32> {
    var ray = initial_ray;
    var throughput = vec3<f32>(1.0);
    var radiance = vec3<f32>(0.0);

    let max_bounces = uniforms.dims.w;
    var bounce = 0u;
    loop {
        if bounce >= max_bounces {
            break;
        }

        let hit = scene_hit(ray, 0.001, 1e20);
        if !is_hit(hit) {
            radiance = radiance + throughput * night_sky(ray.direction);
            break;
        }

        let fog = integrate_fog(ray, hit.t);
        radiance = radiance + throughput * fog.xyz;
        throughput = throughput * fog.w;

        let prim = primitives[hit.prim_index];
        radiance = radiance + throughput * emitted(prim);

        let scatter_result = scatter(ray, hit, prim, rng_state);
        if !scatter_result.valid {
            break;
        }

        throughput = throughput * scatter_result.attenuation;

        if bounce >= 3u {
            let survive_probability = clamp(max_component(throughput), 0.05, 0.95);
            if rand_f32(rng_state) > survive_probability {
                break;
            }
            throughput = throughput / survive_probability;
        }

        ray = scatter_result.ray;
        bounce = bounce + 1u;
    }

    return radiance;
}

fn linear_to_display(color: vec3<f32>) -> vec3<f32> {
    let positive = max(color, vec3<f32>(0.0)) * DISPLAY_EXPOSURE;
    let mapped = positive / (vec3<f32>(1.0) + positive);
    return pow(mapped, vec3<f32>(1.0 / 2.2));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = uniforms.dims.x;
    let height = uniforms.dims.y;
    if gid.x >= width || gid.y >= height {
        return;
    }

    let render_mode = uniforms.frame.z;
    var rng_state = init_rng(gid.xy, uniforms.frame.x, uniforms.frame.y);

    let denom_x = f32(max(width, 2u) - 1u);
    let denom_y = f32(max(height, 2u) - 1u);

    var jitter = vec2<f32>(0.5, 0.5);
    if render_mode == RENDER_MODE_PATH {
        jitter = vec2<f32>(rand_f32(&rng_state), rand_f32(&rng_state));
    }

    let u = (f32(gid.x) + jitter.x) / denom_x;
    let v = (f32(height - 1u - gid.y) + jitter.y) / denom_y;

    let origin = uniforms.origin.xyz;
    let direction = safe_normalize(
        uniforms.lower_left.xyz + uniforms.horizontal.xyz * u + uniforms.vertical.xyz * v - origin,
    );

    let idx = gid.y * width + gid.x;

    if render_mode == RENDER_MODE_FAST {
        let fast_color = trace_fast(Ray(origin, direction));
        accumulation[idx] = vec4<f32>(0.0);
        let gamma = linear_to_display(fast_color);
        textureStore(output_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(gamma, 1.0));
        return;
    }

    let sample = trace_ray(Ray(origin, direction), &rng_state);

    let prev = accumulation[idx];
    let sum = prev.xyz + sample;
    let spp = prev.w + 1.0;
    accumulation[idx] = vec4<f32>(sum, spp);

    let average = sum / max(spp, 1.0);
    let gamma = linear_to_display(average);
    textureStore(output_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(gamma, 1.0));
}
