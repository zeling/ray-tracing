use std::ops::Deref;

use approx::abs_diff_eq;
use nalgebra_glm as glm;
use rand::{thread_rng, Rng};

struct World {
    camera: Camera,
    objects: Vec<Box<dyn Hittable>>,
}

struct CameraOpts {
    vfov: f32,
    aspect_ratio: f32,
    focal_length: f32,
    eye: glm::Vec3,
    center: glm::Vec3,
    up: glm::Vec3,
    aperture: f32,
    focus_dist: f32,
}

struct RenderOpts {
    image_width: u32,
    image_height: u32,
    samples_per_pixel: u32,
    max_sample_depth: u32,
}

struct Lambertian {
    albedo: Color,
}

impl Material for Lambertian {
    fn scatter(&self, _ray: &Ray, record: &HitRecord) -> Option<(Ray, Color)> {
        let random_in_unit = random_in_unit_sphere();
        let dir = {
            let dir = record.normal + glm::normalize(&random_in_unit);
            if abs_diff_eq!(dir, glm::zero(), epsilon = f32::EPSILON) {
                record.normal
            } else {
                dir
            }
        };
        Some((
            Ray {
                orig: record.p,
                dir,
            },
            self.albedo,
        ))
    }
}

struct Metal {
    albedo: Color,
    fuzz: f32,
}

impl Material for Metal {
    fn scatter(&self, ray: &Ray, record: &HitRecord) -> Option<(Ray, Color)> {
        let reflected = Ray {
            orig: record.p,
            dir: ray.dir - 2.0 * glm::dot(&ray.dir, &record.normal) * record.normal
                + self.fuzz * random_in_unit_sphere(),
        };
        (glm::dot(&reflected.dir, &record.normal) > 0.0).then_some((reflected, self.albedo))
    }
}

struct Dielectrics {
    ir: f32,
}

impl Material for Dielectrics {
    fn scatter(&self, ray: &Ray, record: &HitRecord) -> Option<(Ray, Color)> {
        let attenuation = Color(glm::vec3(1.0, 1.0, 1.0));
        let refraction_ratio = if record.front_face {
            1.0 / self.ir
        } else {
            self.ir
        };
        let ray_dir = glm::normalize(&ray.dir);
        let cosine_theta = -glm::dot(&ray_dir, &record.normal);

        let reflectance = {
            let r0 = (1.0 - refraction_ratio) / (1.0 + refraction_ratio);
            let r0 = r0 * r0;
            r0 + (1.0 - r0) * f32::powi(1.0 - cosine_theta, 5)
        };
        let dir = if refraction_ratio * f32::sqrt(1.0 - cosine_theta * cosine_theta) > 1.0
            || reflectance > thread_rng().gen_range(0.0..1.0)
        {
            ray_dir - 2.0 * glm::dot(&ray_dir, &record.normal) * record.normal
        } else {
            let out_x = (ray_dir + cosine_theta * record.normal) * refraction_ratio;
            let out_y = -f32::sqrt(f32::max(0.0, 1.0 - glm::length2(&out_x))) * record.normal;
            out_x + out_y
        };
        Some((
            Ray {
                orig: record.p,
                dir,
            },
            attenuation,
        ))
    }
}

fn main() -> Result<(), std::io::Error> {
    let world = World {
        objects: vec![
            Box::new(Sphere {
                center: glm::vec3(0.0, 0.0, -1.0),
                radius: 0.5,
                material: Lambertian {
                    albedo: Color(glm::vec3(0.1, 0.2, 0.5)),
                },
            }),
            Box::new(Sphere {
                center: glm::vec3(0.0, -100.5, -1.0),
                radius: 100.0,
                material: Lambertian {
                    albedo: Color(glm::vec3(0.8, 0.8, 0.0)),
                },
            }),
            Box::new(Sphere {
                center: glm::vec3(-1.0, 0.0, -1.0),
                radius: 0.5,
                material: Dielectrics { ir: 1.5 },
            }),
            Box::new(Sphere {
                center: glm::vec3(-1.0, 0.0, -1.0),
                radius: -0.4,
                material: Dielectrics { ir: 1.5 },
            }),
            Box::new(Sphere {
                center: glm::vec3(1.0, 0.0, -1.0),
                radius: 0.5,
                material: Metal {
                    albedo: Color(glm::vec3(0.8, 0.6, 0.2)),
                    fuzz: 0.0,
                },
            }),
        ],
        camera: Camera::new(CameraOpts {
            vfov: std::f32::consts::PI / 9.0,
            aspect_ratio: 16.0 / 9.0,
            focal_length: 1.0,
            eye: glm::vec3(3.0, 3.0, 2.0),
            center: glm::vec3(0.0, 0.0, -1.0),
            up: glm::vec3(0.0, 1.0, 0.0),
            aperture: 2.0,
            focus_dist: glm::vec3::<f32>(3.0, 3.0, 3.0).norm(),
        }),
    };
    world.render(
        &mut std::io::stdout().lock(),
        RenderOpts {
            image_width: 400,
            image_height: 225,
            samples_per_pixel: 100,
            max_sample_depth: 50,
        },
    )
}

struct Camera {
    eye: glm::Vec3,
    viewport_origin: glm::Vec3,
    horizontal: glm::Vec3,
    vertical: glm::Vec3,
    lens_radius: f32,
    u: glm::Vec3,
    v: glm::Vec3,
    w: glm::Vec3,
}

impl Camera {
    fn new(opts: CameraOpts) -> Self {
        let viewport_height = 2.0 * f32::tan(opts.vfov / 2.0) * opts.focal_length;
        let viewport_width = viewport_height * opts.aspect_ratio;

        let w = glm::normalize(&(opts.eye - opts.center));
        let u = glm::cross(&opts.up, &w);
        let v = glm::cross(&w, &u);

        let horizontal = opts.focus_dist * viewport_width * u;
        let vertical = opts.focus_dist * viewport_height * v;
        let viewport_origin = opts.eye - horizontal / 2.0 - vertical / 2.0 - opts.focus_dist * w;
        let lens_radius = opts.aperture / 2.0;
        Self {
            eye: opts.eye,
            viewport_origin,
            horizontal,
            vertical,
            lens_radius,
            w,
            u,
            v,
        }
    }

    fn ray(&self, u: f32, v: f32) -> Ray {
        let rd = {
            loop {
                let candidate = glm::vec3(
                    thread_rng().gen_range(0.0..self.lens_radius),
                    thread_rng().gen_range(0.0..self.lens_radius),
                    0.0,
                );
                if candidate.norm() <= self.lens_radius {
                    break candidate;
                }
            }
        };
        let orig = self.eye + (rd[0] * self.u) + (rd[1] * self.v);
        Ray {
            orig,
            dir: self.viewport_origin + self.horizontal * u + self.vertical * v - orig,
        }
    }
}

impl World {
    fn render<W: std::io::Write>(&self, out: &mut W, opts: RenderOpts) -> std::io::Result<()> {
        write!(out, "P3\n{} {}\n255\n", opts.image_width, opts.image_height)?;
        for j in (0..opts.image_height).rev() {
            for i in 0..opts.image_width {
                let color = {
                    let color =
                        (0..opts.samples_per_pixel).fold(glm::zero(), |acc: glm::Vec3, _| {
                            let u = (i as f32 + thread_rng().gen_range(0.0..1.0))
                                / (opts.image_width as f32 - 1.0);
                            let v = (j as f32 + thread_rng().gen_range(0.0..1.0))
                                / (opts.image_height as f32 - 1.0);
                            let ray = self.camera.ray(u, v);
                            acc + self.sample(&ray, opts.max_sample_depth)
                        }) / opts.samples_per_pixel as f32;
                    Color(glm::sqrt(&color))
                };
                color.write(out)?;
            }
        }
        out.flush()
    }

    fn sample(&self, ray: &Ray, sample_depth: u32) -> glm::Vec3 {
        if sample_depth == 0 {
            return glm::zero();
        }
        if let Some((record, material)) = self.objects.hit(&ray, 0.001, f32::MAX) {
            if let Some((scattered, Color(attenuation))) = material.scatter(&ray, &record) {
                attenuation.component_mul(&self.sample(&scattered, sample_depth - 1))
            } else {
                glm::zero()
            }
        } else {
            let white = glm::vec3(1.0, 1.0, 1.0);
            let blue = glm::vec3(0.5, 0.7, 1.0);
            let dir = glm::normalize(&ray.dir);
            glm::lerp(&white, &blue, 0.5 * (dir[1] + 1.0))
        }
    }
}

trait Hittable {
    fn hit(&self, ray: &Ray, tmin: f32, tmax: f32) -> Option<(HitRecord, &dyn Material)>;
}

trait Material {
    fn scatter(&self, ray: &Ray, record: &HitRecord) -> Option<(Ray, Color)>;
}

#[derive(Debug)]
struct HitRecord {
    t: f32,
    p: glm::Vec3,
    front_face: bool,
    normal: glm::Vec3,
}

struct Ray {
    orig: glm::Vec3,
    dir: glm::Vec3,
}

#[derive(Clone, Copy)]
struct Color(glm::Vec3);

#[derive(Debug)]
struct Sphere<M: Material> {
    center: glm::Vec3,
    radius: f32,
    material: M,
}

impl Hittable for Vec<Box<dyn Hittable>> {
    fn hit(&self, ray: &Ray, tmin: f32, tmax: f32) -> Option<(HitRecord, &(dyn Material + '_))> {
        let (record, _) = self
            .iter()
            .fold((None, tmax), |(record, closest_so_far), obj| {
                let cur = obj.deref().hit(ray, tmin, closest_so_far);
                let closest_so_far = cur.as_ref().map(|(r, _m)| r.t).unwrap_or(closest_so_far);
                (cur.or(record), closest_so_far)
            });
        record
    }
}

impl<M: Material> Hittable for Sphere<M> {
    fn hit(&self, ray: &Ray, tmin: f32, tmax: f32) -> Option<(HitRecord, &dyn Material)> {
        let oc = ray.orig - self.center;
        let a = glm::dot(&ray.dir, &ray.dir);
        let h = glm::dot(&ray.dir, &oc);
        let c = glm::dot(&oc, &oc) - self.radius * self.radius;
        let delta = h * h - a * c;
        if delta < 0.0 {
            return None;
        }
        let record = |t| -> (HitRecord, &dyn Material) {
            let p = ray.at(t);
            let outward_normal = (p - self.center) / self.radius;
            let front_face = glm::dot(&ray.dir, &outward_normal) < 0.0;
            let normal = if front_face {
                outward_normal
            } else {
                -outward_normal
            };
            (
                HitRecord {
                    t,
                    p,
                    front_face,
                    normal,
                },
                &self.material,
            )
        };
        let t1 = (-h - delta.sqrt()) / a;
        if t1 >= tmin && t1 <= tmax {
            return Some(record(t1));
        }
        let t2 = (-h + delta.sqrt()) / a;
        if t2 >= tmin && t2 <= tmax {
            return Some(record(t2));
        }
        None
    }
}

impl Color {
    fn write<W: std::io::Write>(&self, out: &mut W) -> Result<(), std::io::Error> {
        write!(
            out,
            "{} {} {}\n",
            (self.0[0] * 255.999) as u32,
            (self.0[1] * 255.999) as u32,
            (self.0[2] * 255.999) as u32,
        )
    }
}

impl Ray {
    fn at(&self, t: f32) -> glm::Vec3 {
        self.orig + t * self.dir
    }
}

fn random_in_unit_sphere() -> glm::Vec3 {
    loop {
        let candidate = glm::Vec3::from_fn(|_, _| thread_rng().gen_range(-1.0..1.0));
        if glm::dot(&candidate, &candidate) <= 1.0 {
            break candidate;
        }
    }
}
