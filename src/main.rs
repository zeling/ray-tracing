use std::ops::Deref;

use approx::abs_diff_eq;
use nalgebra_glm as glm;
use rand::{thread_rng, Rng};

struct World {
    objects: Vec<Box<dyn Hittable>>,
}

struct CameraOpts {
    viewport_width: f32,
    viewport_height: f32,
    focal_length: f32,
}

struct RenderOpts {
    camera: CameraOpts,
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
        let random_in_unit = loop {
            let candidate = glm::Vec3::from_fn(|_, _| thread_rng().gen_range(-1.0..1.0));
            if glm::dot(&candidate, &candidate) <= 1.0 {
                break candidate;
            }
        };
        let dir = {
            let dir = record.normal + glm::normalize(&random_in_unit);
            if abs_diff_eq!(dir, glm::zero(), epsilon = 1e-8 /* f32::EPSILON */) {
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
}

impl Material for Metal {
    fn scatter(&self, ray: &Ray, record: &HitRecord) -> Option<(Ray, Color)> {
        let reflected = Ray {
            orig: record.p,
            dir: ray.dir - 2.0 * glm::dot(&ray.dir, &record.normal) * record.normal,
        };
        (glm::dot(&reflected.dir, &record.normal) > 0.0).then_some((reflected, self.albedo))
    }
}

fn main() -> Result<(), std::io::Error> {
    let world = World {
        objects: vec![
            Box::new(Sphere {
                center: glm::vec3(0.0, 0.0, -1.0),
                radius: 0.5,
                material: Lambertian {
                    albedo: Color(glm::vec3(0.7, 0.3, 0.3)),
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
                material: Metal {
                    albedo: Color(glm::vec3(0.8, 0.8, 0.8)),
                },
            }),
            Box::new(Sphere {
                center: glm::vec3(1.0, 0.0, -1.0),
                radius: 0.5,
                material: Metal {
                    albedo: Color(glm::vec3(0.8, 0.6, 0.2)),
                },
            }),
        ],
    };
    world.render(
        &mut std::io::stdout().lock(),
        RenderOpts {
            camera: CameraOpts {
                viewport_width: 2.0 * 16.0 / 9.0,
                viewport_height: 2.0,
                focal_length: 1.0,
            },
            image_width: 400,
            image_height: 225,
            samples_per_pixel: 100,
            max_sample_depth: 50,
        },
    )
}

struct Camera {
    viewport_origin: glm::Vec3,
    horizontal: glm::Vec3,
    vertical: glm::Vec3,
}

impl Camera {
    fn new(opts: CameraOpts) -> Self {
        let horizontal = glm::vec3(opts.viewport_width, 0.0, 0.0);
        let vertical = glm::vec3(0.0, opts.viewport_height, 0.0);
        let viewport_origin = glm::zero::<glm::Vec3>()
            - horizontal / 2.0
            - vertical / 2.0
            - glm::vec3(0.0, 0.0, opts.focal_length);
        Self {
            viewport_origin,
            horizontal,
            vertical,
        }
    }

    fn ray(&self, u: f32, v: f32) -> Ray {
        Ray {
            orig: glm::zero(),
            dir: self.viewport_origin + self.horizontal * u + self.vertical * v
                - glm::zero::<glm::Vec3>(),
        }
    }
}

impl World {
    fn render<W: std::io::Write>(&self, out: &mut W, opts: RenderOpts) -> std::io::Result<()> {
        let camera = Camera::new(opts.camera);
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
                            let ray = camera.ray(u, v);
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
            let outward_normal = glm::normalize(&(p - self.center));
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
