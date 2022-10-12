use nalgebra_glm as glm;

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
}

fn main() -> Result<(), std::io::Error> {
    let world = World {
        objects: vec![
            Box::new(Sphere {
                center: glm::vec3(0.0, 0.0, -1.0),
                radius: 0.5,
            }),
            Box::new(Sphere {
                center: glm::vec3(0.0, -100.5, -1.0),
                radius: 100.0,
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
        },
    )
}

impl World {
    fn render<W: std::io::Write>(&self, out: &mut W, opts: RenderOpts) -> std::io::Result<()> {
        let origin = glm::vec3(0.0, 0.0, 0.0);
        let horizontal = glm::vec3(opts.camera.viewport_width, 0.0, 0.0);
        let vertical = glm::vec3(0.0, opts.camera.viewport_height, 0.0);
        let lower_left_corner = origin
            - horizontal * 0.5
            - vertical * 0.5
            - glm::vec3(0.0, 0.0, opts.camera.focal_length);
        write!(out, "P3\n{} {}\n255\n", opts.image_width, opts.image_height)?;
        for j in (0..opts.image_height).rev() {
            for i in 0..opts.image_width {
                let u = (i as f32) / (opts.image_width as f32 - 1.0);
                let v = (j as f32) / (opts.image_height as f32 - 1.0);
                let ray = Ray {
                    orig: origin,
                    dir: lower_left_corner + u * horizontal + v * vertical - origin,
                };
                let color = {
                    let dir = glm::normalize(&ray.dir);
                    if let Some(record) = self.hit(&ray, 0.0, f32::MAX) {
                        Color(0.5 * (record.normal + glm::vec3(1.0, 1.0, 1.0)))
                    } else {
                        let white = glm::vec3(1.0, 1.0, 1.0);
                        let blue = glm::vec3(0.0, 0.0, 1.0);
                        Color(glm::lerp(&white, &blue, 0.5 * (dir[1] + 1.0)))
                    }
                };
                color.write(out)?;
            }
        }
        out.flush()
    }
}

trait Hittable {
    fn hit(&self, ray: &Ray, tmin: f32, tmax: f32) -> Option<HitRecord>;
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

struct Color(glm::Vec3);

#[derive(Debug)]
struct Sphere {
    center: glm::Vec3,
    radius: f32,
}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, tmin: f32, tmax: f32) -> Option<HitRecord> {
        let oc = ray.orig - self.center;
        let a = glm::dot(&ray.dir, &ray.dir);
        let h = glm::dot(&ray.dir, &oc);
        let c = glm::dot(&oc, &oc) - self.radius * self.radius;
        let delta = h * h - a * c;
        if delta < 0.0 {
            return None;
        }
        let record = |t| -> HitRecord {
            let p = ray.at(t);
            let outward_normal = glm::normalize(&(p - self.center));
            let front_face = glm::dot(&ray.dir, &outward_normal) < 0.0;
            let normal = if front_face {
                outward_normal
            } else {
                -outward_normal
            };
            HitRecord {
                t,
                p,
                front_face,
                normal,
            }
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

impl Hittable for World {
    fn hit(&self, ray: &Ray, tmin: f32, tmax: f32) -> Option<HitRecord> {
        let (record, _) =
            self.objects
                .iter()
                .fold((None, tmax), |(record, closest_so_far), obj| {
                    let cur = obj.hit(ray, tmin, closest_so_far);
                    let closest_so_far = cur.as_ref().map(|r| r.t).unwrap_or(closest_so_far);
                    (cur.or(record), closest_so_far)
                });
        record
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
