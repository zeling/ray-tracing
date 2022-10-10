use nalgebra_glm as glm;

trait Renderer {
    fn render<W: std::io::Write>(&self, out: &mut W) -> Result<(), std::io::Error>;
}

struct FirstRenderer;

impl Renderer for FirstRenderer {
    fn render<W: std::io::Write>(&self, out: &mut W) -> Result<(), std::io::Error> {
        const HEIGHT: u32 = 256;
        const WIDTH: u32 = 256;
        write!(out, "P3\n{} {}\n255\n", HEIGHT, WIDTH)?;
        for i in 0..HEIGHT {
            for j in 0..WIDTH {
                write!(out, "{} {} {}\n", j, 255 - i, 64)?;
            }
        }
        out.flush()
    }
}

struct GradientRenderer;

impl Renderer for GradientRenderer {
    fn render<W: std::io::Write>(&self, out: &mut W) -> Result<(), std::io::Error> {
        const ASPECT_RATIO: f32 = 16.0 / 9.0;
        const IMAGE_WIDTH: u32 = 400;
        const IMAGE_HEIGHT: u32 = (IMAGE_WIDTH as f32 / ASPECT_RATIO) as u32;

        const VIEWPORT_HEIGHT: f32 = 2.0;
        const VIEWPORT_WIDTH: f32 = VIEWPORT_HEIGHT * ASPECT_RATIO;
        const FOCAL_LENGTH: f32 = 1.0;

        let origin = glm::vec3(0.0, 0.0, 0.0);
        let horizontal = glm::vec3(VIEWPORT_WIDTH, 0.0, 0.0);
        let vertical = glm::vec3(0.0, VIEWPORT_HEIGHT, 0.0);
        let lower_left_corner =
            origin - horizontal * 0.5 - vertical * 0.5 - glm::vec3(0.0, 0.0, FOCAL_LENGTH);
        write!(out, "P3\n{} {}\n255\n", IMAGE_WIDTH, IMAGE_HEIGHT)?;
        for j in (0..IMAGE_HEIGHT).rev() {
            for i in 0..IMAGE_WIDTH {
                let u = (i as f32) / (IMAGE_WIDTH as f32 - 1.0);
                let v = (j as f32) / (IMAGE_HEIGHT as f32 - 1.0);
                let ray = Ray {
                    orig: origin,
                    dir: lower_left_corner + u * horizontal + v * vertical - origin,
                };
                let color = ray.color();
                color.write(out)?;
            }
        }
        out.flush()
    }
}

fn main() -> Result<(), std::io::Error> {
    GradientRenderer.render(&mut std::io::stdout().lock())
}

struct Ray {
    orig: glm::Vec3,
    dir: glm::Vec3,
}

struct Color(glm::Vec3);

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

    fn color(&self) -> Color {
        let dir = glm::normalize(&self.dir);
        let t = 0.5 * (dir[1] + 1.0);
        Color((1.0 - t) * glm::vec3(1.0, 1.0, 1.0) + t * glm::vec3(0.0, 0.0, 1.0))
    }
}
