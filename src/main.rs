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

fn main() -> Result<(), std::io::Error> {
    FirstRenderer.render(&mut std::io::stdout().lock())
}
