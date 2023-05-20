use eframe::egui;
use eframe::egui::*;
use image;
use image::imageops::{resize, Nearest, Triangle};
use image::{GrayImage, ImageBuffer, Luma, Pixel};
use imageproc::drawing::{draw_filled_circle_mut, draw_filled_rect_mut, draw_line_segment_mut};
use imageproc::geometric_transformations::translate;
use imageproc::rect::Region;

#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct Painting {
    /// in 0-1 normalized coordinates
    pub(crate) lines: Vec<Vec<Pos2>>,
    stroke: Stroke,
    pub submit: bool,
}

const CANVAS_SIZE: f32 = 200.;
impl Painting {
    pub fn ui(&mut self, ui: &mut Ui) {
        self.ui_control(ui);

        let canvas_size = Vec2::new(CANVAS_SIZE, CANVAS_SIZE);
        let scroll_area = ScrollArea::neither()
            .max_height(CANVAS_SIZE)
            .max_width(CANVAS_SIZE);
        scroll_area.show(ui, |ui| {
            ui.set_max_size(canvas_size);
            Frame::canvas(ui.style()).show(ui, |ui| {
                self.ui_content(ui);
            });
        });
    }

    pub(crate) fn new() -> Self {
        Self {
            lines: Default::default(),
            stroke: Stroke::new(3.0, Color32::from_rgb(255, 255, 255)),
            submit: false,
        }
    }
    pub fn ui_control(&mut self, ui: &mut egui::Ui) -> egui::Response {
        ui.horizontal(|ui| {
            egui::stroke_ui(ui, &mut self.stroke, "Stroke");
            ui.separator();
            if ui.button("Feed forward").clicked() {
                self.submit = true;
            }
        })
        .response
    }

    pub fn ui_content(&mut self, ui: &mut Ui) -> egui::Response {
        let (mut response, painter) =
            ui.allocate_painter(ui.available_size_before_wrap(), Sense::drag());

        let to_screen = emath::RectTransform::from_to(
            Rect::from_min_size(Pos2::ZERO, response.rect.square_proportions()),
            response.rect,
        );
        let from_screen = to_screen.inverse();

        if self.lines.is_empty() {
            self.lines.push(vec![]);
        }

        let current_line = self.lines.last_mut().unwrap();

        if let Some(pointer_pos) = response.interact_pointer_pos() {
            let canvas_pos = from_screen * pointer_pos;
            if current_line.last() != Some(&canvas_pos) {
                current_line.push(canvas_pos);
                response.mark_changed();
            }
        } else if !current_line.is_empty() {
            self.lines.push(vec![]);
            response.mark_changed();
        }

        let shapes = self
            .lines
            .iter()
            .filter(|line| line.len() >= 2)
            .map(|line| {
                let points: Vec<Pos2> = line.iter().map(|p| to_screen * *p).collect();
                egui::Shape::line(points, self.stroke)
            });

        painter.extend(shapes);

        response
    }

    pub(crate) fn to_mnist(&self) -> [f32; 28 * 28] {
        let mut img = GrayImage::new(CANVAS_SIZE as _, CANVAS_SIZE as _);

        for line in &self.lines {
            if line.len() < 2 {
                continue;
            }
            for i in 0..(line.len() - 1) {
                let start = line[i];
                let end = line[i + 1];

                let x1 = (start.x * CANVAS_SIZE) as i32;
                let y1 = (start.y * CANVAS_SIZE) as i32;
                let x2 = (end.x * CANVAS_SIZE) as i32;
                let y2 = (end.y * CANVAS_SIZE) as i32;

                // interpolate between points and fill in pixels
                let dx = (x2 - x1).abs();
                let dy = (y2 - y1).abs();
                let sx = if x1 < x2 { 1 } else { -1 };
                let sy = if y1 < y2 { 1 } else { -1 };

                let mut err = if dx > dy { dx } else { -dy } / 2;
                let mut err2;

                let mut x = x1;
                let mut y = y1;

                let line_thiccness = 5;

                loop {
                    // add depth around the pixel
                    for i in -line_thiccness..=line_thiccness {
                        for j in -line_thiccness..=line_thiccness {
                            let nx = x + i;
                            let ny = y + j;
                            if nx >= 0
                                && nx < (CANVAS_SIZE as _)
                                && ny >= 0
                                && ny < (CANVAS_SIZE as _)
                            {
                                img.put_pixel(nx as _, ny as _, Luma([255]));
                            }
                        }
                    }
                    if x == x2 && y == y2 {
                        break;
                    }
                    err2 = err;
                    if err2 > -dx {
                        err -= dy;
                        x += sx;
                    }
                    if err2 < dy {
                        err += dx;
                        y += sy;
                    }
                }
            }
        }

        center(&img)
            .into_raw()
            .iter()
            .map(|&x| x as f32 / 255.0)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }
}

fn binarize_and_invert(image: &mut GrayImage, threshold: u8) {
    for pixel in image.pixels_mut() {
        *pixel = if pixel.0[0] > threshold {
            Luma([0u8])
        } else {
            Luma([255u8])
        };
    }
}

fn center_of_mass(img: &GrayImage) -> (f32, f32) {
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut count = 0.0;

    for (x, y, pixel) in img.enumerate_pixels() {
        if pixel.0[0] > 0 {
            sum_x += x as f32;
            sum_y += y as f32;
            count += 1.0;
        }
    }

    if count > 0.0 {
        (sum_x / count, sum_y / count)
    } else {
        (0.0, 0.0)
    }
}

fn center(img: &GrayImage) -> GrayImage {
    let mut img = resize(img, 20, 20, Triangle);

    let (com_x, com_y) = center_of_mass(&img);
    let (com_x, com_y) = (com_x.round() as i32, com_y.round() as i32);

    let mut centered_img = GrayImage::new(28, 28);

    let (center_x, center_y) = (14, 14);

    let (start_x, start_y) = (center_x - com_x, center_y - com_y);

    for (x, y, pixel) in img.enumerate_pixels() {
        let new_x = x as i32 + start_x;
        let new_y = y as i32 + start_y;
        if new_x >= 0
            && new_x < (centered_img.width() as _)
            && new_y >= 0
            && new_y < (centered_img.height() as _)
        {
            centered_img.put_pixel(new_x as _, new_y as _, *pixel);
        }
    }

    centered_img
}
