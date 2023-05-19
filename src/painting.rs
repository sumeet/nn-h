use eframe::egui;
use eframe::egui::*;

#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct Painting {
    /// in 0-1 normalized coordinates
    lines: Vec<Vec<Pos2>>,
    stroke: Stroke,
    pub submit: bool,
}

impl Painting {
    pub fn ui(&mut self, ui: &mut Ui) {
        self.ui_control(ui);
        ui.label("Paint with your mouse/touch!");
        // Set the canvas size to 28x28 pixels
        let canvas_size = Vec2::new(28.0, 28.0);
        let scroll_area = ScrollArea::neither().max_height(28.).max_width(28.);
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
            stroke: Stroke::new(1.0, Color32::from_rgb(255, 255, 255)),
            submit: false,
        }
    }
    pub fn ui_control(&mut self, ui: &mut egui::Ui) -> egui::Response {
        ui.horizontal(|ui| {
            egui::stroke_ui(ui, &mut self.stroke, "Stroke");
            ui.separator();
            if ui.button("Clear Painting").clicked() {
                self.lines.clear();
            }
            if ui.button("Submit").clicked() {
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

    pub fn to_mnist(&self) -> [f32; 28 * 28] {
        let mut image = [0.0; 28 * 28];

        for line in &self.lines {
            if line.len() < 2 {
                continue;
            }

            for i in 0..(line.len() - 1) {
                let start = line[i];
                let end = line[i + 1];

                let x1 = (start.x * 28.0) as i32;
                let y1 = (start.y * 28.0) as i32;
                let x2 = (end.x * 28.0) as i32;
                let y2 = (end.y * 28.0) as i32;

                // interpolate between points and fill in pixels
                let dx = (x2 - x1).abs();
                let dy = (y2 - y1).abs();
                let sx = if x1 < x2 { 1 } else { -1 };
                let sy = if y1 < y2 { 1 } else { -1 };

                let mut err = if dx > dy { dx } else { -dy } / 2;
                let mut err2;

                let mut x = x1;
                let mut y = y1;

                loop {
                    // set the pixel to 1.0 (white)
                    if x >= 0 && x < 28 && y >= 0 && y < 28 {
                        image[x as usize + y as usize * 28] = 1.0;

                        // thicken the line
                        for i in -1..=1 {
                            for j in -1..=1 {
                                let nx = x + i;
                                let ny = y + j;

                                if nx >= 0 && nx < 28 && ny >= 0 && ny < 28 {
                                    image[nx as usize + ny as usize * 28] = 1.0;
                                }
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

        // Return the final mnist-style image
        image
    }
}
