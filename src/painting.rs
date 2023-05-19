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

    // fn interpolate_line(&self, p1: Pos2, p2: Pos2) -> Vec<Pos2> {
    //     let dx = (p2.x - p1.x).abs();
    //     let dy = (p2.y - p1.y).abs();
    //
    //     let sx = if p1.x < p2.x { 1.0 } else { -1.0 };
    //     let sy = if p1.y < p2.y { 1.0 } else { -1.0 };
    //
    //     let mut err = dx - dy;
    //     let mut x = p1.x;
    //     let mut y = p1.y;
    //
    //     let mut points = vec![];
    //
    //     while (x.floor() != p2.x.floor() || y.floor() != p2.y.floor()) && (x >= p1.x && y >= p1.y) {
    //         points.push(Pos2::new(x, y));
    //
    //         let e2 = 2.0 * err;
    //
    //         if e2 > -dy {
    //             err -= dy;
    //             x += sx;
    //         }
    //         if e2 < dx {
    //             err += dx;
    //             y += sy;
    //         }
    //     }
    //
    //     points.push(Pos2::new(x, y)); // Include the last point
    //
    //     points
    // }

    fn interpolate_line(&self, p1: Pos2, p2: Pos2) -> Vec<Pos2> {
        let dx = p2.x - p1.x;
        let dy = p2.y - p1.y;

        let distance = dx.abs().max(dy.abs());

        let dx_normalized = dx / distance;
        let dy_normalized = dy / distance;

        let mut x = p1.x;
        let mut y = p1.y;

        let mut points = vec![];

        for _ in 0..=distance.round() as i32 {
            points.push(Pos2::new(x, y));

            x += dx_normalized;
            y += dy_normalized;
        }

        points
    }

    pub fn to_mnist(&self) -> [f32; 28 * 28] {
        let mut img = [[0.0; 28]; 28];

        // Convert lines to image matrix
        for line in &self.lines {
            for i in 1..line.len() {
                let p1 = line[i - 1];
                let p2 = line[i];
                let points = self.interpolate_line(p1, p2);
                for point in points {
                    let x = (point.x * 28.0) as usize;
                    let y = (point.y * 28.0) as usize;
                    if x < 28 && y < 28 {
                        img[y][x] = 1.0;
                    }
                }
            }
        }

        // Flatten the 2D image matrix into a 1D array
        let mut flat_img = [0.0; 28 * 28];
        for (i, row) in img.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                flat_img[i * 28 + j] = value;
            }
        }

        flat_img
    }
}
