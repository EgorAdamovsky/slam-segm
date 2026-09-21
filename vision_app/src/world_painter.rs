use eframe::egui::{Align2, Color32, FontId, Painter, Pos2, Stroke, Ui};
use glam::Vec2;

pub(crate) struct WorldPainter {
    painter: Painter,
    scale: f32,
}

impl WorldPainter {
    pub(crate) fn new(ui: &Ui, scale: f32) -> Self {
        let painter = ui.painter().with_clip_rect(ui.available_rect_before_wrap());
        let min_side = painter.clip_rect().size().min_elem();
        let scale = min_side / scale;
        Self { painter, scale }
    }

    fn map_position(&self, pos: Vec2) -> Pos2 {
        let clip_rect = self.painter.clip_rect();
        Pos2::new(
            pos.x * self.scale + clip_rect.center().x,
            -pos.y * self.scale + clip_rect.center().y,
        )
    }

    pub(crate) fn draw_grid(&self) {
        let color = Color32::from_gray(50);
        let rect = self.painter.clip_rect();
        let center = rect.center();
        let grid_size = 10.0;
        let lines_x = f32::ceil(rect.width() / (self.scale * grid_size) / 2.0) as usize;
        let lines_y = f32::ceil(rect.height() / (self.scale * grid_size) / 2.0) as usize;
        self.painter.line_segment(
            [
                Pos2::new(center.x, rect.min.y),
                Pos2::new(center.x, rect.max.y),
            ],
            Stroke::new(1.0, color),
        );
        for i in 1..lines_x {
            self.painter.line_segment(
                [
                    Pos2::new(center.x + self.scale * grid_size * (i as f32), rect.min.y),
                    Pos2::new(center.x + self.scale * grid_size * (i as f32), rect.max.y),
                ],
                Stroke::new(1.0, color),
            );
            self.painter.line_segment(
                [
                    Pos2::new(center.x - self.scale * grid_size * (i as f32), rect.min.y),
                    Pos2::new(center.x - self.scale * grid_size * (i as f32), rect.max.y),
                ],
                Stroke::new(1.0, color),
            );
        }
        for i in 1..lines_y {
            self.painter.line_segment(
                [
                    Pos2::new(rect.min.x, center.y + self.scale * grid_size * (i as f32)),
                    Pos2::new(rect.max.x, center.y + self.scale * grid_size * (i as f32)),
                ],
                Stroke::new(1.0, color),
            );
            self.painter.line_segment(
                [
                    Pos2::new(rect.min.x, center.y - self.scale * grid_size * (i as f32)),
                    Pos2::new(rect.max.x, center.y - self.scale * grid_size * (i as f32)),
                ],
                Stroke::new(1.0, color),
            );
        }
        self.painter.line_segment(
            [
                Pos2::new(rect.min.x, center.y),
                Pos2::new(rect.max.x, center.y),
            ],
            Stroke::new(1.0, color),
        );
    }

    pub(crate) fn circle_filled(&self, pos: Vec2, radius: f32, fill_color: eframe::egui::Color32) {
        let pos = self.map_position(pos);
        self.painter.circle_filled(pos, radius, fill_color);
    }

    pub(crate) fn label(&self, pos: Vec2, text: &str, text_color: eframe::egui::Color32) {
        let pos = self.map_position(pos);
        self.painter.text(
            pos,
            Align2::LEFT_BOTTOM,
            text,
            FontId::default(),
            text_color,
        );
    }
}
