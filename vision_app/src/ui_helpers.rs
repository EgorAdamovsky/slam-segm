use std::sync::Arc;

use eframe::egui::{
    self, Align2, Color32, ColorImage, DragValue, FontId, Image, Pos2, Rect, Spinner, Stroke,
    TextureHandle, TextureOptions, Ui, Vec2, vec2,
};
use glam::Mat4;
use image::RgbImage;

pub(crate) fn show_mat_editor(ui: &mut Ui, mat: &mut Mat4) {
    ui.vertical(|ui| {
        for row in [
            &mut mat.x_axis,
            &mut mat.y_axis,
            &mut mat.z_axis,
            &mut mat.w_axis,
        ] {
            ui.horizontal(|ui| {
                ui.add_sized([80.0, 20.0], DragValue::new(&mut row.x));
                ui.add_sized([80.0, 20.0], DragValue::new(&mut row.y));
                ui.add_sized([80.0, 20.0], DragValue::new(&mut row.z));
                ui.add_sized([80.0, 20.0], DragValue::new(&mut row.w));
            });
        }
    });
}

struct TextAnnotation {
    x: f32,
    y: f32,
    text: String,
}

struct RectAnnotation {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    color: Color32,
    fill_color: Color32,
}

#[derive(Default)]
pub(crate) struct ImageAnnotations {
    text: Vec<TextAnnotation>,
    rects: Vec<RectAnnotation>,
}

impl ImageAnnotations {
    pub(crate) fn add_text(&mut self, x: f32, y: f32, text: &str) {
        self.text.push(TextAnnotation {
            x,
            y,
            text: text.to_string(),
        });
    }

    pub(crate) fn add_rect(&mut self, x: f32, y: f32, w: f32, h: f32) {
        self.rects.push(RectAnnotation {
            x,
            y,
            w,
            h,
            color: Color32::WHITE,
            fill_color: Color32::TRANSPARENT,
        })
    }

    pub(crate) fn add_filled_rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: Color32) {
        self.rects.push(RectAnnotation {
            x,
            y,
            w,
            h,
            color,
            fill_color: color,
        })
    }
}

pub(crate) struct VideoPlayer {
    ctx: egui::Context,
    texture: Option<TextureHandle>,
    annotations: Option<Arc<ImageAnnotations>>,
}

impl VideoPlayer {
    pub(crate) fn new(ctx: &egui::Context) -> Self {
        Self {
            ctx: ctx.clone(),
            texture: None,
            annotations: None,
        }
    }

    pub(crate) fn reset(&mut self) {
        self.texture = None;
        self.annotations = None;
    }

    pub(crate) fn set_image(&mut self, image: &RgbImage) {
        let image = ColorImage::from_rgb(
            [
                image.width().try_into().unwrap(),
                image.height().try_into().unwrap(),
            ],
            image,
        );
        self.texture = Some(
            self.ctx
                .load_texture("video_frame", image, TextureOptions::LINEAR),
        );
    }

    pub(crate) fn set_annotations(&mut self, annotations: Arc<ImageAnnotations>) {
        self.annotations = Some(annotations);
    }

    fn show_annotations(&self, ui: &mut Ui, rect: Rect) {
        if let Some(annotations) = &self.annotations {
            let painter = ui.painter_at(rect);
            let bx = rect.left();
            let by = rect.top();
            let w = rect.width();
            let h = rect.height();
            let font_size = w / 100.0 * 1.0;
            for text in &annotations.text {
                painter.text(
                    Pos2::new(bx + text.x * w, by + text.y * h),
                    Align2::CENTER_CENTER,
                    &text.text,
                    FontId::proportional(font_size),
                    Color32::GREEN,
                );
            }
            for rect in &annotations.rects {
                painter.rect(
                    Rect::from_min_size(
                        Pos2::new(bx + rect.x * w, by + rect.y * h),
                        vec2(rect.w * w, rect.h * h),
                    ),
                    0.0,
                    rect.fill_color,
                    Stroke::new(1.0, rect.color),
                    egui::StrokeKind::Middle,
                );
            }
        }
    }

    pub(crate) fn show(&self, ui: &mut Ui) {
        self.show2(ui, true, false);
    }

    pub(crate) fn show2(&self, ui: &mut Ui, add_hover_ui: bool, stretch_to_fill: bool) {
        if let Some(texture) = &self.texture {
            let image = if stretch_to_fill {
                Image::new(texture).fit_to_exact_size(ui.available_size())
            } else {
                Image::new(texture).max_size(ui.available_size())
            };
            let resp = ui.add(image.clone());
            let rect = image_rect(ui, &image, resp.rect);
            self.show_annotations(ui, rect);
            if add_hover_ui {
                resp.on_hover_ui(|ui| {
                    let okay_size = ui.ctx().content_rect().size() * 0.75;
                    let resp = ui.add(Image::new(texture).max_size(okay_size));
                    self.show_annotations(ui, resp.rect);
                });
            }
        } else {
            let side = ui.available_size().min_elem();
            ui.add_sized((side, side), Spinner::new());
        }
    }
}

/// Возвращает Rect для области, которую будет занимать изображение.
fn image_rect(ui: &mut Ui, image: &Image<'_>, response_rect: Rect) -> Rect {
    let image_size = image
        .load_and_calc_size(ui, ui.available_size())
        .unwrap_or(Vec2::splat(24.0));
    Rect::from_center_size(response_rect.center(), image_size)
}
