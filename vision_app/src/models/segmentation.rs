use std::fmt::Display;

use eframe::egui::Color32;
use eyre::Context;
use image::{
    RgbImage,
    imageops::{FilterType, resize},
};
use ndarray::{Array2, Axis, Ix3, Ix4};
use ort::{
    execution_providers::{ExecutionProvider as _, MIGraphXExecutionProvider},
    inputs,
    session::Session,
    value::TensorRef,
};
use tracing::info;

use crate::{pipeline_settings::UsedBackend, ui_helpers::ImageAnnotations};

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Получает индекс класса с наибольшей уверенностью, вовзращает индекс и уверенность.
fn find_most_confident_class_in_row(row: ndarray::ArrayView1<f32>) -> (usize, f32) {
    let max = row.iter().copied().max_by(f32::total_cmp).unwrap();
    let max_pos = row.iter().position(|x| *x == max).unwrap();
    (max_pos, sigmoid(max))
}

pub(crate) struct BBox {
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
}

pub(crate) struct SegmentationDetection {
    class: SegmentationClass,
    #[expect(unused)]
    class_raw: usize,
    confidence: f32,
    bbox: BBox,
    mask: Array2<bool>,
}

pub(crate) struct ObjectInfo {
    pub(crate) class: SegmentationClass,
    pub(crate) confidence: f32,
}

impl ObjectInfo {
    pub(crate) fn dummy() -> Self {
        ObjectInfo {
            class: SegmentationClass::Other,
            confidence: 1.0,
        }
    }
}

#[derive(PartialEq, Eq)]
pub(crate) struct SegmentationObjectAt {
    pub(crate) object_id: u32,
}

pub(crate) struct SegmentationResult {
    orig_width: u32,
    orig_height: u32,
    detections: Vec<SegmentationDetection>,
}

impl SegmentationResult {
    pub(crate) fn object_infos(&self) -> impl Iterator<Item = ObjectInfo> {
        self.detections.iter().map(|det| ObjectInfo {
            class: det.class,
            confidence: det.confidence,
        })
    }

    pub(crate) fn object_at_pixel_coords(
        &self,
        orig_x: f32,
        orig_y: f32,
    ) -> Option<SegmentationObjectAt> {
        let width_m = SegmentationModel::MASK_SIDE_SIZE as f32 / self.orig_width as f32;
        let height_m = SegmentationModel::MASK_SIDE_SIZE as f32 / self.orig_height as f32;
        let x = (orig_x * width_m).floor() as usize;
        let y = (orig_y * height_m).floor() as usize;
        for (object_id, det) in self.detections.iter().enumerate() {
            if det.mask.get([y, x]).copied()? {
                return Some(SegmentationObjectAt {
                    object_id: (object_id as u32),
                });
            }
        }
        None
    }

    pub(crate) fn total_objects(&self) -> u32 {
        u32::try_from(self.detections.len()).unwrap()
    }

    pub(crate) fn as_annotations(&self) -> ImageAnnotations {
        let mut annotations = ImageAnnotations::default();

        for detection in &self.detections {
            if detection.class != SegmentationClass::Other {
                let bbox = &detection.bbox;
                annotations.add_text(
                    bbox.cx,
                    bbox.cy,
                    &format!("{} {:.2}", detection.class, detection.confidence),
                );
                annotations.add_rect(
                    bbox.cx - bbox.w / 2.0,
                    bbox.cy - bbox.h / 2.0,
                    bbox.w,
                    bbox.h,
                );
                let mask = &detection.mask;
                let mask_side = mask.raw_dim()[0] as f32;
                if detection.class != SegmentationClass::Road {
                    for ((y, x), pixel) in mask.indexed_iter() {
                        if *pixel {
                            annotations.add_filled_rect(
                                x as f32 / mask_side,
                                y as f32 / mask_side,
                                1.0 / mask_side,
                                1.0 / mask_side,
                                Color32::from_white_alpha(50),
                            );
                        }
                    }
                }
            }
        }

        annotations
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SegmentationClass {
    Road,
    Person,
    Car,
    Other,
}

impl Display for SegmentationClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            SegmentationClass::Road => "Дорога",
            SegmentationClass::Person => "Человек",
            SegmentationClass::Car => "Автомобиль",
            SegmentationClass::Other => "Другое",
        };
        write!(f, "{}", label)
    }
}

impl SegmentationClass {
    fn from_raw(val: usize) -> Self {
        match val {
            0 => Self::Road,
            3 | 4 => Self::Person,
            5 | 7 | 8 => Self::Car,
            _ => Self::Other,
        }
    }
}

pub(crate) struct SegmentationModel {
    model: Session,
}

// {1: 'road', 2: 'sidewalk', 3: 'parking railtrack', 4: 'person', 5: 'rider', 6: 'car', 7: 'truck', 8: 'bus', 9: 'on rails', 10: 'motorcycle', 11: 'bicycle', 12: 'caravan', 13: 'trailer', 14: 'building', 15: 'wall', 16: 'fence', 17: 'guard rail', 18: 'bridge', 19: 'tunnel', 20: 'pole', 21: 'pole group', 22: 'traffic sign', 23: 'traffic light', 24: 'vegetation', 25: 'terrain', 26: 'sky', 27: 'ground', 28: 'dynamic', 29: 'static'}
impl SegmentationModel {
    const SIDE_SIZE: usize = 432;
    const MASK_SIDE_SIZE: usize = 108;

    pub(crate) fn new(backend: UsedBackend) -> eyre::Result<Self> {
        info!("Creating Segmentation Model");
        let session = if true {
            Session::builder()?
                .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
                .with_execution_providers(backend.get_execution_providers())?
                .commit_from_memory(include_bytes!("../../weights/onnx/rf-detr.onnx"))
                .wrap_err("Failed to build segmentation model")?
        } else {
            let mut builder = Session::builder()?
                .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?;
            MIGraphXExecutionProvider::default().register(&mut builder)?;
            builder
                .commit_from_memory(include_bytes!("../../weights/onnx/rf-detr.onnx"))
                .wrap_err("Failed to build segmentation model")?
        };

        Ok(Self { model: session })
    }

    pub(crate) fn process(
        &mut self,
        input_imgs: Vec<&RgbImage>,
    ) -> eyre::Result<Vec<SegmentationResult>> {
        let mut input =
            ndarray::Array::zeros((input_imgs.len(), 3, Self::SIDE_SIZE, Self::SIDE_SIZE));
        for original_img in &input_imgs {
            let img = resize(
                *original_img,
                Self::SIDE_SIZE as _,
                Self::SIDE_SIZE as _,
                FilterType::CatmullRom,
            );
            for pixel in img.enumerate_pixels() {
                let x = pixel.0 as _;
                let y = pixel.1 as _;
                let [r, g, b] = pixel.2.0;
                input[[0, 0, y, x]] = (r as f32) / 255.;
                input[[0, 1, y, x]] = (g as f32) / 255.;
                input[[0, 2, y, x]] = (b as f32) / 255.;
            }
        }
        let outputs = self
            .model
            .run(inputs!["input" => TensorRef::from_array_view(&input)?])?;
        let dets = outputs["dets"]
            .try_extract_array::<f32>()?
            .into_dimensionality::<Ix3>()?; // batch, dets, bbox
        let labels = outputs["labels"]
            .try_extract_array::<f32>()?
            .into_dimensionality::<Ix3>()?; // batch, dets, classes
        let masks = outputs[2]
            .try_extract_array::<f32>()?
            .into_dimensionality::<Ix4>()?; // batch, dets, mask (108x108)

        let min_conf = 0.4;

        let result = input_imgs
            .iter()
            .enumerate()
            .map(|(batch_id, original_img)| {
                let (orig_width, orig_height) = (original_img.width(), original_img.height());
                let dets = dets.index_axis(Axis(0), batch_id);
                let labels = labels.index_axis(Axis(0), batch_id);
                let masks = masks.index_axis(Axis(0), batch_id);

                let detections = labels
                    .outer_iter()
                    .map(find_most_confident_class_in_row)
                    .enumerate()
                    .filter(|(_index, (_cls, confidence))| *confidence > min_conf)
                    .map(|(index, (class, confidence))| {
                        let [cx, cy, w, h] = *dets.index_axis(Axis(0), index).as_slice().unwrap()
                        else {
                            panic!("Expected exactly 4 values for a bbox")
                        };
                        let bbox = BBox { cx, cy, w, h };
                        let mask = masks.index_axis(Axis(0), index).map(|x| *x > 0.0);

                        SegmentationDetection {
                            class: SegmentationClass::from_raw(class),
                            class_raw: class,
                            confidence,
                            bbox,
                            mask,
                        }
                    })
                    .collect::<Vec<_>>();
                SegmentationResult {
                    orig_width,
                    orig_height,
                    detections,
                }
            })
            .collect::<Vec<_>>();
        Ok(result)
    }
}

#[cfg(test)]
mod test {
    use super::SegmentationModel;

    #[test]
    fn test_segmentation() -> eyre::Result<()> {
        let mut model = SegmentationModel::new(crate::pipeline_settings::UsedBackend::Cpu)?;
        let _res = model.process(vec![&image::open("/home/quant/datasets/drivingstereo/rainy/left-image-half-size/2018-08-17-09-45-58_2018-08-17-10-13-11-340.jpg")?.into_rgb8()])?;
        dbg!(_res[0].detections[0].mask.shape());
        // dbg!(res.dets.shape());
        // dbg!(res.labels.shape());
        Ok(())
    }
}
