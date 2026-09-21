use eyre::{Context, bail};
use image::{Rgb, RgbImage};
use ndarray::{Array2, Axis, Ix3, s};
use ort::{
    execution_providers::{ExecutionProvider, MIGraphXExecutionProvider},
    inputs,
    session::Session,
    value::TensorRef,
};
use tracing::info;

use crate::pipeline_settings::UsedBackend;

pub(crate) struct DisparityMap(pub(crate) Array2<f32>);

pub(crate) struct DepthModel {
    model: Session,
}

impl DepthModel {
    pub(crate) fn new(backend: UsedBackend) -> eyre::Result<Self> {
        info!("Creating Depth Model");
        let session = if true {
            Session::builder()?
                .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
                .with_execution_providers(backend.get_execution_providers())?
                .commit_from_memory(include_bytes!("../../weights/onnx/banet_1.onnx"))
                .wrap_err("Failed to build depth model")?
        } else {
            let mut builder = Session::builder()?
                .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?;
            MIGraphXExecutionProvider::default().register(&mut builder)?;
            builder
                .commit_from_memory(include_bytes!("../../weights/onnx/banet_1.onnx"))
                .wrap_err("Failed to build depth model")?
        };
        Ok(Self { model: session })
    }

    pub(crate) fn process(
        &mut self,
        input_imgs: (&RgbImage, &RgbImage),
    ) -> eyre::Result<DisparityMap> {
        fn pad_size(in_size: usize) -> usize {
            let divis_by = 32;
            let left = in_size % divis_by;
            if left == 0 {
                in_size
            } else {
                in_size + divis_by - left
            }
        }

        let (orig_width, orig_height) = (input_imgs.0.width(), input_imgs.0.height());
        if orig_width != input_imgs.1.width() || orig_height != input_imgs.1.height() {
            bail!("Size of the left image isn't the same as the size of the right image");
        }
        let mut input_left = ndarray::Array::zeros((
            1,
            3,
            pad_size(orig_height as usize),
            pad_size(orig_width as usize),
        ));
        let mut input_right = ndarray::Array::zeros((
            1,
            3,
            pad_size(orig_height as usize),
            pad_size(orig_width as usize),
        ));

        for pixel in input_imgs.0.enumerate_pixels() {
            let x = pixel.0 as _;
            let y = pixel.1 as _;
            let [r, g, b] = pixel.2.0;
            input_left[[0, 0, y, x]] = r as f32;
            input_left[[0, 1, y, x]] = g as f32;
            input_left[[0, 2, y, x]] = b as f32;
        }

        for pixel in input_imgs.1.enumerate_pixels() {
            let x = pixel.0 as _;
            let y = pixel.1 as _;
            let [r, g, b] = pixel.2.0;
            input_right[[0, 0, y, x]] = r as f32;
            input_right[[0, 1, y, x]] = g as f32;
            input_right[[0, 2, y, x]] = b as f32;
        }

        let outputs = self
            .model
            .run(inputs![
                "left_image" => TensorRef::from_array_view(&input_left)?,
                "right_image" => TensorRef::from_array_view(&input_right)?,
            ])
            .wrap_err("Failed to run the model")?;

        let depth = outputs[0]
            .try_extract_array::<f32>()
            .wrap_err("Failed to extract depth array")?;
        let depth = depth
            .index_axis(Axis(0), 0)
            .into_dimensionality::<Ix3>()
            .wrap_err("Failed to convert to 3d array")?
            .index_axis(Axis(0), 0)
            .slice(s![..orig_height as usize, ..orig_width as usize])
            .into_owned();

        Ok(DisparityMap(depth))
    }
}

pub(crate) struct DisparityUnreliability {
    unreliability_map: Array2<bool>,
    margin: usize,
}

impl DisparityUnreliability {
    pub(crate) fn new(input_depth: &DisparityMap) -> Self {
        let orig_shape = input_depth.0.raw_dim();
        let downsampled =
            Array2::from_shape_fn([orig_shape[0] / 2, orig_shape[1] / 2], |(x, y)| {
                let sum = input_depth
                    .0
                    .get([x * 2, y * 2])
                    .copied()
                    .unwrap_or_default()
                    + input_depth
                        .0
                        .get([x * 2, y * 2 + 1])
                        .copied()
                        .unwrap_or_default()
                    + input_depth
                        .0
                        .get([x * 2 + 1, y * 2])
                        .copied()
                        .unwrap_or_default()
                    + input_depth
                        .0
                        .get([x * 2 + 1, y * 2 + 1])
                        .copied()
                        .unwrap_or_default();
                sum / 4.0
            });
        let downsampled_shape = downsampled.raw_dim();
        let c = 2;
        let edges_dim = [downsampled_shape[0] - 2 * c, downsampled_shape[1] - 2 * c];
        let unreliability_map = Array2::from_shape_fn(edges_dim, |(x, y)| {
            let sum = downsampled[[x + c, y + c]] * 4.0
                - downsampled[[x + c, y]]
                - downsampled[[x, y + c]]
                - downsampled[[x + 2 * c, y + c]]
                - downsampled[[x + c, y + 2 * c]];
            sum * downsampled[[x + c, y + c]] > 100.0
        });

        Self {
            unreliability_map,
            margin: c,
        }
    }

    pub(crate) fn is_unreliable(&self, x: usize, y: usize) -> bool {
        let x = x / 2;
        let y = y / 2;
        let dim = self.unreliability_map.raw_dim();
        if x < self.margin
            || y < self.margin
            || x - self.margin >= dim[0]
            || y - self.margin >= dim[1]
        {
            true
        } else {
            self.unreliability_map[[x - self.margin, y - self.margin]]
        }
    }
}

pub(crate) fn disparity_to_image(res: &Array2<f32>) -> image::ImageBuffer<Rgb<u8>, Vec<u8>> {
    RgbImage::from_fn(res.shape()[1] as u32, res.shape()[0] as u32, |x, y| {
        let disp = f32::clamp(res[(y as usize, x as usize)] * 1.0, 0.0, 255.0) as u8;
        Rgb([disp, disp, disp])
    })
}

#[cfg(test)]
mod test {
    use eyre::Context;

    use crate::models::depth::disparity_to_image;

    use super::DepthModel;

    #[test]
    fn test_depth() -> eyre::Result<()> {
        let mut model = DepthModel::new(crate::pipeline_settings::UsedBackend::Cpu)?;
        // let _res = model.process((&image::open("/home/quant/datasets/drivingstereo/rainy/left-image-half-size/2018-08-17-09-45-58_2018-08-17-10-13-11-340.jpg")?.into_rgb8(), &image::open("/home/quant/datasets/drivingstereo/rainy/right-image-half-size/2018-08-17-09-45-58_2018-08-17-10-13-11-340.jpg")?.into_rgb8())).wrap_err("Failed to process the first pair")?;

        let left = image::open(
            "/home/quant/datasets/dsec/interlaken_00_c_images_rectified_left/000172.png",
        )?
        .into_rgb8();
        let right = image::open(
            "/home/quant/datasets/dsec/interlaken_00_c_images_rectified_right/000172.png",
        )?
        .into_rgb8();

        let res = model
            .process((&left, &right))
            .wrap_err("Failed to process the second pair")?;
        disparity_to_image(&res.0).save("target/disp-img.png")?;
        // array_to_image(&depth_find_unreliable(&res)).save("target/disp-img-unreliable.png")?;

        Ok(())
    }
}
