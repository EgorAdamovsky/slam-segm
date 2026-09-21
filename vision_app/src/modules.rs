use std::sync::Arc;

use crate::{
    models::{
        depth::{DepthModel, DisparityMap, disparity_to_image},
        segmentation::{SegmentationModel, SegmentationResult},
    },
    modules::{
        object_detector::{ObjectExtractor, ObjectExtractorResult},
        pointcloud::{CameraCalibration, PointcloudGeneratorModule},
    },
    pipeline_settings::PipelineSettings,
    ui_helpers::ImageAnnotations,
};
use image::RgbImage;
use node_plumbing::{InputChannel, Module, Output, Pipeline};
use tracing::info;

pub(crate) mod object_detector;
pub(crate) mod pointcloud;

pub(crate) struct SensorSnapshot {
    pub(crate) stereoimages: Vec<(RgbImage, RgbImage)>,
}

pub(crate) struct SensorSource {
    index: usize,
    left_path: std::path::PathBuf,
    right_path: std::path::PathBuf,
}

impl Module for SensorSource {
    type In = ();

    type Out = Arc<SensorSnapshot>;

    fn process(&mut self, _input: Self::In) -> eyre::Result<Self::Out> {
        let index = self.index;
        self.index += 2;

        // let left = image::open(format!(
        //     "/home/quant/datasets/dsec/zurich_city_03_a_images_rectified_left/{index:06}.png"
        // ))?
        // .into_rgb8();
        // let right = image::open(format!(
        //     "/home/quant/datasets/dsec/zurich_city_03_a_images_rectified_right/{index:06}.png"
        // ))?
        // .into_rgb8();
        let left = image::open(self.left_path.join(format!("{index:06}.png")))?.into_rgb8();
        let right = image::open(self.right_path.join(format!("{index:06}.png")))?.into_rgb8();

        Ok(SensorSnapshot {
            stereoimages: vec![(left, right)],
        }
        .into())
    }
}

impl Module for DepthModel {
    type In = Arc<SensorSnapshot>;

    type Out = Arc<Vec<DisparityMap>>;

    fn process(&mut self, input: Self::In) -> eyre::Result<Self::Out> {
        info!("Starting depth estimation...");
        let disparity_maps = input
            .stereoimages
            .iter()
            .map(|stereoimage| self.process((&stereoimage.0, &stereoimage.1)))
            .collect::<eyre::Result<Vec<_>>>()?;
        info!("Complete depth");
        Ok(disparity_maps.into())
    }
}

impl Module for SegmentationModel {
    type In = Arc<SensorSnapshot>;

    type Out = Arc<Vec<SegmentationResult>>;

    fn process(&mut self, input: Self::In) -> eyre::Result<Self::Out> {
        let input_imgs = input.stereoimages.iter().map(|x| &x.0).collect::<Vec<_>>();
        info!("Starting segm...");
        let ret = self.process(input_imgs)?;
        info!("Complete segm");
        Ok(ret.into())
    }
}

pub(crate) struct DisplayResult {
    pub(crate) sensor_data: Arc<SensorSnapshot>,
    pub(crate) objects: Arc<ObjectExtractorResult>,
    pub(crate) disparity_image: RgbImage,
    pub(crate) segm_annotations: Arc<ImageAnnotations>,
}

struct DisplayResultCollector;

impl Module for DisplayResultCollector {
    type In = (
        Arc<SensorSnapshot>,
        Arc<ObjectExtractorResult>,
        Arc<Vec<DisparityMap>>,
        Arc<Vec<SegmentationResult>>,
    );
    type Out = Arc<DisplayResult>;

    fn process(&mut self, input: Self::In) -> eyre::Result<Self::Out> {
        Ok(DisplayResult {
            sensor_data: input.0.clone(),
            objects: input.1.clone(),
            disparity_image: disparity_to_image(&input.2.first().unwrap().0),
            segm_annotations: Arc::new(input.3.first().unwrap().as_annotations()),
        }
        .into())
    }
}

pub(crate) fn build_pipeline(
    settings: PipelineSettings,
) -> eyre::Result<InputChannel<Arc<DisplayResult>>> {
    let calibration = Arc::new(CameraCalibration {
        camera_matrices: vec![settings.camera_perspecrive_matrix],
    });

    let pipeline = Pipeline::new();

    let pipeline_out = {
        let mut source = pipeline.source_node(SensorSource {
            index: 0,
            left_path: settings.dataset_left,
            right_path: settings.dataset_right,
        });
        let mut depth = pipeline.node(DepthModel::new(settings.backend)?, source.output());
        let mut segm = pipeline.node(SegmentationModel::new(settings.backend)?, source.output());
        let mut pointcloud = pipeline.node(
            PointcloudGeneratorModule::new(calibration, settings.pointcloud_generator),
            &mut (depth.output(), segm.output()),
        );
        let mut object_extractor = pipeline.node(ObjectExtractor, pointcloud.output());
        let mut display_collector = pipeline.node(
            DisplayResultCollector,
            &mut (
                source.output(),
                object_extractor.output(),
                depth.output(),
                segm.output(),
            ),
        );

        display_collector.output().linked_input()
    };

    pipeline.start();

    Ok(pipeline_out)
}

#[cfg(test)]
mod test {
    use crate::{modules::build_pipeline, pipeline_settings::PipelineSettings};

    #[test]
    fn test_pipeline() -> eyre::Result<()> {
        build_pipeline(PipelineSettings::default())?;
        Ok(())
    }
}
