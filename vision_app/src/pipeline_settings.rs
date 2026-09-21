use std::path::PathBuf;

use glam::Mat4;
use ort::{
    ep::{DirectML, ExecutionProviderDispatch},
    execution_providers::TensorRTExecutionProvider,
};

use crate::modules::pointcloud::PointcloudGeneratorSettings;

#[derive(PartialEq, Eq, Clone, Copy)]
pub(crate) enum VideoSource {
    Dataset,
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub(crate) enum UsedBackend {
    Cpu,
    TensorRT,
    DirectML,
}

impl UsedBackend {
    pub(crate) fn get_execution_providers(self) -> Vec<ExecutionProviderDispatch> {
        match self {
            UsedBackend::Cpu => vec![],
            UsedBackend::TensorRT => vec![
                TensorRTExecutionProvider::default()
                    .with_engine_cache(true)
                    .build(),
            ],
            UsedBackend::DirectML => vec![DirectML::default().build()],
        }
    }
}

#[derive(Clone)]
pub(crate) struct PipelineSettings {
    pub(crate) video_source: VideoSource,

    pub(crate) dataset_left: PathBuf,
    pub(crate) dataset_right: PathBuf,

    pub(crate) camera_perspecrive_matrix: Mat4,
    pub(crate) pointcloud_generator: PointcloudGeneratorSettings,

    pub(crate) backend: UsedBackend,
}

impl Default for PipelineSettings {
    fn default() -> Self {
        Self {
            video_source: VideoSource::Dataset,

            dataset_left: "D:/dsec/interlaken_00_c_images_rectified_left/".into(),
            dataset_right: "D:/dsec/interlaken_00_c_images_rectified_right/".into(),

            camera_perspecrive_matrix: Mat4::from_cols_array_2d(&[
                [1.0, 0.0, 0.0, -713.5791168212891],
                [0.0, 1.0, 0.0, -570.9349365234375],
                [0.0, 0.0, 0.0, 1164.6238115833075],
                [0.0, 0.0, 1.9625989469856626, -0.0],
            ])
            .transpose(),
            pointcloud_generator: PointcloudGeneratorSettings {
                filter_depth_edges: false,
                filter_semantic: false,
            },

            backend: UsedBackend::TensorRT,
        }
    }
}
