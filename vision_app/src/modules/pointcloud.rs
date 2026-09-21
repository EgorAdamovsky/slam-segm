use std::{iter, num::NonZeroU32, sync::Arc};

use glam::{Mat4, Vec3, Vec3Swizzles as _};
use ordered_float::OrderedFloat;

use crate::models::{
    depth::{DisparityMap, DisparityUnreliability},
    segmentation::{ObjectInfo, SegmentationResult},
};
use node_plumbing::Module;

pub(crate) struct CameraCalibration {
    pub(crate) camera_matrices: Vec<Mat4>,
}

pub(crate) struct Point {
    pub(crate) position: Vec3,
    object_id: u32,
}

pub(crate) struct Pointcloud {
    points: Vec<Point>,
    object_infos: Vec<ObjectInfo>,
    object_point_indexes: Vec<Vec<usize>>,
}

pub(crate) struct ObjectData<'a> {
    pointcloud: &'a Pointcloud,
    pub(crate) info: &'a ObjectInfo,
    pub(crate) point_indexes: &'a [usize],
}

impl ObjectData<'_> {
    pub(crate) fn iter_points(&self) -> impl Iterator<Item = &Point> {
        self.point_indexes
            .iter()
            .copied()
            .map(|index| &self.pointcloud.points[index])
    }

    pub(crate) fn point_count(&self) -> usize {
        self.point_indexes.len()
    }

    pub(crate) fn calculate_center(&self) -> Vec3 {
        let total_point: Vec3 = self.iter_points().map(|p| p.position).sum();
        total_point / (self.point_count() as f32)
    }

    pub(crate) fn furtherst_from_2d(&self, point: Vec3) -> &Point {
        self.iter_points()
            .max_by_key(|x| OrderedFloat(x.position.xz().distance_squared(point.xz())))
            .unwrap()
    }

    pub(crate) fn extents(&self) -> (&Point, &Point) {
        let point = self.iter_points().next().unwrap();
        let ext1 = self.furtherst_from_2d(point.position);
        let ext2 = self.furtherst_from_2d(ext1.position);
        (ext1, ext2)
    }

    pub(crate) fn closest_to_zero(&self) -> Vec3 {
        self.iter_points()
            .min_by_key(|x| OrderedFloat(x.position.length_squared()))
            .unwrap()
            .position
    }
}

impl Pointcloud {
    #[expect(unused)]
    pub(crate) fn object_info(&self, object_id: NonZeroU32) -> &ObjectInfo {
        &self.object_infos[(u32::from(object_id)) as usize]
    }

    #[expect(unused)]
    pub(crate) fn iter_points(&self) -> impl Iterator<Item = &Point> {
        self.points.iter()
    }

    pub(crate) fn iter_objects(&self) -> impl Iterator<Item = ObjectData<'_>> {
        self.object_infos
            .iter()
            .zip(self.object_point_indexes.iter())
            .map(|(info, indexes)| ObjectData {
                pointcloud: self,
                info,
                point_indexes: indexes.as_slice(),
            })
            .filter(|obj| !obj.point_indexes.is_empty())
    }
}

#[derive(Clone, Copy)]
pub(crate) struct PointcloudGeneratorSettings {
    pub(crate) filter_depth_edges: bool,
    pub(crate) filter_semantic: bool,
}

pub(crate) struct PointcloudGeneratorModule {
    calibration: Arc<CameraCalibration>,
    settings: PointcloudGeneratorSettings,
}

impl PointcloudGeneratorModule {
    pub(crate) fn new(
        calibration: Arc<CameraCalibration>,
        settings: PointcloudGeneratorSettings,
    ) -> Self {
        Self {
            calibration,
            settings,
        }
    }
}

impl Module for PointcloudGeneratorModule {
    type In = (Arc<Vec<DisparityMap>>, Arc<Vec<SegmentationResult>>);

    type Out = Arc<Pointcloud>;

    fn process(&mut self, input: Self::In) -> eyre::Result<Self::Out> {
        let disparities = &input.0;
        let segmentations = &input.1;

        let mut object_id_offset = 1;

        let object_infos = iter::once_with(ObjectInfo::dummy)
            .chain(segmentations.iter().flat_map(|segm| segm.object_infos()))
            .collect::<Vec<_>>();

        let filter_semantic = self.settings.filter_semantic;
        let filter_depth_edges = self.settings.filter_depth_edges;
        let points = disparities
            .iter()
            .zip(segmentations.iter())
            .zip(self.calibration.camera_matrices.iter())
            .flat_map(|((disparity_map, segmentation_result), &camera_matrix)| {
                let unreliability = DisparityUnreliability::new(disparity_map);
                let points = disparity_map
                    .0
                    .indexed_iter()
                    .filter_map(move |((y, x), v)| {
                        let point_camera_space = Vec3::new(x as f32, y as f32, *v);
                        let point_world_space = camera_matrix.project_point3(point_camera_space);
                        let segm_obj =
                            segmentation_result.object_at_pixel_coords(x as f32, y as f32);
                        let d = 6.0;
                        if filter_semantic
                            && (segm_obj
                                != segmentation_result
                                    .object_at_pixel_coords(x as f32 + d, y as f32)
                                || segm_obj
                                    != segmentation_result
                                        .object_at_pixel_coords(x as f32 - d, y as f32)
                                || segm_obj
                                    != segmentation_result
                                        .object_at_pixel_coords(x as f32, y as f32 - d)
                                || segm_obj
                                    != segmentation_result
                                        .object_at_pixel_coords(x as f32, y as f32 + d))
                        {
                            return None;
                        };
                        let object_id = segm_obj
                            .as_ref()
                            .map(|seg| seg.object_id.saturating_add(object_id_offset))
                            .unwrap_or(0);
                        if !filter_depth_edges || !unreliability.is_unreliable(y, x) {
                            Some(Point {
                                position: point_world_space,
                                object_id,
                            })
                        } else {
                            None
                        }
                    });
                object_id_offset += segmentation_result.total_objects();
                points
            })
            .collect::<Vec<_>>();
        // opencv::calib3d::reproject_image_to_3d(disparity, 3d_image, q, handle_missing_values, ddepth)

        // TODO join same objects across different cameras

        let mut object_point_indexes: Vec<Vec<usize>> = Vec::new();
        object_point_indexes.resize_with(object_infos.len() + 1, Default::default);
        for (i, point) in points.iter().enumerate() {
            object_point_indexes[point.object_id as usize].push(i);
        }

        Ok(Pointcloud {
            points,
            object_infos,
            object_point_indexes,
        }
        .into())
    }
}
