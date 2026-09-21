use std::sync::Arc;

use crate::{models::segmentation::SegmentationClass, modules::pointcloud::Pointcloud};
use glam::Vec3;
use node_plumbing::Module;

pub(crate) struct Vehicle {
    pub(crate) position: Vec3,
    pub(crate) extents: (Vec3, Vec3),
    pub(crate) confidence: f32,
    // class: SegmentationClass,
}

pub(crate) struct Person {
    pub(crate) position: Vec3,
    pub(crate) closest: Vec3,
    pub(crate) confidence: f32,
}

pub(crate) struct ObjectExtractorResult {
    pub(crate) vehicles: Vec<Vehicle>,
    pub(crate) people: Vec<Person>,
    pub(crate) source_pointcloud: Arc<Pointcloud>,
}

pub(crate) struct ObjectExtractor;

impl Module for ObjectExtractor {
    type In = Arc<Pointcloud>;
    type Out = Arc<ObjectExtractorResult>;

    fn process(&mut self, input: Self::In) -> eyre::Result<Self::Out> {
        let mut vehicles = Vec::new();
        let mut people = Vec::new();
        for object in input.iter_objects() {
            match object.info.class {
                SegmentationClass::Car => {
                    let center = object.closest_to_zero();
                    let (ext1, ext2) = object.extents();
                    vehicles.push(Vehicle {
                        position: center,
                        extents: (ext1.position, ext2.position),
                        confidence: object.info.confidence,
                    });
                }
                SegmentationClass::Person => {
                    let closest = object.closest_to_zero();
                    let center = object.calculate_center();
                    people.push(Person {
                        closest,
                        position: center,
                        confidence: object.info.confidence,
                    });
                }
                SegmentationClass::Road => {}
                SegmentationClass::Other => {}
            }
        }
        Ok(ObjectExtractorResult {
            vehicles,
            people,
            source_pointcloud: input,
        }
        .into())
    }
}
