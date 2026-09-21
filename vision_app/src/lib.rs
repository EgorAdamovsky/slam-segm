use std::{f32, path::PathBuf, sync::Arc};

use eframe::{
    CreationContext,
    egui::{self, CentralPanel, Color32, ScrollArea, SidePanel, Slider, TopBottomPanel},
};
use eyre::Context;
use glam::{Vec2, Vec3Swizzles};
use node_plumbing::{Input, InputChannel};
use tracing::info;

use crate::{
    models::segmentation::SegmentationClass,
    modules::{DisplayResult, build_pipeline},
    pipeline_settings::PipelineSettings,
    tasks::TaskRunner,
    ui_helpers::{VideoPlayer, show_mat_editor},
    world_painter::WorldPainter,
};

pub(crate) mod models;
pub(crate) mod modules;
pub(crate) mod pipeline_settings;
pub(crate) mod tasks;
pub(crate) mod ui_helpers;
pub(crate) mod world_painter;

#[derive(Default, PartialEq, Eq)]
enum MainPanelContent {
    #[default]
    SegmVideo,
    BirdView,
}

#[derive(Default)]
enum AppScreen {
    #[default]
    Init,
    RunningPipeline,
}

pub struct VisionApp {
    ctx: egui::Context,
    screen: AppScreen,
    task: Option<TaskRunner>,
    pipeline_settings: PipelineSettings,
    pipeline_out: Option<InputChannel<Arc<DisplayResult>>>,
    pipeline_last: Option<Arc<DisplayResult>>,
    main_panel_content: MainPanelContent,
    world_scale: f32,
    display_vehicles: bool,
    display_road: bool,
    display_others: bool,
    display_grid: bool,

    video_left: VideoPlayer,
    video_right: VideoPlayer,
    video_depth: VideoPlayer,
    last_error: Option<eyre::Error>,
}

impl VisionApp {
    pub fn new(cc: &CreationContext) -> Self {
        cc.egui_ctx.set_theme(egui::Theme::Light);
        // cc.egui_ctx.set_zoom_factor(cc.egui_ctx.zoom_factor() * 1.5);

        Self {
            ctx: cc.egui_ctx.clone(),
            screen: AppScreen::default(),
            task: None,
            pipeline_settings: PipelineSettings::default(),
            pipeline_out: None,
            pipeline_last: None,
            world_scale: 100.0,
            display_grid: true,
            display_vehicles: true,
            display_road: true,
            display_others: false,
            main_panel_content: MainPanelContent::default(),
            video_left: VideoPlayer::new(&cc.egui_ctx),
            video_right: VideoPlayer::new(&cc.egui_ctx),
            video_depth: VideoPlayer::new(&cc.egui_ctx),
            last_error: None,
        }
    }

    fn build_pipeline(&mut self) {
        let pipeline_settings = self.pipeline_settings.clone();
        self.task = Some(TaskRunner::start(&self.ctx, move |status| {
            info!("Building pipeline...");
            status.update(tasks::TaskState::SpinnerWithMessage {
                message: "Собирается цепочка обработки...\n(Может занять несколько минут при первом запуске при использовании TensorRT)".to_string(),
            });
            let pipeline_out = build_pipeline(pipeline_settings).wrap_err(
                "Не удалось собрать цепочку обработки.\nСкорее всего выбранное устройство для выполнения модели нельзя использовать на этом компьютере.\nВыбранное устройство можно поменять на начальном экране"
            );
            info!("Complete");
            |app| {
                match pipeline_out {
                    Ok(pipeline_out) => {
                        app.pipeline_out = Some(pipeline_out);
                    },
                    Err(err) => {
                        app.last_error = Some(err);
                        app.screen = AppScreen::Init;
                    },
                }
            }
        }));
        self.screen = AppScreen::RunningPipeline;
        self.pipeline_out = None;
        self.pipeline_last = None;
        self.video_left.reset();
        self.video_right.reset();
        self.video_depth.reset();
    }

    fn show_init_screen(&mut self, ctx: &eframe::egui::Context) {
        CentralPanel::default().show(ctx, |ui| {
            ui.vertical(|ui| {
                egui::Frame::group(&ui.style())
                    .outer_margin(64.0)
                    .show(ui, |ui| {
                        ui.take_available_space();
                        ui.horizontal(|ui| {
                            ui.group(|ui| {
                                ui.set_min_width(ui.available_width() / 3.0 * 1.2);
                                // ui.take_available_width();
                                ui.vertical(|ui| {
                                    ui.label("Источник видео");
                                    ui.radio_value(
                                        &mut self.pipeline_settings.video_source,
                                        pipeline_settings::VideoSource::Dataset,
                                        "Датасет",
                                    );
                                    ui.add_space(4.0);
                                    ui.label("Папка с левыми изображениями:");
                                    ui.label(
                                        self.pipeline_settings.dataset_left.display().to_string(),
                                    );
                                    if ui.button("Выбрать...").clicked() {
                                        if let Some(new_folder) = rfd::FileDialog::new()
                                            .set_title(
                                                "Выберите новую папку с левыми изображениями",
                                            )
                                            .set_directory(&self.pipeline_settings.dataset_left)
                                            .pick_folder()
                                        {
                                            self.pipeline_settings.dataset_left = new_folder;
                                        }
                                    }
                                    ui.add_space(4.0);
                                    ui.label("Папка с правыми изображениями:");
                                    ui.label(
                                        self.pipeline_settings.dataset_right.display().to_string(),
                                    );
                                    if ui.button("Выбрать...").clicked() {
                                        if let Some(new_folder) = rfd::FileDialog::new()
                                            .set_title(
                                                "Выберите новую папку с правыми изображениями",
                                            )
                                            .set_directory(&self.pipeline_settings.dataset_left)
                                            .pick_folder()
                                        {
                                            self.pipeline_settings.dataset_right = new_folder;
                                        }
                                    }
                                });
                            });
                            ui.group(|ui| {
                                ui.take_available_width();
                                ui.vertical(|ui| {
                                    ui.label("Матрица проекции из системы координат камеры в систему координат автомобиля");
                                    show_mat_editor(
                                        ui,
                                        &mut self.pipeline_settings.camera_perspecrive_matrix,
                                    );
                                    ui.take_available_height();
                                });
                            });
                        });
                        ui.horizontal(|ui| {
                            ui.group(|ui| {
                                ui.set_min_width(ui.available_width() / 2.0);
                                ui.vertical(|ui| {
                                    ui.label("Параметры генератора облака точек");
                                    ui.checkbox(&mut self.pipeline_settings.pointcloud_generator.filter_depth_edges, "Фильтр по значениям краёв в карте глубины");
                                    ui.checkbox(&mut self.pipeline_settings.pointcloud_generator.filter_semantic, "Не учитывать края семантических масок объектов");
                                } );
                            });
                        
                            ui.group(|ui| {
                                ui.set_min_width(ui.available_width() / 2.0);
                                ui.vertical(|ui| {
                                    ui.label("Устройство для вычисления моделей");
                                    ui.radio_value(&mut self.pipeline_settings.backend, pipeline_settings::UsedBackend::Cpu, "Процессор");
                                    ui.radio_value(&mut self.pipeline_settings.backend, pipeline_settings::UsedBackend::TensorRT, "Видеокарта NVIDIA (TensortRT)");
                                    ui.radio_value(&mut self.pipeline_settings.backend, pipeline_settings::UsedBackend::DirectML, "Видеокарта (DirectML)");
                                } );
                        })});
                        

                        if ui.button("Начать обработку").clicked() {
                            self.build_pipeline();
                        }
                    });
            });
        });
    }

    fn show_running_screen(&mut self, ctx: &eframe::egui::Context) {
        ctx.request_repaint_after_secs(1.0 / 60.0);
        TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Остановить обработку").clicked() {
                        self.pipeline_last = None;
                        self.pipeline_out = None;
                        self.screen = AppScreen::Init;
                    }
                })
            });
        });
        SidePanel::right("settings").show(ctx, |ui| {
            ui.take_available_width();
            ui.label("Содержимое главного экрана");
            ui.radio_value(
                &mut self.main_panel_content,
                MainPanelContent::SegmVideo,
                "Видео с сегментацией",
            );
            ui.radio_value(
                &mut self.main_panel_content,
                MainPanelContent::BirdView,
                "Вид сверху",
            );

            ui.separator();

            ui.label("Параметры отображения вида сверху");
            ui.add(Slider::new(&mut self.world_scale, 5.0..=500.0).text("Масштаб, м"));
            ui.checkbox(&mut self.display_grid, "Отображать сетку");
            ui.checkbox(&mut self.display_vehicles, "Показывать автомобили");
            ui.checkbox(&mut self.display_road, "Показывать дорогу");
            ui.checkbox(&mut self.display_others, "Показывать другие объекты");
        });
        SidePanel::left("videos").resizable(true).show(ctx, |ui| {
            ui.set_min_width(200.0);
            ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
                ui.set_max_height(f32::INFINITY);
                ui.vertical_centered(|ui| {
                    ui.vertical(|ui| {
                        ui.group(|ui| {
                            ui.label("Левое изображение");
                            self.video_left.show(ui);
                        });
                    });
                    ui.vertical(|ui| {
                        ui.group(|ui| {
                            ui.label("Правое изображение");
                            self.video_right.show(ui);
                        });
                    });
                    ui.vertical(|ui| {
                        ui.group(|ui| {
                            ui.label("Карта глубины");
                            self.video_depth.show(ui);
                        });
                    });
                });
            });
        });
        CentralPanel::default().show(ctx, |ui| match self.main_panel_content {
            MainPanelContent::SegmVideo => {
                // ui.take_available_space();
                ui.centered_and_justified(|ui| {
                    self.video_left.show2(ui, false, true);
                });
            }
            MainPanelContent::BirdView => self.show_birdview(ui),
        });
    }

    fn show_birdview(&mut self, ui: &mut egui::Ui) {
        if let Some(value) = self.pipeline_last.as_ref() {
            egui::Frame::dark_canvas(&ui.style()).show(ui, |ui| {
                ui.take_available_space();
                let painter = WorldPainter::new(ui, self.world_scale);
                if self.display_grid {
                    painter.draw_grid();
                }
                painter.circle_filled(Vec2::new(0.0, 0.0), 3.0, Color32::GREEN);

                // for point in value.source_pointcloud.iter_points() {
                for obj in value.objects.source_pointcloud.iter_objects() {
                    for point in obj.iter_points() {
                        if self.display_vehicles && obj.info.class == SegmentationClass::Car {
                            painter.circle_filled(point.position.xz(), 1.0, Color32::ORANGE);
                        }
                        if self.display_road && obj.info.class == SegmentationClass::Road {
                            painter.circle_filled(point.position.xz(), 0.1, Color32::BLUE);
                        }
                        if self.display_others && obj.info.class == SegmentationClass::Other {
                            painter.circle_filled(point.position.xz(), 0.1, Color32::GRAY);
                        }
                    }
                }

                for vehicle in &value.objects.vehicles {
                    let pos = vehicle.position.xz();
                    let distance = pos.length();
                    painter.circle_filled(pos, 3.0, Color32::RED);
                    painter.circle_filled(vehicle.extents.0.xz(), 1.0, Color32::WHITE);
                    painter.circle_filled(vehicle.extents.1.xz(), 1.0, Color32::WHITE);
                    painter.label(
                        pos,
                        &format!("Автомобиль {distance:.2}м conf: {:.2}", vehicle.confidence),
                        Color32::WHITE,
                    );
                }
                for person in &value.objects.people {
                    let pos = person.position.xz();
                    let distance = pos.length();
                    painter.circle_filled(pos, 3.0, Color32::BLUE);
                    painter.circle_filled(person.closest.xz(), 1.0, Color32::BLUE);
                    painter.label(
                        pos,
                        &format!("Человек {distance:.2}м conf: {:.2}", person.confidence),
                        Color32::WHITE,
                    );
                }
            });
        } else {
            ui.label("No value yet");
        }
    }
}

impl eframe::App for VisionApp {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint_after_secs(1.0);

        if let Some(pipeline_out) = &mut self.pipeline_out {
            if let Ok(Some(val)) = pipeline_out.try_recv() {
                self.video_left
                    .set_image(&val.sensor_data.stereoimages[0].0);
                self.video_right
                    .set_image(&val.sensor_data.stereoimages[0].1);
                self.video_depth.set_image(&val.disparity_image);
                self.video_left
                    .set_annotations(val.segm_annotations.clone());
                self.video_depth
                    .set_annotations(val.segm_annotations.clone());
                self.pipeline_last = Some(val);
            }
        }

        if let Some(task) = &mut self.task {
            CentralPanel::default().show(ctx, |ui| {
                task.show(ui);
            });
            if let Some(task) = self.task.take_if(TaskRunner::is_complete) {
                task.finish(self);
            }
        } else if let Some(err) = &self.last_error {
            let err = format!("{:?}", err);
            CentralPanel::default().show(ctx, |ui| {
                egui::Frame::group(&ctx.style()).outer_margin(20.0).show(ui, |ui| {
                    ui.take_available_space();
                    ui.label("Произошла ошибка:");
                    ui.label(err);
                    if ui.button("Ок").clicked() {
                        self.last_error = None;
                    }
                })
            });
        } else {
            match self.screen {
                AppScreen::Init => self.show_init_screen(ctx),
                AppScreen::RunningPipeline => self.show_running_screen(ctx),
            }
        }
    }
}

pub fn self_test(dataset_path: PathBuf) -> eyre::Result<()> {
    let mut pipeline = build_pipeline(PipelineSettings { 
        dataset_left: dataset_path.join("interlaken_00_c_images_rectified_left"),
            dataset_right: dataset_path.join("interlaken_00_c_images_rectified_right"),
        ..PipelineSettings::default()
    })?;
    info!("Built pipeline - OK");
    pipeline.recv()?;
    info!("Received item from pipeline - OK");
    Ok(())
}
