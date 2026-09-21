#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::PathBuf;

use argh::FromArgs;
use eframe::egui;
use tracing_subscriber::{EnvFilter, FmtSubscriber};
use vision_app::self_test;

#[derive(FromArgs)]
/// Automotive vision app.
struct VisionArgs {
    /// test model runtime to see if it works
    #[argh(option)]
    self_test_with_dataset: Option<PathBuf>,
}

pub fn main() -> eyre::Result<()> {
    FmtSubscriber::builder()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args: VisionArgs = argh::from_env();
    if let Some(dataset_path) = args.self_test_with_dataset {
        return self_test(dataset_path);
    }

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_min_inner_size([1200.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Прототип системы автомобильного зрения",
        native_options,
        Box::new(|cc| Ok(Box::new(vision_app::VisionApp::new(cc)))),
    )
    .unwrap();

    Ok(())
}
