use std::{
    sync::{Arc, Mutex},
    thread,
};

use eframe::egui::{Spinner, Ui};

use crate::VisionApp;

pub(crate) struct TaskStateUpdater {
    ctx: eframe::egui::Context,
    state: Arc<Mutex<TaskState>>,
}

impl TaskStateUpdater {
    pub(crate) fn update(&self, new_state: TaskState) {
        *self.state.lock().unwrap() = new_state;
        self.ctx.request_repaint();
    }
}

pub(crate) enum TaskState {
    SpinnerWithMessage { message: String },
}

pub(crate) struct TaskRunner {
    state: Arc<Mutex<TaskState>>,
    handle: thread::JoinHandle<Box<dyn FnOnce(&mut VisionApp) + Send>>,
}

impl TaskRunner {
    pub(crate) fn start<FInner>(
        ctx: &eframe::egui::Context,
        f: impl FnOnce(TaskStateUpdater) -> FInner + Send + 'static,
    ) -> Self
    where
        FInner: FnOnce(&mut VisionApp) + Send + 'static,
    {
        let state = Arc::new(Mutex::new(TaskState::SpinnerWithMessage {
            message: "".to_string(),
        }));
        let updater = TaskStateUpdater {
            ctx: ctx.clone(),
            state: state.clone(),
        };
        let handle = thread::spawn(move || {
            let inner_fn = f(updater);
            Box::new(inner_fn) as Box<dyn FnOnce(&mut VisionApp) + Send>
        });
        Self { handle, state }
    }

    pub(crate) fn show(&mut self, ui: &mut Ui) {
        ui.vertical_centered_justified(|ui| match &*self.state.lock().unwrap() {
            TaskState::SpinnerWithMessage { message } => {
                ui.add_space(ui.available_height() / 2.0 - 64.0 / 2.0);
                ui.add(Spinner::new().size(64.0));
                ui.label(message);
            }
        });
    }

    pub(crate) fn is_complete(&mut self) -> bool {
        self.handle.is_finished()
    }

    pub(crate) fn finish(self, app: &mut VisionApp) {
        let final_fn = self.handle.join().unwrap();
        final_fn(app)
    }
}
