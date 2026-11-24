//! Headless UI tests for the two-column Start Screen: layout, i18n language switching, mode
//! card launching, New Project action, Recent Projects opening and async Open Project… flow.

use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};

use egui_kittest::Harness;
use filmcraft_engine::Session;
use filmcraft_ui_egui::FilmcraftApp;
use filmcraft_ui_egui::control::ControlRequest;
use filmcraft_ui_egui::native_dialogs::FileDialog;
use filmcraft_ui_egui::state::{AppExperience, Mode};
use serde_json::{Value, json};

type Asked = Arc<Mutex<Vec<(FileDialog, Sender<Vec<String>>)>>>;

struct Driver {
    harness: Harness<'static, FilmcraftApp>,
    tx: Sender<ControlRequest>,
    asked: Asked,
}

impl Driver {
    fn start(recent: Vec<String>) -> Self {
        let mut session = Session::default();
        if !recent.is_empty() {
            let mut prefs = session.prefs.clone();
            prefs.general.recent_projects = recent;
            session.set_prefs(prefs).expect("recent prefs");
        }
        let (tx, rx) = channel();
        let mut app = FilmcraftApp::new(session).with_control(rx);
        let asked: Asked = Arc::default();
        let a = asked.clone();
        app.hooks.run_dialog = Some(Box::new(move |dialog: FileDialog| -> Receiver<Vec<String>> {
            let (tx, rx) = channel();
            a.lock().unwrap().push((dialog, tx));
            rx
        }));
        let harness = Harness::builder().with_size(egui::vec2(1600.0, 980.0)).with_max_steps(10_000).build_eframe(move |_cc| app);
        let mut d = Driver { harness, tx, asked };
        d.frames(4);
        d
    }

    fn frames(&mut self, n: usize) {
        for _ in 0..n {
            let ctx = self.harness.ctx.clone();
            let mut raw = std::mem::take(self.harness.input_mut());
            eframe::App::raw_input_hook(self.harness.state_mut(), &ctx, &mut raw);
            *self.harness.input_mut() = raw;
            self.harness.step();
        }
    }

    fn ok(&mut self, method: &str, params: Value) -> Value {
        let (req, reply) = ControlRequest::new(method, params.clone());
        self.tx.send(req).unwrap();
        for _ in 0..600 {
            self.frames(1);
            if let Ok(v) = reply.try_recv() {
                assert_eq!(v["ok"], json!(true), "{method} {params} failed: {v}");
                return v["result"].clone();
            }
        }
        panic!("no reply to {method} {params}");
    }

    fn elem(&mut self, id: &str) -> Value {
        self.find_elem("start.", id)
    }

    fn find_elem(&mut self, prefix: &str, id: &str) -> Value {
        let list = self.ok("ui.elements", json!({"prefix": prefix}));
        list.as_array().unwrap().iter().find(|e| e["id"] == id).cloned().unwrap_or_else(|| panic!("missing element {id} in {list}"))
    }
}

#[test]
fn two_column_layout_and_dynamic_i18n() {
    let mut d = Driver::start(vec!["/work/Reel_One.fcproj".into()]);
    assert!(d.harness.state().ui.start_screen);

    let creator = d.elem("start.creator");
    let studio = d.elem("start.studio");
    let new_btn = d.elem("start.newProject");
    let open_btn = d.elem("start.openProject");
    let recent0 = d.elem("start.recent.0");

    // Two-column split geometry: Creator | Studio side-by-side, buttons below, Recent on right
    assert_eq!(creator["rect"][1], studio["rect"][1]);
    assert!(creator["rect"][0].as_f64().unwrap() < studio["rect"][0].as_f64().unwrap());
    assert!(new_btn["rect"][1].as_f64().unwrap() > creator["rect"][1].as_f64().unwrap() + creator["rect"][3].as_f64().unwrap());
    assert_eq!(new_btn["rect"][1], open_btn["rect"][1]);
    assert!(recent0["rect"][0].as_f64().unwrap() > studio["rect"][0].as_f64().unwrap() + studio["rect"][2].as_f64().unwrap());

    assert_eq!(new_btn["label"], "New Project");
    assert_eq!(open_btn["label"], "Open Project…");

    // Switching languages dynamically updates all welcome buttons
    d.ok("ui.menu.invoke", json!({"command": "app.language.bulgarian"}));
    d.frames(2);
    assert_eq!(d.elem("start.newProject")["label"], "Нов проект");
    assert_eq!(d.elem("start.openProject")["label"], "Отваряне на проект…");

    d.ok("ui.menu.invoke", json!({"command": "app.language.german"}));
    d.frames(2);
    assert_eq!(d.elem("start.newProject")["label"], "Neues Projekt");
    assert_eq!(d.elem("start.openProject")["label"], "Projekt öffnen…");
}

#[test]
fn mode_cards_new_project_and_async_open_work() {
    // 1. Clicking Creator card launches Creator workspace
    let mut d = Driver::start(vec![]);
    d.ok("ui.click", json!({"id": "start.creator"}));
    d.frames(2);
    assert!(!d.harness.state().ui.start_screen);
    assert_eq!(d.harness.state().ui.experience, AppExperience::Creator);
    assert_eq!(d.harness.state().ui.mode, Mode::Edit);

    // 2. Clicking Studio card launches Studio workspace
    let mut d = Driver::start(vec![]);
    d.ok("ui.click", json!({"id": "start.studio"}));
    d.frames(2);
    assert!(!d.harness.state().ui.start_screen);
    assert_eq!(d.harness.state().ui.experience, AppExperience::Studio);

    // 3. Clicking New Project starts a fresh untitled project in the selected experience
    let mut d = Driver::start(vec![]);
    d.harness.state_mut().ui.experience = AppExperience::Creator;
    d.ok("engine.execute", json!({"command": "file.openDemoProject", "params": {}}));
    assert_eq!(d.harness.state().session.project.name, "Orpheus – Demo Project");
    d.ok("ui.click", json!({"id": "start.newProject"}));
    d.frames(2);
    assert!(!d.harness.state().ui.start_screen);
    assert_eq!(d.harness.state().ui.mode, Mode::Edit);
    assert_eq!(d.harness.state().ui.experience, AppExperience::Creator);
    assert_eq!(d.harness.state().session.project.name, "Untitled");
    assert_eq!(d.harness.state().session.path, None);

    // 4. Clicking Open Project… via async file picker dismisses start screen and restores mode
    let dir = std::env::temp_dir().join(format!("orpheus-start-open-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let proj_path = dir.join("saved.fcproj").to_string_lossy().into_owned();
    let mut seed = Session::default();
    seed.execute("file.newProject", json!({"name": "Saved Launch"})).unwrap();
    seed.execute("file.saveAs", json!({"path": proj_path})).unwrap();

    let mut d = Driver::start(vec![proj_path.clone()]);
    d.harness.state_mut().ui.remember_project_experience(&proj_path, AppExperience::Creator);
    d.ok("ui.click", json!({"id": "start.openProject"}));
    d.frames(2);
    let (dlg, sender) = d.asked.lock().unwrap().pop().expect("open dialog requested");
    assert!(matches!(dlg, FileDialog::OpenProject));
    sender.send(vec![proj_path.clone()]).unwrap();
    d.frames(3);
    assert!(!d.harness.state().ui.start_screen);
    assert_eq!(d.harness.state().session.path.as_deref(), Some(proj_path.as_str()));
    assert_eq!(d.harness.state().ui.experience, AppExperience::Creator);
    assert_eq!(d.harness.state().ui.mode, Mode::Edit);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn ai_tools_panel_and_creator_integration() {
    let mut d = Driver::start(vec![]);
    d.ok("engine.execute", json!({"command": "file.openDemoProject", "params": {}}));
    d.ok("ui.click", json!({"id": "start.studio"}));
    d.frames(2);

    // Activate AI Tools tab in Studio inspector and verify Auto-Cut Silence + AI Subtitles controls
    d.ok("ui.menu.invoke", json!({"command": "window.panel.AiTools"}));
    d.frames(2);
    assert_eq!(d.find_elem("ai.", "ai.silence.status")["label"], "Orpheus AI Ready");
    assert_eq!(d.find_elem("ai.", "ai.thresholdDb")["label"], "Silence Threshold");
    assert_eq!(d.find_elem("ai.", "ai.minPauseMs")["label"], "Minimum Pause");
    assert_eq!(d.find_elem("ai.", "ai.paddingMs")["label"], "Cut Padding");
    assert_eq!(d.find_elem("ai.", "ai.removePauses")["label"], "Remove Pauses");
    assert_eq!(d.find_elem("ai.", "ai.subtitles.cliStatus")["label"].is_string(), true);
    assert_eq!(d.find_elem("ai.", "ai.subtitles.language")["label"], "Language");
    assert_eq!(d.find_elem("ai.", "ai.subtitles.model")["label"], "AI Model");
    assert_eq!(d.find_elem("ai.", "ai.subtitles.compute.gpu")["label"], "GPU");
    assert_eq!(d.find_elem("ai.", "ai.subtitles.compute.cpu")["label"], "CPU");
    assert_eq!(d.find_elem("ai.", "ai.subtitles.mode.reels")["label"], "Reels (1–3 words)");
    assert_eq!(d.find_elem("ai.", "ai.subtitles.mode.sentences")["label"], "Standard sentences");
    assert_eq!(d.find_elem("ai.", "ai.subtitles.generate")["label"], "Generate Subtitles");

    // Select CPU compute card and verify state + ui.set "auto" reset
    d.ok("ui.click", json!({"id": "ai.subtitles.compute.cpu"}));
    d.frames(2);
    assert_eq!(d.harness.state().ui.ai_tools.subtitle_compute, Some(filmcraft_engine::subtitles::SubtitleComputeDevice::Cpu));
    d.ok("ui.set", json!({"aiTools": {"subtitleCompute": "auto"}}));
    d.frames(2);
    assert_eq!(d.harness.state().ui.ai_tools.subtitle_compute, None);

    // Toggle Reels mode via UI click
    assert!(!d.harness.state().ui.ai_tools.subtitle_reels_mode);
    d.ok("ui.click", json!({"id": "ai.subtitles.mode.reels"}));
    d.frames(2);
    assert!(d.harness.state().ui.ai_tools.subtitle_reels_mode);

    // Switch to Bulgarian and verify localized AI Tools + Subtitles labels in Studio & Creator
    d.ok("ui.menu.invoke", json!({"command": "app.language.bulgarian"}));
    d.frames(2);
    assert_eq!(d.find_elem("ai.", "ai.silence.status")["label"], "Orpheus AI е готов");
    assert_eq!(d.find_elem("ai.", "ai.thresholdDb")["label"], "Праг на тишина");
    assert_eq!(d.find_elem("ai.", "ai.minPauseMs")["label"], "Минимална пауза");
    assert_eq!(d.find_elem("ai.", "ai.paddingMs")["label"], "Буфер при рязане");
    assert_eq!(d.find_elem("ai.", "ai.removePauses")["label"], "Премахване на паузи");
    assert_eq!(d.find_elem("ai.", "ai.subtitles.language")["label"], "Език");
    assert_eq!(d.find_elem("ai.", "ai.subtitles.model")["label"], "AI Модел");
    assert_eq!(d.find_elem("ai.", "ai.subtitles.compute.gpu")["label"], "GPU");
    assert_eq!(d.find_elem("ai.", "ai.subtitles.compute.cpu")["label"], "CPU");
    assert_eq!(d.find_elem("ai.", "ai.subtitles.mode.reels")["label"], "Reels / TikTok (1–3 думи)");
    assert_eq!(d.find_elem("ai.", "ai.subtitles.mode.sentences")["label"], "Стандартни изречения");
    assert_eq!(d.find_elem("ai.", "ai.subtitles.generate")["label"], "Генерирай субтитри");

    // Switch to Creator workspace and verify AI Tools + AI Subtitles buttons & stacked compute cards in narrow drawer
    d.ok("ui.click", json!({"id": "header.experience.creator"}));
    d.frames(2);
    assert_eq!(d.find_elem("creator.", "creator.action.aiTools")["label"], "AI Инструменти");
    assert_eq!(d.find_elem("creator.", "creator.action.removePauses")["label"], "Премахване на паузи");
    assert_eq!(d.find_elem("creator.", "creator.action.subtitles")["label"], "AI Субтитри");
    d.ok("ui.click", json!({"id": "creator.action.aiTools"}));
    d.frames(2);
    assert_eq!(d.find_elem("ai.", "ai.silence.status")["label"], "Orpheus AI е готов");
    assert_eq!(d.find_elem("ai.", "ai.removePauses")["label"], "Премахване на паузи");
    assert_eq!(d.find_elem("ai.", "ai.subtitles.mode.reels")["label"], "Reels / TikTok (1–3 думи)");
    assert_eq!(d.find_elem("ai.", "ai.subtitles.generate")["label"], "Генерирай субтитри");
    let gpu_rect = d.find_elem("ai.", "ai.subtitles.compute.gpu")["rect"].clone();
    let cpu_rect = d.find_elem("ai.", "ai.subtitles.compute.cpu")["rect"].clone();
    assert!(cpu_rect[1].as_f64().unwrap() > gpu_rect[1].as_f64().unwrap(), "Creator narrow drawer must stack GPU and CPU compute cards vertically");
    // None of the AI panel widgets should clip left of Creator's drawer body (rail_w=64 + gap=4 = 68px)
    let ai_elems = d.ok("ui.elements", json!({"prefix": "ai."}));
    for el in ai_elems.as_array().unwrap() {
        assert!(el["rect"][0].as_f64().unwrap() >= 68.0, "AI element {} clipped on the left: {}", el["id"], el["rect"]);
    }
}

#[test]
fn redesigned_import_workspace_and_i18n() {
    let mut d = Driver::start(vec![]);
    d.ok("engine.execute", json!({"command": "file.openDemoProject", "params": {}}));
    d.ok("ui.click", json!({"id": "start.studio"}));
    d.frames(2);

    // Navigate to Import mode via top header Home/Import tab
    d.ok("ui.click", json!({"id": "header.home"}));
    d.frames(2);
    assert_eq!(d.harness.state().ui.mode, Mode::Import);

    // Top row: 4 equal-width source cards aligned horizontally across viewport
    let browse = d.find_elem("import.", "import.file.import");
    let demo_clips = d.find_elem("import.", "import.file.importDemoFootage");
    let demo_proj = d.find_elem("import.", "import.file.openDemoProject");
    let bars = d.find_elem("import.", "import.file.newBarsAndTone");
    assert_eq!(browse["rect"][1], demo_clips["rect"][1]);
    assert_eq!(demo_clips["rect"][1], demo_proj["rect"][1]);
    assert_eq!(demo_proj["rect"][1], bars["rect"][1]);
    assert!(browse["rect"][0].as_f64().unwrap() < demo_clips["rect"][0].as_f64().unwrap());
    assert!(demo_clips["rect"][0].as_f64().unwrap() < demo_proj["rect"][0].as_f64().unwrap());
    assert!(demo_proj["rect"][0].as_f64().unwrap() < bars["rect"][0].as_f64().unwrap());
    assert!((browse["rect"][2].as_f64().unwrap() - bars["rect"][2].as_f64().unwrap()).abs() < 2.0);

    // Bottom row: Left Quick Drag & Drop Zone + Right Imported Media Pool
    let drop = d.find_elem("import.", "import.dropZone");
    let search = d.find_elem("import.", "import.media.search");
    let filter_all = d.find_elem("import.", "import.media.filter.all");
    let filter_video = d.find_elem("import.", "import.media.filter.video");
    let filter_audio = d.find_elem("import.", "import.media.filter.audio");
    let filter_images = d.find_elem("import.", "import.media.filter.images");
    assert!(drop["rect"][0].as_f64().unwrap() < filter_all["rect"][0].as_f64().unwrap());
    assert!(drop["rect"][1].as_f64().unwrap() > browse["rect"][1].as_f64().unwrap() + browse["rect"][3].as_f64().unwrap());
    assert_eq!(filter_all["label"], "All");
    assert_eq!(filter_video["label"], "Video");
    assert_eq!(filter_audio["label"], "Audio");
    assert_eq!(filter_images["label"], "Images");
    assert_eq!(search["label"], "Search media…");

    // Demo project populates media cards with quick-add `+` buttons
    let cards_before = d.ok("ui.elements", json!({"prefix": "import.media.card."})).as_array().unwrap().len();
    assert!(cards_before > 0, "Expected imported media cards from demo project");
    let first_add = d.ok("ui.elements", json!({"prefix": "import.media.add."})).as_array().unwrap()[0]["id"].as_str().unwrap().to_string();

    // Quick-add `+` appends item onto active timeline
    let v1_len_before = d.harness.state().session.active_sequence().unwrap().video_tracks[0].items.len();
    d.ok("ui.click", json!({"id": first_add}));
    d.frames(2);
    let v1_len_after = d.harness.state().session.active_sequence().unwrap().video_tracks[0].items.len();
    assert!(v1_len_after >= v1_len_before);

    // Filter tabs switch category
    d.ok("ui.click", json!({"id": "import.media.filter.images"}));
    d.frames(2);
    assert_eq!(d.harness.state().ui.import_filter, filmcraft_ui_egui::state::ImportMediaFilter::Images);
    d.ok("ui.click", json!({"id": "import.media.filter.all"}));
    d.frames(2);

    // Switching to Bulgarian updates all new Import labels cleanly
    d.ok("ui.menu.invoke", json!({"command": "app.language.bulgarian"}));
    d.frames(2);
    assert_eq!(d.find_elem("import.", "import.dropZone")["label"], "Провлачете и пуснете файлове тук за бърз импорт");
    assert_eq!(d.find_elem("import.", "import.media.search")["label"], "Търсене на медии…");
    assert_eq!(d.find_elem("import.", "import.media.filter.all")["label"], "Всички");
    assert_eq!(d.find_elem("import.", "import.media.filter.images")["label"], "Изображения");
}

#[test]
fn redesigned_creator_mode_dock_aspect_toggle_and_mode_switch() {
    let mut d = Driver::start(vec![]);
    d.ok("engine.execute", json!({"command": "file.openDemoProject", "params": {}}));
    d.ok("ui.click", json!({"id": "start.creator"}));
    d.frames(2);
    assert_eq!(d.harness.state().ui.experience, AppExperience::Creator);

    // Top All-In-One prompt bar + 1-Click Creator Actions dock + Timeline bar controls are visible
    let _ = d.find_elem("creator.", "creator.prompt.input");
    let _ = d.find_elem("creator.", "creator.prompt.run");
    let _ = d.find_elem("creator.", "creator.card.autoShorts");
    let _ = d.find_elem("creator.", "creator.card.gameStream");
    let _ = d.find_elem("creator.", "creator.subtitles.language");
    let _ = d.find_elem("creator.", "creator.subtitles.style.pop");
    let _ = d.find_elem("creator.", "creator.card.addBoldTitle");
    let _ = d.find_elem("creator.", "creator.card.removePauses");
    let _ = d.find_elem("creator.", "creator.card.enhanceAudio");
    let _ = d.find_elem("creator.", "creator.card.quickExport");
    let _ = d.find_elem("creator.", "creator.action.split");
    let _ = d.find_elem("creator.", "creator.action.delete");
    let _ = d.find_elem("creator.", "creator.action.speed");

    // Aspect ratio toggle under preview switches between 9:16 Vertical and 16:9 Widescreen
    d.ok("ui.click", json!({"id": "creator.aspect.vertical"}));
    d.frames(2);
    let seq = d.harness.state().session.active_sequence().unwrap();
    assert_eq!((seq.settings.width, seq.settings.height), (1080, 1920));
    let clips_before = seq.video_tracks[0].items.len();

    d.ok("ui.click", json!({"id": "creator.aspect.widescreen"}));
    d.frames(2);
    let seq = d.harness.state().session.active_sequence().unwrap();
    assert_eq!((seq.settings.width, seq.settings.height), (1920, 1080));

    // Seamless Studio <-> Creator mode switch preserves timeline state
    d.ok("ui.click", json!({"id": "header.experience.studio"}));
    d.frames(2);
    assert_eq!(d.harness.state().ui.experience, AppExperience::Studio);
    assert_eq!(d.harness.state().session.active_sequence().unwrap().video_tracks[0].items.len(), clips_before);

    d.ok("ui.click", json!({"id": "header.experience.creator"}));
    d.frames(2);
    assert_eq!(d.harness.state().ui.experience, AppExperience::Creator);
    assert_eq!(d.harness.state().session.active_sequence().unwrap().video_tracks[0].items.len(), clips_before);
}
