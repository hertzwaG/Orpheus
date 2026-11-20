//! Non-intrusive GitHub Releases update checker & Windows self-updater UI.

use std::sync::{Arc, mpsc};

use filmcraft_engine::updater::{
    DEFAULT_LATEST_RELEASE_PAGE_URL, ORPHEUS_VERSION, ReleaseInfo, UpdateCheckOutcome, UpdateDownloadProgress, cleanup_stale_backup_exe,
    install_windows_update_and_relaunch_with_progress, resolve_update_endpoint, spawn_update_check,
};
use serde_json::Value;

use crate::FilmcraftApp;

#[derive(Default)]
pub(crate) struct Runtime {
    startup_check: bool,
    startup_dispatched: bool,
    cleaned_bak: bool,
    check_requested: bool,
    manual_check: bool,
    pub(crate) manual_window_open: bool,
    pub(crate) banner_dismissed: bool,
    pending_rx: Option<mpsc::Receiver<UpdateCheckOutcome>>,
    last_outcome: Option<UpdateCheckOutcome>,
    install_rx: Option<mpsc::Receiver<Result<std::path::PathBuf, String>>>,
    install_progress: Option<Arc<UpdateDownloadProgress>>,
    install_status: Option<String>,
    repaint: Option<egui::Context>,
}

impl FilmcraftApp {
    /// Whether the app asks for a newer release when it starts (off until the desktop app turns it on).
    pub fn set_startup_update_check(&mut self, on: bool) {
        self.updates_rt.startup_check = on;
    }
}

pub(crate) fn check(app: &mut FilmcraftApp) -> Result<Value, String> {
    app.updates_rt.check_requested = true;
    app.updates_rt.manual_check = true;
    app.updates_rt.manual_window_open = true;
    app.updates_rt.banner_dismissed = false;
    app.updates_rt.last_outcome = None;
    app.updates_rt.install_status = None;
    if let Some(ctx) = &app.updates_rt.repaint {
        ctx.request_repaint();
    }
    Ok(Value::Null)
}

fn active_endpoint(app: &FilmcraftApp) -> String {
    resolve_update_endpoint(&app.ui.update_endpoint)
}

pub(crate) fn tick(app: &mut FilmcraftApp, ctx: &egui::Context) {
    if !app.updates_rt.cleaned_bak {
        app.updates_rt.cleaned_bak = true;
        cleanup_stale_backup_exe();
    }
    app.updates_rt.repaint = Some(ctx.clone());

    if std::mem::take(&mut app.updates_rt.check_requested) {
        let endpoint = active_endpoint(app);
        app.updates_rt.pending_rx = Some(spawn_update_check(endpoint, ORPHEUS_VERSION.to_string()));
    } else if app.updates_rt.startup_check && app.ui.check_updates_on_startup && !app.updates_rt.startup_dispatched {
        app.updates_rt.startup_dispatched = true;
        app.updates_rt.manual_check = false;
        let endpoint = active_endpoint(app);
        app.updates_rt.pending_rx = Some(spawn_update_check(endpoint, ORPHEUS_VERSION.to_string()));
    }

    if let Some(rx) = &app.updates_rt.pending_rx
        && let Ok(outcome) = rx.try_recv()
    {
        app.updates_rt.pending_rx = None;
        match &outcome {
            UpdateCheckOutcome::Available(_) => {
                app.updates_rt.last_outcome = Some(outcome);
                app.updates_rt.banner_dismissed = false;
                app.updates_rt.manual_window_open = true;
            }
            UpdateCheckOutcome::UpToDate(_) | UpdateCheckOutcome::OfflineOrError(_) => {
                // Startup checks fail silently when offline or already up to date.
                if app.updates_rt.manual_check {
                    app.updates_rt.last_outcome = Some(outcome);
                } else {
                    app.updates_rt.manual_window_open = false;
                }
            }
        }
        ctx.request_repaint();
    }

    if let Some(rx) = &app.updates_rt.install_rx
        && let Ok(res) = rx.try_recv()
    {
        app.updates_rt.install_rx = None;
        app.updates_rt.install_progress = None;
        match res {
            Ok(_) => {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
            Err(err) => {
                app.updates_rt.install_status = Some(err);
            }
        }
        ctx.request_repaint();
    }
}

pub(crate) fn show(app: &mut FilmcraftApp, ctx: &egui::Context) {
    if !app.updates_rt.manual_window_open {
        return;
    }
    let mut open = true;
    let checking = app.updates_rt.pending_rx.is_some();
    let installing = app.updates_rt.install_rx.is_some();
    let outcome = app.updates_rt.last_outcome.clone();
    let install_status = app.updates_rt.install_status.clone();
    let mut action_open_url: Option<String> = None;
    let mut action_install: Option<ReleaseInfo> = None;
    let mut action_recheck = false;
    let mut action_close = false;

    let title = match &outcome {
        Some(UpdateCheckOutcome::Available(_)) => tl!("Update Available"),
        _ => tl!("Orpheus Updates"),
    };

    egui::Window::new(title)
        .id(egui::Id::unique("orpheus-updates-modal"))
        .open(&mut open)
        .collapsible(false)
        .resizable(false)
        .pivot(egui::Align2::CENTER_CENTER)
        .default_pos(crate::widgets::dialog_center(ctx))
        .show(ctx, |ui| {
            ui.set_width(420.0);
            if checking {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(tl!("Checking for updates…"));
                });
            } else if installing {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(tl!("Downloading and installing Orpheus update…"));
                });
                if let Some(prog) = &app.updates_rt.install_progress {
                    if let Some(frac) = prog.fraction() {
                        ui.add(egui::ProgressBar::new(frac).show_percentage());
                    } else {
                        let mb = prog.downloaded.load(std::sync::atomic::Ordering::Relaxed) as f64 / (1024.0 * 1024.0);
                        ui.label(format!("{mb:.1} MB downloaded…"));
                    }
                    ctx.request_repaint();
                }
            } else {
                match &outcome {
                    Some(UpdateCheckOutcome::Available(info)) => {
                        ui.heading(tlf!("Orpheus v{ver} is available", ver = info.version));
                        ui.label(tlf!("Installed version: v{cur}", cur = ORPHEUS_VERSION));
                        if !info.notes.is_empty() {
                            ui.add_space(6.0);
                            egui::ScrollArea::vertical().max_height(140.0).show(ui, |ui| {
                                ui.label(&info.notes);
                            });
                        }
                        if let Some(err) = &install_status {
                            ui.add_space(6.0);
                            ui.colored_label(app.tokens.danger, err);
                        }
                        ui.add_space(10.0);
                        ui.horizontal(|ui| {
                            if ui.button(tl!("Open Download Page")).clicked() {
                                action_open_url = Some(info.html_url.clone());
                            }
                            if let Some(_url) = &info.download_url
                                && cfg!(windows)
                                && ui.button(tl!("Update Now")).clicked()
                            {
                                action_install = Some(info.clone());
                            }
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.button(tl!("Later")).clicked() {
                                    action_close = true;
                                }
                            });
                        });
                    }
                    Some(UpdateCheckOutcome::UpToDate(v)) => {
                        ui.label(tlf!("You are on the latest version (v{ver}).", ver = v));
                        ui.add_space(8.0);
                        ui.horizontal(|ui| {
                            if ui.button(tl!("Check Again")).clicked() {
                                action_recheck = true;
                            }
                            if ui.button(tl!("Open Releases Page")).clicked() {
                                action_open_url = Some(DEFAULT_LATEST_RELEASE_PAGE_URL.to_string());
                            }
                            if ui.button(tl!("OK")).clicked() {
                                action_close = true;
                            }
                        });
                    }
                    Some(UpdateCheckOutcome::OfflineOrError(msg)) => {
                        ui.label(tl!("Could not reach the update server (offline or timed out)."));
                        ui.label(egui::RichText::new(msg).small().color(app.tokens.text_dim));
                        ui.add_space(8.0);
                        ui.horizontal(|ui| {
                            if ui.button(tl!("Open Download Page")).clicked() {
                                action_open_url = Some(DEFAULT_LATEST_RELEASE_PAGE_URL.to_string());
                            }
                            if ui.button(tl!("Check Again")).clicked() {
                                action_recheck = true;
                            }
                            if ui.button(tl!("OK")).clicked() {
                                action_close = true;
                            }
                        });
                    }
                    None => {
                        ui.label(tl!("Ready to check for updates."));
                    }
                }
            }
        });

    if let Some(url) = action_open_url {
        crate::links::open(app, ctx, &url);
    }
    if let Some(info) = action_install
        && let Some(dl_url) = info.download_url
    {
        start_install(app, dl_url);
    }
    if action_recheck {
        let _ = check(app);
    }
    if !open || action_close {
        app.updates_rt.manual_window_open = false;
    }
}

fn start_install(app: &mut FilmcraftApp, dl_url: String) {
    let (tx, rx) = mpsc::channel();
    let prog = Arc::new(UpdateDownloadProgress::default());
    app.updates_rt.install_rx = Some(rx);
    app.updates_rt.install_progress = Some(prog.clone());
    app.updates_rt.install_status = None;
    std::thread::spawn(move || {
        let res = install_windows_update_and_relaunch_with_progress(&dl_url, Some(&prog));
        let _ = tx.send(res);
    });
}

/// Render the **Updates** section inside Settings ▸ General.
pub(crate) fn draw_settings_updates_section(app: &mut FilmcraftApp, ui: &mut egui::Ui) {
    let t = app.tokens;
    ui.horizontal(|ui| {
        ui.label(format!("Orpheus v{ORPHEUS_VERSION}"));
        let checking = app.updates_rt.pending_rx.is_some();
        let installing = app.updates_rt.install_rx.is_some();
        let check_btn = ui.add_enabled(!checking && !installing, egui::Button::new(tl!("Check for Updates")));
        app.auto.add("settings.updates.check", check_btn.rect, tl!("Check for Updates"));
        if check_btn.clicked() {
            app.updates_rt.check_requested = true;
            app.updates_rt.manual_check = true;
            app.updates_rt.last_outcome = None;
            app.updates_rt.install_status = None;
        }
    });
    let chk = ui.checkbox(&mut app.ui.check_updates_on_startup, tl!("Automatically check for updates on startup"));
    app.auto.add("settings.updates.autoCheck", chk.rect, tl!("Automatically check for updates on startup"));

    if app.updates_rt.pending_rx.is_some() {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label(tl!("Checking for updates…"));
        });
    } else if app.updates_rt.install_rx.is_some() {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label(tl!("Downloading and installing Orpheus update…"));
        });
        if let Some(prog) = &app.updates_rt.install_progress {
            if let Some(frac) = prog.fraction() {
                ui.add(egui::ProgressBar::new(frac).show_percentage());
            }
            ui.ctx().request_repaint();
        }
    } else if let Some(outcome) = app.updates_rt.last_outcome.clone() {
        match outcome {
            UpdateCheckOutcome::Available(info) => {
                ui.colored_label(Color32::from_rgb(0x10, 0xB9, 0x81), tlf!("Orpheus v{ver} is available", ver = info.version));
                if !info.notes.is_empty() {
                    egui::ScrollArea::vertical().max_height(90.0).show(ui, |ui| {
                        ui.label(egui::RichText::new(&info.notes).small().color(t.text_dim));
                    });
                }
                if let Some(dl_url) = info.download_url.clone()
                    && cfg!(windows)
                    && ui.button(tl!("Download & Install Update")).clicked()
                {
                    start_install(app, dl_url);
                }
            }
            UpdateCheckOutcome::UpToDate(v) => {
                ui.label(egui::RichText::new(tlf!("You are on the latest version (v{ver}).", ver = v)).small().color(t.text_dim));
            }
            UpdateCheckOutcome::OfflineOrError(msg) => {
                ui.label(egui::RichText::new(msg).small().color(t.text_dim));
            }
        }
    }
    if let Some(err) = &app.updates_rt.install_status {
        ui.colored_label(t.danger, err);
    }
}

use egui::Color32;

pub(crate) fn banner_height(app: &FilmcraftApp) -> f32 {
    if app.updates_rt.install_rx.is_some()
        || (!app.updates_rt.banner_dismissed && matches!(app.updates_rt.last_outcome, Some(UpdateCheckOutcome::Available(_))))
    {
        36.0
    } else {
        0.0
    }
}

pub(crate) fn show_banner(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: egui::Rect) {
    let t = app.tokens;
    let bg = Color32::from_rgb(0x13, 0x26, 0x4A);
    let border = Color32::from_rgb(0x3B, 0x7A, 0xF5);
    ui.painter().rect_filled(rect, egui::CornerRadius::same(6), bg);
    ui.painter().rect_stroke(rect, egui::CornerRadius::same(6), egui::Stroke::new(1.0, border), egui::StrokeKind::Inside);

    let mut action_install: Option<String> = None;
    let mut action_open_url: Option<String> = None;
    let mut action_later = false;

    let inner = rect.shrink2(egui::vec2(12.0, 4.0));
    ui.scope_builder(egui::UiBuilder::new().max_rect(inner), |ui| {
        ui.horizontal_centered(|ui| {
            if app.updates_rt.install_rx.is_some() {
                ui.spinner();
                ui.label(egui::RichText::new(tl!("Downloading and installing Orpheus update…")).strong().color(Color32::WHITE));
                if let Some(prog) = &app.updates_rt.install_progress {
                    if let Some(frac) = prog.fraction() {
                        ui.add(egui::ProgressBar::new(frac).desired_width(160.0).show_percentage());
                    }
                    ui.ctx().request_repaint();
                }
            } else if let Some(UpdateCheckOutcome::Available(info)) = app.updates_rt.last_outcome.clone() {
                let headline = tlf!("Available: Orpheus v{ver} — Recommended update", ver = info.version);
                ui.label(egui::RichText::new(headline).strong().color(Color32::WHITE));
                if !info.notes.is_empty() {
                    let first_line = info.notes.lines().next().unwrap_or("").trim();
                    if !first_line.is_empty() {
                        let summary: String = first_line.chars().take(72).collect();
                        let summary_resp = ui.label(egui::RichText::new(format!("· {summary}")).small().color(Color32::from_rgb(0xCB, 0xD5, 0xE1)));
                        summary_resp.on_hover_text(&info.notes);
                    }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let later_btn = ui.button(tl!("Later"));
                    app.auto.add("updates.banner.later", later_btn.rect, tl!("Later"));
                    if later_btn.clicked() {
                        action_later = true;
                    }
                    let update_btn = ui.add(
                        egui::Button::new(egui::RichText::new(tl!("Update Now")).strong().color(Color32::WHITE)).fill(Color32::from_rgb(0x2F, 0x6F, 0xED)),
                    );
                    app.auto.add("updates.banner.updateNow", update_btn.rect, tl!("Update Now"));
                    if update_btn.clicked() {
                        if let Some(dl) = info.download_url.clone()
                            && cfg!(windows)
                        {
                            action_install = Some(dl);
                        } else {
                            action_open_url = Some(info.html_url.clone());
                        }
                    }
                });
            } else {
                let _ = t;
            }
        });
    });

    if let Some(dl) = action_install {
        start_install(app, dl);
    }
    if let Some(url) = action_open_url {
        let ctx = ui.ctx().clone();
        crate::links::open(app, &ctx, &url);
    }
    if action_later {
        app.updates_rt.banner_dismissed = true;
    }
}

pub(crate) fn exiting(_app: &mut FilmcraftApp) {}
