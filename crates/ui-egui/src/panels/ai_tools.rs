//! AI Tools panel (`PanelKind::AiTools`): local offline AI / DSP editing tools:
//! 1. **Auto-Cut Silence (Remove Pauses)** (`ai.thresholdDb`, `ai.minPauseMs`, `ai.paddingMs`,
//!    `ai.reset`, `ai.removePauses`).
//! 2. **AI Subtitles** (`ai.subtitles.language`, `ai.subtitles.model`, `ai.subtitles.compute.gpu`,
//!    `ai.subtitles.compute.cpu`, `ai.subtitles.download`, `ai.subtitles.downloadCancel`,
//!    `ai.subtitles.mode.reels`, `ai.subtitles.mode.sentences`, `ai.subtitles.generate`,
//!    `ai.subtitles.cancel`, `ai.subtitles.cliStatus`).

use std::sync::atomic::Ordering;
use std::sync::{Arc, PoisonError};

use egui::{Align2, Color32, CornerRadius, Rect, Sense, Stroke, StrokeKind, pos2, vec2};
use filmcraft_engine::audio_enhance::NoiseReductionStrength;
use filmcraft_engine::auto_reframe::{FacecamRegion, ReframeAspect, ReframeSmoothing};
use filmcraft_engine::beat_sync::{self, BeatInterval, BeatSensitivity};
use filmcraft_engine::bg_removal::{BgEdgeSoftness, BgRemovalMode};
use filmcraft_engine::silence::{SilenceOptions, detect_sequence_silence};
use filmcraft_engine::smart_triggers::TriggerDensity;
use filmcraft_engine::subtitles::{
    self, AiFeatureWeight, ComputeRecommendationReason, SUBTITLE_LANGUAGES, SubtitleComputeDevice, SubtitleStatus, WHISPER_MODELS,
};
use filmcraft_engine::viral_clips::{self, ViralClip, ViralDurationPreset};
use filmcraft_export::Progress;
use filmcraft_time::TICKS_PER_SECOND;
use serde_json::json;

use crate::FilmcraftApp;
use crate::icons::{self, Icon};
use crate::state::{AiToolsState, AiToolsTab};
use crate::theme::Tokens;

const PAD: f32 = 6.0;
const ROYAL_BLUE: Color32 = Color32::from_rgb(0x2F, 0x6F, 0xED);
const ROYAL_BLUE_HOVER: Color32 = Color32::from_rgb(0x3B, 0x7A, 0xF5);
const ROYAL_BLUE_ACTIVE: Color32 = Color32::from_rgb(0x25, 0x5B, 0xCA);
const READY_GREEN: Color32 = Color32::from_rgb(0x34, 0xD3, 0x99);
const WARN_AMBER: Color32 = Color32::from_rgb(0xE0, 0x9A, 0x3C);
const SLOW_RED: Color32 = Color32::from_rgb(0xF8, 0x71, 0x71);

#[derive(Clone, Debug, PartialEq, Eq)]
struct PreviewKey {
    revision: u64,
    seq: Option<u64>,
    selection: Vec<u64>,
    threshold_tenths: i64,
    min_pause_ms: i64,
    padding_ms: i64,
}

enum FinishedOutcome {
    Running,
    Ok(u64),
    Err(String),
}

fn check_job_finished(app: &FilmcraftApp, id: u64) -> Option<FinishedOutcome> {
    let job = app.session.jobs.iter().find(|j| j.id == id)?;
    if !job.progress.finished.load(Ordering::Relaxed) {
        return Some(FinishedOutcome::Running);
    }
    let res = job.result.lock().unwrap_or_else(PoisonError::into_inner);
    match res.as_ref() {
        Some(Ok(rep)) => Some(FinishedOutcome::Ok(rep.frames)),
        Some(Err(e)) => Some(FinishedOutcome::Err(e.clone())),
        None => Some(FinishedOutcome::Ok(0)),
    }
}

/// Poll finished subtitle download / generation jobs and update user status cleanly.
fn poll_vod_chapters_job(app: &mut FilmcraftApp) {
    let Some(id) = app.ui.ai_tools.vod_chapters_job else { return };
    match check_job_finished(app, id) {
        None => app.ui.ai_tools.vod_chapters_job = None,
        Some(FinishedOutcome::Running) => {}
        Some(FinishedOutcome::Ok(_)) => {
            app.ui.ai_tools.vod_chapters_job = None;
            let count =
                app.session.active_sequence().map_or(0, |seq| seq.markers.iter().filter(|m| m.kind == filmcraft_engine::project::MarkerKind::Chapter).count());
            // the worker finished but nothing was attached (no chapters found, project replaced): say so
            let failed = count == 0;
            app.ui.status = if failed {
                tlf!("Stream VOD Chapterizer failed: {err}", err = "no chapters were created")
            } else {
                tlf!("Created {n} stream chapter(s)", n = count)
            };
            app.ui.ai_tools.push_toast(tl!("Stream VOD Chapterizer"), app.ui.status.clone(), failed);
        }
        Some(FinishedOutcome::Err(e)) => {
            app.ui.ai_tools.vod_chapters_job = None;
            app.ui.status = tlf!("Stream VOD Chapterizer failed: {err}", err = e);
            app.ui.ai_tools.push_toast(tl!("Stream VOD Chapterizer"), app.ui.status.clone(), true);
        }
    }
}

pub fn poll_jobs(app: &mut FilmcraftApp) {
    if let Some(id) = app.ui.ai_tools.subtitle_engine_download_job {
        match check_job_finished(app, id) {
            None => app.ui.ai_tools.subtitle_engine_download_job = None,
            Some(FinishedOutcome::Running) => {}
            Some(FinishedOutcome::Ok(_)) => {
                app.ui.ai_tools.subtitle_engine_download_job = None;
                app.ui.ai_tools.subtitle_engine_download_error = None;
                app.ui.status = tl!("AI Engine ready").into();
                app.ui.ai_tools.push_toast(tl!("AI Subtitles"), app.ui.status.clone(), false);
            }
            Some(FinishedOutcome::Err(e)) => {
                app.ui.ai_tools.subtitle_engine_download_job = None;
                app.ui.ai_tools.subtitle_engine_download_error = Some(e.clone());
                app.ui.status = tlf!("AI Engine download failed: {err}", err = e);
                app.ui.ai_tools.push_toast(tl!("AI Subtitles"), app.ui.status.clone(), true);
            }
        }
    }
    if let Some(id) = app.ui.ai_tools.subtitle_download_job {
        match check_job_finished(app, id) {
            None => app.ui.ai_tools.subtitle_download_job = None,
            Some(FinishedOutcome::Running) => {}
            Some(FinishedOutcome::Ok(_)) => {
                app.ui.ai_tools.subtitle_download_job = None;
                app.ui.status = tl!("Model ready offline").into();
                app.ui.ai_tools.push_toast(tl!("AI Subtitles"), app.ui.status.clone(), false);
            }
            Some(FinishedOutcome::Err(e)) => {
                app.ui.ai_tools.subtitle_download_job = None;
                app.ui.status = tlf!("Model download failed: {err}", err = e);
                app.ui.ai_tools.push_toast(tl!("AI Subtitles"), app.ui.status.clone(), true);
            }
        }
    }
    if let Some(id) = app.ui.ai_tools.subtitle_generate_job {
        match check_job_finished(app, id) {
            None => app.ui.ai_tools.subtitle_generate_job = None,
            Some(FinishedOutcome::Running) => {}
            Some(FinishedOutcome::Ok(n)) => {
                app.ui.ai_tools.subtitle_generate_job = None;
                app.ui.status = if n > 0 { tlf!("Generated {n} subtitles", n = n) } else { tl!("No speech detected in audio").into() };
                app.ui.ai_tools.push_toast(tl!("AI Subtitles"), app.ui.status.clone(), false);
            }
            Some(FinishedOutcome::Err(e)) => {
                app.ui.ai_tools.subtitle_generate_job = None;
                app.ui.status = tlf!("Subtitle generation failed: {err}", err = e);
                app.ui.ai_tools.push_toast(tl!("AI Subtitles"), app.ui.status.clone(), true);
            }
        }
    }
    if let Some(id) = app.ui.ai_tools.beat_sync_job {
        match check_job_finished(app, id) {
            None => app.ui.ai_tools.beat_sync_job = None,
            Some(FinishedOutcome::Running) => {}
            Some(FinishedOutcome::Ok(n)) => {
                app.ui.ai_tools.beat_sync_job = None;
                if let Some(cache) = beat_sync::cached_for_session(&app.session, app.ui.ai_tools.beat_sensitivity)
                    && !cache.beats_ms.is_empty()
                {
                    app.ui.status = tlf!("Detected {count} beats ({bpm} BPM)", count = cache.beats_ms.len(), bpm = format!("{:.0}", cache.bpm));
                } else if n == 0 {
                    app.ui.status = tl!("No beats detected in audio").into();
                }
                app.ui.ai_tools.push_toast(tl!("AI Beat Sync"), app.ui.status.clone(), false);
            }
            Some(FinishedOutcome::Err(e)) => {
                app.ui.ai_tools.beat_sync_job = None;
                app.ui.status = tlf!("Beat Sync failed: {err}", err = e);
                app.ui.ai_tools.push_toast(tl!("AI Beat Sync"), app.ui.status.clone(), true);
            }
        }
    }
    if let Some(id) = app.ui.ai_tools.viral_clips_job {
        match check_job_finished(app, id) {
            None => app.ui.ai_tools.viral_clips_job = None,
            Some(FinishedOutcome::Running) => {}
            Some(FinishedOutcome::Ok(n)) => {
                app.ui.ai_tools.viral_clips_job = None;
                app.ui.status = if n > 0 { tlf!("Found {n} viral clips", n = n) } else { tl!("No speech detected in audio").into() };
                app.ui.ai_tools.push_toast(tl!("AI Viral Clip Finder"), app.ui.status.clone(), false);
            }
            Some(FinishedOutcome::Err(e)) => {
                app.ui.ai_tools.viral_clips_job = None;
                app.ui.status = tlf!("Viral Clip Finder failed: {err}", err = e);
                app.ui.ai_tools.push_toast(tl!("AI Viral Clip Finder"), app.ui.status.clone(), true);
            }
        }
    }
    if let Some(id) = app.ui.ai_tools.smart_triggers_job {
        match check_job_finished(app, id) {
            None => app.ui.ai_tools.smart_triggers_job = None,
            Some(FinishedOutcome::Running) => {}
            Some(FinishedOutcome::Ok(n)) => {
                app.ui.ai_tools.smart_triggers_job = None;
                let count =
                    app.session.smart_trigger_report.as_ref().filter(|r| Some(r.seq) == app.session.state.active_sequence).map_or(n, |r| r.applied as u64);
                app.ui.status = if count > 0 { tlf!("Applied {n} smart triggers", n = count) } else { tl!("No emphasis triggers detected in speech").into() };
                app.ui.ai_tools.push_toast(tl!("AI Smart Triggers"), app.ui.status.clone(), false);
            }
            Some(FinishedOutcome::Err(e)) => {
                app.ui.ai_tools.smart_triggers_job = None;
                app.ui.status = tlf!("Smart Triggers failed: {err}", err = e);
                app.ui.ai_tools.push_toast(tl!("AI Smart Triggers"), app.ui.status.clone(), true);
            }
        }
    }
    if let Some(id) = app.ui.ai_tools.audio_enhance_job {
        match check_job_finished(app, id) {
            None => app.ui.ai_tools.audio_enhance_job = None,
            Some(FinishedOutcome::Running) => {}
            Some(FinishedOutcome::Ok(n)) => {
                app.ui.ai_tools.audio_enhance_job = None;
                let count = app
                    .session
                    .audio_enhance_report
                    .as_ref()
                    .filter(|r| Some(r.seq) == app.session.state.active_sequence)
                    .map_or(n, |r| r.enhanced_clips as u64);
                app.ui.status = tlf!("Enhanced {n} audio clips", n = count);
                app.ui.ai_tools.push_toast(tl!("AI Audio Enhance"), app.ui.status.clone(), false);
            }
            Some(FinishedOutcome::Err(e)) => {
                app.ui.ai_tools.audio_enhance_job = None;
                app.ui.status = tlf!("Audio Enhance failed: {err}", err = e);
                app.ui.ai_tools.push_toast(tl!("AI Audio Enhance"), app.ui.status.clone(), true);
            }
        }
    }
    if let Some(id) = app.ui.ai_tools.auto_reframe_job {
        match check_job_finished(app, id) {
            None => app.ui.ai_tools.auto_reframe_job = None,
            Some(FinishedOutcome::Running) => {}
            Some(FinishedOutcome::Ok(n)) => {
                app.ui.ai_tools.auto_reframe_job = None;
                let rep = app.session.auto_reframe_report.as_ref().filter(|r| Some(r.seq) == app.session.state.active_sequence);
                let count = rep.map_or(n, |r| r.reframed_clips as u64);
                let kf = rep.map_or(1, |r| r.total_keyframes as u64);
                let asp = rep.map_or_else(|| app.ui.ai_tools.reframe_aspect.as_id(), |r| r.aspect.as_id());
                let split_fallback = rep.is_some_and(|r| r.split_screen && r.split_screen_clips == 0);
                app.ui.status = if split_fallback {
                    tl!("No webcam detected — fell back to single-frame 9:16 Auto-Reframe").into()
                } else {
                    tlf!("Reframed {n} clip(s) to {aspect} ({kf} keyframes)", n = count, aspect = asp, kf = kf)
                };
                app.ui.ai_tools.push_toast(tl!("AI Auto-Reframe"), app.ui.status.clone(), split_fallback);
            }
            Some(FinishedOutcome::Err(e)) => {
                app.ui.ai_tools.auto_reframe_job = None;
                app.ui.status = tlf!("Auto-Reframe failed: {err}", err = e);
                app.ui.ai_tools.push_toast(tl!("AI Auto-Reframe"), app.ui.status.clone(), true);
            }
        }
    }
    if let Some(id) = app.ui.ai_tools.bg_removal_job {
        match check_job_finished(app, id) {
            None => app.ui.ai_tools.bg_removal_job = None,
            Some(FinishedOutcome::Running) => {}
            Some(FinishedOutcome::Ok(n)) => {
                app.ui.ai_tools.bg_removal_job = None;
                let count =
                    app.session.bg_removal_report.as_ref().filter(|r| Some(r.seq) == app.session.state.active_sequence).map_or(n, |r| r.processed_clips as u64);
                app.ui.status = tlf!("Removed background on {n} clip(s)", n = count);
                app.ui.ai_tools.push_toast(tl!("AI Background Removal"), app.ui.status.clone(), false);
            }
            Some(FinishedOutcome::Err(e)) => {
                app.ui.ai_tools.bg_removal_job = None;
                app.ui.status = tlf!("Background Removal failed: {err}", err = e);
                app.ui.ai_tools.push_toast(tl!("AI Background Removal"), app.ui.status.clone(), true);
            }
        }
    }
    if let Some(id) = app.ui.ai_tools.bg_model_download_job {
        match check_job_finished(app, id) {
            None => app.ui.ai_tools.bg_model_download_job = None,
            Some(FinishedOutcome::Running) => {}
            Some(FinishedOutcome::Ok(_)) => {
                app.ui.ai_tools.bg_model_download_job = None;
                app.ui.ai_tools.bg_model_download_error = None;
                app.ui.status = tl!("AI matting model ready offline").into();
                app.ui.ai_tools.push_toast(tl!("AI Background Removal"), app.ui.status.clone(), false);
            }
            Some(FinishedOutcome::Err(e)) => {
                app.ui.ai_tools.bg_model_download_job = None;
                app.ui.ai_tools.bg_model_download_error = Some(e.clone());
                app.ui.status = tlf!("AI matting model download failed: {err}", err = e);
                app.ui.ai_tools.push_toast(tl!("AI Background Removal"), app.ui.status.clone(), true);
            }
        }
    }
    if let Some(id) = app.ui.ai_tools.ai_models_download_job {
        match check_job_finished(app, id) {
            None => app.ui.ai_tools.ai_models_download_job = None,
            Some(FinishedOutcome::Running) => {}
            Some(FinishedOutcome::Ok(_)) => {
                app.ui.ai_tools.ai_models_download_job = None;
                app.ui.ai_tools.ai_models_download_error = None;
                app.ui.status = tl!("ONNX Neural Model Ready").into();
                app.ui.ai_tools.push_toast(tl!("AI Models"), app.ui.status.clone(), false);
            }
            Some(FinishedOutcome::Err(e)) => {
                app.ui.ai_tools.ai_models_download_job = None;
                app.ui.ai_tools.ai_models_download_error = Some(e.clone());
                app.ui.status = tlf!("Model download failed: {err}", err = e);
                app.ui.ai_tools.push_toast(tl!("AI Models"), app.ui.status.clone(), true);
            }
        }
    }
    if let Some(id) = app.ui.ai_tools.all_in_one_job {
        match check_job_finished(app, id) {
            None => app.ui.ai_tools.all_in_one_job = None,
            Some(FinishedOutcome::Running) => {}
            Some(FinishedOutcome::Ok(_)) => {
                app.ui.ai_tools.all_in_one_job = None;
                app.ui.status = tl!("Orpheus All-In-One complete").into();
                app.ui.ai_tools.push_toast(tl!("Orpheus All-In-One"), app.ui.status.clone(), false);
            }
            Some(FinishedOutcome::Err(e)) => {
                app.ui.ai_tools.all_in_one_job = None;
                app.ui.status = tlf!("Orpheus All-In-One failed: {err}", err = e);
                app.ui.ai_tools.push_toast(tl!("Orpheus All-In-One"), app.ui.status.clone(), true);
            }
        }
    }
    poll_vod_chapters_job(app);
    if !app.ui.ai_tools.has_active_mutation_job()
        && let Some(next) = app.ui.ai_tools.job_queue.pending.pop_front()
    {
        let ctx = egui::Context::default();
        let _ = crate::menus::invoke(app, &ctx, &next.id, next.params);
    }
}

/// Non-blocking bottom-right AI Toast & Queue Progress overlay.
pub fn show_toast_overlay(app: &mut FilmcraftApp, ctx: &egui::Context) {
    let running_job = app.ui.ai_tools.has_active_mutation_job()
        || app.ui.ai_tools.bg_model_download_job.is_some()
        || app.ui.ai_tools.ai_models_download_job.is_some()
        || app.ui.ai_tools.subtitle_engine_download_job.is_some()
        || app.ui.ai_tools.subtitle_download_job.is_some();
    let queued_count = app.ui.ai_tools.job_queue.pending.len();
    let now = ctx.input(|i| i.time);

    if let Some(ref mut toast) = app.ui.ai_tools.toast {
        let start = *toast.shown_at.get_or_insert(now);
        if now - start >= 4.5 {
            app.ui.ai_tools.toast = None;
        } else {
            ctx.request_repaint_after(std::time::Duration::from_millis(200));
        }
    }

    if !running_job && app.ui.ai_tools.toast.is_none() {
        return;
    }

    let t = app.tokens;
    let mut dismiss = false;
    let toast_snapshot = app.ui.ai_tools.toast.clone();

    egui::Area::new(egui::Id::unique("orpheus-ai-toast-overlay")).order(egui::Order::Foreground).anchor(egui::Align2::RIGHT_BOTTOM, vec2(-18.0, -30.0)).show(
        ctx,
        |ui| {
            let bg = if app.ui.dark { Color32::from_rgb(0x16, 0x19, 0x24) } else { t.panel_bg };
            let border = if toast_snapshot.as_ref().is_some_and(|ts| ts.is_error) { t.danger } else { Color32::from_rgb(0x10, 0xB9, 0x81) };
            egui::Frame::new()
                .fill(bg)
                .stroke(Stroke::new(1.2, border))
                .corner_radius(CornerRadius::same(8))
                .inner_margin(egui::Margin::symmetric(12, 9))
                .show(ui, |ui| {
                    ui.set_max_width(320.0);
                    if running_job {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            let active_label = if app.ui.ai_tools.bg_removal_job.is_some() {
                                tl!("Removing background…")
                            } else if app.ui.ai_tools.bg_model_download_job.is_some() {
                                tl!("Downloading AI matting model…")
                            } else if app.ui.ai_tools.ai_models_download_job.is_some() {
                                tl!("Downloading AI models…")
                            } else if app.ui.ai_tools.audio_enhance_job.is_some() {
                                tl!("Enhancing audio…")
                            } else if app.ui.ai_tools.auto_reframe_job.is_some() {
                                tl!("Tracking subject for Auto-Reframe…")
                            } else if app.ui.ai_tools.smart_triggers_job.is_some() {
                                tl!("Analyzing speech for Smart Triggers…")
                            } else if app.ui.ai_tools.viral_clips_job.is_some() {
                                tl!("Analyzing viral clips…")
                            } else if app.ui.ai_tools.beat_sync_job.is_some() {
                                tl!("Detecting beats…")
                            } else if app.ui.ai_tools.all_in_one_job.is_some() {
                                tl!("Running Orpheus All-In-One…")
                            } else if app.ui.ai_tools.vod_chapters_job.is_some() {
                                tl!("Chapterizing stream VOD…")
                            } else if app.ui.ai_tools.subtitle_download_job.is_some() {
                                tl!("Downloading AI model…")
                            } else {
                                tl!("Generating AI subtitles…")
                            };
                            let msg = if queued_count > 0 { format!("{active_label} (+{queued_count})") } else { active_label.to_string() };
                            ui.label(egui::RichText::new(msg).font(Tokens::semibold(11.5)).color(t.text));
                        });
                    }
                    if let Some(ts) = &toast_snapshot {
                        if running_job {
                            ui.add_space(4.0);
                        }
                        ui.horizontal(|ui| {
                            let badge_col = if ts.is_error { t.danger } else { Color32::from_rgb(0x10, 0xB9, 0x81) };
                            let badge_sym = if ts.is_error { "!" } else { "✓" };
                            ui.label(egui::RichText::new(badge_sym).font(Tokens::semibold(13.5)).color(badge_col));
                            ui.vertical(|ui| {
                                ui.label(egui::RichText::new(&ts.title).font(Tokens::semibold(12.0)).color(t.text));
                                ui.label(egui::RichText::new(&ts.summary).font(Tokens::ui(11.0)).color(t.text_dim));
                            });
                            let close_btn = ui.small_button("×");
                            app.auto.add("ai.toast.dismiss", close_btn.rect, "Dismiss");
                            if close_btn.clicked() {
                                dismiss = true;
                            }
                        });
                    }
                });
        },
    );

    if dismiss {
        app.ui.ai_tools.toast = None;
    }
}

pub fn show(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect) {
    let t = app.tokens;
    ui.painter().rect_filled(rect, 0.0, t.panel_bg);
    if rect.width() < 80.0 || rect.height() < 80.0 {
        return;
    }

    let inner = rect.shrink(PAD);
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(inner).layout(egui::Layout::top_down(egui::Align::Min)));
    child.set_clip_rect(inner);

    let refresh_status = app.ui.ai_tools.subtitle_download_job.is_none() && app.ui.ai_tools.subtitle_engine_download_job.is_none();
    let status = cached_subtitle_status(ui.ctx(), !refresh_status);

    draw_section_tabs(app, &mut child, &t);
    child.add_space(6.0);

    let scroll_w = (inner.width() - 6.0).max(40.0);
    egui::ScrollArea::vertical().auto_shrink([false, false]).id_salt("ai-tools-scroll").show(&mut child, |ui| {
        ui.set_width(scroll_w);
        match app.ui.ai_tools.active_tab {
            AiToolsTab::AllInOne => {
                draw_all_in_one_tab(app, ui, &t);
            }
            AiToolsTab::AudioText => {
                draw_subtitles_card(app, ui, &t, &status);
                ui.add_space(4.0);
                draw_auto_cut_card(app, ui, &t, &status);
                ui.add_space(4.0);
                draw_beat_sync_card(app, ui, &t, &status);
                ui.add_space(4.0);
                draw_viral_clips_card(app, ui, &t, &status);
                ui.add_space(4.0);
                draw_audio_enhance_card(app, ui, &t, &status);
            }
            AiToolsTab::VideoVisuals => {
                draw_smart_triggers_card(app, ui, &t, &status);
                ui.add_space(4.0);
                draw_auto_reframe_card(app, ui, &t, &status);
                ui.add_space(4.0);
                draw_bg_removal_card(app, ui, &t, &status);
                ui.add_space(4.0);
                draw_vod_chapters_card(app, ui, &t, &status);
            }
        }
    });
}

fn draw_section_tabs(app: &mut FilmcraftApp, ui: &mut egui::Ui, t: &Tokens) {
    let w = ui.available_width().max(60.0);
    let (bar_r, _) = ui.allocate_exact_size(vec2(w, 28.0), Sense::hover());
    let bg = if app.ui.dark { Color32::from_rgb(0x11, 0x13, 0x1A) } else { t.field_bg };
    ui.painter().rect_filled(bar_r, CornerRadius::same(7), bg);
    ui.painter().rect_stroke(bar_r, CornerRadius::same(7), Stroke::new(1.0, t.control_border), StrokeKind::Inside);

    let pad = 3.0;
    let gap = 3.0;
    let tab_w = ((bar_r.width() - pad * 2.0 - gap * 2.0) / 3.0).max(20.0);
    let tab_h = (bar_r.height() - pad * 2.0).max(16.0);
    let t0_r = Rect::from_min_size(pos2(bar_r.min.x + pad, bar_r.min.y + pad), vec2(tab_w, tab_h));
    let t1_r = Rect::from_min_size(pos2(t0_r.max.x + gap, bar_r.min.y + pad), vec2(tab_w, tab_h));
    let t2_r = Rect::from_min_size(pos2(t1_r.max.x + gap, bar_r.min.y + pad), vec2(tab_w, tab_h));

    if draw_section_tab(
        ui,
        &mut app.auto,
        t,
        app.ui.dark,
        t0_r,
        "ai.tab.allInOne",
        Icon::Sparkle,
        tl!("Orpheus All-In-One"),
        app.ui.ai_tools.active_tab == AiToolsTab::AllInOne,
    ) {
        app.ui.ai_tools.active_tab = AiToolsTab::AllInOne;
    }
    if draw_section_tab(
        ui,
        &mut app.auto,
        t,
        app.ui.dark,
        t1_r,
        "ai.tab.audioText",
        Icon::Audio,
        tl!("Audio & Text"),
        app.ui.ai_tools.active_tab == AiToolsTab::AudioText,
    ) {
        app.ui.ai_tools.active_tab = AiToolsTab::AudioText;
    }
    if draw_section_tab(
        ui,
        &mut app.auto,
        t,
        app.ui.dark,
        t2_r,
        "ai.tab.videoVisuals",
        Icon::Film,
        tl!("Video & Visuals"),
        app.ui.ai_tools.active_tab == AiToolsTab::VideoVisuals,
    ) {
        app.ui.ai_tools.active_tab = AiToolsTab::VideoVisuals;
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_section_tab(
    ui: &mut egui::Ui,
    auto: &mut crate::automation::Registry,
    t: &Tokens,
    dark: bool,
    rect: Rect,
    id: &str,
    icon: Icon,
    label: &str,
    active: bool,
) -> bool {
    let resp = ui.interact(rect, egui::Id::unique(id), Sense::click());
    auto.add(id, rect, label);

    let fill = if active {
        if dark { Color32::from_rgb(0x1E, 0x26, 0x38) } else { t.control_surface }
    } else if resp.hovered() {
        t.hover
    } else {
        Color32::TRANSPARENT
    };
    if fill != Color32::TRANSPARENT {
        ui.painter().rect_filled(rect, CornerRadius::same(5), fill);
    }
    if active {
        ui.painter().rect_stroke(rect, CornerRadius::same(5), Stroke::new(1.0, Color32::from_rgba_unmultiplied(0x2F, 0x6F, 0xED, 150)), StrokeKind::Inside);
    }

    let fg = if active { t.text } else { t.text_dim };
    let icon_col = if active { ROYAL_BLUE } else { t.text_dim };
    let icon_s = 12.0;
    let gap = 5.0;
    let font = if active { Tokens::semibold(11.5) } else { Tokens::ui(11.5) };
    let max_text_w = (rect.width() - icon_s - gap - 10.0).max(8.0);
    let g = layout_truncated(ui.painter(), label, font, fg, max_text_w);
    let total_w = icon_s + gap + g.size().x;
    let sx = (rect.center().x - total_w / 2.0).max(rect.min.x + 5.0);
    icons::paint(ui.painter(), Rect::from_center_size(pos2(sx + icon_s / 2.0, rect.center().y), vec2(icon_s, icon_s)), icon, icon_col);
    ui.painter().galley(pos2(sx + icon_s + gap, rect.center().y - g.size().y / 2.0), g, fg);

    resp.clicked()
}

fn card_frame(t: &Tokens, dark: bool) -> (Color32, Color32, egui::Frame) {
    let card_bg = if dark { Color32::from_rgb(0x16, 0x19, 0x22) } else { t.control_surface };
    let card_border = if dark { Color32::from_rgba_unmultiplied(255, 255, 255, 22) } else { t.control_border };
    let frame = egui::Frame::new().fill(card_bg).stroke(Stroke::new(1.0, card_border)).corner_radius(CornerRadius::same(8)).inner_margin(egui::Margin::same(7));
    (card_bg, card_border, frame)
}

fn draw_help_badge(ui: &mut egui::Ui, auto: &mut crate::automation::Registry, t: &Tokens, id: &str, tip: &str) {
    let (r, resp) = ui.allocate_exact_size(vec2(16.0, 16.0), Sense::click());
    let pin_id = egui::Id::unique(("ai-help-pin", id));
    let mut pinned = ui.ctx().data(|d| d.get_temp::<bool>(pin_id).unwrap_or(false));
    if resp.clicked() {
        pinned = !pinned;
    } else if pinned && ui.input(|i| i.pointer.any_pressed()) && !resp.hovered() {
        pinned = false;
    }
    ui.ctx().data_mut(|d| d.insert_temp(pin_id, pinned));

    let active = resp.hovered() || pinned;
    let center = r.center();
    let radius = 7.0;
    let fill = if active { t.hover } else { t.field_bg };
    let stroke_col = if active { ROYAL_BLUE } else { t.control_border };
    let text_col = if active { t.text } else { t.text_dim };
    ui.painter().circle_filled(center, radius, fill);
    ui.painter().circle_stroke(center, radius, Stroke::new(1.0, stroke_col));
    ui.painter().text(center, Align2::CENTER_CENTER, "?", Tokens::semibold(10.0), text_col);

    // Zero-delay instant tooltip on hover (or pinned via click).
    if active {
        let anchor = if resp.hovered() { egui::PopupAnchor::Pointer } else { egui::PopupAnchor::from(r) };
        egui::Tooltip::always_open(ui.ctx().clone(), ui.layer_id(), egui::Id::unique(("ai-help-tip", id)), anchor).show(|ui| {
            ui.set_max_width(260.0);
            ui.label(tip);
        });
    }
    auto.add(id, r, tip);
}

struct BadgeStyle {
    label: String,
    tip: String,
    fg: Color32,
    fill: Color32,
}

fn badge_style(weight: AiFeatureWeight, status: &SubtitleStatus, dark: bool) -> BadgeStyle {
    use filmcraft_engine::ai_models::{ResolvedExecutionProvider, active_execution_mode, assess_machine_specs, detect_system_ram_mb};

    let green_fg = if dark { READY_GREEN } else { Color32::from_rgb(0x0E, 0x7C, 0x52) };
    let green_fill = Color32::from_rgba_unmultiplied(0x34, 0xD3, 0x99, if dark { 28 } else { 38 });
    let yellow_fg = if dark { WARN_AMBER } else { Color32::from_rgb(0xA1, 0x62, 0x07) };
    let yellow_fill = Color32::from_rgba_unmultiplied(0xE0, 0x9A, 0x3C, if dark { 28 } else { 38 });
    let red_fg = if dark { SLOW_RED } else { Color32::from_rgb(0xC8, 0x1E, 0x1E) };
    let red_fill = Color32::from_rgba_unmultiplied(0xF8, 0x71, 0x71, if dark { 28 } else { 38 });

    if weight == AiFeatureWeight::Lightweight {
        return BadgeStyle { label: tl!("Fast (CPU)").into(), tip: tl!("Optimal performance on your hardware").into(), fg: green_fg, fill: green_fill };
    }

    let mode = active_execution_mode();
    let ram_mb = detect_system_ram_mb();
    let specs = assess_machine_specs(mode, status.gpu.as_ref(), ram_mb, status.cpu_cores);
    let has_gpu_accel = match weight {
        AiFeatureWeight::HeavyVisual => specs.resolved_provider == ResolvedExecutionProvider::DirectML,
        AiFeatureWeight::NeuralSpeech => {
            specs.resolved_provider == ResolvedExecutionProvider::DirectML
                || (status.vulkan_dll_found && mode != filmcraft_engine::ai_models::AiExecutionMode::Cpu)
        }
        AiFeatureWeight::Lightweight => false,
    };

    if has_gpu_accel && specs.dedicated_vram_mb >= 4096 {
        let vram_gb = ((specs.dedicated_vram_mb as f64) / 1024.0).round() as u64;
        let gpu_name = specs.gpu_name.clone().unwrap_or_else(|| "GPU".into());
        let backend = if weight == AiFeatureWeight::HeavyVisual || !status.vulkan_dll_found { "DirectML" } else { "Vulkan" };
        BadgeStyle {
            label: tl!("Fast (GPU)").into(),
            tip: tlf!("Fast GPU acceleration with {gpu} ({vram} GB VRAM, {backend})", gpu = gpu_name, vram = vram_gb, backend = backend),
            fg: green_fg,
            fill: green_fill,
        }
    } else if has_gpu_accel || (specs.cpu_cores >= 6 && specs.system_ram_mb >= 8192) {
        let lbl = if has_gpu_accel { tl!("Moderate (GPU)") } else { tl!("Moderate (CPU)") };
        BadgeStyle { label: lbl.into(), tip: tl!("Moderate performance on CPU or < 4 GB VRAM").into(), fg: yellow_fg, fill: yellow_fill }
    } else {
        BadgeStyle { label: tl!("Slow (CPU)").into(), tip: tl!("Slow on CPU-only or low-RAM/VRAM hardware").into(), fg: red_fg, fill: red_fill }
    }
}

fn draw_hardware_badge_row(
    ui: &mut egui::Ui,
    auto: &mut crate::automation::Registry,
    badge_id: &str,
    weight: AiFeatureWeight,
    status: &SubtitleStatus,
    dark: bool,
) {
    let style = badge_style(weight, status, dark);
    let max_w = ui.available_width().max(40.0);
    let g = layout_truncated(ui.painter(), &style.label, Tokens::semibold(9.5), style.fg, (max_w - 22.0).max(16.0));
    let pill_w = (g.size().x + 20.0).min(max_w);
    let (pill_r, resp) = ui.allocate_exact_size(vec2(pill_w, 16.0), Sense::hover());
    resp.on_hover_text(&style.tip);
    ui.painter().rect_filled(pill_r, CornerRadius::same(4), style.fill);
    ui.painter().circle_filled(pos2(pill_r.min.x + 8.0, pill_r.center().y), 3.0, style.fg);
    ui.painter().galley(pos2(pill_r.min.x + 14.0, pill_r.center().y - g.size().y / 2.0), g, style.fg);
    auto.add(badge_id, pill_r, &style.label);
}

fn beat_sensitivity_label(s: BeatSensitivity) -> &'static str {
    match s {
        BeatSensitivity::Low => tl!("Low"),
        BeatSensitivity::Medium => tl!("Medium"),
        BeatSensitivity::High => tl!("High"),
    }
}

fn beat_interval_label(iv: BeatInterval) -> &'static str {
    match iv {
        BeatInterval::EveryBeat => tl!("Every Beat"),
        BeatInterval::Every2Beats => tl!("Every 2 Beats"),
        BeatInterval::Every4Beats => tl!("Every 4 Beats"),
    }
}

struct ComboRowSpec<'a> {
    salt: &'a str,
    auto_id: &'a str,
    label: &'a str,
}

fn draw_combo_row<T: Copy + PartialEq>(
    ui: &mut egui::Ui,
    auto: &mut crate::automation::Registry,
    t: &Tokens,
    spec: &ComboRowSpec<'_>,
    choices: &[(T, &str)],
    current: &mut T,
) {
    let cur_label = choices.iter().find(|(v, _)| v == current).map_or("", |(_, s)| *s);
    let total_w = ui.available_width();
    let combo_w = (total_w * 0.56).clamp(84.0, 165.0).min((total_w - 64.0).max(60.0));
    let label_w = (total_w - combo_w - 6.0).max(40.0);
    let shown_selected = layout_truncated(ui.painter(), cur_label, Tokens::ui(11.5), t.text, (combo_w - 28.0).max(24.0)).text().to_string();
    let mut rect_out = Rect::NOTHING;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        ui.add_sized(vec2(label_w, 18.0), egui::Label::new(egui::RichText::new(spec.label).font(Tokens::ui(11.5)).color(t.text_dim)).truncate());
        let cb = craft_ui::ComboBox::from_id_salt(spec.salt).selected_text(&shown_selected).width(combo_w).show_ui(ui, |ui| {
            for &(val, shown) in choices {
                if ui.selectable_label(*current == val, shown).clicked() {
                    *current = val;
                }
            }
        });
        rect_out = cb.response.rect;
    });
    auto.add(spec.auto_id, rect_out, spec.label);
}

fn draw_beat_sync_card(app: &mut FilmcraftApp, ui: &mut egui::Ui, t: &Tokens, status: &SubtitleStatus) {
    let (_card_bg, _card_border, frame) = card_frame(t, app.ui.dark);
    let card_w = (ui.available_width() - 14.0).max(40.0);
    let sens = app.ui.ai_tools.beat_sensitivity;
    let cached = beat_sync::cached_for_session(&app.session, sens).map(|c| (c.beats_ms.len(), c.bpm));

    frame.show(ui, |ui| {
        ui.set_width(card_w);
        ui.spacing_mut().item_spacing.y = 3.0;
        ui.spacing_mut().interact_size.y = 18.0;

        ui.horizontal(|ui| {
            let (icon_r, _) = ui.allocate_exact_size(vec2(14.0, 14.0), Sense::hover());
            icons::paint(ui.painter(), icon_r, Icon::Audio, Color32::from_rgb(0x38, 0xBD, 0xF8));
            let title_w = (ui.available_width() - 22.0).max(40.0);
            ui.add_sized(vec2(title_w, 18.0), egui::Label::new(egui::RichText::new(tl!("AI Beat Sync")).font(Tokens::semibold(13.0)).color(t.text)).truncate())
                .on_hover_text(tl!("Automatically analyze music transients and align edits to the beat."));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                draw_help_badge(
                    ui,
                    &mut app.auto,
                    t,
                    "ai.help.beatSync",
                    tl!("Finds musical beats and places markers or auto-cuts video to the rhythm. Needs a sequence with a music or audio track."),
                );
            });
        });

        draw_hardware_badge_row(ui, &mut app.auto, "ai.badge.beatSync", AiFeatureWeight::Lightweight, status, app.ui.dark);

        let has_seq = app.session.active_sequence().is_some();
        if let Some((count, bpm)) = cached {
            let (summary, col) = if count > 0 {
                (tlf!("Detected {count} beats ({bpm} BPM)", count = count, bpm = format!("{:.0}", bpm)), Color32::from_rgb(0x7C, 0xB3, 0xFF))
            } else {
                (tl!("No beats detected in audio").into(), t.text_dim)
            };
            let resp = ui.add(egui::Label::new(egui::RichText::new(summary.clone()).font(Tokens::ui(10.5)).color(col)).wrap());
            app.auto.add("ai.beatSync.status", resp.rect, &summary);
        } else if !has_seq {
            let hint = tl!("Open a sequence with audio first");
            let resp = ui.add(egui::Label::new(egui::RichText::new(hint).font(Tokens::ui(10.5)).color(t.text_dim)).wrap());
            app.auto.add("ai.beatSync.status", resp.rect, hint);
        } else {
            let desc = tl!("Automatically analyze music transients and align edits to the beat.");
            let resp = ui.add(egui::Label::new(egui::RichText::new(desc).font(Tokens::ui(10.5)).color(t.text_dim)).wrap());
            app.auto.add("ai.beatSync.status", resp.rect, desc);
        }

        let sens_choices = [
            (BeatSensitivity::Low, beat_sensitivity_label(BeatSensitivity::Low)),
            (BeatSensitivity::Medium, beat_sensitivity_label(BeatSensitivity::Medium)),
            (BeatSensitivity::High, beat_sensitivity_label(BeatSensitivity::High)),
        ];
        draw_combo_row(
            ui,
            &mut app.auto,
            t,
            &ComboRowSpec { salt: "ai-beat-sens", auto_id: "ai.beatSync.sensitivity", label: tl!("Sensitivity") },
            &sens_choices,
            &mut app.ui.ai_tools.beat_sensitivity,
        );

        let iv_choices = [
            (BeatInterval::EveryBeat, beat_interval_label(BeatInterval::EveryBeat)),
            (BeatInterval::Every2Beats, beat_interval_label(BeatInterval::Every2Beats)),
            (BeatInterval::Every4Beats, beat_interval_label(BeatInterval::Every4Beats)),
        ];
        draw_combo_row(
            ui,
            &mut app.auto,
            t,
            &ComboRowSpec { salt: "ai-beat-interval", auto_id: "ai.beatSync.interval", label: tl!("Cut Interval") },
            &iv_choices,
            &mut app.ui.ai_tools.beat_interval,
        );

        ui.add_space(1.0);
        if let Some(prog) = active_job_progress(app, app.ui.ai_tools.beat_sync_job) {
            let shown = nonblocking_progress_status(&prog, tl!("Analyzing audio beats…"));
            draw_job_bar(ui, &mut app.auto, t, prog.fraction(), &shown, &prog, "ai.beatSync.cancel");
            return;
        }

        let sens_param = app.ui.ai_tools.beat_sensitivity.as_id();
        let iv_step = app.ui.ai_tools.beat_interval.step();

        if cached.is_none() && has_seq {
            let analyze_label = tl!("Analyze Beats");
            if draw_secondary_button(ui, &mut app.auto, t, "ai.beatSync.analyze", analyze_label, analyze_label) {
                let ctx = ui.ctx().clone();
                let _ = crate::menus::invoke(app, &ctx, "ai.analyzeBeats", json!({ "sensitivity": sens_param, "interval": iv_step }));
            }
        }

        let marker_tip = (!has_seq).then_some(tl!("Open a sequence with audio first"));
        let marker_clicked =
            draw_primary_icon_button(ui, &mut app.auto, t, "ai.beatSync.addMarkers", tl!("Add Beat Markers"), Icon::Marker, has_seq, true, marker_tip);
        if has_seq && marker_clicked {
            let ctx = ui.ctx().clone();
            let _ = crate::menus::invoke(app, &ctx, "ai.addBeatMarkers", json!({ "sensitivity": sens_param, "interval": iv_step }));
        }

        let cut_tip = (!has_seq).then_some(tl!("Open a sequence with audio first"));
        let cut_clicked =
            draw_primary_icon_button(ui, &mut app.auto, t, "ai.beatSync.cutVideo", tl!("Auto-Cut Video to Beats"), Icon::Razor, has_seq, false, cut_tip);
        if has_seq && cut_clicked {
            let ctx = ui.ctx().clone();
            let _ = crate::menus::invoke(app, &ctx, "ai.cutVideoToBeats", json!({ "sensitivity": sens_param, "interval": iv_step }));
        }
    });
}

fn viral_duration_label(p: ViralDurationPreset) -> &'static str {
    match p {
        ViralDurationPreset::Short15To30 => tl!("15–30s"),
        ViralDurationPreset::Medium30To60 => tl!("30–60s"),
        ViralDurationPreset::Long60To90 => tl!("60–90s"),
    }
}

fn viral_score_colors(score: u8, dark: bool) -> (Color32, Color32) {
    if score >= 80 {
        let fg = if dark { READY_GREEN } else { Color32::from_rgb(0x0E, 0x7C, 0x52) };
        (fg, Color32::from_rgba_unmultiplied(0x34, 0xD3, 0x99, if dark { 28 } else { 38 }))
    } else if score >= 60 {
        let fg = if dark { WARN_AMBER } else { Color32::from_rgb(0xA1, 0x62, 0x07) };
        (fg, Color32::from_rgba_unmultiplied(0xE0, 0x9A, 0x3C, if dark { 28 } else { 38 }))
    } else {
        let fg = if dark { Color32::from_rgb(0x93, 0xC5, 0xFD) } else { Color32::from_rgb(0x1D, 0x4E, 0xD8) };
        (fg, Color32::from_rgba_unmultiplied(0x2F, 0x6F, 0xED, if dark { 28 } else { 34 }))
    }
}

struct CompactActionSpec<'a> {
    id: &'a str,
    label: &'a str,
    tooltip: &'a str,
    icon: Icon,
    accent: bool,
}

fn draw_compact_action_button(ui: &mut egui::Ui, auto: &mut crate::automation::Registry, t: &Tokens, dark: bool, w: f32, spec: &CompactActionSpec<'_>) -> bool {
    let (r, resp) = ui.allocate_exact_size(vec2(w.max(40.0), 22.0), Sense::click());
    let resp = resp.on_hover_text(spec.tooltip);
    auto.add(spec.id, r, spec.label);
    let fill = if spec.accent {
        if resp.is_pointer_button_down_on() {
            ROYAL_BLUE_ACTIVE
        } else if resp.hovered() {
            ROYAL_BLUE_HOVER
        } else {
            ROYAL_BLUE
        }
    } else if resp.hovered() {
        t.hover
    } else if dark {
        Color32::from_rgb(0x1E, 0x24, 0x33)
    } else {
        t.control_surface
    };
    ui.painter().rect_filled(r, CornerRadius::same(5), fill);
    if !spec.accent {
        ui.painter().rect_stroke(r, CornerRadius::same(5), Stroke::new(1.0, t.control_border), StrokeKind::Inside);
    }
    let fg = if spec.accent { Color32::WHITE } else { t.text };
    let icon_s = 11.0;
    let gap = 4.0;
    let g = layout_truncated(ui.painter(), spec.label, Tokens::semibold(10.5), fg, (r.width() - icon_s - gap - 10.0).max(12.0));
    let total = icon_s + gap + g.size().x;
    let sx = (r.center().x - total / 2.0).max(r.min.x + 5.0);
    icons::paint(ui.painter(), Rect::from_center_size(pos2(sx + icon_s / 2.0, r.center().y), vec2(icon_s, icon_s)), spec.icon, fg);
    ui.painter().galley(pos2(sx + icon_s + gap, r.center().y - g.size().y / 2.0), g, fg);
    resp.clicked()
}

fn draw_viral_clip_item(app: &mut FilmcraftApp, ui: &mut egui::Ui, t: &Tokens, clip: &ViralClip, preset: ViralDurationPreset) {
    let bg = if app.ui.dark { Color32::from_rgb(0x11, 0x14, 0x1D) } else { t.field_bg };
    let frame = egui::Frame::new().fill(bg).stroke(Stroke::new(1.0, t.control_border)).corner_radius(CornerRadius::same(6)).inner_margin(egui::Margin::same(6));
    let inner_w = ui.available_width().max(40.0);

    frame.show(ui, |ui| {
        ui.set_width((inner_w - 14.0).max(40.0));
        ui.spacing_mut().item_spacing.y = 3.0;

        let dur_sec = ((clip.duration_ms as f64) / 1000.0).round().max(1.0) as u64;
        let tc_text = format!("{} · {}s", clip.start_timecode, dur_sec);
        let tc_tip = format!("{} – {}", clip.start_timecode, clip.end_timecode);
        let badge_text = format!("#{} · {}", clip.rank, tlf!("Score {score}/100", score = clip.score));
        let (badge_fg, badge_fill) = viral_score_colors(clip.score, app.ui.dark);

        ui.horizontal(|ui| {
            let max_badge_w = (ui.available_width() * 0.52).max(52.0);
            let g = layout_truncated(ui.painter(), &badge_text, Tokens::semibold(9.5), badge_fg, (max_badge_w - 10.0).max(16.0));
            let pill_w = (g.size().x + 12.0).min(max_badge_w);
            let (pill_r, _) = ui.allocate_exact_size(vec2(pill_w, 16.0), Sense::hover());
            ui.painter().rect_filled(pill_r, CornerRadius::same(4), badge_fill);
            ui.painter().galley(pos2(pill_r.min.x + 6.0, pill_r.center().y - g.size().y / 2.0), g, badge_fg);

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add(egui::Label::new(egui::RichText::new(&tc_text).font(Tokens::mono(10.0)).color(t.hot_text)).truncate()).on_hover_text(tc_tip);
            });
        });

        let hook_resp = ui
            .add(egui::Label::new(egui::RichText::new(&clip.title).font(Tokens::semibold(11.0)).color(t.text)).wrap().sense(Sense::click()))
            .on_hover_cursor(egui::CursorIcon::PointingHand)
            .on_hover_text(tl!("Jump to clip start and add timeline In/Out range marker"));
        app.auto.add(&format!("ai.viralClips.clip.{}", clip.rank), hook_resp.rect, &clip.title);

        let btn_gap = 5.0;
        let half_w = ((ui.available_width() - btn_gap) / 2.0).max(40.0);
        let payload = json!({
            "rank": clip.rank,
            "preset": preset.as_id(),
            "start": clip.start.0,
            "end": clip.end.0,
            "score": clip.score,
            "title": clip.title,
        });
        if hook_resp.clicked() {
            let ctx = ui.ctx().clone();
            let _ = crate::menus::invoke(app, &ctx, "ai.jumpToViralClip", payload.clone());
        }

        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = btn_gap;
            let jump_id = format!("ai.viralClips.jump.{}", clip.rank);
            if draw_compact_action_button(
                ui,
                &mut app.auto,
                t,
                app.ui.dark,
                half_w,
                &CompactActionSpec {
                    id: &jump_id,
                    label: tl!("Jump / Mark"),
                    tooltip: tl!("Jump to clip start and add timeline In/Out range marker"),
                    icon: Icon::Marker,
                    accent: false,
                },
            ) {
                let ctx = ui.ctx().clone();
                let _ = crate::menus::invoke(app, &ctx, "ai.jumpToViralClip", payload.clone());
            }

            let extract_id = format!("ai.viralClips.extract.{}", clip.rank);
            if draw_compact_action_button(
                ui,
                &mut app.auto,
                t,
                app.ui.dark,
                half_w,
                &CompactActionSpec {
                    id: &extract_id,
                    label: tl!("Extract Clip"),
                    tooltip: tl!("Create trimmed sequence copy for this clip"),
                    icon: Icon::Extract,
                    accent: true,
                },
            ) {
                let ctx = ui.ctx().clone();
                let _ = crate::menus::invoke(app, &ctx, "ai.extractViralClip", payload);
            }
        });
    });
}

fn draw_viral_clips_card(app: &mut FilmcraftApp, ui: &mut egui::Ui, t: &Tokens, status: &SubtitleStatus) {
    let (_card_bg, _card_border, frame) = card_frame(t, app.ui.dark);
    let card_w = (ui.available_width() - 14.0).max(40.0);
    let preset = app.ui.ai_tools.viral_duration;
    let has_seq = app.session.active_sequence().is_some();
    let has_subtitles = app.session.active_sequence().is_some_and(|q| viral_clips::sequence_has_subtitles(&app.session.project, q));
    let cached = viral_clips::latest_cached_clips(&app.session, preset).map(<[ViralClip]>::to_vec);

    frame.show(ui, |ui| {
        ui.set_width(card_w);
        ui.spacing_mut().item_spacing.y = 3.0;
        ui.spacing_mut().interact_size.y = 18.0;

        ui.horizontal(|ui| {
            let (icon_r, _) = ui.allocate_exact_size(vec2(14.0, 14.0), Sense::hover());
            icons::paint(ui.painter(), icon_r, Icon::Sparkle, Color32::from_rgb(0xF4, 0x72, 0xB6));
            let title_w = (ui.available_width() - 22.0).max(40.0);
            ui.add_sized(
                vec2(title_w, 18.0),
                egui::Label::new(egui::RichText::new(tl!("AI Viral Clip Finder")).font(Tokens::semibold(13.0)).color(t.text)).truncate(),
            )
            .on_hover_text(tl!("Identify high-impact speech moments for short-form social clips."));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                draw_help_badge(
                    ui,
                    &mut app.auto,
                    t,
                    "ai.help.viralClips",
                    tl!("Scores spoken highlights to find short-form hooks you can jump to or extract. Needs a sequence with spoken audio or subtitles."),
                );
            });
        });

        draw_hardware_badge_row(ui, &mut app.auto, "ai.badge.viralClips", AiFeatureWeight::Lightweight, status, app.ui.dark);

        let (summary, col) = if let Some(ref clips) = cached {
            if clips.is_empty() {
                (tl!("No speech detected in audio").into(), t.text_dim)
            } else {
                (tlf!("Found {n} viral clips", n = clips.len()), Color32::from_rgb(0x7C, 0xB3, 0xFF))
            }
        } else if !has_seq {
            (tl!("Open a sequence first").into(), t.text_dim)
        } else if !has_subtitles {
            (tl!("No subtitles yet — click to transcribe and find clips").into(), t.text_dim)
        } else {
            (tl!("Subtitles ready — scan for top short-form hooks").into(), t.text_dim)
        };
        let status_resp = ui.add(egui::Label::new(egui::RichText::new(summary.clone()).font(Tokens::ui(10.5)).color(col)).wrap());
        app.auto.add("ai.viralClips.status", status_resp.rect, &summary);

        let preset_choices = [
            (ViralDurationPreset::Short15To30, viral_duration_label(ViralDurationPreset::Short15To30)),
            (ViralDurationPreset::Medium30To60, viral_duration_label(ViralDurationPreset::Medium30To60)),
            (ViralDurationPreset::Long60To90, viral_duration_label(ViralDurationPreset::Long60To90)),
        ];
        draw_combo_row(
            ui,
            &mut app.auto,
            t,
            &ComboRowSpec { salt: "ai-viral-duration", auto_id: "ai.viralClips.duration", label: tl!("Target Length") },
            &preset_choices,
            &mut app.ui.ai_tools.viral_duration,
        );

        ui.add_space(1.0);
        if let Some(prog) = active_job_progress(app, app.ui.ai_tools.viral_clips_job) {
            let shown = nonblocking_progress_status(&prog, tl!("Scoring viral highlights…"));
            draw_job_bar(ui, &mut app.auto, t, prog.fraction(), &shown, &prog, "ai.viralClips.cancel");
            return;
        }

        let model_installed = status.models.iter().find(|m| m.id == app.ui.ai_tools.subtitle_model).is_some_and(|m| m.installed);
        let ready = has_seq && (has_subtitles || (status.cli_found && model_installed));
        let tip = if !has_seq {
            Some(tl!("Open a sequence first"))
        } else if !has_subtitles && !status.cli_found {
            Some(tl!("Install the Orpheus AI engine first"))
        } else if !has_subtitles && !model_installed {
            Some(tl!("Download the selected AI model first"))
        } else {
            None
        };
        let btn_label = if has_subtitles { tl!("Find Viral Clips") } else { tl!("Analyze & Find Clips") };
        let clicked = draw_primary_icon_button(ui, &mut app.auto, t, "ai.viralClips.find", btn_label, Icon::Sparkle, ready, true, tip);
        if ready && clicked {
            let ctx = ui.ctx().clone();
            let _ = crate::menus::invoke(app, &ctx, "ai.findViralClips", json!({ "preset": app.ui.ai_tools.viral_duration.as_id(), "force": true }));
        }

        if let Some(ref clips) = cached
            && !clips.is_empty()
        {
            ui.add_space(3.0);
            for clip in clips {
                draw_viral_clip_item(app, ui, t, clip, app.ui.ai_tools.viral_duration);
                ui.add_space(2.0);
            }
        }
    });
}

fn enhance_strength_label(s: NoiseReductionStrength) -> &'static str {
    match s {
        NoiseReductionStrength::Light => crate::i18n::tr_ctx("audioEnhance", "Light"),
        NoiseReductionStrength::Balanced => tl!("Balanced"),
        NoiseReductionStrength::StudioStrong => tl!("Studio Strong"),
    }
}

fn draw_audio_enhance_card(app: &mut FilmcraftApp, ui: &mut egui::Ui, t: &Tokens, status: &SubtitleStatus) {
    let (_card_bg, _card_border, frame) = card_frame(t, app.ui.dark);
    let card_w = (ui.available_width() - 14.0).max(40.0);
    let active_seq = app.session.state.active_sequence;
    let has_seq = app.session.active_sequence().is_some();
    let has_unlocked_audio = app.session.active_sequence().is_some_and(|q| q.audio_tracks.iter().any(|tr| !tr.locked && !tr.items.is_empty()));
    let report = app.session.audio_enhance_report.as_ref().filter(|r| Some(r.seq) == active_seq).cloned();

    frame.show(ui, |ui| {
        ui.set_width(card_w);
        ui.spacing_mut().item_spacing.y = 3.0;
        ui.spacing_mut().interact_size.y = 18.0;

        ui.horizontal(|ui| {
            let (icon_r, _) = ui.allocate_exact_size(vec2(14.0, 14.0), Sense::hover());
            icons::paint(ui.painter(), icon_r, Icon::Mic, Color32::from_rgb(0x34, 0xD3, 0x99));
            let title_w = (ui.available_width() - 22.0).max(40.0);
            ui.add_sized(
                vec2(title_w, 18.0),
                egui::Label::new(egui::RichText::new(tl!("AI Audio Enhance")).font(Tokens::semibold(13.0)).color(t.text)).truncate(),
            )
            .on_hover_text(tl!("Isolate dialogue, reduce room echo, and balance vocal loudness locally."));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                draw_help_badge(
                    ui,
                    &mut app.auto,
                    t,
                    "ai.help.audioEnhance",
                    tl!("Reduces background noise, clarifies vocals, and balances loudness to -14 LUFS. Needs unlocked audio clips on the timeline."),
                );
            });
        });

        draw_hardware_badge_row(ui, &mut app.auto, "ai.badge.audioEnhance", AiFeatureWeight::Lightweight, status, app.ui.dark);

        let (summary, col) = if let Some(ref rep) = report {
            (tlf!("Enhanced {n} audio clips", n = rep.enhanced_clips), Color32::from_rgb(0x34, 0xD3, 0x99))
        } else if !has_seq {
            (tl!("Open a sequence first").into(), t.text_dim)
        } else if !has_unlocked_audio {
            (tl!("No audio clips found on unlocked tracks").into(), t.text_dim)
        } else {
            (tl!("Ready — enhances selected clips or unlocked audio").into(), t.text_dim)
        };
        let status_resp = ui.add(egui::Label::new(egui::RichText::new(summary.clone()).font(Tokens::ui(10.5)).color(col)).wrap());
        app.auto.add("ai.audioEnhance.status", status_resp.rect, &summary);

        let strength_choices = [
            (NoiseReductionStrength::Light, enhance_strength_label(NoiseReductionStrength::Light)),
            (NoiseReductionStrength::Balanced, enhance_strength_label(NoiseReductionStrength::Balanced)),
            (NoiseReductionStrength::StudioStrong, enhance_strength_label(NoiseReductionStrength::StudioStrong)),
        ];
        draw_combo_row(
            ui,
            &mut app.auto,
            t,
            &ComboRowSpec { salt: "ai-audio-enhance-strength", auto_id: "ai.audioEnhance.strength", label: tl!("Noise Reduction") },
            &strength_choices,
            &mut app.ui.ai_tools.enhance_strength,
        );

        draw_trigger_checkbox(
            ui,
            &mut app.auto,
            t,
            "ai.audioEnhance.voiceClarity",
            tl!("Voice Clarity EQ"),
            tl!("Boost vocal presence (3 kHz), warmth, and soften harsh sibilance"),
            &mut app.ui.ai_tools.enhance_voice_clarity,
        );
        draw_trigger_checkbox(
            ui,
            &mut app.auto,
            t,
            "ai.audioEnhance.autoLevel",
            tl!("Auto-Level (-14 LUFS)"),
            tl!("Apply smooth vocal compression and normalize loudness to -14 LUFS without clipping"),
            &mut app.ui.ai_tools.enhance_auto_level,
        );

        ui.add_space(1.0);
        if let Some(prog) = active_job_progress(app, app.ui.ai_tools.ai_models_download_job) {
            let shown = nonblocking_progress_status(&prog, tl!("Downloading AI models…"));
            draw_job_bar(ui, &mut app.auto, t, prog.fraction(), &shown, &prog, "ai.audioEnhance.cancelModel");
            return;
        }
        let speech_onnx_ready = filmcraft_engine::ai_models::detect_model(filmcraft_engine::ai_models::AiModelKind::SpeechDenoise, None).is_some();
        if speech_onnx_ready {
            ui.label(egui::RichText::new(tl!("ONNX Neural Model Ready")).font(Tokens::ui(10.5)).color(Color32::from_rgb(0x10, 0xB9, 0x81)));
        } else {
            let dl_btn = ui.button(tl!("Download Speech Denoise Model (~2 MB)"));
            app.auto.add("ai.audioEnhance.downloadModel", dl_btn.rect, tl!("Download Speech Denoise Model (~2 MB)"));
            if dl_btn.on_hover_text(tl!("Download rnnoise.onnx into local models folder for offline neural speech denoising")).clicked() {
                let ctx = ui.ctx().clone();
                let _ = crate::menus::invoke(app, &ctx, "ai.downloadModel", json!({ "model": "speech_denoise" }));
            }
        }
        if let Some(prog) = active_job_progress(app, app.ui.ai_tools.audio_enhance_job) {
            let shown = nonblocking_progress_status(&prog, tl!("Enhancing audio…"));
            draw_job_bar(ui, &mut app.auto, t, prog.fraction(), &shown, &prog, "ai.audioEnhance.cancel");
            return;
        }

        let ready = has_seq && has_unlocked_audio;
        let tip = if !has_seq {
            Some(tl!("Open a sequence first"))
        } else if !has_unlocked_audio {
            Some(tl!("No audio clips found on unlocked tracks"))
        } else {
            None
        };
        let label = tl!("Enhance Audio");
        let clicked = draw_primary_icon_button(ui, &mut app.auto, t, "ai.audioEnhance.apply", label, Icon::Mic, ready, true, tip);
        if ready && clicked {
            let ctx = ui.ctx().clone();
            let _ = crate::menus::invoke(
                app,
                &ctx,
                "ai.enhanceAudio",
                json!({
                    "strength": app.ui.ai_tools.enhance_strength.as_id(),
                    "voiceClarity": app.ui.ai_tools.enhance_voice_clarity,
                    "autoLevel": app.ui.ai_tools.enhance_auto_level,
                }),
            );
        }
    });
}

fn reframe_aspect_label(a: ReframeAspect) -> &'static str {
    match a {
        ReframeAspect::Vertical9x16 => tl!("9:16 Vertical"),
        ReframeAspect::Portrait4x5 => tl!("4:5 Portrait"),
        ReframeAspect::Square1x1 => tl!("1:1 Square"),
    }
}

fn reframe_smoothing_label(s: ReframeSmoothing) -> &'static str {
    match s {
        ReframeSmoothing::Locked => tl!("Locked"),
        ReframeSmoothing::SmoothPan => tl!("Smooth Pan"),
        ReframeSmoothing::DynamicFollow => tl!("Dynamic Follow"),
    }
}

fn draw_auto_reframe_card(app: &mut FilmcraftApp, ui: &mut egui::Ui, t: &Tokens, status: &SubtitleStatus) {
    let (_card_bg, _card_border, frame) = card_frame(t, app.ui.dark);
    let card_w = (ui.available_width() - 14.0).max(40.0);
    let active_seq = app.session.state.active_sequence;
    let has_seq = app.session.active_sequence().is_some();
    let has_unlocked_video = app.session.active_sequence().is_some_and(|q| q.video_tracks.iter().any(|tr| !tr.locked && tr.items.iter().any(|it| it.enabled)));
    let report = app.session.auto_reframe_report.as_ref().filter(|r| Some(r.seq) == active_seq).cloned();

    frame.show(ui, |ui| {
        ui.set_width(card_w);
        ui.spacing_mut().item_spacing.y = 3.0;
        ui.spacing_mut().interact_size.y = 18.0;

        ui.horizontal(|ui| {
            let (icon_r, _) = ui.allocate_exact_size(vec2(14.0, 14.0), Sense::hover());
            icons::paint(ui.painter(), icon_r, Icon::Sequence, Color32::from_rgb(0x6E, 0x9F, 0xFF));
            let title_w = (ui.available_width() - 22.0).max(40.0);
            ui.add_sized(
                vec2(title_w, 18.0),
                egui::Label::new(egui::RichText::new(tl!("AI Auto-Reframe (16:9 -> 9:16)")).font(Tokens::semibold(13.0)).color(t.text)).truncate(),
            )
            .on_hover_text(tl!("Track key subjects to convert widescreen edits into vertical 9:16 framing."));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                draw_help_badge(
                    ui,
                    &mut app.auto,
                    t,
                    "ai.help.autoReframe",
                    tl!("Tracks the main subject and reframes widescreen video for vertical or square formats. Needs unlocked video clips on the timeline."),
                );
            });
        });

        draw_hardware_badge_row(ui, &mut app.auto, "ai.badge.autoReframe", AiFeatureWeight::HeavyVisual, status, app.ui.dark);

        let (summary, col) = if let Some(ref rep) = report {
            (
                tlf!("Reframed {n} clip(s) to {aspect} ({kf} keyframes)", n = rep.reframed_clips, aspect = rep.aspect.as_id(), kf = rep.total_keyframes),
                Color32::from_rgb(0x6E, 0x9F, 0xFF),
            )
        } else if !has_seq {
            (tl!("Open a sequence first").into(), t.text_dim)
        } else if !has_unlocked_video {
            (tl!("No video clips found on unlocked tracks").into(), t.text_dim)
        } else {
            (tl!("Ready — tracks subjects and reframes selected or V1 clips").into(), t.text_dim)
        };
        let status_resp = ui.add(egui::Label::new(egui::RichText::new(summary.clone()).font(Tokens::ui(10.5)).color(col)).wrap());
        app.auto.add("ai.autoReframe.status", status_resp.rect, &summary);

        let aspect_choices = [
            (ReframeAspect::Vertical9x16, reframe_aspect_label(ReframeAspect::Vertical9x16)),
            (ReframeAspect::Portrait4x5, reframe_aspect_label(ReframeAspect::Portrait4x5)),
            (ReframeAspect::Square1x1, reframe_aspect_label(ReframeAspect::Square1x1)),
        ];
        draw_combo_row(
            ui,
            &mut app.auto,
            t,
            &ComboRowSpec { salt: "ai-auto-reframe-aspect", auto_id: "ai.autoReframe.aspect", label: tl!("Target Format") },
            &aspect_choices,
            &mut app.ui.ai_tools.reframe_aspect,
        );

        let smoothing_choices = [
            (ReframeSmoothing::Locked, reframe_smoothing_label(ReframeSmoothing::Locked)),
            (ReframeSmoothing::SmoothPan, reframe_smoothing_label(ReframeSmoothing::SmoothPan)),
            (ReframeSmoothing::DynamicFollow, reframe_smoothing_label(ReframeSmoothing::DynamicFollow)),
        ];
        draw_combo_row(
            ui,
            &mut app.auto,
            t,
            &ComboRowSpec { salt: "ai-auto-reframe-smoothing", auto_id: "ai.autoReframe.smoothing", label: tl!("Tracking Motion") },
            &smoothing_choices,
            &mut app.ui.ai_tools.reframe_smoothing,
        );

        draw_trigger_checkbox(
            ui,
            &mut app.auto,
            t,
            "ai.autoReframe.resizeSequence",
            tl!("Resize Sequence to Vertical (9:16)"),
            tl!("Automatically set the sequence canvas resolution to match the selected vertical or portrait aspect ratio"),
            &mut app.ui.ai_tools.reframe_resize_sequence,
        );
        draw_trigger_checkbox(
            ui,
            &mut app.auto,
            t,
            "ai.autoReframe.splitScreen",
            tl!("Gaming Split-Screen"),
            tl!("Detect corner FaceCam and split gameplay (65%) + FaceCam (35%) across two vertical panels"),
            &mut app.ui.ai_tools.reframe_split_screen,
        );
        if app.ui.ai_tools.reframe_split_screen {
            let fc_choices = [
                (FacecamRegion::Auto, tl!("Auto-Detect")),
                (FacecamRegion::LeftMiddle, tl!("Left-Middle")),
                (FacecamRegion::BottomLeft, tl!("Bottom-Left")),
                (FacecamRegion::BottomRight, tl!("Bottom-Right")),
                (FacecamRegion::TopLeft, tl!("Top-Left")),
                (FacecamRegion::TopRight, tl!("Top-Right")),
                (FacecamRegion::RightMiddle, tl!("Right-Middle")),
            ];
            draw_combo_row(
                ui,
                &mut app.auto,
                t,
                &ComboRowSpec { salt: "ai-auto-reframe-facecam", auto_id: "ai.autoReframe.facecam", label: tl!("FaceCam Position") },
                &fc_choices,
                &mut app.ui.ai_tools.reframe_facecam_region,
            );
        }

        ui.add_space(1.0);
        if let Some(prog) = active_job_progress(app, app.ui.ai_tools.ai_models_download_job) {
            let shown = nonblocking_progress_status(&prog, tl!("Downloading AI models…"));
            draw_job_bar(ui, &mut app.auto, t, prog.fraction(), &shown, &prog, "ai.autoReframe.cancelModel");
            return;
        }
        let face_onnx_ready = filmcraft_engine::ai_models::detect_model(filmcraft_engine::ai_models::AiModelKind::FaceDetect, None).is_some();
        if face_onnx_ready {
            ui.label(egui::RichText::new(tl!("ONNX Neural Model Ready")).font(Tokens::ui(10.5)).color(Color32::from_rgb(0x10, 0xB9, 0x81)));
        } else {
            let dl_btn = ui.button(tl!("Download Face Detection Model (~1.2 MB)"));
            app.auto.add("ai.autoReframe.downloadModel", dl_btn.rect, tl!("Download Face Detection Model (~1.2 MB)"));
            if dl_btn.on_hover_text(tl!("Download ultraface-rfb-320.onnx into local models folder for offline neural face tracking")).clicked() {
                let ctx = ui.ctx().clone();
                let _ = crate::menus::invoke(app, &ctx, "ai.downloadModel", json!({ "model": "face_detect" }));
            }
        }
        if let Some(prog) = active_job_progress(app, app.ui.ai_tools.auto_reframe_job) {
            let shown = nonblocking_progress_status(&prog, tl!("Tracking subject for Auto-Reframe…"));
            draw_job_bar(ui, &mut app.auto, t, prog.fraction(), &shown, &prog, "ai.autoReframe.cancel");
            return;
        }

        let ready = has_seq && has_unlocked_video;
        let tip = if !has_seq {
            Some(tl!("Open a sequence first"))
        } else if !has_unlocked_video {
            Some(tl!("No video clips found on unlocked tracks"))
        } else {
            None
        };
        let label = tl!("Auto-Reframe");
        let clicked = draw_primary_icon_button(ui, &mut app.auto, t, "ai.autoReframe.apply", label, Icon::Sequence, ready, true, tip);
        if ready && clicked {
            let ctx = ui.ctx().clone();
            let _ = crate::menus::invoke(
                app,
                &ctx,
                "ai.autoReframe",
                json!({
                    "aspect": app.ui.ai_tools.reframe_aspect.as_id(),
                    "smoothing": app.ui.ai_tools.reframe_smoothing.as_id(),
                    "resizeSequence": app.ui.ai_tools.reframe_resize_sequence,
                    "splitScreen": app.ui.ai_tools.reframe_split_screen,
                    "facecamPlacement": app.ui.ai_tools.reframe_facecam_placement.as_id(),
                    "facecamRegion": app.ui.ai_tools.reframe_facecam_region.as_id(),
                }),
            );
        }
    });
}

fn bg_softness_label(s: BgEdgeSoftness) -> &'static str {
    match s {
        BgEdgeSoftness::Crisp => tl!("Crisp"),
        BgEdgeSoftness::Natural => tl!("Natural"),
        BgEdgeSoftness::Soft => tl!("Soft"),
    }
}

fn bg_mode_label(m: BgRemovalMode) -> &'static str {
    match m {
        BgRemovalMode::TransparentAlpha => tl!("Transparent Alpha"),
        BgRemovalMode::CustomMedia => tl!("Custom Image / Video…"),
        BgRemovalMode::SolidColor => tl!("Solid Color"),
        BgRemovalMode::BlurBackground => tl!("Blur Original"),
    }
}

fn draw_bg_removal_card(app: &mut FilmcraftApp, ui: &mut egui::Ui, t: &Tokens, status: &SubtitleStatus) {
    let (_card_bg, _card_border, frame) = card_frame(t, app.ui.dark);
    let card_w = (ui.available_width() - 14.0).max(40.0);
    let active_seq = app.session.state.active_sequence;
    let has_seq = app.session.active_sequence().is_some();
    let has_unlocked_video = app.session.active_sequence().is_some_and(|q| q.video_tracks.iter().any(|tr| !tr.locked && tr.items.iter().any(|it| it.enabled)));
    let report = app.session.bg_removal_report.as_ref().filter(|r| Some(r.seq) == active_seq).cloned();

    frame.show(ui, |ui| {
        ui.set_width(card_w);
        ui.spacing_mut().item_spacing.y = 3.0;
        ui.spacing_mut().interact_size.y = 18.0;

        ui.horizontal(|ui| {
            let (icon_r, _) = ui.allocate_exact_size(vec2(14.0, 14.0), Sense::hover());
            icons::paint(ui.painter(), icon_r, Icon::Eye, Color32::from_rgb(0xA7, 0x8B, 0xFA));
            let title_w = (ui.available_width() - 22.0).max(40.0);
            ui.add_sized(
                vec2(title_w, 18.0),
                egui::Label::new(egui::RichText::new(tl!("AI Background Removal")).font(Tokens::semibold(13.0)).color(t.text)).truncate(),
            )
            .on_hover_text(tl!("Remove or blur video backgrounds locally without a green screen."));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                draw_help_badge(
                    ui,
                    &mut app.auto,
                    t,
                    "ai.help.bgRemoval",
                    tl!("Separates the foreground subject to make the background transparent, blurred, or solid. Needs unlocked video clips on the timeline."),
                );
            });
        });

        draw_hardware_badge_row(ui, &mut app.auto, "ai.badge.backgroundRemoval", AiFeatureWeight::HeavyVisual, status, app.ui.dark);

        let (summary, col) = if let Some(ref rep) = report {
            if rep.used_onnx {
                let provider =
                    filmcraft_engine::ai_models::resolve_execution_provider(filmcraft_engine::ai_models::active_execution_mode(), status.gpu.as_ref());
                (format!("ONNX Neural Matting ({}) — {} clip(s)", provider.short_label(), rep.processed_clips), READY_GREEN)
            } else {
                (format!("Heuristic Fallback ({} clip(s)) — Download model in Settings", rep.processed_clips), WARN_AMBER)
            }
        } else if !has_seq {
            (tl!("Open a sequence first").into(), t.text_dim)
        } else if !has_unlocked_video {
            (tl!("No video clips found on unlocked tracks").into(), t.text_dim)
        } else {
            (tl!("Ready — isolates foreground on selected or V1 clips").into(), t.text_dim)
        };
        let status_resp = ui.add(egui::Label::new(egui::RichText::new(summary.clone()).font(Tokens::ui(10.5)).color(col)).wrap());
        app.auto.add("ai.bgRemoval.status", status_resp.rect, &summary);

        let soft_choices = [
            (BgEdgeSoftness::Crisp, bg_softness_label(BgEdgeSoftness::Crisp)),
            (BgEdgeSoftness::Natural, bg_softness_label(BgEdgeSoftness::Natural)),
            (BgEdgeSoftness::Soft, bg_softness_label(BgEdgeSoftness::Soft)),
        ];
        draw_combo_row(
            ui,
            &mut app.auto,
            t,
            &ComboRowSpec { salt: "ai-bg-softness", auto_id: "ai.bgRemoval.softness", label: tl!("Edge Softness") },
            &soft_choices,
            &mut app.ui.ai_tools.bg_softness,
        );

        let mode_choices = [
            (BgRemovalMode::TransparentAlpha, bg_mode_label(BgRemovalMode::TransparentAlpha)),
            (BgRemovalMode::CustomMedia, bg_mode_label(BgRemovalMode::CustomMedia)),
            (BgRemovalMode::SolidColor, bg_mode_label(BgRemovalMode::SolidColor)),
            (BgRemovalMode::BlurBackground, bg_mode_label(BgRemovalMode::BlurBackground)),
        ];
        draw_combo_row(
            ui,
            &mut app.auto,
            t,
            &ComboRowSpec { salt: "ai-bg-mode", auto_id: "ai.bgRemoval.mode", label: tl!("Background Mode") },
            &mode_choices,
            &mut app.ui.ai_tools.bg_mode,
        );

        if app.ui.ai_tools.bg_mode == BgRemovalMode::SolidColor {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(tl!("Background Color")).font(Tokens::ui(11.0)).color(t.text_dim));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    egui::color_picker::color_edit_button_srgb(ui, &mut app.ui.ai_tools.bg_solid_rgb);
                });
            });
        } else if app.ui.ai_tools.bg_mode == BgRemovalMode::CustomMedia
            && let Some(path) = &app.ui.ai_tools.bg_custom_path
        {
            let fname = std::path::Path::new(path).file_name().and_then(|n| n.to_str()).unwrap_or(path);
            ui.label(egui::RichText::new(format!("{}: {fname}", tl!("Background"))).font(Tokens::ui(10.5)).color(t.text_dim));
        }

        ui.add_space(1.0);
        if let Some(prog) = active_job_progress(app, app.ui.ai_tools.bg_model_download_job) {
            let shown = nonblocking_progress_status(&prog, tl!("Downloading AI matting model…"));
            draw_job_bar(ui, &mut app.auto, t, prog.fraction(), &shown, &prog, "ai.bgRemoval.cancelModel");
            return;
        }
        let onnx_installed = filmcraft_engine::bg_removal::detect_local_matting_onnx(None).is_some();
        if onnx_installed {
            let provider = filmcraft_engine::ai_models::resolve_execution_provider(filmcraft_engine::ai_models::active_execution_mode(), status.gpu.as_ref());
            let msg = format!("{} ({})", tl!("AI matting model ready offline"), provider.short_label());
            ui.label(egui::RichText::new(msg).font(Tokens::ui(10.5)).color(Color32::from_rgb(0x10, 0xB9, 0x81)));
        } else {
            ui.add(
                egui::Label::new(
                    egui::RichText::new(tl!("Using built-in heuristic mask — download ONNX model in Settings ▸ AI Models for neural matting"))
                        .font(Tokens::ui(10.0))
                        .color(WARN_AMBER),
                )
                .wrap(),
            );
            let dl_btn = ui.button(tl!("Download AI Matting Model (~25 MB)"));
            app.auto.add("ai.bgRemoval.downloadModel", dl_btn.rect, tl!("Download AI Matting Model (~25 MB)"));
            if dl_btn.on_hover_text(tl!("Download u2netp.onnx into local models folder for offline neural matting")).clicked() {
                let ctx = ui.ctx().clone();
                let _ = crate::menus::invoke(app, &ctx, "ai.downloadBgModel", json!({}));
            }
            if let Some(err) = &app.ui.ai_tools.bg_model_download_error {
                ui.add(egui::Label::new(egui::RichText::new(err).font(Tokens::ui(10.5)).color(t.danger)).wrap());
            }
        }
        if let Some(prog) = active_job_progress(app, app.ui.ai_tools.bg_removal_job) {
            let shown = nonblocking_progress_status(&prog, tl!("Removing background…"));
            draw_job_bar(ui, &mut app.auto, t, prog.fraction(), &shown, &prog, "ai.bgRemoval.cancel");
            return;
        }

        let ready = has_seq && has_unlocked_video;
        let tip = if !has_seq {
            Some(tl!("Open a sequence first"))
        } else if !has_unlocked_video {
            Some(tl!("No video clips found on unlocked tracks"))
        } else {
            None
        };
        let mut do_apply = false;
        let mut do_pick_bg = false;
        ui.horizontal(|ui| {
            let half_w = ((ui.available_width() - 6.0) * 0.5).max(80.0);
            ui.allocate_ui(vec2(half_w, 26.0), |ui| {
                do_apply = draw_primary_icon_button(ui, &mut app.auto, t, "ai.bgRemoval.apply", tl!("Remove Background"), Icon::Eye, ready, true, tip);
            });
            ui.allocate_ui(vec2(half_w, 26.0), |ui| {
                let replace_btn = ui.add_enabled(ready, egui::Button::new(tl!("Replace Background…")).min_size(vec2(half_w, 24.0)));
                app.auto.add("ai.bgRemoval.replace", replace_btn.rect, tl!("Replace Background…"));
                if let Some(tip_text) = tip {
                    replace_btn.clone().on_disabled_hover_text(tip_text);
                } else {
                    replace_btn.clone().on_hover_text(tl!("Pick a custom image or video to place behind the cut-out subject"));
                }
                if ready && replace_btn.clicked() {
                    do_pick_bg = true;
                }
            });
        });

        if ready && do_apply && app.ui.ai_tools.bg_mode == BgRemovalMode::CustomMedia && app.ui.ai_tools.bg_custom_path.is_none() {
            do_pick_bg = true;
        }

        if ready && do_pick_bg {
            let ctx = ui.ctx().clone();
            let exts = ["png", "jpg", "jpeg", "webp", "mp4", "mov", "mkv"].into_iter().map(str::to_string).collect();
            let _ = app.pick(crate::native_dialogs::FileDialog::OpenFiles { exts }, move |app, paths| {
                if let Some(path) = paths.into_iter().next() {
                    app.ui.ai_tools.bg_mode = BgRemovalMode::CustomMedia;
                    app.ui.ai_tools.bg_custom_path = Some(path.clone());
                    let _ = crate::menus::invoke(
                        app,
                        &ctx,
                        "ai.removeBackground",
                        json!({
                            "softness": app.ui.ai_tools.bg_softness.as_id(),
                            "mode": BgRemovalMode::CustomMedia.as_id(),
                            "backgroundPath": path,
                        }),
                    );
                }
                Ok(serde_json::Value::Null)
            });
        } else if ready && do_apply {
            let ctx = ui.ctx().clone();
            let [r, g, b] = app.ui.ai_tools.bg_solid_rgb;
            let mut params = json!({
                "softness": app.ui.ai_tools.bg_softness.as_id(),
                "mode": app.ui.ai_tools.bg_mode.as_id(),
                "color": format!("#{r:02x}{g:02x}{b:02x}"),
            });
            if let Some(path) = &app.ui.ai_tools.bg_custom_path {
                params["backgroundPath"] = json!(path);
            }
            let _ = crate::menus::invoke(app, &ctx, "ai.removeBackground", params);
        }
    });
}

fn draw_vod_chapters_card(app: &mut FilmcraftApp, ui: &mut egui::Ui, t: &Tokens, status: &SubtitleStatus) {
    let (_card_bg, _card_border, frame) = card_frame(t, app.ui.dark);
    let card_w = (ui.available_width() - 14.0).max(40.0);

    frame.show(ui, |ui| {
        ui.set_width(card_w);
        ui.spacing_mut().item_spacing.y = 4.0;
        ui.horizontal(|ui| {
            let (icon_r, _) = ui.allocate_exact_size(vec2(14.0, 14.0), Sense::hover());
            icons::paint(ui.painter(), icon_r, Icon::Marker, Color32::from_rgb(0x0E, 0xA5, 0xE9));
            let title_w = (ui.available_width() - 22.0).max(40.0);
            ui.add_sized(
                vec2(title_w, 18.0),
                egui::Label::new(egui::RichText::new(tl!("Stream VOD Chapterizer")).font(Tokens::semibold(13.0)).color(t.text)).truncate(),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                draw_help_badge(
                    ui,
                    &mut app.auto,
                    t,
                    "ai.help.vodChapters",
                    tl!("Scans long recordings for scene changes and splits them into labeled chapter markers. Needs unlocked video clips on the timeline."),
                );
            });
        });
        draw_hardware_badge_row(ui, &mut app.auto, "ai.badge.vodChapters", AiFeatureWeight::HeavyVisual, status, app.ui.dark);

        ui.add(
            egui::Label::new(
                egui::RichText::new(tl!("Sparse 10s layout scan + transcript keywords to split long VODs into named chapters."))
                    .font(Tokens::ui(10.5))
                    .color(t.text_dim),
            )
            .wrap(),
        );

        let chapter_count =
            app.session.active_sequence().map_or(0, |seq| seq.markers.iter().filter(|m| m.kind == filmcraft_engine::project::MarkerKind::Chapter).count());
        if chapter_count > 0 {
            ui.label(egui::RichText::new(tlf!("{n} stream chapter(s) on timeline", n = chapter_count)).font(Tokens::ui(10.5)).color(t.accent));
        }

        if let Some(prog) = active_job_progress(app, app.ui.ai_tools.vod_chapters_job) {
            let shown = nonblocking_progress_status(&prog, tl!("Chapterizing stream VOD…"));
            draw_job_bar(ui, &mut app.auto, t, prog.fraction(), &shown, &prog, "ai.vodChapters.cancel");
            return;
        }

        let has_video = app.session.active_sequence().is_some_and(|seq| seq.video_tracks.iter().any(|tr| !tr.locked && !tr.items.is_empty()));
        let ready = has_video && !app.ui.ai_tools.has_active_mutation_job();
        let tip = if !has_video { Some(tl!("Add a video clip to the timeline first")) } else { None };
        let clicked = draw_primary_icon_button(ui, &mut app.auto, t, "ai.vodChapters.apply", tl!("Auto-Split Stream Chapters"), Icon::Marker, ready, true, tip);
        if ready && clicked {
            let ctx = ui.ctx().clone();
            let ollama = app.ui.ai_tools.all_in_one_ollama_url.trim().to_string();
            let _ = crate::menus::invoke(
                app,
                &ctx,
                "ai.vodChapters",
                json!({
                    "splitClips": true,
                    "ollamaUrl": ollama,
                }),
            );
        }
    });
}

fn draw_all_in_one_tab(app: &mut FilmcraftApp, ui: &mut egui::Ui, t: &Tokens) {
    let (_card_bg, _card_border, frame) = card_frame(t, app.ui.dark);
    let card_w = (ui.available_width() - 14.0).max(40.0);

    frame.show(ui, |ui| {
        ui.set_width(card_w);
        ui.spacing_mut().item_spacing.y = 4.0;

        ui.horizontal(|ui| {
            let (icon_r, _) = ui.allocate_exact_size(vec2(14.0, 14.0), Sense::hover());
            icons::paint(ui.painter(), icon_r, Icon::Sparkle, ROYAL_BLUE);
            let title_w = (ui.available_width() - 22.0).max(40.0);
            ui.add_sized(
                vec2(title_w, 18.0),
                egui::Label::new(egui::RichText::new(tl!("Orpheus All-In-One")).font(Tokens::semibold(13.0)).color(t.text)).truncate(),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                draw_help_badge(
                    ui,
                    &mut app.auto,
                    t,
                    "ai.help.allInOne",
                    tl!("Describe the edit or scene you want in plain text and Orpheus will build or automate it. Needs a prompt, attached files, or clips on the timeline."),
                );
            });
        });

        ui.add(
            egui::Label::new(
                egui::RichText::new(tl!("Describe your edit, 2.5D corkboard scene, or AI workflow in plain text.")).font(Tokens::ui(10.5)).color(t.text_dim),
            )
            .wrap(),
        );

        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = egui::vec2(5.0, 5.0);
            let presets = [
                (tl!("Cinematic Story"), "Build a 2.5D corkboard documentary intro with dynamic shadows"),
                (tl!("Vertical Short (9:16)"), "Build a 50/50 TikTok split with talking head, B-roll, and center pop subtitles"),
                (tl!("Clean & Subtitles"), "Auto-Cut & Polish: remove pauses, clean audio, and generate Shorts subtitles"),
            ];
            for (label, prompt_text) in presets {
                if ui.small_button(label).clicked() {
                    app.ui.ai_tools.all_in_one_prompt = prompt_text.into();
                }
            }
        });

        let prompt_resp = ui.add(
            egui::TextEdit::multiline(&mut app.ui.ai_tools.all_in_one_prompt)
                .desired_rows(4)
                .desired_width(ui.available_width())
                .hint_text(tl!("e.g. Build a 50/50 TikTok split with talking head and article highlight, or a 2.5D corkboard scene")),
        );
        app.auto.add("ai.allInOne.prompt", prompt_resp.rect, tl!("Prompt"));

        ui.horizontal(|ui| {
            let attach_btn = ui.button(tl!("Attach Files…"));
            app.auto.add("ai.allInOne.attach", attach_btn.rect, tl!("Attach Files…"));
            if attach_btn.clicked() {
                let exts = ["png", "jpg", "jpeg", "webp", "mp4", "mov", "mkv", "wav", "mp3"].into_iter().map(str::to_string).collect();
                let _ = app.pick(crate::native_dialogs::FileDialog::OpenFiles { exts }, |app, paths| {
                    for path in paths {
                        if !app.ui.ai_tools.all_in_one_attachments.contains(&path) && app.ui.ai_tools.all_in_one_attachments.len() < 16 {
                            app.ui.ai_tools.all_in_one_attachments.push(path);
                        }
                    }
                    Ok(serde_json::Value::Null)
                });
            }
            if !app.ui.ai_tools.all_in_one_attachments.is_empty() && ui.small_button(tl!("Clear")).clicked() {
                app.ui.ai_tools.all_in_one_attachments.clear();
            }
        });

        if !app.ui.ai_tools.all_in_one_attachments.is_empty() {
            let mut remove_idx: Option<usize> = None;
            for (idx, path) in app.ui.ai_tools.all_in_one_attachments.iter().enumerate() {
                let name = std::path::Path::new(path).file_name().and_then(|n| n.to_str()).unwrap_or(path);
                ui.horizontal(|ui| {
                    ui.add(egui::Label::new(egui::RichText::new(name).font(Tokens::ui(10.5)).color(t.text)).truncate());
                    if ui.small_button("×").clicked() {
                        remove_idx = Some(idx);
                    }
                });
            }
            if let Some(i) = remove_idx {
                app.ui.ai_tools.all_in_one_attachments.remove(i);
            }
        }

        draw_trigger_checkbox(
            ui,
            &mut app.auto,
            t,
            "ai.allInOne.fetchWikimedia",
            tl!("Auto-Search Stock Images"),
            tl!("Fetch public-domain portraits and photos from Wikimedia Commons when building documentary scenes"),
            &mut app.ui.ai_tools.all_in_one_fetch_wikimedia,
        );

        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(tl!("Local Ollama URL (optional)")).font(Tokens::ui(10.5)).color(t.text_dim));
            ui.add(egui::TextEdit::singleline(&mut app.ui.ai_tools.all_in_one_ollama_url).desired_width(ui.available_width()));
        });

        ui.add_space(2.0);
        if let Some(prog) = active_job_progress(app, app.ui.ai_tools.all_in_one_job) {
            let shown = nonblocking_progress_status(&prog, tl!("Running Orpheus All-In-One…"));
            draw_job_bar(ui, &mut app.auto, t, prog.fraction(), &shown, &prog, "ai.allInOne.cancel");
            return;
        }

        let ready = !app.ui.ai_tools.all_in_one_prompt.trim().is_empty() || !app.ui.ai_tools.all_in_one_attachments.is_empty();
        let tip = if !ready { Some(tl!("Enter a prompt or attach local files first")) } else { None };
        let clicked = draw_primary_icon_button(ui, &mut app.auto, t, "ai.allInOne.run", tl!("Generate & Apply"), Icon::Sparkle, ready, true, tip);
        if ready && clicked {
            let ctx = ui.ctx().clone();
            let _ = crate::menus::invoke(
                app,
                &ctx,
                "ai.allInOne",
                json!({
                    "prompt": app.ui.ai_tools.all_in_one_prompt,
                    "attachments": app.ui.ai_tools.all_in_one_attachments,
                    "fetchWikimedia": app.ui.ai_tools.all_in_one_fetch_wikimedia,
                    "ollamaUrl": app.ui.ai_tools.all_in_one_ollama_url,
                }),
            );
        }

        if let Some(ref rep) = app.session.all_in_one_report {
            ui.add_space(3.0);
            let joined = rep.steps_applied.join(" · ");
            ui.add(egui::Label::new(egui::RichText::new(joined).font(Tokens::ui(10.5)).color(Color32::from_rgb(0x34, 0xD3, 0x99))).wrap());
        }
    });
}

fn trigger_density_label(d: TriggerDensity) -> &'static str {
    match d {
        TriggerDensity::Subtle => tl!("Subtle"),
        TriggerDensity::Balanced => tl!("Balanced"),
        TriggerDensity::Dynamic => tl!("Dynamic"),
    }
}

fn draw_trigger_checkbox(ui: &mut egui::Ui, auto: &mut crate::automation::Registry, t: &Tokens, id: &str, label: &str, tip: &str, checked: &mut bool) {
    let resp = ui.checkbox(checked, egui::RichText::new(label).font(Tokens::ui(11.0)).color(t.text)).on_hover_text(tip);
    auto.add(id, resp.rect, label);
}

fn draw_smart_triggers_card(app: &mut FilmcraftApp, ui: &mut egui::Ui, t: &Tokens, status: &SubtitleStatus) {
    let (_card_bg, _card_border, frame) = card_frame(t, app.ui.dark);
    let card_w = (ui.available_width() - 14.0).max(40.0);
    let has_seq = app.session.active_sequence().is_some();
    let has_subtitles = app.session.active_sequence().is_some_and(|q| viral_clips::sequence_has_subtitles(&app.session.project, q));
    let last_report = app.session.smart_trigger_report.as_ref().filter(|r| Some(r.seq) == app.session.state.active_sequence).cloned();

    frame.show(ui, |ui| {
        ui.set_width(card_w);
        ui.spacing_mut().item_spacing.y = 3.0;
        ui.spacing_mut().interact_size.y = 18.0;

        ui.horizontal(|ui| {
            let (icon_r, _) = ui.allocate_exact_size(vec2(14.0, 14.0), Sense::hover());
            icons::paint(ui.painter(), icon_r, Icon::Fx, Color32::from_rgb(0xFB, 0xBF, 0x24));
            let title_w = (ui.available_width() - 22.0).max(40.0);
            ui.add_sized(
                vec2(title_w, 18.0),
                egui::Label::new(egui::RichText::new(tl!("AI Smart Triggers")).font(Tokens::semibold(13.0)).color(t.text)).truncate(),
            )
            .on_hover_text(tl!("Trigger automated B-roll, zoom keyframes, and graphics from dialogue cues."));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                draw_help_badge(
                    ui,
                    &mut app.auto,
                    t,
                    "ai.help.smartTriggers",
                    tl!("Adds zoom punch-ins, SFX accents, and callout markers on key speech moments. Needs a sequence with spoken audio or subtitles."),
                );
            });
        });

        draw_hardware_badge_row(ui, &mut app.auto, "ai.badge.smartTriggers", AiFeatureWeight::Lightweight, status, app.ui.dark);

        let (summary, col) = if let Some(ref rep) = last_report {
            if rep.applied > 0 {
                (tlf!("Applied {n} smart triggers", n = rep.applied), Color32::from_rgb(0x7C, 0xB3, 0xFF))
            } else {
                (tl!("No emphasis triggers detected in speech").into(), t.text_dim)
            }
        } else if !has_seq {
            (tl!("Open a sequence first").into(), t.text_dim)
        } else if !has_subtitles {
            (tl!("No subtitles yet — click to transcribe and apply triggers").into(), t.text_dim)
        } else {
            (tl!("Subtitles ready — apply zoom, SFX & callout accents").into(), t.text_dim)
        };
        let status_resp = ui.add(egui::Label::new(egui::RichText::new(summary.clone()).font(Tokens::ui(10.5)).color(col)).wrap());
        app.auto.add("ai.smartTriggers.status", status_resp.rect, &summary);

        let density_choices = [
            (TriggerDensity::Subtle, trigger_density_label(TriggerDensity::Subtle)),
            (TriggerDensity::Balanced, trigger_density_label(TriggerDensity::Balanced)),
            (TriggerDensity::Dynamic, trigger_density_label(TriggerDensity::Dynamic)),
        ];
        draw_combo_row(
            ui,
            &mut app.auto,
            t,
            &ComboRowSpec { salt: "ai-smart-triggers-density", auto_id: "ai.smartTriggers.density", label: tl!("Trigger Density") },
            &density_choices,
            &mut app.ui.ai_tools.trigger_density,
        );

        draw_trigger_checkbox(
            ui,
            &mut app.auto,
            t,
            "ai.smartTriggers.zoom",
            tl!("Zoom Punch-Ins"),
            tl!("Adds smooth 100% -> 108% -> 100% scale keyframes on emphasis cues"),
            &mut app.ui.ai_tools.trigger_zoom_punch_ins,
        );
        draw_trigger_checkbox(
            ui,
            &mut app.auto,
            t,
            "ai.smartTriggers.sfx",
            tl!("SFX Accents"),
            tl!("Synthesizes pop, whoosh & ding WAV accents onto an AI SFX track"),
            &mut app.ui.ai_tools.trigger_sfx_accents,
        );
        draw_trigger_checkbox(
            ui,
            &mut app.auto,
            t,
            "ai.smartTriggers.markers",
            tl!("Callout Markers"),
            tl!("Places labeled amber timeline markers for quick overlay drops"),
            &mut app.ui.ai_tools.trigger_callout_markers,
        );

        ui.add_space(1.0);
        if let Some(prog) = active_job_progress(app, app.ui.ai_tools.smart_triggers_job) {
            let shown = nonblocking_progress_status(&prog, tl!("Detecting smart triggers…"));
            draw_job_bar(ui, &mut app.auto, t, prog.fraction(), &shown, &prog, "ai.smartTriggers.cancel");
            return;
        }

        let any_action = app.ui.ai_tools.trigger_zoom_punch_ins || app.ui.ai_tools.trigger_sfx_accents || app.ui.ai_tools.trigger_callout_markers;
        let model_installed = status.models.iter().find(|m| m.id == app.ui.ai_tools.subtitle_model).is_some_and(|m| m.installed);
        let ready = has_seq && any_action && (has_subtitles || (status.cli_found && model_installed));
        let tip = if !has_seq {
            Some(tl!("Open a sequence first"))
        } else if !any_action {
            Some(tl!("Enable at least one Smart Trigger action"))
        } else if !has_subtitles && !status.cli_found {
            Some(tl!("Install the Orpheus AI engine first"))
        } else if !has_subtitles && !model_installed {
            Some(tl!("Download the selected AI model first"))
        } else {
            None
        };
        let label = tl!("Apply Smart Triggers");
        let clicked = draw_primary_icon_button(ui, &mut app.auto, t, "ai.smartTriggers.apply", label, Icon::Fx, ready, true, tip);
        if ready && clicked {
            let ctx = ui.ctx().clone();
            let _ = crate::menus::invoke(
                app,
                &ctx,
                "ai.applySmartTriggers",
                json!({
                    "density": app.ui.ai_tools.trigger_density.as_id(),
                    "zoomPunchIns": app.ui.ai_tools.trigger_zoom_punch_ins,
                    "sfxAccents": app.ui.ai_tools.trigger_sfx_accents,
                    "calloutMarkers": app.ui.ai_tools.trigger_callout_markers,
                }),
            );
        }
    });
}

fn draw_ai_ready_pill(ui: &mut egui::Ui, auto: &mut crate::automation::Registry, t: &Tokens, dark: bool, id: &str) {
    let (r, _) = ui.allocate_exact_size(vec2(ui.available_width().max(40.0), 18.0), Sense::hover());
    let bg = if dark { Color32::from_rgb(0x11, 0x1A, 0x17) } else { t.field_bg };
    ui.painter().rect_filled(r, CornerRadius::same(5), bg);
    ui.painter().circle_filled(pos2(r.min.x + 10.0, r.center().y), 3.5, READY_GREEN);
    let label = tl!("Orpheus AI Ready");
    let g = layout_truncated(ui.painter(), label, Tokens::ui(11.0), t.text, (r.width() - 26.0).max(12.0));
    ui.painter().galley(pos2(r.min.x + 20.0, r.center().y - g.size().y / 2.0), g, t.text);
    auto.add(id, r, label);
}

fn draw_auto_cut_card(app: &mut FilmcraftApp, ui: &mut egui::Ui, t: &Tokens, status: &SubtitleStatus) {
    let (_card_bg, card_border, frame) = card_frame(t, app.ui.dark);
    let card_w = (ui.available_width() - 14.0).max(40.0);

    frame.show(ui, |ui| {
        ui.set_width(card_w);
        ui.spacing_mut().item_spacing.y = 2.0;
        ui.spacing_mut().interact_size.y = 18.0;

        let (pauses, saved_secs) = cached_preview(app, ui.ctx());
        ui.horizontal(|ui| {
            let (icon_r, _) = ui.allocate_exact_size(vec2(14.0, 14.0), Sense::hover());
            icons::paint(ui.painter(), icon_r, Icon::Sparkle, Color32::from_rgb(0x6E, 0x9F, 0xFF));
            let title_w = (ui.available_width() - 44.0).max(40.0);
            ui.add_sized(
                vec2(title_w, 18.0),
                egui::Label::new(egui::RichText::new(tl!("Auto-Cut Silence")).font(Tokens::semibold(13.0)).color(t.text)).truncate(),
            )
            .on_hover_text(tl!("Automatically detect and ripple-delete silent pauses in speech."));

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                draw_help_badge(
                    ui,
                    &mut app.auto,
                    t,
                    "ai.help.silence",
                    tl!("Detects silent pauses in dialogue and ripple-deletes them automatically. Needs a sequence with audio clips."),
                );
                let (reset_r, reset_resp) = ui.allocate_exact_size(vec2(18.0, 18.0), Sense::click());
                let reset_resp = reset_resp.on_hover_text(tl!("Reset"));
                app.auto.add("ai.reset", reset_r, tl!("Reset"));
                if reset_resp.hovered() {
                    ui.painter().rect_filled(reset_r, 4.0, t.hover);
                }
                icons::paint(ui.painter(), reset_r.shrink(3.0), Icon::Reset, if reset_resp.hovered() { t.text } else { t.text_dim });
                if reset_resp.clicked() {
                    let d = AiToolsState::default();
                    app.ui.ai_tools.threshold_db = d.threshold_db;
                    app.ui.ai_tools.min_pause_ms = d.min_pause_ms;
                    app.ui.ai_tools.padding_ms = d.padding_ms;
                }
            });
        });

        draw_hardware_badge_row(ui, &mut app.auto, "ai.badge.silence", AiFeatureWeight::Lightweight, status, app.ui.dark);

        draw_ai_ready_pill(ui, &mut app.auto, t, app.ui.dark, "ai.silence.status");

        let summary = if pauses > 0 {
            tlf!("{n} pauses detected ({sec}s saved)", n = pauses, sec = format!("{saved_secs:.1}"))
        } else {
            tl!("No silent pauses detected").into()
        };
        let summary_col = if pauses > 0 { Color32::from_rgb(0x7C, 0xB3, 0xFF) } else { t.text_dim };
        ui.add(egui::Label::new(egui::RichText::new(summary).font(Tokens::ui(10.5)).color(summary_col)).wrap());

        draw_slider_row(
            ui,
            &mut app.auto,
            t,
            "ai.thresholdDb",
            tl!("Silence Threshold"),
            &format!("{:.1} dB", app.ui.ai_tools.threshold_db),
            &mut app.ui.ai_tools.threshold_db,
            -60.0..=-15.0,
        );

        draw_slider_row(
            ui,
            &mut app.auto,
            t,
            "ai.minPauseMs",
            tl!("Minimum Pause"),
            &format!("{:.0} ms", app.ui.ai_tools.min_pause_ms),
            &mut app.ui.ai_tools.min_pause_ms,
            150.0..=2000.0,
        );

        draw_slider_row(
            ui,
            &mut app.auto,
            t,
            "ai.paddingMs",
            tl!("Cut Padding"),
            &format!("{:.0} ms", app.ui.ai_tools.padding_ms),
            &mut app.ui.ai_tools.padding_ms,
            0.0..=500.0,
        );
        ui.add_space(2.0);

        let _ = card_border;
        let has_seq = app.session.active_sequence().is_some();
        let clicked = draw_primary_icon_button(ui, &mut app.auto, t, "ai.removePauses", tl!("Remove Pauses"), Icon::Sparkle, has_seq, true, None);
        if has_seq && clicked {
            let ctx = ui.ctx().clone();
            let params = json!({
                "thresholdDb": app.ui.ai_tools.threshold_db,
                "minPauseMs": app.ui.ai_tools.min_pause_ms,
                "paddingMs": app.ui.ai_tools.padding_ms,
            });
            let _ = crate::menus::invoke(app, &ctx, "ai.removePauses", params);
        }
    });
}

fn draw_subtitles_card(app: &mut FilmcraftApp, ui: &mut egui::Ui, t: &Tokens, status: &SubtitleStatus) {
    let (_card_bg, _card_border, frame) = card_frame(t, app.ui.dark);
    let card_w = (ui.available_width() - 14.0).max(40.0);

    frame.show(ui, |ui| {
        ui.set_width(card_w);
        ui.spacing_mut().item_spacing.y = 2.0;
        ui.spacing_mut().interact_size.y = 18.0;

        ui.horizontal(|ui| {
            let (icon_r, _) = ui.allocate_exact_size(vec2(14.0, 14.0), Sense::hover());
            icons::paint(ui.painter(), icon_r, Icon::Captions, Color32::from_rgb(0xA7, 0x8B, 0xFA));
            let title_w = (ui.available_width() - 22.0).max(40.0);
            ui.add_sized(vec2(title_w, 18.0), egui::Label::new(egui::RichText::new(tl!("AI Subtitles")).font(Tokens::semibold(13.0)).color(t.text)).truncate())
                .on_hover_text(tl!("Generate offline subtitles locally with Orpheus AI."));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                draw_help_badge(
                    ui,
                    &mut app.auto,
                    t,
                    "ai.help.subtitles",
                    tl!("Transcribes speech offline into timed captions on the timeline. Needs a sequence with spoken audio and a downloaded speech model."),
                );
            });
        });

        draw_hardware_badge_row(ui, &mut app.auto, "ai.badge.subtitles", AiFeatureWeight::NeuralSpeech, status, app.ui.dark);

        draw_cli_status(app, ui, t, status);

        draw_language_selector(app, ui, t);

        draw_model_selector(app, ui, t);

        let compute = draw_compute_selector(app, ui, t, status);

        let selected_installed = draw_model_download_section(app, ui, t, status);

        draw_style_selector(app, ui, t);
        ui.add_space(2.0);

        draw_generate_section(app, ui, t, status, selected_installed, compute);
    });
}

fn draw_cli_status(app: &mut FilmcraftApp, ui: &mut egui::Ui, t: &Tokens, status: &SubtitleStatus) {
    if let Some(prog) = active_job_progress(app, app.ui.ai_tools.subtitle_engine_download_job) {
        let shown = nonblocking_progress_status(&prog, tl!("Downloading Whisper AI engine…"));
        draw_job_bar(ui, &mut app.auto, t, prog.fraction(), &shown, &prog, "ai.subtitles.cancelEngine");
        return;
    }
    if status.cli_found {
        draw_ai_ready_pill(ui, &mut app.auto, t, app.ui.dark, "ai.subtitles.cliStatus");
        if !status.vulkan_dll_found && status.recommendation.recommended == SubtitleComputeDevice::Gpu {
            ui.add_space(2.0);
            let upgrade_btn = ui.small_button(tl!("Reinstall / Upgrade to GPU Engine"));
            app.auto.add("ai.subtitles.upgradeGpuEngine", upgrade_btn.rect, tl!("Reinstall / Upgrade to GPU Engine"));
            if upgrade_btn.on_hover_text(tl!("Download the Vulkan-accelerated whisper-cli.exe and ggml-vulkan.dll into %APPDATA%\\Orpheus\\bin\\")).clicked() {
                let ctx = ui.ctx().clone();
                let _ = crate::menus::invoke(app, &ctx, "ai.downloadSubtitleEngine", json!({"force": true}));
            }
            if let Some(err) = &app.ui.ai_tools.subtitle_engine_download_error {
                ui.add(egui::Label::new(egui::RichText::new(err).font(Tokens::ui(10.0)).color(t.danger)).wrap());
            }
        }
    } else {
        let title = tl!("Orpheus AI engine not found");
        let hint = tl!("Place whisper-cli.exe next to Orpheus or in %APPDATA%\\Orpheus\\bin\\whisper-cli.exe");
        let resp = egui::Frame::new()
            .fill(Color32::from_rgba_unmultiplied(0xE0, 0x9A, 0x3C, 22))
            .stroke(Stroke::new(1.0, Color32::from_rgba_unmultiplied(0xE0, 0x9A, 0x3C, 120)))
            .corner_radius(CornerRadius::same(5))
            .inner_margin(egui::Margin::same(5))
            .show(ui, |ui| {
                ui.set_width((ui.available_width() - 10.0).max(40.0));
                ui.label(egui::RichText::new(title).font(Tokens::semibold(10.5)).color(WARN_AMBER));
                ui.label(egui::RichText::new(hint).font(Tokens::ui(10.0)).color(t.text_dim));
                ui.add_space(3.0);
                let dl_btn = ui.button(tl!("Download AI Engine"));
                app.auto.add("ai.subtitles.downloadEngine", dl_btn.rect, tl!("Download AI Engine"));
                if dl_btn.on_hover_text(tl!("Download whisper-cli.exe and ggml-vulkan.dll into %APPDATA%\\Orpheus\\bin\\")).clicked() {
                    let ctx = ui.ctx().clone();
                    let _ = crate::menus::invoke(app, &ctx, "ai.downloadSubtitleEngine", json!({"force": true}));
                }
                if let Some(err) = &app.ui.ai_tools.subtitle_engine_download_error {
                    ui.add(egui::Label::new(egui::RichText::new(err).font(Tokens::ui(10.0)).color(t.danger)).wrap());
                }
            });
        app.auto.add("ai.subtitles.cliStatus", resp.response.rect, title);
    }
}

fn model_display_label(model_id: &str, language: &str) -> String {
    match model_id {
        "large-v3-turbo" => {
            let lang_tag = if language.eq_ignore_ascii_case("auto") { tl!("All languages").to_string() } else { language.trim().to_ascii_uppercase() };
            tlf!("Large v3 Turbo — Most accurate ({lang})", lang = lang_tag)
        }
        "base" => tl!("Base — Fast").to_string(),
        other => WHISPER_MODELS.iter().find(|m| m.id == other).map_or_else(|| other.to_string(), |m| crate::i18n::t(m.label).to_string()),
    }
}

fn draw_language_selector(app: &mut FilmcraftApp, ui: &mut egui::Ui, t: &Tokens) {
    let cur_label = SUBTITLE_LANGUAGES
        .iter()
        .find(|(id, _)| *id == app.ui.ai_tools.subtitle_language)
        .map_or_else(|| app.ui.ai_tools.subtitle_language.clone(), |(_, name)| crate::i18n::t(name).to_string());
    let label = tl!("Language");
    let total_w = ui.available_width();
    let combo_w = (total_w * 0.56).clamp(84.0, 165.0).min((total_w - 64.0).max(60.0));
    let label_w = (total_w - combo_w - 6.0).max(40.0);
    let shown_selected = layout_truncated(ui.painter(), &cur_label, Tokens::ui(11.5), t.text, (combo_w - 28.0).max(24.0)).text().to_string();
    let mut rect_out = Rect::NOTHING;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        ui.add_sized(vec2(label_w, 18.0), egui::Label::new(egui::RichText::new(label).font(Tokens::ui(11.5)).color(t.text_dim)).truncate());
        let cb = craft_ui::ComboBox::from_id_salt("ai-sub-lang").selected_text(&shown_selected).width(combo_w).show_ui(ui, |ui| {
            for (code, raw_name) in SUBTITLE_LANGUAGES {
                let shown = crate::i18n::t(raw_name);
                let resp = ui.selectable_label(app.ui.ai_tools.subtitle_language == *code, shown);
                if resp.clicked() {
                    app.ui.ai_tools.subtitle_language = (*code).to_string();
                }
            }
        });
        rect_out = cb.response.rect;
    });
    app.auto.add("ai.subtitles.language", rect_out, label);
}

fn draw_model_selector(app: &mut FilmcraftApp, ui: &mut egui::Ui, t: &Tokens) {
    let cur_label = model_display_label(&app.ui.ai_tools.subtitle_model, &app.ui.ai_tools.subtitle_language);
    let label = tl!("AI Model");
    let total_w = ui.available_width();
    let combo_w = (total_w * 0.56).clamp(84.0, 165.0).min((total_w - 64.0).max(60.0));
    let label_w = (total_w - combo_w - 6.0).max(40.0);
    let shown_selected = layout_truncated(ui.painter(), &cur_label, Tokens::ui(11.5), t.text, (combo_w - 28.0).max(24.0)).text().to_string();
    let gpu = subtitles::detect_primary_gpu();
    let ram_mb = filmcraft_engine::ai_models::detect_system_ram_mb();
    let mut rect_out = Rect::NOTHING;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        ui.add_sized(vec2(label_w, 18.0), egui::Label::new(egui::RichText::new(label).font(Tokens::ui(11.5)).color(t.text_dim)).truncate());
        let cb = craft_ui::ComboBox::from_id_salt("ai-sub-model").selected_text(&shown_selected).width(combo_w).show_ui(ui, |ui| {
            for m in WHISPER_MODELS {
                let base_shown = model_display_label(m.id, &app.ui.ai_tools.subtitle_language);
                let (_, rec_for_specs) = filmcraft_engine::ai_models::whisper_tier_readiness_for_specs(m.id, gpu.as_ref(), ram_mb);
                let shown = if rec_for_specs { format!("{base_shown} ★ {}", tl!("Recommended for your specs")) } else { base_shown };
                let resp = ui.selectable_label(app.ui.ai_tools.subtitle_model == m.id, shown);
                if resp.clicked() {
                    app.ui.ai_tools.subtitle_model = m.id.to_string();
                }
            }
        });
        rect_out = cb.response.rect;
    });
    app.auto.add("ai.subtitles.model", rect_out, label);
}

fn resolve_compute(stored: Option<SubtitleComputeDevice>, status: &SubtitleStatus) -> SubtitleComputeDevice {
    let chosen = stored.unwrap_or(status.recommendation.recommended);
    if chosen == SubtitleComputeDevice::Gpu && !status.recommendation.gpu_available { SubtitleComputeDevice::Cpu } else { chosen }
}

fn layout_truncated(painter: &egui::Painter, text: &str, font: egui::FontId, color: Color32, max_w: f32) -> Arc<egui::Galley> {
    let full = painter.layout_no_wrap(text.to_string(), font.clone(), color);
    if full.size().x <= max_w || max_w <= 8.0 || text.is_empty() {
        return full;
    }
    let mut end = text.len();
    while end > 0 {
        // Step back one Unicode scalar value so slicing stays on a valid UTF-8 char boundary.
        let prev = text[..end].char_indices().next_back().map_or(0, |(idx, _)| idx);
        end = prev;
        let candidate = if end == 0 { "…".to_string() } else { format!("{}…", &text[..end]) };
        let g = painter.layout_no_wrap(candidate, font.clone(), color);
        if g.size().x <= max_w || end == 0 {
            return g;
        }
    }
    full
}

struct ComputeCardSpec<'a> {
    id: &'a str,
    title: &'a str,
    subtitle: &'a str,
    badge: Option<&'a str>,
    active: bool,
    enabled: bool,
    disabled_tip: Option<&'a str>,
}

fn draw_compute_card(ui: &mut egui::Ui, auto: &mut crate::automation::Registry, t: &Tokens, dark: bool, w: f32, spec: &ComputeCardSpec<'_>) -> bool {
    let sense = if spec.enabled { Sense::click() } else { Sense::hover() };
    let (r, mut resp) = ui.allocate_exact_size(vec2(w.max(48.0), 30.0), sense);
    if let Some(tip) = spec.disabled_tip.filter(|_| !spec.enabled) {
        resp = resp.on_hover_text(tip);
    }
    auto.add(spec.id, r, spec.title);
    if let Some(el) = auto.elements.last_mut() {
        el.enabled = Some(spec.enabled);
    }

    let fill = if !spec.enabled {
        t.disabled_clip_bg
    } else if spec.active {
        t.control_surface
    } else if resp.hovered() {
        t.hover
    } else {
        t.field_bg
    };
    let border = if spec.enabled && spec.active { ROYAL_BLUE } else { t.control_border };
    let title_col = if spec.enabled && spec.active { t.text } else { t.text_dim };
    let sub_col = t.text_dim;

    ui.painter().rect_filled(r, CornerRadius::same(6), fill);
    ui.painter().rect_stroke(r, CornerRadius::same(6), Stroke::new(1.0, border), StrokeKind::Inside);

    let title_font = if spec.enabled && spec.active { Tokens::semibold(11.5) } else { Tokens::ui(11.5) };
    let title_g = ui.painter().layout_no_wrap(spec.title.to_string(), title_font, title_col);
    let top_y = r.min.y + 9.5;
    ui.painter().galley(pos2(r.min.x + 8.0, top_y - title_g.size().y / 2.0), title_g.clone(), title_col);

    if let Some(badge) = spec.badge {
        let title_end = r.min.x + 8.0 + title_g.size().x + 6.0;
        let max_pill_w = (r.max.x - 6.0 - title_end).max(0.0);
        if max_pill_w >= 28.0 {
            let badge_col = if dark { READY_GREEN } else { Color32::from_rgb(0x0E, 0x7C, 0x52) };
            let bg = Color32::from_rgba_unmultiplied(0x34, 0xD3, 0x99, if dark { 32 } else { 44 });
            let bg_g = layout_truncated(ui.painter(), badge, Tokens::semibold(9.0), badge_col, (max_pill_w - 8.0).max(8.0));
            let pill_w = (bg_g.size().x + 8.0).min(max_pill_w);
            let pill_r = Rect::from_min_size(pos2(r.max.x - 6.0 - pill_w, top_y - 6.5), vec2(pill_w, 13.0));
            ui.painter().rect_filled(pill_r, CornerRadius::same(4), bg);
            ui.painter().galley(pos2(pill_r.center().x - bg_g.size().x / 2.0, top_y - bg_g.size().y / 2.0), bg_g, badge_col);
        }
    }

    let sub_g = layout_truncated(ui.painter(), spec.subtitle, Tokens::ui(9.5), sub_col, (r.width() - 16.0).max(16.0));
    ui.painter().galley(pos2(r.min.x + 8.0, (r.min.y + 21.5) - sub_g.size().y / 2.0), sub_g, sub_col);

    spec.enabled && resp.clicked()
}

fn draw_compute_selector(app: &mut FilmcraftApp, ui: &mut egui::Ui, t: &Tokens, status: &SubtitleStatus) -> SubtitleComputeDevice {
    ui.label(egui::RichText::new(tl!("Compute Device")).font(Tokens::ui(11.5)).color(t.text_dim));
    ui.add_space(2.0);

    let rec = &status.recommendation;
    let active = resolve_compute(app.ui.ai_tools.subtitle_compute, status);
    let rec_badge = tl!("Recommended");
    let gpu_sub: String = if !status.vulkan_dll_found {
        if rec.recommended == SubtitleComputeDevice::Gpu { tl!("Click Download AI Engine to enable").into() } else { tl!("Vulkan DLL missing").into() }
    } else if let Some(name) = rec.gpu_display_name.as_deref() {
        name.to_string()
    } else {
        tl!("No GPU detected").into()
    };
    let gpu_tip = if !status.cli_found {
        Some(tl!("Install whisper-cli.exe with ggml-vulkan.dll for GPU acceleration"))
    } else if !status.vulkan_dll_found {
        Some(tl!("ggml-vulkan.dll was not found alongside whisper-cli.exe"))
    } else if !rec.gpu_available {
        Some(tl!("No Vulkan-compatible discrete GPU detected"))
    } else {
        None
    };

    let gpu_spec = ComputeCardSpec {
        id: "ai.subtitles.compute.gpu",
        title: tl!("GPU"),
        subtitle: &gpu_sub,
        badge: (rec.recommended == SubtitleComputeDevice::Gpu).then_some(rec_badge),
        active: active == SubtitleComputeDevice::Gpu,
        enabled: rec.gpu_available,
        disabled_tip: gpu_tip,
    };
    let cpu_spec = ComputeCardSpec {
        id: "ai.subtitles.compute.cpu",
        title: tl!("CPU"),
        subtitle: tl!("Compatibility"),
        badge: (rec.recommended == SubtitleComputeDevice::Cpu).then_some(rec_badge),
        active: active == SubtitleComputeDevice::Cpu,
        enabled: true,
        disabled_tip: None,
    };

    let avail_w = ui.available_width();
    if avail_w < 270.0 {
        if draw_compute_card(ui, &mut app.auto, t, app.ui.dark, avail_w, &gpu_spec) {
            app.ui.ai_tools.subtitle_compute = Some(SubtitleComputeDevice::Gpu);
        }
        ui.add_space(4.0);
        if draw_compute_card(ui, &mut app.auto, t, app.ui.dark, avail_w, &cpu_spec) {
            app.ui.ai_tools.subtitle_compute = Some(SubtitleComputeDevice::Cpu);
        }
    } else {
        let half_w = ((avail_w - 6.0) / 2.0).max(48.0);
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            if draw_compute_card(ui, &mut app.auto, t, app.ui.dark, half_w, &gpu_spec) {
                app.ui.ai_tools.subtitle_compute = Some(SubtitleComputeDevice::Gpu);
            }
            if draw_compute_card(ui, &mut app.auto, t, app.ui.dark, half_w, &cpu_spec) {
                app.ui.ai_tools.subtitle_compute = Some(SubtitleComputeDevice::Cpu);
            }
        });
    }

    ui.add_space(3.0);
    let reason_line: String = match rec.reason {
        ComputeRecommendationReason::DiscreteGpuSufficientVram => {
            if let Some(gpu) = rec.gpu_display_name.as_deref() {
                tlf!("Recommended for your system ({gpu})", gpu = gpu)
            } else {
                tl!("Recommended for your system").into()
            }
        }
        ComputeRecommendationReason::VulkanDllMissing => {
            if status.cli_found {
                tl!("Vulkan DLL missing — using CPU compatibility mode").into()
            } else {
                tl!("Install whisper-cli.exe with ggml-vulkan.dll for GPU acceleration").into()
            }
        }
        ComputeRecommendationReason::LowVram => tl!("Recommended for systems with < 4 GB VRAM").into(),
        ComputeRecommendationReason::IntegratedGpu => tl!("Recommended for integrated graphics (< 4 GB VRAM)").into(),
        ComputeRecommendationReason::NoGpuDetected => tl!("No hardware GPU detected — using CPU mode").into(),
    };
    ui.add(egui::Label::new(egui::RichText::new(reason_line).font(Tokens::ui(10.5)).color(t.text_dim)).wrap());

    resolve_compute(app.ui.ai_tools.subtitle_compute, status)
}

fn nonblocking_progress_status(prog: &Progress, fallback: &'static str) -> String {
    match prog.status.try_lock() {
        Ok(s) if !s.is_empty() => crate::i18n::t(&s).into(),
        _ => fallback.into(),
    }
}

fn active_job_progress(app: &FilmcraftApp, job_id: Option<u64>) -> Option<Arc<Progress>> {
    let id = job_id?;
    app.session.jobs.iter().find(|j| j.id == id && !j.progress.finished.load(Ordering::Relaxed)).map(|j| j.progress.clone())
}

fn draw_model_download_section(app: &mut FilmcraftApp, ui: &mut egui::Ui, t: &Tokens, status: &SubtitleStatus) -> bool {
    let selected_id = app.ui.ai_tools.subtitle_model.clone();
    let model_info = status.models.iter().find(|m| m.id == selected_id);
    let installed = model_info.is_some_and(|m| m.installed);
    let approx_mb = model_info.map_or(1625, |m| m.approx_mb);

    if let Some(prog) = active_job_progress(app, app.ui.ai_tools.subtitle_download_job) {
        let done_mb = prog.done.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0);
        let total_mb = prog.total.load(Ordering::Relaxed).max(1) as f64 / (1024.0 * 1024.0);
        let pct = (prog.fraction() * 100.0).round().clamp(0.0, 100.0);
        let status_txt = tlf!("Downloading… {pct}% ({done}/{total} MB)", pct = pct as u64, done = format!("{done_mb:.0}"), total = format!("{total_mb:.0}"));
        draw_job_bar(ui, &mut app.auto, t, prog.fraction(), &status_txt, &prog, "ai.subtitles.downloadCancel");
        return false;
    }

    if installed {
        let (r, _) = ui.allocate_exact_size(vec2(ui.available_width(), 22.0), Sense::hover());
        ui.painter().circle_filled(pos2(r.min.x + 8.0, r.center().y), 3.5, READY_GREEN);
        ui.painter().text(pos2(r.min.x + 18.0, r.center().y), Align2::LEFT_CENTER, tl!("Model ready offline"), Tokens::ui(11.5), t.text_dim);
        return true;
    }

    let dl_label = tlf!("Download Model (~{mb} MB)", mb = approx_mb);
    let clicked = draw_secondary_button(ui, &mut app.auto, t, "ai.subtitles.download", &dl_label, tl!("Download Model"));
    if clicked {
        let ctx = ui.ctx().clone();
        let _ = crate::menus::invoke(app, &ctx, "ai.downloadSubtitleModel", json!({"model": selected_id}));
    }
    false
}

fn draw_style_selector(app: &mut FilmcraftApp, ui: &mut egui::Ui, t: &Tokens) {
    use filmcraft_engine::subtitles::SubtitleLengthMode;

    ui.label(egui::RichText::new(tl!("Caption Style")).font(Tokens::ui(11.5)).color(t.text_dim));
    let w = ui.available_width();
    let cur = app.ui.ai_tools.subtitle_style;

    let set_style = |ai: &mut crate::state::AiToolsState, mode: SubtitleLengthMode| {
        ai.subtitle_style = mode;
        ai.subtitle_reels_mode = mode == SubtitleLengthMode::ReelsShort;
    };

    if w < 230.0 {
        if draw_toggle_pill(ui, &mut app.auto, t, w, "ai.subtitles.mode.reels", tl!("Reels (1–3 words)"), cur == SubtitleLengthMode::ReelsShort) {
            set_style(&mut app.ui.ai_tools, SubtitleLengthMode::ReelsShort);
        }
        ui.add_space(3.0);
        if draw_toggle_pill(ui, &mut app.auto, t, w, "ai.subtitles.mode.sentences", tl!("Standard sentences"), cur == SubtitleLengthMode::Sentences) {
            set_style(&mut app.ui.ai_tools, SubtitleLengthMode::Sentences);
        }
    } else {
        let half_w = ((w - 6.0) / 2.0).max(40.0);
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            if draw_toggle_pill(ui, &mut app.auto, t, half_w, "ai.subtitles.mode.reels", tl!("Reels (1–3 words)"), cur == SubtitleLengthMode::ReelsShort) {
                set_style(&mut app.ui.ai_tools, SubtitleLengthMode::ReelsShort);
            }
            if draw_toggle_pill(ui, &mut app.auto, t, half_w, "ai.subtitles.mode.sentences", tl!("Standard sentences"), cur == SubtitleLengthMode::Sentences) {
                set_style(&mut app.ui.ai_tools, SubtitleLengthMode::Sentences);
            }
        });
    }
    ui.add_space(3.0);
    if draw_toggle_pill(ui, &mut app.auto, t, w, "ai.subtitles.mode.tiktok", tl!("Bold Pop-In Captions"), cur == SubtitleLengthMode::TikTokCenterPop) {
        set_style(&mut app.ui.ai_tools, SubtitleLengthMode::TikTokCenterPop);
    }
}

fn draw_generate_section(
    app: &mut FilmcraftApp,
    ui: &mut egui::Ui,
    t: &Tokens,
    status: &SubtitleStatus,
    model_installed: bool,
    compute: SubtitleComputeDevice,
) {
    if let Some(prog) = active_job_progress(app, app.ui.ai_tools.subtitle_generate_job) {
        let shown = nonblocking_progress_status(&prog, tl!("Generating AI subtitles…"));
        draw_job_bar(ui, &mut app.auto, t, prog.fraction(), &shown, &prog, "ai.subtitles.cancel");
        return;
    }

    let has_seq = app.session.active_sequence().is_some();
    let ready = has_seq && status.cli_found && model_installed;
    let tip = if !has_seq {
        Some(tl!("Open a sequence first"))
    } else if !status.cli_found {
        Some(tl!("Install the Orpheus AI engine first"))
    } else if !model_installed {
        Some(tl!("Download the selected AI model first"))
    } else {
        None
    };
    let label = tl!("Generate Subtitles");
    let clicked = draw_primary_icon_button(ui, &mut app.auto, t, "ai.subtitles.generate", label, Icon::Captions, ready, false, tip);
    if ready && clicked {
        let ctx = ui.ctx().clone();
        let mode = app.ui.ai_tools.subtitle_style.as_id();
        let _ = crate::menus::invoke(
            app,
            &ctx,
            "ai.generateSubtitles",
            json!({
                "language": app.ui.ai_tools.subtitle_language,
                "model": app.ui.ai_tools.subtitle_model,
                "mode": mode,
                "compute": compute.as_id(),
            }),
        );
    }
}

fn draw_toggle_pill(ui: &mut egui::Ui, auto: &mut crate::automation::Registry, t: &Tokens, w: f32, id: &str, label: &str, active: bool) -> bool {
    let (r, resp) = ui.allocate_exact_size(vec2(w.max(40.0), 22.0), Sense::click());
    auto.add(id, r, label);
    let fill = if active {
        t.control_surface
    } else if resp.hovered() {
        t.hover
    } else {
        t.field_bg
    };
    let border = if active { ROYAL_BLUE } else { t.control_border };
    let fg = if active { t.text } else { t.text_dim };
    ui.painter().rect_filled(r, CornerRadius::same(6), fill);
    ui.painter().rect_stroke(r, CornerRadius::same(6), Stroke::new(1.0, border), StrokeKind::Inside);
    let g = layout_truncated(ui.painter(), label, if active { Tokens::semibold(11.0) } else { Tokens::ui(11.0) }, fg, (r.width() - 12.0).max(12.0));
    ui.painter().galley(pos2((r.center().x - g.size().x / 2.0).max(r.min.x + 6.0), r.center().y - g.size().y / 2.0), g, fg);
    resp.clicked()
}

fn draw_secondary_button(ui: &mut egui::Ui, auto: &mut crate::automation::Registry, t: &Tokens, id: &str, shown: &str, auto_label: &str) -> bool {
    let (r, resp) = ui.allocate_exact_size(vec2(ui.available_width().max(40.0), 28.0), Sense::click());
    auto.add(id, r, auto_label);
    let fill = if resp.hovered() { t.hover } else { t.control_surface };
    ui.painter().rect_filled(r, CornerRadius::same(6), fill);
    ui.painter().rect_stroke(r, CornerRadius::same(6), Stroke::new(1.0, t.control_border), StrokeKind::Inside);
    let g = layout_truncated(ui.painter(), shown, Tokens::semibold(12.0), t.text, (r.width() - 12.0).max(12.0));
    ui.painter().galley(pos2((r.center().x - g.size().x / 2.0).max(r.min.x + 6.0), r.center().y - g.size().y / 2.0), g, t.text);
    resp.clicked()
}

#[allow(clippy::too_many_arguments)]
fn draw_primary_icon_button(
    ui: &mut egui::Ui,
    auto: &mut crate::automation::Registry,
    t: &Tokens,
    id: &str,
    label: &str,
    icon: Icon,
    enabled: bool,
    primary_blue: bool,
    disabled_tip: Option<&str>,
) -> bool {
    let (btn_r, mut btn_resp) = ui.allocate_exact_size(vec2(ui.available_width().max(40.0), 26.0), if enabled { Sense::click() } else { Sense::hover() });
    if let Some(tip) = disabled_tip.filter(|_| !enabled) {
        btn_resp = btn_resp.on_hover_text(tip);
    }
    auto.add(id, btn_r, label);
    if let Some(el) = auto.elements.last_mut() {
        el.enabled = Some(enabled);
    }

    let base = if primary_blue { ROYAL_BLUE } else { Color32::from_rgb(0x6D, 0x4A, 0xFF) };
    let hover = if primary_blue { ROYAL_BLUE_HOVER } else { Color32::from_rgb(0x7D, 0x5D, 0xFF) };
    let active = if primary_blue { ROYAL_BLUE_ACTIVE } else { Color32::from_rgb(0x5B, 0x39, 0xE6) };

    let btn_fill = if !enabled {
        t.disabled_clip_bg
    } else if btn_resp.is_pointer_button_down_on() {
        active
    } else if btn_resp.hovered() {
        hover
    } else {
        base
    };
    ui.painter().rect_filled(btn_r, CornerRadius::same(8), btn_fill);

    let fg = if enabled { Color32::WHITE } else { t.text_dim };
    let icon_s = 14.0;
    let gap = 7.0;
    let max_text_w = (btn_r.width() - icon_s - gap - 14.0).max(16.0);
    let text_g = layout_truncated(ui.painter(), label, Tokens::semibold(13.0), fg, max_text_w);
    let total_w = icon_s + gap + text_g.size().x;
    let sx = (btn_r.center().x - total_w / 2.0).max(btn_r.min.x + 7.0);
    icons::paint(ui.painter(), Rect::from_center_size(pos2(sx + icon_s / 2.0, btn_r.center().y), vec2(icon_s, icon_s)), icon, fg);
    ui.painter().galley(pos2(sx + icon_s + gap, btn_r.center().y - text_g.size().y / 2.0), text_g, fg);

    enabled && btn_resp.clicked()
}

fn draw_job_bar(ui: &mut egui::Ui, auto: &mut crate::automation::Registry, t: &Tokens, fraction: f32, label: &str, prog: &Arc<Progress>, cancel_id: &str) {
    let w = ui.available_width();
    ui.horizontal(|ui| {
        let label_w = (w - 64.0).max(40.0);
        ui.add_sized(vec2(label_w, 18.0), egui::Label::new(egui::RichText::new(label).font(Tokens::ui(11.5)).color(t.text)).truncate());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let b = ui.small_button(tl!("Cancel"));
            auto.add(cancel_id, b.rect, tl!("Cancel"));
            if b.clicked() {
                prog.cancel.store(true, Ordering::Relaxed);
            }
        });
    });
    let (bar_r, _) = ui.allocate_exact_size(vec2(ui.available_width(), 6.0), Sense::hover());
    ui.painter().rect_filled(bar_r, 3.0, t.separator);
    let fill_w = (bar_r.width() * fraction.clamp(0.0, 1.0)).max(2.0);
    ui.painter().rect_filled(Rect::from_min_size(bar_r.min, vec2(fill_w, bar_r.height())), 3.0, ROYAL_BLUE);
}

#[allow(clippy::too_many_arguments)]
fn draw_slider_row(
    ui: &mut egui::Ui,
    auto: &mut crate::automation::Registry,
    t: &Tokens,
    id: &str,
    label: &str,
    readout: &str,
    value: &mut f64,
    range: std::ops::RangeInclusive<f64>,
) {
    let w = ui.available_width().max(120.0);
    let label_w = (w * 0.40).clamp(64.0, 120.0);
    let readout_w = 48.0;
    let slider_w = (w - label_w - readout_w - 10.0).max(24.0);
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        ui.spacing_mut().slider_width = (slider_w - 8.0).max(16.0);
        ui.add_sized(vec2(label_w, 18.0), egui::Label::new(egui::RichText::new(label).font(Tokens::ui(11.0)).color(t.text)).truncate());
        let resp = ui.add_sized(vec2(slider_w, 18.0), egui::Slider::new(value, range).show_value(false));
        auto.add(id, resp.rect, label);
        ui.add_sized(vec2(readout_w, 18.0), egui::Label::new(egui::RichText::new(readout).font(Tokens::mono(11.0)).color(t.hot_text)).truncate());
    });
}

fn cached_preview(app: &FilmcraftApp, ctx: &egui::Context) -> (usize, f64) {
    if app.session.active_sequence().is_none() {
        return (0, 0.0);
    }
    let key = PreviewKey {
        revision: app.session.revision,
        seq: app.session.state.active_sequence.map(|i| i.0),
        selection: app.session.state.selection.iter().map(|c| c.0).collect(),
        threshold_tenths: (app.ui.ai_tools.threshold_db * 10.0).round() as i64,
        min_pause_ms: app.ui.ai_tools.min_pause_ms.round() as i64,
        padding_ms: app.ui.ai_tools.padding_ms.round() as i64,
    };
    let cache_id = egui::Id::unique("ai-silence-preview-cache");
    if let Some((saved_key, cached)) = ctx.data(|d| d.get_temp::<(PreviewKey, (usize, f64))>(cache_id))
        && saved_key == key
    {
        return cached;
    }
    let opts =
        SilenceOptions { threshold_db: app.ui.ai_tools.threshold_db, min_pause_ms: app.ui.ai_tools.min_pause_ms, padding_ms: app.ui.ai_tools.padding_ms };
    let ranges = detect_sequence_silence(&app.session, &json!({}), opts);
    let total_ticks: i64 = ranges.iter().map(|r| r.duration.0).sum();
    let res = (ranges.len(), total_ticks as f64 / TICKS_PER_SECOND as f64);
    ctx.data_mut(|d| d.insert_temp(cache_id, (key, res)));
    res
}

fn cached_subtitle_status(ctx: &egui::Context, downloading: bool) -> SubtitleStatus {
    let now = ctx.input(|i| i.time);
    let cache_id = egui::Id::unique("ai-subtitles-status-cache");
    if let Some((saved_at, saved_dl, st)) = ctx.data(|d| d.get_temp::<(f64, bool, SubtitleStatus)>(cache_id))
        && saved_dl == downloading
        && (now - saved_at).abs() < 10.0
    {
        return st;
    }
    let st = subtitles::status();
    ctx.data_mut(|d| d.insert_temp(cache_id, (now, downloading, st.clone())));
    st
}
