//! Streamlined, action-first Creator workspace for TikTok, YouTube Shorts, and Reels.
//!
//! Designed to stay visually and functionally distinct from Studio Mode:
//! - Top: full-width **Orpheus All-In-One** natural-language prompt bar
//! - Left: compact rail (**Actions · Media · Text · AI**) opening into a spacious 1-Click
//!   **Creator Actions** dock (Auto Shorts / Game Stream Split, inline AI Subtitles, Bold
//!   Title / Hook, Remove Pauses, Enhance Voice, and Quick 1080×1920 60fps Export)
//! - Center: maximized **Video Preview Monitor** with a `9:16 Vertical` / `16:9` aspect toggle
//!   right underneath
//! - Bottom: simplified magnetic-style **Timeline** with prominent **Split · Delete · Speed**
//!   controls and larger clip rows (hiding Studio's audio mixer, scopes, dense parameter trees,
//!   and keyframe graphs)

use std::sync::Arc;
use std::sync::atomic::Ordering;

use egui::{Align2, Color32, CornerRadius, Rect, Sense, Stroke, StrokeKind, pos2, vec2};
use filmcraft_engine::subtitles::{SUBTITLE_LANGUAGES, SubtitleComputeDevice, SubtitleLengthMode, SubtitleStatus};
use filmcraft_export::Progress;
use serde_json::json;

use crate::FilmcraftApp;
use crate::dock::PanelKind;
use crate::icons::{self, Icon};
use crate::state::CreatorTool;
use crate::theme::Tokens;

const GAP: f32 = 5.0;
const PROMPT_BAR_H: f32 = 42.0;
const RAIL_W: f32 = 64.0;
const ACTIONS_DRAWER_W: f32 = 316.0;
const TOOL_DRAWER_W: f32 = 264.0;
const HEADER_H: f32 = 32.0;
const ASPECT_BAR_H: f32 = 36.0;
const TIMELINE_BAR_H: f32 = 38.0;

const ROYAL_BLUE: Color32 = Color32::from_rgb(0x2F, 0x6F, 0xED);
const ROYAL_BLUE_HOVER: Color32 = Color32::from_rgb(0x3B, 0x7A, 0xF5);
const EMERALD: Color32 = Color32::from_rgb(0x10, 0xB9, 0x81);
const EMERALD_HOVER: Color32 = Color32::from_rgb(0x16, 0xCE, 0x91);
const VIOLET: Color32 = Color32::from_rgb(0x8B, 0x5C, 0xF6);
const AMBER: Color32 = Color32::from_rgb(0xE0, 0x9A, 0x3C);

const TIKTOK_EXPORT_PRESET: &str = "TikTok / Reels / Shorts (9:16 • 1080×1920 • 60fps)";

pub fn show(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect) {
    let t = app.tokens;
    ui.painter().rect_filled(rect, 0.0, t.app_bg);

    if rect.width() < 240.0 || rect.height() < 220.0 {
        return;
    }

    let prompt_rect = Rect::from_min_max(rect.min, pos2(rect.max.x, (rect.min.y + PROMPT_BAR_H).min(rect.max.y)));
    draw_prompt_bar(app, ui, prompt_rect, &t);

    let body = Rect::from_min_max(pos2(rect.min.x, prompt_rect.max.y + GAP), rect.max);
    if body.height() < 160.0 {
        return;
    }

    let top_h = (body.height() * 0.61).round().clamp(180.0, (body.height() - 140.0).max(180.0));
    let top_rect = Rect::from_min_max(body.min, pos2(body.max.x, body.min.y + top_h));
    let bot_rect = Rect::from_min_max(pos2(body.min.x, top_rect.max.y + GAP), body.max);

    let rail_rect = Rect::from_min_max(top_rect.min, pos2((top_rect.min.x + RAIL_W).min(top_rect.max.x), top_rect.max.y));
    draw_tool_rail(app, ui, rail_rect, &t);

    let mut left_x = rail_rect.max.x + GAP;
    if let Some(tool) = app.ui.creator_tool {
        let target_w = if tool == CreatorTool::Actions { ACTIONS_DRAWER_W } else { TOOL_DRAWER_W };
        let max_frac = if tool == CreatorTool::Actions { 0.32 } else { 0.27 };
        let drawer_w = target_w.min((top_rect.width() * max_frac).max(200.0));
        let drawer_rect = Rect::from_min_max(pos2(left_x, top_rect.min.y), pos2((left_x + drawer_w).min(top_rect.max.x), top_rect.max.y));
        draw_tool_drawer(app, ui, drawer_rect, tool, &t);
        left_x = drawer_rect.max.x + GAP;
    }

    let preview_rect = Rect::from_min_max(pos2(left_x, top_rect.min.y), pos2(top_rect.max.x.max(left_x + 80.0), top_rect.max.y));
    draw_preview_pane(app, ui, preview_rect, &t);

    draw_bottom_timeline(app, ui, bot_rect, &t);
}

fn frame_panel(ui: &egui::Ui, rect: Rect, t: &Tokens, focused: bool) {
    let border = if focused { t.accent } else { t.control_border };
    ui.painter().rect_filled(rect, CornerRadius::same(8), t.panel_bg);
    ui.painter().rect_stroke(rect, CornerRadius::same(8), Stroke::new(1.0, border), StrokeKind::Inside);
}

fn active_job_progress(app: &FilmcraftApp, job_id: Option<u64>) -> Option<Arc<Progress>> {
    let id = job_id?;
    app.session.jobs.iter().find(|j| j.id == id && !j.progress.finished.load(Ordering::Relaxed)).map(|j| j.progress.clone())
}

fn draw_prompt_bar(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect, t: &Tokens) {
    let bg = if app.ui.dark { Color32::from_rgb(0x14, 0x18, 0x24) } else { t.panel_bg };
    let border = Color32::from_rgba_unmultiplied(0x2F, 0x6F, 0xED, if app.ui.dark { 110 } else { 95 });
    ui.painter().rect_filled(rect, CornerRadius::same(8), bg);
    ui.painter().rect_stroke(rect, CornerRadius::same(8), Stroke::new(1.0, border), StrokeKind::Inside);

    let inner = rect.shrink2(vec2(10.0, 5.0));
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(inner).layout(egui::Layout::left_to_right(egui::Align::Center)));
    child.set_clip_rect(inner);
    child.spacing_mut().item_spacing.x = 8.0;

    let (icon_r, _) = child.allocate_exact_size(vec2(16.0, 16.0), Sense::hover());
    icons::paint(child.painter(), icon_r, Icon::Sparkle, ROYAL_BLUE);
    child.label(egui::RichText::new(tl!("Orpheus All-In-One")).font(Tokens::semibold(12.5)).color(t.text));

    if let Some(prog) = active_job_progress(app, app.ui.ai_tools.all_in_one_job) {
        let pct = (prog.fraction().clamp(0.0, 1.0) * 100.0).round() as u32;
        let stage = match prog.status.try_lock() {
            Ok(s) if !s.is_empty() => s.clone(),
            _ => tl!("Running Orpheus All-In-One…").to_string(),
        };
        child.spinner();
        let status_w = (child.available_width() - 96.0).max(80.0);
        child.add_sized(
            vec2(status_w, 22.0),
            egui::Label::new(egui::RichText::new(format!("{stage} ({pct}%)")).font(Tokens::semibold(11.5)).color(ROYAL_BLUE)).truncate(),
        );
        let cancel_btn = child.add(
            egui::Button::new(egui::RichText::new(tl!("Cancel")).font(Tokens::semibold(11.5)).color(Color32::WHITE))
                .fill(t.danger)
                .corner_radius(CornerRadius::same(6))
                .min_size(vec2(72.0, 26.0)),
        );
        app.auto.add("creator.prompt.cancel", cancel_btn.rect, tl!("Cancel"));
        if cancel_btn.clicked() {
            prog.cancel.store(true, Ordering::Relaxed);
        }
        return;
    }

    let show_chip = inner.width() > 760.0;
    let right_reserve = if show_chip { 255.0 } else { 112.0 };
    let input_w = (child.available_width() - right_reserve).max(120.0);

    let edit = child.add_sized(
        vec2(input_w, 28.0),
        egui::TextEdit::singleline(&mut app.ui.ai_tools.all_in_one_prompt)
            .hint_text("Ask Orpheus AI… e.g. \"9:16 Shorts with FaceCam at bottom + subtitles\"")
            .font(Tokens::ui(12.0)),
    );
    app.auto.add("creator.prompt.input", edit.rect, tl!("Prompt"));
    let submitted = edit.lost_focus() && child.input(|i| i.key_pressed(egui::Key::Enter));

    if show_chip {
        let chip_label = "Shorts + FaceCam";
        let chip_r = child.add(
            egui::Button::new(egui::RichText::new(chip_label).font(Tokens::ui(11.0)).color(t.text_dim))
                .fill(t.control_surface)
                .stroke(Stroke::new(1.0, t.control_border))
                .corner_radius(CornerRadius::same(6))
                .min_size(vec2(124.0, 26.0)),
        );
        app.auto.add("creator.prompt.presetShorts", chip_r.rect, chip_label);
        if chip_r.clicked() {
            app.ui.ai_tools.all_in_one_prompt = "Make 9:16 Shorts with FaceCam at the bottom and add subtitles".into();
        }
    }

    let ready = !app.ui.ai_tools.all_in_one_prompt.trim().is_empty();
    let run_label = tl!("Generate & Apply");
    let run_btn = child.add_enabled(
        ready,
        egui::Button::new(egui::RichText::new(run_label).font(Tokens::semibold(11.5)).color(Color32::WHITE))
            .fill(if ready { ROYAL_BLUE } else { t.control_surface })
            .corner_radius(CornerRadius::same(6))
            .min_size(vec2(104.0, 28.0)),
    );
    app.auto.add("creator.prompt.run", run_btn.rect, run_label);
    if ready && (run_btn.clicked() || submitted) {
        let ctx = child.ctx().clone();
        let _ = crate::menus::invoke(
            app,
            &ctx,
            "ai.allInOne",
            json!({
                "prompt": app.ui.ai_tools.all_in_one_prompt,
                "attachments": app.ui.ai_tools.all_in_one_attachments,
                "ollamaUrl": app.ui.ai_tools.all_in_one_ollama_url,
            }),
        );
    }
}

fn draw_tool_rail(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect, t: &Tokens) {
    frame_panel(ui, rect, t, false);
    let mut y = rect.min.y + 8.0;
    let items = [
        (CreatorTool::Actions, "actions", Icon::Fx, tl!("Create")),
        (CreatorTool::Media, "media", Icon::Film, tl!("Media")),
        (CreatorTool::Text, "text", Icon::Type, tl!("Text")),
        (CreatorTool::Ai, "ai", Icon::Sparkle, tl!("AI")),
    ];
    for (tool, id, icon, label) in items {
        let active = app.ui.creator_tool == Some(tool);
        let btn = Rect::from_min_size(pos2(rect.min.x + 6.0, y), vec2((rect.width() - 12.0).max(20.0), 54.0));
        let resp = ui.interact(btn, egui::Id::unique(("creator-rail", id)), Sense::click());
        app.auto.add(&format!("creator.rail.{id}"), btn, label);
        let bg = if active {
            if app.ui.dark { Color32::from_rgb(0x1C, 0x26, 0x3D) } else { t.control_surface }
        } else if resp.hovered() {
            t.hover
        } else {
            Color32::TRANSPARENT
        };
        if bg != Color32::TRANSPARENT {
            ui.painter().rect_filled(btn, 7.0, bg);
        }
        if active {
            ui.painter().rect_stroke(btn, 7.0, Stroke::new(1.2, ROYAL_BLUE), StrokeKind::Inside);
        }
        let fg = if active || resp.hovered() { t.text } else { t.text_dim };
        let icon_col = if active { ROYAL_BLUE } else { fg };
        let icon_center = pos2(btn.center().x, btn.min.y + 19.0);
        icons::paint(ui.painter(), Rect::from_center_size(icon_center, vec2(16.0, 16.0)), icon, icon_col);
        ui.painter().text(pos2(btn.center().x, btn.max.y - 12.0), Align2::CENTER_CENTER, label, Tokens::semibold(10.5), fg);
        if resp.clicked() {
            app.ui.creator_tool = if active { None } else { Some(tool) };
        }
        y = btn.max.y + 6.0;
    }
}

fn draw_tool_drawer(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect, tool: CreatorTool, t: &Tokens) {
    let kind = match tool {
        CreatorTool::Actions => PanelKind::AiTools,
        CreatorTool::Media => PanelKind::Project,
        CreatorTool::Text => PanelKind::Text,
        CreatorTool::Ai => PanelKind::AiTools,
    };
    if ui.input(|i| i.pointer.any_pressed() && i.pointer.interact_pos().is_some_and(|p| rect.contains(p))) {
        app.ui.focused = kind;
    }
    frame_panel(ui, rect, t, app.ui.focused == kind);

    let hdr = Rect::from_min_max(rect.min, pos2(rect.max.x, (rect.min.y + HEADER_H).min(rect.max.y)));
    ui.painter().line_segment([hdr.left_bottom(), hdr.right_bottom()], Stroke::new(1.0, t.control_border));
    let title = match tool {
        CreatorTool::Actions => tl!("Creator Actions"),
        CreatorTool::Media => tl!("Media"),
        CreatorTool::Text => tl!("Text"),
        CreatorTool::Ai => tl!("AI Tools"),
    };

    let col_r = Rect::from_center_size(pos2(hdr.max.x - 16.0, hdr.center().y), vec2(22.0, 22.0));
    let col_resp = ui.interact(col_r, egui::Id::unique("creator-drawer-close"), Sense::click()).on_hover_text(tl!("Close Panel"));
    app.auto.add("creator.drawer.close", col_r, tl!("Close Panel"));
    if col_resp.hovered() {
        ui.painter().rect_filled(col_r, 4.0, t.hover);
    }
    ui.painter().text(col_r.center(), Align2::CENTER_CENTER, "◀", Tokens::ui(10.0), t.text_dim);
    if col_resp.clicked() {
        app.ui.creator_tool = None;
    }

    let mut title_max_x = col_r.min.x - 6.0;
    if tool == CreatorTool::Media {
        let imp_label = format!("+ {}", tl!("Import"));
        let imp_g = ui.painter().layout_no_wrap(imp_label.clone(), Tokens::semibold(11.0), Color32::WHITE);
        let imp_w = imp_g.size().x + 14.0;
        let imp_r = Rect::from_min_size(pos2(col_r.min.x - imp_w - 6.0, hdr.center().y - 11.0), vec2(imp_w, 22.0));
        title_max_x = imp_r.min.x - 6.0;
        let imp_resp = ui.interact(imp_r, egui::Id::unique("creator-import"), Sense::click());
        app.auto.add("creator.import", imp_r, &imp_label);
        ui.painter().rect_filled(imp_r, 5.0, if imp_resp.hovered() { ROYAL_BLUE_HOVER } else { ROYAL_BLUE });
        ui.painter().galley(pos2(imp_r.min.x + 7.0, imp_r.center().y - imp_g.size().y / 2.0), imp_g, Color32::WHITE);
        if imp_resp.clicked() {
            let ctx = ui.ctx().clone();
            if let Err(e) = crate::menus::invoke(app, &ctx, "file.import", json!({})) {
                app.ui.status = e;
            }
        }
    } else if tool == CreatorTool::Text {
        let txt_label = format!("+ {}", tl!("New Text"));
        let txt_g = ui.painter().layout_no_wrap(txt_label.clone(), Tokens::semibold(11.0), Color32::WHITE);
        let txt_w = txt_g.size().x + 14.0;
        let txt_r = Rect::from_min_size(pos2(col_r.min.x - txt_w - 6.0, hdr.center().y - 11.0), vec2(txt_w, 22.0));
        title_max_x = txt_r.min.x - 6.0;
        let txt_resp = ui.interact(txt_r, egui::Id::unique("creator-drawer-add-text"), Sense::click());
        app.auto.add("creator.drawer.addText", txt_r, &txt_label);
        ui.painter().rect_filled(txt_r, 5.0, if txt_resp.hovered() { ROYAL_BLUE_HOVER } else { ROYAL_BLUE });
        ui.painter().galley(pos2(txt_r.min.x + 7.0, txt_r.center().y - txt_g.size().y / 2.0), txt_g, Color32::WHITE);
        if txt_resp.clicked() {
            let ctx = ui.ctx().clone();
            add_bold_title(app, &ctx, "WATCH THIS!");
        }
    }

    let title_clip = Rect::from_min_max(pos2(hdr.min.x + 8.0, hdr.min.y), pos2(title_max_x.max(hdr.min.x + 24.0), hdr.max.y));
    ui.painter().with_clip_rect(title_clip).text(pos2(hdr.min.x + 12.0, hdr.center().y), Align2::LEFT_CENTER, title, Tokens::semibold(12.5), t.text);

    let body = Rect::from_min_max(pos2(rect.min.x + 2.0, hdr.max.y + 1.0), pos2(rect.max.x - 2.0, rect.max.y - 2.0));
    if body.height() <= 10.0 {
        return;
    }
    if tool == CreatorTool::Actions {
        draw_actions_dock(app, ui, body, t);
    } else {
        let mut child = ui.new_child(egui::UiBuilder::new().max_rect(body).id_salt(("creator-drawer-body", kind)));
        child.set_clip_rect(body);
        super::show(app, &mut child, kind, body);
    }
}

fn creator_card_frame(t: &Tokens, dark: bool) -> egui::Frame {
    let fill = if dark { Color32::from_rgb(0x16, 0x1B, 0x27) } else { t.control_surface };
    let border = if dark { Color32::from_rgba_unmultiplied(255, 255, 255, 22) } else { t.control_border };
    egui::Frame::new().fill(fill).stroke(Stroke::new(1.0, border)).corner_radius(CornerRadius::same(9)).inner_margin(egui::Margin::same(10))
}

fn draw_card_header(ui: &mut egui::Ui, t: &Tokens, icon: Icon, icon_col: Color32, title: &str, subtitle: &str) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 7.0;
        let (icon_r, _) = ui.allocate_exact_size(vec2(16.0, 16.0), Sense::hover());
        icons::paint(ui.painter(), icon_r, icon, icon_col);
        ui.label(egui::RichText::new(title).font(Tokens::semibold(13.0)).color(t.text));
    });
    ui.add(egui::Label::new(egui::RichText::new(subtitle).font(Tokens::ui(10.5)).color(t.text_dim)).wrap());
}

#[allow(clippy::too_many_arguments)]
fn draw_action_cta(
    ui: &mut egui::Ui,
    auto: &mut crate::automation::Registry,
    t: &Tokens,
    id: &str,
    label: &str,
    w: f32,
    fill: Color32,
    hover_fill: Color32,
    enabled: bool,
) -> bool {
    let sense = if enabled { Sense::click() } else { Sense::hover() };
    let (r, resp) = ui.allocate_exact_size(vec2(w.max(48.0), 30.0), sense);
    auto.add(id, r, label);
    if let Some(el) = auto.elements.last_mut() {
        el.enabled = Some(enabled);
    }
    let bg = if !enabled {
        t.disabled_clip_bg
    } else if resp.hovered() {
        hover_fill
    } else {
        fill
    };
    let fg = if enabled { Color32::WHITE } else { t.text_dim };
    ui.painter().rect_filled(r, CornerRadius::same(7), bg);
    ui.painter().text(r.center(), Align2::CENTER_CENTER, label, Tokens::semibold(11.5), fg);
    enabled && resp.clicked()
}

fn draw_inline_progress(ui: &mut egui::Ui, t: &Tokens, prog: &Arc<Progress>, fallback: &str) {
    let frac = prog.fraction().clamp(0.0, 1.0);
    let msg = match prog.status.try_lock() {
        Ok(s) if !s.is_empty() => s.clone(),
        _ => fallback.to_string(),
    };
    let w = ui.available_width();
    ui.horizontal(|ui| {
        let lbl_w = (w - 60.0).max(40.0);
        ui.add_sized(vec2(lbl_w, 18.0), egui::Label::new(egui::RichText::new(msg).font(Tokens::ui(10.5)).color(ROYAL_BLUE)).truncate());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.small_button(tl!("Cancel")).clicked() {
                prog.cancel.store(true, Ordering::Relaxed);
            }
        });
    });
    let (r, _) = ui.allocate_exact_size(vec2(ui.available_width().max(40.0), 6.0), Sense::hover());
    ui.painter().rect_filled(r, 3.0, t.field_bg);
    let fill = Rect::from_min_max(r.min, pos2(r.min.x + r.width() * frac, r.max.y));
    ui.painter().rect_filled(fill, 3.0, ROYAL_BLUE);
}

fn draw_actions_dock(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect, t: &Tokens) {
    let inner = rect.shrink2(vec2(8.0, 6.0));
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(inner).layout(egui::Layout::top_down(egui::Align::Min)));
    child.set_clip_rect(inner);

    let refreshing = app.ui.ai_tools.subtitle_download_job.is_none() && app.ui.ai_tools.subtitle_engine_download_job.is_none();
    let sub_status = super::ai_tools::cached_subtitle_status(ui.ctx(), !refreshing);

    let scroll_w = (inner.width() - 4.0).max(80.0);
    egui::ScrollArea::vertical().auto_shrink([false, false]).id_salt("creator-actions-dock").show(&mut child, |ui| {
        ui.set_width(scroll_w);
        ui.spacing_mut().item_spacing.y = 8.0;

        draw_reframe_card(app, ui, t);
        draw_subtitles_quick_card(app, ui, t, &sub_status);
        draw_bold_title_card(app, ui, t);
        draw_clean_audio_cards(app, ui, t);
        draw_quick_export_card(app, ui, t);
    });
}

fn draw_reframe_card(app: &mut FilmcraftApp, ui: &mut egui::Ui, t: &Tokens) {
    let has_video = app.session.active_sequence().is_some_and(|q| q.video_tracks.iter().any(|tr| !tr.locked && tr.items.iter().any(|it| it.enabled)));
    creator_card_frame(t, app.ui.dark).show(ui, |ui| {
        ui.set_width((ui.available_width() - 4.0).max(60.0));
        ui.spacing_mut().item_spacing.y = 6.0;
        draw_card_header(
            ui,
            t,
            Icon::Sequence,
            ROYAL_BLUE,
            "Auto Shorts & Game Stream",
            "Convert 16:9 clips to 9:16 vertical or top gameplay + bottom FaceCam.",
        );

        if let Some(prog) = active_job_progress(app, app.ui.ai_tools.auto_reframe_job) {
            draw_inline_progress(ui, t, &prog, tl!("Tracking subject for Auto-Reframe…"));
            return;
        }

        let half = ((ui.available_width() - 6.0) / 2.0).max(60.0);
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            if draw_action_cta(ui, &mut app.auto, t, "creator.card.autoShorts", "9:16 Shorts", half, ROYAL_BLUE, ROYAL_BLUE_HOVER, has_video) {
                let ctx = ui.ctx().clone();
                let _ = crate::menus::invoke(app, &ctx, "ai.autoReframe", json!({ "aspect": "9:16", "resizeSequence": true, "splitScreen": false }));
            }
            if draw_action_cta(ui, &mut app.auto, t, "creator.card.gameStream", "Game Stream", half, VIOLET, Color32::from_rgb(0x9D, 0x73, 0xFA), has_video) {
                let ctx = ui.ctx().clone();
                let _ = crate::menus::invoke(
                    app,
                    &ctx,
                    "ai.autoReframe",
                    json!({
                        "aspect": "9:16",
                        "resizeSequence": true,
                        "splitScreen": true,
                        "facecamPlacement": "bottom",
                        "facecamRegion": "auto",
                    }),
                );
            }
        });
    });
}

fn draw_subtitles_quick_card(app: &mut FilmcraftApp, ui: &mut egui::Ui, t: &Tokens, status: &SubtitleStatus) {
    creator_card_frame(t, app.ui.dark).show(ui, |ui| {
        ui.set_width((ui.available_width() - 4.0).max(60.0));
        ui.spacing_mut().item_spacing.y = 6.0;
        draw_card_header(ui, t, Icon::Captions, VIOLET, tl!("AI Subtitles"), "Offline Whisper captions with Bold Pop-In & Reels styling.");

        // Language + GPU/CPU row
        let cur_lang = SUBTITLE_LANGUAGES
            .iter()
            .find(|(code, _)| *code == app.ui.ai_tools.subtitle_language)
            .map_or_else(|| app.ui.ai_tools.subtitle_language.clone(), |(_, name)| crate::i18n::t(name).to_string());
        let chosen_compute = app.ui.ai_tools.subtitle_compute.unwrap_or(status.recommendation.recommended);
        let active_compute =
            if chosen_compute == SubtitleComputeDevice::Gpu && !status.recommendation.gpu_available { SubtitleComputeDevice::Cpu } else { chosen_compute };

        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            let lang_w = (ui.available_width() - 96.0).max(80.0);
            let cb = craft_ui::ComboBox::from_id_salt("creator-sub-lang").selected_text(&cur_lang).width(lang_w).show_ui(ui, |ui| {
                for (code, name) in SUBTITLE_LANGUAGES {
                    if ui.selectable_label(app.ui.ai_tools.subtitle_language == *code, crate::i18n::t(name)).clicked() {
                        app.ui.ai_tools.subtitle_language = (*code).to_string();
                    }
                }
            });
            app.auto.add("creator.subtitles.language", cb.response.rect, tl!("Language"));

            let gpu_ok = status.recommendation.gpu_available;
            for (dev, id, lbl, enabled) in
                [(SubtitleComputeDevice::Gpu, "creator.subtitles.gpu", "GPU", gpu_ok), (SubtitleComputeDevice::Cpu, "creator.subtitles.cpu", "CPU", true)]
            {
                let on = active_compute == dev;
                let btn = ui.add_enabled(
                    enabled,
                    egui::Button::new(egui::RichText::new(lbl).font(Tokens::semibold(10.5)).color(if on { Color32::WHITE } else { t.text_dim }))
                        .fill(if on { ROYAL_BLUE } else { t.field_bg })
                        .corner_radius(CornerRadius::same(5))
                        .min_size(vec2(40.0, 22.0)),
                );
                app.auto.add(id, btn.rect, lbl);
                if enabled && btn.clicked() {
                    app.ui.ai_tools.subtitle_compute = Some(dev);
                }
            }
        });

        // Style pills: Bold Pop-In / Reels / Standard
        let third = ((ui.available_width() - 8.0) / 3.0).max(52.0);
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            for (mode, id, lbl) in [
                (SubtitleLengthMode::TikTokCenterPop, "creator.subtitles.style.pop", "Bold Pop-In"),
                (SubtitleLengthMode::ReelsShort, "creator.subtitles.style.reels", "Reels"),
                (SubtitleLengthMode::Sentences, "creator.subtitles.style.standard", "Standard"),
            ] {
                let on = app.ui.ai_tools.subtitle_style == mode;
                let btn = ui.add(
                    egui::Button::new(egui::RichText::new(lbl).font(Tokens::semibold(10.5)).color(if on { Color32::WHITE } else { t.text_dim }))
                        .fill(if on { VIOLET } else { t.field_bg })
                        .stroke(Stroke::new(1.0, if on { VIOLET } else { t.control_border }))
                        .corner_radius(CornerRadius::same(5))
                        .min_size(vec2(third, 24.0)),
                );
                app.auto.add(id, btn.rect, lbl);
                if btn.clicked() {
                    app.ui.ai_tools.subtitle_style = mode;
                    app.ui.ai_tools.subtitle_reels_mode = mode == SubtitleLengthMode::ReelsShort;
                }
            }
        });

        if let Some(prog) = active_job_progress(app, app.ui.ai_tools.subtitle_generate_job) {
            draw_inline_progress(ui, t, &prog, tl!("Generating AI subtitles…"));
            return;
        }
        if let Some(prog) = active_job_progress(app, app.ui.ai_tools.subtitle_download_job) {
            draw_inline_progress(ui, t, &prog, tl!("Downloading AI model…"));
            return;
        }
        if let Some(prog) = active_job_progress(app, app.ui.ai_tools.subtitle_engine_download_job) {
            draw_inline_progress(ui, t, &prog, tl!("Downloading Whisper AI engine…"));
            return;
        }

        let w = ui.available_width();
        let has_seq = app.session.active_sequence().is_some();
        let model_installed = status.models.iter().any(|m| m.id == app.ui.ai_tools.subtitle_model && m.installed);
        if !status.cli_found {
            if draw_action_cta(ui, &mut app.auto, t, "creator.subtitles.downloadEngine", tl!("Download AI Engine"), w, AMBER, AMBER, true) {
                let ctx = ui.ctx().clone();
                let _ = crate::menus::invoke(app, &ctx, "ai.downloadSubtitleEngine", json!({"force": true}));
            }
        } else if !model_installed {
            if draw_action_cta(ui, &mut app.auto, t, "creator.subtitles.downloadModel", tl!("Download Model"), w, AMBER, AMBER, true) {
                let ctx = ui.ctx().clone();
                let model = app.ui.ai_tools.subtitle_model.clone();
                let _ = crate::menus::invoke(app, &ctx, "ai.downloadSubtitleModel", json!({"model": model}));
            }
        } else if draw_action_cta(ui, &mut app.auto, t, "creator.card.generateSubtitles", tl!("Generate Subtitles"), w, ROYAL_BLUE, ROYAL_BLUE_HOVER, has_seq) {
            let ctx = ui.ctx().clone();
            let _ = crate::menus::invoke(
                app,
                &ctx,
                "ai.generateSubtitles",
                json!({
                    "language": app.ui.ai_tools.subtitle_language,
                    "model": app.ui.ai_tools.subtitle_model,
                    "mode": app.ui.ai_tools.subtitle_style.as_id(),
                    "compute": active_compute.as_id(),
                }),
            );
        }
    });
}

fn add_bold_title(app: &mut FilmcraftApp, ctx: &egui::Context, text: &str) {
    let cleaned = text.trim();
    let final_text = if cleaned.is_empty() { "WATCH THIS!" } else { cleaned };
    let mut params = json!({ "text": final_text, "newClip": true });
    if let Some(seq) = app.session.active_sequence()
        && let Some(split_y) = filmcraft_engine::subtitles::detect_game_stream_split_y(seq)
    {
        let cx = f64::from(seq.settings.width.max(1)) * 0.5;
        let cy = (f64::from(seq.settings.height.max(1)) * (split_y - 0.03).clamp(0.45, 0.70)).round();
        params["position"] = json!([cx, cy]);
    }
    if let Err(e) = crate::menus::invoke(app, ctx, "graphics.newText", params) {
        app.ui.status = e;
    }
}

fn draw_bold_title_card(app: &mut FilmcraftApp, ui: &mut egui::Ui, t: &Tokens) {
    let has_seq = app.session.active_sequence().is_some();
    creator_card_frame(t, app.ui.dark).show(ui, |ui| {
        ui.set_width((ui.available_width() - 4.0).max(60.0));
        ui.spacing_mut().item_spacing.y = 6.0;
        draw_card_header(ui, t, Icon::Type, AMBER, "Bold Title / Hook", "Drop an instant punchy headline onto your video.");

        let edit = ui.add_sized(
            vec2(ui.available_width(), 26.0),
            egui::TextEdit::singleline(&mut app.ui.creator_hook_text).hint_text("WATCH THIS!").font(Tokens::ui(12.0)),
        );
        app.auto.add("creator.hook.input", edit.rect, "Bold Title Text");
        let submit = edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));

        let clicked = draw_action_cta(
            ui,
            &mut app.auto,
            t,
            "creator.card.addBoldTitle",
            "+ Add Bold Title / Hook",
            ui.available_width(),
            t.accent,
            t.accent_hover,
            has_seq,
        );
        if has_seq && (clicked || submit) {
            let ctx = ui.ctx().clone();
            let text = app.ui.creator_hook_text.clone();
            add_bold_title(app, &ctx, &text);
        }
    });
}

fn draw_clean_audio_cards(app: &mut FilmcraftApp, ui: &mut egui::Ui, t: &Tokens) {
    let has_seq = app.session.active_sequence().is_some();
    creator_card_frame(t, app.ui.dark).show(ui, |ui| {
        ui.set_width((ui.available_width() - 4.0).max(60.0));
        ui.spacing_mut().item_spacing.y = 6.0;
        draw_card_header(ui, t, Icon::Audio, EMERALD, "Smart Pacing & Voice", "Trim silent dead air and boost studio voice clarity in 1 click.");

        if let Some(prog) = active_job_progress(app, app.ui.ai_tools.silence_job) {
            draw_inline_progress(ui, t, &prog, tl!("Scanning audio for silent pauses…"));
            return;
        }
        if let Some(prog) = active_job_progress(app, app.ui.ai_tools.audio_enhance_job) {
            draw_inline_progress(ui, t, &prog, tl!("Enhancing audio…"));
            return;
        }

        let half = ((ui.available_width() - 6.0) / 2.0).max(60.0);
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            if draw_action_cta(
                ui,
                &mut app.auto,
                t,
                "creator.card.removePauses",
                tl!("Remove Pauses"),
                half,
                Color32::from_rgb(0x33, 0x41, 0x55),
                Color32::from_rgb(0x47, 0x55, 0x69),
                has_seq,
            ) {
                let ctx = ui.ctx().clone();
                if let Err(e) = crate::menus::invoke(app, &ctx, "ai.removePauses", json!({ "async": true })) {
                    app.ui.status = e;
                }
            }
            if draw_action_cta(ui, &mut app.auto, t, "creator.card.enhanceAudio", "Enhance Voice", half, EMERALD, EMERALD_HOVER, has_seq) {
                let ctx = ui.ctx().clone();
                let _ = crate::menus::invoke(app, &ctx, "ai.enhanceAudio", json!({ "strength": "balanced", "voiceClarity": true, "autoLevel": true }));
            }
        });
    });
}

fn draw_quick_export_card(app: &mut FilmcraftApp, ui: &mut egui::Ui, t: &Tokens) {
    let has_seq = app.session.active_sequence().is_some();
    creator_card_frame(t, app.ui.dark).show(ui, |ui| {
        ui.set_width((ui.available_width() - 4.0).max(60.0));
        ui.spacing_mut().item_spacing.y = 6.0;
        draw_card_header(ui, t, Icon::Export, EMERALD, "Quick TikTok / Shorts Export", "1-click 1080×1920 60fps H.264 MP4 ready to upload.");

        if draw_action_cta(
            ui,
            &mut app.auto,
            t,
            "creator.card.quickExport",
            "Export 1080×1920 60fps MP4",
            ui.available_width(),
            EMERALD,
            EMERALD_HOVER,
            has_seq,
        ) {
            let seq_name =
                app.session.state.active_sequence.and_then(|id| app.session.project.item(id)).map(|it| it.name.clone()).unwrap_or_else(|| "Shorts".into());
            let dir = filmcraft_engine::export_tools::default_export_dir(&app.session);
            let file = format!("{}_9x16.mp4", filmcraft_engine::export_tools::file_safe(&seq_name));
            let path = dir.join(file).to_string_lossy().into_owned();
            match app.session.execute("export.quick", json!({ "preset": TIKTOK_EXPORT_PRESET, "path": path })) {
                Ok(v) => {
                    let out_path = v["path"].as_str().unwrap_or(&path);
                    app.ui.status = tlf!("Quick Export: {path}", path = out_path);
                    app.ui.ai_tools.push_toast("Quick TikTok / Shorts Export", app.ui.status.clone(), false);
                }
                Err(e) => {
                    app.ui.status = e.to_string();
                    app.ui.ai_tools.push_toast("Quick TikTok / Shorts Export", app.ui.status.clone(), true);
                }
            }
        }
    });
}

fn draw_preview_pane(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect, t: &Tokens) {
    if ui.input(|i| i.pointer.any_pressed() && i.pointer.interact_pos().is_some_and(|p| rect.contains(p))) {
        app.ui.focused = PanelKind::Program;
    }
    frame_panel(ui, rect, t, app.ui.focused == PanelKind::Program);

    let hdr = Rect::from_min_max(rect.min, pos2(rect.max.x, (rect.min.y + HEADER_H).min(rect.max.y)));
    ui.painter().line_segment([hdr.left_bottom(), hdr.right_bottom()], Stroke::new(1.0, t.control_border));
    ui.painter().text(pos2(hdr.min.x + 12.0, hdr.center().y), Align2::LEFT_CENTER, tl!("Preview"), Tokens::semibold(12.5), t.text);

    // Resolution / FPS pill on header right
    let mut rx = hdr.max.x - 10.0;
    if let Some(seq) = app.session.active_sequence() {
        let meta = format!("{}×{} · {:.0} fps", seq.settings.width, seq.settings.height, seq.settings.frame_rate.as_f64());
        let mg = ui.painter().layout_no_wrap(meta, Tokens::ui(11.0), t.text_dim);
        let pill_w = mg.size().x + 14.0;
        let pill_r = Rect::from_min_size(pos2(rx - pill_w, hdr.center().y - 10.0), vec2(pill_w, 20.0));
        ui.painter().rect_filled(pill_r, 5.0, t.field_bg);
        ui.painter().galley(pos2(pill_r.min.x + 7.0, pill_r.center().y - mg.size().y / 2.0), mg, t.text_dim);
        rx = pill_r.min.x - 8.0;
    }

    if app.session.state.active_sequence.is_none() {
        let seq_label = "+ 9:16";
        let seq_g = ui.painter().layout_no_wrap(seq_label.to_string(), Tokens::semibold(11.0), Color32::WHITE);
        let seq_w = seq_g.size().x + 16.0;
        let seq_r = Rect::from_min_size(pos2(rx - seq_w, hdr.center().y - 11.0), vec2(seq_w, 22.0));
        let seq_resp = ui.interact(seq_r, egui::Id::unique("creator-new-vertical"), Sense::click());
        app.auto.add("creator.newVertical", seq_r, seq_label);
        ui.painter().rect_filled(seq_r, 5.0, if seq_resp.hovered() { ROYAL_BLUE_HOVER } else { ROYAL_BLUE });
        ui.painter().galley(pos2(seq_r.min.x + 8.0, seq_r.center().y - seq_g.size().y / 2.0), seq_g, Color32::WHITE);
        if seq_resp.clicked()
            && let Err(e) = app.session.execute("file.newSequence", json!({ "name": "Shorts", "width": 1080, "height": 1920, "fps": 60.0 }))
        {
            app.ui.status = e.to_string();
        }
    }

    let aspect_bar = Rect::from_min_max(pos2(rect.min.x + 2.0, (rect.max.y - ASPECT_BAR_H).max(hdr.max.y + 12.0)), pos2(rect.max.x - 2.0, rect.max.y - 2.0));
    let body = Rect::from_min_max(pos2(rect.min.x + 2.0, hdr.max.y + 1.0), pos2(rect.max.x - 2.0, (aspect_bar.min.y - 1.0).max(hdr.max.y + 1.0)));
    if body.height() > 16.0 {
        let mut child = ui.new_child(egui::UiBuilder::new().max_rect(body).id_salt("creator-program-body"));
        child.set_clip_rect(body);
        super::show(app, &mut child, PanelKind::Program, body);
    }

    draw_aspect_toggle_bar(app, ui, aspect_bar, t);
}

fn draw_aspect_toggle_bar(app: &mut FilmcraftApp, ui: &mut egui::Ui, bar: Rect, t: &Tokens) {
    if bar.height() < 20.0 {
        return;
    }
    ui.painter().line_segment([bar.left_top(), bar.right_top()], Stroke::new(1.0, t.control_border));
    let is_vertical = app.session.active_sequence().is_some_and(|s| s.settings.height > s.settings.width);
    let has_seq = app.session.active_sequence().is_some();

    let v_label = "9:16 Vertical";
    let w_label = "16:9 Widescreen";
    let vg = ui.painter().layout_no_wrap(v_label.to_string(), Tokens::semibold(11.5), Color32::WHITE);
    let wg = ui.painter().layout_no_wrap(w_label.to_string(), Tokens::semibold(11.5), Color32::WHITE);
    let vw = vg.size().x + 22.0;
    let ww = wg.size().x + 22.0;
    let pill_w = vw + ww + 4.0;
    let pill = Rect::from_center_size(bar.center(), vec2(pill_w, 26.0));
    ui.painter().rect_filled(pill, 7.0, t.field_bg);
    ui.painter().rect_stroke(pill, 7.0, Stroke::new(1.0, t.control_border), StrokeKind::Inside);

    let v_rect = Rect::from_min_size(pos2(pill.min.x + 2.0, pill.min.y + 2.0), vec2(vw, 22.0));
    let w_rect = Rect::from_min_size(pos2(v_rect.max.x, pill.min.y + 2.0), vec2(ww, 22.0));

    for (target_vert, r, g, id, lbl, dims) in [
        (true, v_rect, vg, "creator.aspect.vertical", v_label, (1080u32, 1920u32)),
        (false, w_rect, wg, "creator.aspect.widescreen", w_label, (1920u32, 1080u32)),
    ] {
        let active = has_seq && is_vertical == target_vert;
        let resp = ui.interact(r, egui::Id::unique(("creator-aspect", id)), Sense::click());
        app.auto.add(id, r, lbl);
        if active {
            ui.painter().rect_filled(r, 5.0, ROYAL_BLUE);
        } else if resp.hovered() {
            ui.painter().rect_filled(r, 5.0, t.hover);
        }
        let col = if active { Color32::WHITE } else { t.text_dim };
        ui.painter().galley(pos2(r.center().x - g.size().x / 2.0, r.center().y - g.size().y / 2.0), g, col);
        if resp.clicked() {
            let res = if has_seq {
                app.session.execute("sequence.settings", json!({ "width": dims.0, "height": dims.1, "scaleMotion": true }))
            } else {
                let name = if target_vert { "Shorts" } else { "Widescreen" };
                app.session.execute("file.newSequence", json!({ "name": name, "width": dims.0, "height": dims.1, "fps": 60.0 }))
            };
            match res {
                Ok(_) => app.ui.status = format!("Canvas set to {}×{}", dims.0, dims.1),
                Err(e) => app.ui.status = e.to_string(),
            }
        }
    }
}

fn target_clips_for_speed(app: &FilmcraftApp) -> Vec<u64> {
    if !app.session.state.selection.is_empty() {
        return app.session.state.selection.iter().map(|c| c.0).collect();
    }
    let Some(seq) = app.session.active_sequence() else {
        return Vec::new();
    };
    seq.video_tracks
        .iter()
        .filter(|tr| !tr.locked)
        .flat_map(|tr| tr.items.iter().map(|it| it.id.0))
        .chain(seq.audio_tracks.iter().filter(|tr| !tr.locked).flat_map(|tr| tr.items.iter().map(|it| it.id.0)))
        .collect()
}

fn draw_bottom_timeline(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect, t: &Tokens) {
    if rect.width() < 16.0 || rect.height() < 56.0 {
        return;
    }
    if ui.input(|i| i.pointer.any_pressed() && i.pointer.interact_pos().is_some_and(|p| rect.contains(p))) {
        app.ui.focused = PanelKind::Timeline;
    }
    frame_panel(ui, rect, t, app.ui.focused == PanelKind::Timeline);

    let bar = Rect::from_min_max(pos2(rect.min.x + 2.0, rect.min.y + 2.0), pos2(rect.max.x - 2.0, (rect.min.y + TIMELINE_BAR_H).min(rect.max.y)));
    ui.painter().line_segment([bar.left_bottom(), bar.right_bottom()], Stroke::new(1.0, t.control_border));

    // Right-aligned quick zoom controls (- | Fit | +)
    let mut rx = bar.max.x - 8.0;
    for (glyph, cmd, tip) in
        [("+", "view.zoomIn", tl!("Zoom In")), ("Fit", "view.zoomToSequence", tl!("Zoom to Sequence")), ("-", "view.zoomOut", tl!("Zoom Out"))]
    {
        let g = ui.painter().layout_no_wrap(glyph.to_string(), Tokens::semibold(11.0), t.text);
        let bw = (g.size().x + 14.0).max(26.0);
        let br = Rect::from_min_size(pos2(rx - bw, bar.center().y - 12.0), vec2(bw, 24.0));
        let resp = ui.interact(br, egui::Id::unique(("creator-tl-zoom", cmd)), Sense::click()).on_hover_text(tip);
        app.auto.add(&format!("creator.zoom.{cmd}"), br, tip);
        ui.painter().rect_filled(br, 5.0, if resp.hovered() { t.hover } else { t.control_surface });
        ui.painter().rect_stroke(br, 5.0, Stroke::new(1.0, t.control_border), StrokeKind::Inside);
        ui.painter().galley(pos2(br.center().x - g.size().x / 2.0, br.center().y - g.size().y / 2.0), g, t.text);
        if resp.clicked() {
            let ctx = ui.ctx().clone();
            let _ = crate::menus::invoke(app, &ctx, cmd, json!({}));
        }
        rx = br.min.x - 5.0;
    }

    let delete_cmd =
        if filmcraft_engine::find_command("edit.rippleDelete").is_some_and(|c| (c.enabled)(&app.session).is_ok()) { "edit.rippleDelete" } else { "edit.clear" };
    let mut x = bar.min.x + 10.0;

    // Prominent Split & Delete buttons
    for (id, cmd, label, icon) in [("split", "sequence.addEdit", tl!("Split"), Some(Icon::Razor)), ("delete", delete_cmd, tl!("Delete"), Some(Icon::Trash))] {
        x = draw_tl_pill(app, ui, t, bar.center().y, x, rx, id, cmd, label, icon);
    }

    // Prominent Speed quick-menu button (0.5x / 0.75x / 1x / 1.25x / 1.5x / 2x)
    x = draw_speed_pill(app, ui, t, bar.center().y, x, rx);

    // Undo / Redo / Remove Pauses
    for (id, cmd, label) in [("undo", "edit.undo", tl!("Undo")), ("redo", "edit.redo", tl!("Redo")), ("removePauses", "ai.removePauses", tl!("Remove Pauses"))]
    {
        x = draw_tl_pill(app, ui, t, bar.center().y, x, rx, id, cmd, label, None);
    }

    // "AI Tools" toggle button opening full AI drawer
    let ai_open = app.ui.creator_tool == Some(CreatorTool::Ai);
    let ai_label = tl!("AI Tools");
    let ai_fg = if ai_open { Color32::WHITE } else { t.text };
    let ai_g = ui.painter().layout_no_wrap(ai_label.to_string(), Tokens::semibold(11.5), ai_fg);
    let ai_btn = Rect::from_min_size(pos2(x, bar.center().y - 12.0), vec2(ai_g.size().x + 20.0, 24.0));
    if ai_btn.max.x <= rx {
        let ai_resp = ui.interact(ai_btn, egui::Id::unique("creator-tl-ai-tools"), Sense::click());
        app.auto.add("creator.action.aiTools", ai_btn, ai_label);
        let ai_bg = if ai_open {
            ROYAL_BLUE
        } else if ai_resp.hovered() {
            t.hover
        } else {
            t.control_surface
        };
        ui.painter().rect_filled(ai_btn, 6.0, ai_bg);
        ui.painter().rect_stroke(ai_btn, 6.0, Stroke::new(1.0, if ai_open { ROYAL_BLUE } else { t.control_border }), StrokeKind::Inside);
        ui.painter().galley(pos2(ai_btn.min.x + 10.0, ai_btn.center().y - ai_g.size().y / 2.0), ai_g, ai_fg);
        if ai_resp.clicked() {
            app.ui.creator_tool = if ai_open { Some(CreatorTool::Actions) } else { Some(CreatorTool::Ai) };
        }
        x = ai_btn.max.x + 6.0;
    }

    // Quick "AI Subtitles" button
    let sub_enabled = app.session.active_sequence().is_some();
    let sub_label = tl!("AI Subtitles");
    let sub_fg = if sub_enabled { t.text } else { t.text_dim };
    let sub_g = ui.painter().layout_no_wrap(sub_label.to_string(), Tokens::semibold(11.5), sub_fg);
    let sub_btn = Rect::from_min_size(pos2(x, bar.center().y - 12.0), vec2(sub_g.size().x + 20.0, 24.0));
    if sub_btn.max.x <= rx {
        let sub_resp = ui.interact(sub_btn, egui::Id::unique("creator-tl-ai-subtitles"), if sub_enabled { Sense::click() } else { Sense::hover() });
        app.auto.add("creator.action.subtitles", sub_btn, sub_label);
        if let Some(el) = app.auto.elements.last_mut() {
            el.enabled = Some(sub_enabled);
        }
        ui.painter().rect_filled(sub_btn, 6.0, if sub_enabled && sub_resp.hovered() { t.hover } else { t.control_surface });
        ui.painter().rect_stroke(sub_btn, 6.0, Stroke::new(1.0, t.control_border), StrokeKind::Inside);
        ui.painter().galley(pos2(sub_btn.min.x + 10.0, sub_btn.center().y - sub_g.size().y / 2.0), sub_g, sub_fg);
        if sub_enabled && sub_resp.clicked() {
            let ctx = ui.ctx().clone();
            if let Err(e) = crate::menus::invoke(app, &ctx, "ai.generateSubtitles", json!({})) {
                app.ui.status = e;
                app.ui.creator_tool = Some(CreatorTool::Actions);
            }
        }
    }

    let tl_body = Rect::from_min_max(pos2(rect.min.x + 2.0, bar.max.y + 1.0), pos2(rect.max.x - 2.0, rect.max.y - 2.0));
    if tl_body.height() > 16.0 {
        let mut child = ui.new_child(egui::UiBuilder::new().max_rect(tl_body).id_salt(("creator-panel", PanelKind::Timeline)));
        child.set_clip_rect(tl_body);
        super::show(app, &mut child, PanelKind::Timeline, tl_body);
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_tl_pill(
    app: &mut FilmcraftApp,
    ui: &mut egui::Ui,
    t: &Tokens,
    cy: f32,
    x: f32,
    max_x: f32,
    id: &str,
    cmd: &str,
    label: &str,
    icon: Option<Icon>,
) -> f32 {
    let enabled = filmcraft_engine::find_command(cmd).is_some_and(|c| (c.enabled)(&app.session).is_ok());
    let fg = if enabled { t.text } else { t.text_dim };
    let g = ui.painter().layout_no_wrap(label.to_string(), Tokens::semibold(11.5), fg);
    let icon_pad = if icon.is_some() { 16.0 } else { 0.0 };
    let btn = Rect::from_min_size(pos2(x, cy - 12.0), vec2(g.size().x + 18.0 + icon_pad, 24.0));
    if btn.max.x > max_x {
        return x;
    }
    let resp = ui.interact(btn, egui::Id::unique(("creator-tl-act", id)), if enabled { Sense::click() } else { Sense::hover() });
    app.auto.add(&format!("creator.action.{id}"), btn, label);
    if let Some(el) = app.auto.elements.last_mut() {
        el.enabled = Some(enabled);
    }
    ui.painter().rect_filled(btn, 6.0, if enabled && resp.hovered() { t.hover } else { t.control_surface });
    ui.painter().rect_stroke(btn, 6.0, Stroke::new(1.0, t.control_border), StrokeKind::Inside);
    if let Some(ic) = icon {
        icons::paint(ui.painter(), Rect::from_center_size(pos2(btn.min.x + 13.0, btn.center().y), vec2(12.0, 12.0)), ic, fg);
    }
    ui.painter().galley(pos2(btn.min.x + 9.0 + icon_pad, btn.center().y - g.size().y / 2.0), g, fg);
    if enabled && resp.clicked() {
        let ctx = ui.ctx().clone();
        if let Err(e) = crate::menus::invoke(app, &ctx, cmd, json!({})) {
            app.ui.status = e;
        }
    }
    btn.max.x + 6.0
}

fn draw_speed_pill(app: &mut FilmcraftApp, ui: &mut egui::Ui, t: &Tokens, cy: f32, x: f32, max_x: f32) -> f32 {
    let clips = target_clips_for_speed(app);
    let enabled = !clips.is_empty();
    let label = tl!("Speed");
    let fg = if enabled { t.text } else { t.text_dim };
    let g = ui.painter().layout_no_wrap(label.to_string(), Tokens::semibold(11.5), fg);
    let btn = Rect::from_min_size(pos2(x, cy - 12.0), vec2(g.size().x + 30.0, 24.0));
    if btn.max.x > max_x {
        return x;
    }
    let resp = ui.interact(btn, egui::Id::unique("creator-tl-speed"), if enabled { Sense::click() } else { Sense::hover() });
    app.auto.add("creator.action.speed", btn, label);
    if let Some(el) = app.auto.elements.last_mut() {
        el.enabled = Some(enabled);
    }
    ui.painter().rect_filled(btn, 6.0, if enabled && resp.hovered() { t.hover } else { t.control_surface });
    ui.painter().rect_stroke(btn, 6.0, Stroke::new(1.0, t.control_border), StrokeKind::Inside);
    ui.painter().galley(pos2(btn.min.x + 9.0, btn.center().y - g.size().y / 2.0), g, fg);
    icons::paint(ui.painter(), Rect::from_center_size(pos2(btn.max.x - 11.0, btn.center().y), vec2(10.0, 10.0)), Icon::ChevronDown, fg);

    if enabled {
        egui::Popup::menu(&resp).close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside).show(|ui| {
            for (lbl, pct) in
                [("0.5× (Slow)", 50.0), ("0.75×", 75.0), ("1.0× (Normal)", 100.0), ("1.25×", 125.0), ("1.5× (Fast)", 150.0), ("2.0× (Double)", 200.0)]
            {
                if ui.button(lbl).clicked() {
                    match app.session.execute("clip.speedDuration", json!({ "clips": clips, "speed": pct })) {
                        Ok(_) => app.ui.status = format!("Clip speed set to {pct:.0}%"),
                        Err(e) => app.ui.status = e.to_string(),
                    }
                    ui.close();
                }
            }
        });
    }
    btn.max.x + 6.0
}
