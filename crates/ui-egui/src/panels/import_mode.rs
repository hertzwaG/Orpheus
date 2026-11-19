//! Import mode (header "Import", also Home): four source cards, Quick Drag & Drop zone,
//! Imported Media pool with search/filter/quick-add, and compact recent projects.

use std::collections::HashSet;

use egui::{Align2, Color32, CornerRadius, Rect, Sense, Stroke, StrokeKind, pos2, vec2};
use filmcraft_engine::project_panel as pp;
use filmcraft_media::MediaKind;
use filmcraft_project::{ItemId, ItemKind, ProjectItem};
use filmcraft_time::Tick;
use serde_json::json;

use crate::FilmcraftApp;
use crate::icons::{self, Icon};
use crate::state::{ImportMediaFilter, Mode};
use crate::theme::Tokens;

const OUTER_PAD: f32 = 24.0;
const CARD_GAP: f32 = 14.0;
const COL_GAP: f32 = 16.0;
const TOP_CARD_H: f32 = 184.0;
const RECENT_ROWS: usize = 6;
const RECENT_ROW_H: f32 = 24.0;

const ROYAL_BLUE: Color32 = Color32::from_rgb(0x2F, 0x6F, 0xED);
const ROYAL_BLUE_HOVER: Color32 = Color32::from_rgb(0x3B, 0x7A, 0xF5);
const FORMAT_PILLS: [&str; 6] = ["MP4", "MOV", "WAV", "MP3", "MKV", "PNG"];

#[derive(Clone, Copy)]
enum SourceKind {
    Browse,
    NewProject,
    DemoProject,
    BarsAndTone,
}

pub fn show(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect) {
    let t = app.tokens;
    let dark = app.ui.dark;
    ui.painter().rect_filled(rect, 0.0, t.panel_bg);
    if rect.width() < 120.0 || rect.height() < 120.0 {
        return;
    }

    let inner = rect.shrink2(vec2(OUTER_PAD, 18.0));
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(inner).layout(egui::Layout::top_down(egui::Align::Min)));
    child.set_clip_rect(rect);

    draw_header(&mut child, &t);
    child.add_space(14.0);

    draw_source_cards(app, &mut child, &t, dark);
    child.add_space(16.0);

    let rem_h = child.available_height().max(260.0);
    draw_bottom_columns(app, &mut child, &t, dark, rem_h);
}

fn draw_header(ui: &mut egui::Ui, t: &Tokens) {
    ui.label(egui::RichText::new(tl!("Import")).font(Tokens::semibold(22.0)).color(t.text));
    ui.add_space(2.0);
    ui.label(egui::RichText::new(tl!("Bring footage, audio, images, or starter projects into your workspace.")).font(Tokens::ui(12.5)).color(t.text_dim));
}

fn draw_source_cards(app: &mut FilmcraftApp, ui: &mut egui::Ui, t: &Tokens, dark: bool) {
    let cards = [
        (SourceKind::Browse, "file.import", tl!("Browse files…"), tl!("Movies, audio and stills from disk")),
        (SourceKind::NewProject, "file.newProject", tl!("New Project"), tl!("Start with a blank timeline and empty media bin")),
        (SourceKind::DemoProject, "file.openDemoProject", tl!("Demo project"), tl!("A cut sequence with transitions, effects and music")),
        (SourceKind::BarsAndTone, "file.newBarsAndTone", tl!("Bars and Tone"), tl!("SMPTE HD bars with 1 kHz reference tone")),
    ];

    let total_w = ui.available_width();
    let cols = if total_w >= 720.0 { 4 } else { 2 };
    let card_w = ((total_w - (cols - 1) as f32 * CARD_GAP) / cols as f32).max(140.0);
    let rows = cards.len().div_ceil(cols);
    let ctx = ui.ctx().clone();

    for row in 0..rows {
        if row > 0 {
            ui.add_space(CARD_GAP);
        }
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = CARD_GAP;
            for col in 0..cols {
                let idx = row * cols + col;
                if let Some((kind, cmd, title, desc)) = cards.get(idx)
                    && draw_source_card(app, ui, t, dark, vec2(card_w, TOP_CARD_H), *kind, cmd, title, desc)
                {
                    invoke_source_action(app, &ctx, *kind, cmd);
                }
            }
        });
    }
}

fn invoke_source_action(app: &mut FilmcraftApp, ctx: &egui::Context, kind: SourceKind, cmd: &str) {
    let params = match kind {
        SourceKind::NewProject => json!({"name": "Untitled"}),
        _ => json!({}),
    };
    let ok = crate::menus::invoke(app, ctx, cmd, params).is_ok();
    match kind {
        SourceKind::Browse => {}
        SourceKind::DemoProject => {
            if ok && app.ui.clip_dialog.is_none() {
                app.ui.mode = Mode::Edit;
            }
        }
        SourceKind::NewProject | SourceKind::BarsAndTone => {
            if ok {
                app.ui.mode = Mode::Edit;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_source_card(
    app: &mut FilmcraftApp,
    ui: &mut egui::Ui,
    t: &Tokens,
    dark: bool,
    size: egui::Vec2,
    kind: SourceKind,
    cmd: &str,
    title: &str,
    desc: &str,
) -> bool {
    let (r, resp) = ui.allocate_exact_size(size, Sense::click());
    let resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand);
    app.auto.add(&format!("import.{cmd}"), r, title);

    let hovered = resp.hovered();
    let is_primary = matches!(kind, SourceKind::Browse);
    let bg = match (is_primary, dark, hovered) {
        (true, true, true) => Color32::from_rgb(0x18, 0x24, 0x3D),
        (true, true, false) => Color32::from_rgb(0x13, 0x1B, 0x2F),
        (true, false, true) => t.hover,
        (true, false, false) => t.control_surface,
        (false, true, true) => Color32::from_rgb(0x1C, 0x20, 0x2B),
        (false, true, false) => Color32::from_rgb(0x16, 0x19, 0x22),
        (false, false, true) => t.hover,
        (false, false, false) => t.control_surface,
    };
    let border = if is_primary {
        if hovered { Color32::from_rgb(0x5B, 0x95, 0xFF) } else { ROYAL_BLUE }
    } else if hovered {
        Color32::from_rgba_unmultiplied(0x5B, 0x95, 0xFF, 140)
    } else if dark {
        Color32::from_rgba_unmultiplied(255, 255, 255, 22)
    } else {
        t.control_border
    };

    ui.painter().rect_filled(r, CornerRadius::same(10), bg);
    ui.painter().rect_stroke(r, CornerRadius::same(10), Stroke::new(if is_primary { 1.3 } else { 1.0 }, border), StrokeKind::Inside);

    let pad = 12.0;
    let banner_h = if is_primary { 54.0 } else { 70.0 };
    let banner_r = Rect::from_min_size(r.min + vec2(pad, pad), vec2((r.width() - 2.0 * pad).max(40.0), banner_h));
    match kind {
        SourceKind::Browse => paint_browse_banner(ui.painter(), banner_r, dark),
        SourceKind::NewProject => paint_new_project_banner(ui.painter(), banner_r),
        SourceKind::DemoProject => paint_demo_project_banner(ui.painter(), banner_r),
        SourceKind::BarsAndTone => paint_smpte_bars_banner(ui.painter(), banner_r),
    }

    let text_x = r.min.x + pad;
    let text_w = (r.width() - 2.0 * pad).max(40.0);
    let title_y = banner_r.max.y + 11.0;
    let title_g = elided_line(ui, title, Tokens::semibold(13.5), t.text, text_w);
    ui.painter().galley(pos2(text_x, title_y), title_g, t.text);

    let desc_y = title_y + 20.0;
    let max_desc_h = if is_primary { (r.max.y - 46.0 - desc_y).max(16.0) } else { (r.max.y - pad - desc_y).max(20.0) };
    let desc_r = Rect::from_min_size(pos2(text_x, desc_y), vec2(text_w, max_desc_h));
    let mut job = egui::text::LayoutJob::single_section(desc.to_string(), egui::TextFormat::simple(Tokens::ui(11.0), t.text_dim));
    job.wrap.max_width = text_w;
    job.wrap.max_rows = if is_primary { 1 } else { 2 };
    job.wrap.overflow_character = Some('…');
    let desc_g = ui.fonts_mut(|f| f.layout_job(job));
    ui.painter().with_clip_rect(desc_r).galley(desc_r.min, desc_g, t.text_dim);

    if is_primary {
        let btn_r = Rect::from_min_size(pos2(text_x, r.max.y - pad - 28.0), vec2(text_w, 28.0));
        ui.painter().rect_filled(btn_r, CornerRadius::same(6), if hovered { ROYAL_BLUE_HOVER } else { ROYAL_BLUE });
        let btn_g = elided_line(ui, title, Tokens::semibold(12.0), Color32::WHITE, (btn_r.width() - 16.0).max(20.0));
        ui.painter().galley(pos2(btn_r.center().x - btn_g.size().x / 2.0, btn_r.center().y - btn_g.size().y / 2.0), btn_g, Color32::WHITE);
    }

    resp.clicked()
}

fn paint_browse_banner(painter: &egui::Painter, r: Rect, dark: bool) {
    let center = r.center();
    painter.circle_filled(center, 25.0, Color32::from_rgba_unmultiplied(0x2F, 0x6F, 0xED, if dark { 28 } else { 36 }));
    painter.circle_filled(center, 18.0, Color32::from_rgba_unmultiplied(0x2F, 0x6F, 0xED, if dark { 52 } else { 64 }));
    let icon_r = Rect::from_center_size(pos2(center.x - 4.0, center.y + 1.0), vec2(20.0, 20.0));
    icons::paint(painter, icon_r, Icon::Folder, Color32::from_rgb(0x8C, 0xB4, 0xFF));
    // Crisp upward upload arrow accent beside the folder
    let ax = center.x + 11.0;
    let ay = center.y;
    let col = Color32::WHITE;
    painter.line_segment([pos2(ax, ay + 6.0), pos2(ax, ay - 6.0)], Stroke::new(1.8, col));
    painter.line_segment([pos2(ax - 4.0, ay - 2.0), pos2(ax, ay - 6.0)], Stroke::new(1.8, col));
    painter.line_segment([pos2(ax + 4.0, ay - 2.0), pos2(ax, ay - 6.0)], Stroke::new(1.8, col));
}

fn paint_new_project_banner(painter: &egui::Painter, r: Rect) {
    painter.rect_filled(r, CornerRadius::same(6), Color32::from_rgb(0x0B, 0x0E, 0x15));
    let inner = r.shrink(8.0);
    let lane_h = ((inner.height() - 8.0) / 3.0).max(6.0);
    for row in 0..3 {
        let y = inner.min.y + row as f32 * (lane_h + 4.0);
        let lane = Rect::from_min_size(pos2(inner.min.x, y), vec2(inner.width(), lane_h));
        painter.rect_filled(lane, CornerRadius::same(2), Color32::from_rgba_unmultiplied(0x2F, 0x6F, 0xED, 22));
        painter.rect_stroke(lane, CornerRadius::same(2), Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 18)), egui::StrokeKind::Inside);
    }
    let center = r.center();
    painter.circle_filled(center, 14.0, Color32::from_rgba_unmultiplied(0x2F, 0x6F, 0xED, 44));
    painter.circle_filled(center, 11.0, Color32::from_rgb(0x14, 0x1A, 0x29));
    painter.circle_stroke(center, 11.0, Stroke::new(1.0, Color32::from_rgb(0x60, 0xA5, 0xFA)));
    let arm = 4.5;
    let stroke = Stroke::new(1.8, Color32::WHITE);
    painter.line_segment([pos2(center.x - arm, center.y), pos2(center.x + arm, center.y)], stroke);
    painter.line_segment([pos2(center.x, center.y - arm), pos2(center.x, center.y + arm)], stroke);
}

fn paint_demo_project_banner(painter: &egui::Painter, r: Rect) {
    painter.rect_filled(r, CornerRadius::same(6), Color32::from_rgb(0x0B, 0x0E, 0x15));
    let inner = r.shrink(7.0);
    let track_h = ((inner.height() - 8.0) / 3.0).max(8.0);
    // V2 title/fx clip
    let v2 = Rect::from_min_size(pos2(inner.min.x + inner.width() * 0.14, inner.min.y), vec2(inner.width() * 0.42, track_h));
    painter.rect_filled(v2, CornerRadius::same(3), Color32::from_rgb(0x7C, 0x5C, 0xFF));
    // V1 cut sequence clips
    let v1_y = inner.min.y + track_h + 4.0;
    let c1 = Rect::from_min_size(pos2(inner.min.x, v1_y), vec2(inner.width() * 0.45, track_h));
    let c2 = Rect::from_min_size(pos2(c1.max.x + 2.0, v1_y), vec2(inner.width() * 0.52 - 2.0, track_h));
    painter.rect_filled(c1, CornerRadius::same(3), Color32::from_rgb(0x2F, 0x6F, 0xED));
    painter.rect_filled(c2, CornerRadius::same(3), Color32::from_rgb(0x3B, 0x82, 0xF6));
    let cut_r = Rect::from_center_size(pos2(c1.max.x + 1.0, v1_y + track_h * 0.5), vec2(10.0, track_h - 2.0));
    painter.rect_filled(cut_r, CornerRadius::same(2), Color32::from_rgba_unmultiplied(255, 255, 255, 200));
    // A1 waveform strip
    let a1_y = v1_y + track_h + 4.0;
    let a1 = Rect::from_min_size(pos2(inner.min.x, a1_y), vec2(inner.width() * 0.90, track_h));
    painter.rect_filled(a1, CornerRadius::same(3), Color32::from_rgb(0x16, 0x65, 0x54));
    let mut x = a1.min.x + 4.0;
    let mut i = 0usize;
    while x < a1.max.x - 3.0 {
        let frac = [0.4, 0.85, 0.65, 0.95, 0.5, 0.75][i % 6];
        let bh = (a1.height() - 4.0) * frac;
        painter.line_segment([pos2(x, a1.center().y - bh * 0.5), pos2(x, a1.center().y + bh * 0.5)], Stroke::new(1.5, Color32::from_rgb(0x34, 0xD3, 0x99)));
        x += 5.0;
        i += 1;
    }
    // Playhead
    let px = inner.min.x + inner.width() * 0.62;
    painter.line_segment([pos2(px, r.min.y + 4.0), pos2(px, r.max.y - 4.0)], Stroke::new(1.6, Color32::from_rgb(0x60, 0xA5, 0xFA)));
    painter.rect_filled(Rect::from_center_size(pos2(px, r.min.y + 6.0), vec2(7.0, 5.0)), 1.0, Color32::from_rgb(0x60, 0xA5, 0xFA));
}

fn paint_smpte_bars_banner(painter: &egui::Painter, r: Rect) {
    let top_cols = [
        Color32::from_rgb(0xB8, 0xB8, 0xB8),
        Color32::from_rgb(0xC4, 0xC4, 0x1C),
        Color32::from_rgb(0x1C, 0xC4, 0xC4),
        Color32::from_rgb(0x1C, 0xC4, 0x1C),
        Color32::from_rgb(0xC4, 0x1C, 0xC4),
        Color32::from_rgb(0xC4, 0x1C, 0x1C),
        Color32::from_rgb(0x1C, 0x1C, 0xC4),
    ];
    let bot_cols = [
        Color32::from_rgb(0x1C, 0x1C, 0xC4),
        Color32::from_rgb(0x14, 0x14, 0x18),
        Color32::from_rgb(0xC4, 0x1C, 0xC4),
        Color32::from_rgb(0x14, 0x14, 0x18),
        Color32::from_rgb(0x1C, 0xC4, 0xC4),
        Color32::from_rgb(0x14, 0x14, 0x18),
        Color32::from_rgb(0xB8, 0xB8, 0xB8),
    ];
    let top_h = (r.height() * 0.72).round();
    let mid_h = (r.height() * 0.12).round();
    let p = painter.with_clip_rect(r);
    p.rect_filled(r, CornerRadius::same(6), Color32::from_rgb(0x0F, 0x11, 0x16));
    let bw = r.width() / top_cols.len() as f32;
    for (i, (&tc, &bc)) in top_cols.iter().zip(&bot_cols).enumerate() {
        let x0 = r.min.x + i as f32 * bw;
        let x1 = if i + 1 == top_cols.len() { r.max.x } else { x0 + bw };
        p.rect_filled(Rect::from_min_max(pos2(x0, r.min.y), pos2(x1, r.min.y + top_h)), 0.0, tc);
        p.rect_filled(Rect::from_min_max(pos2(x0, r.min.y + top_h), pos2(x1, r.min.y + top_h + mid_h)), 0.0, bc);
    }
    let bottom_r = Rect::from_min_max(pos2(r.min.x, r.min.y + top_h + mid_h), r.max);
    p.rect_filled(bottom_r, 0.0, Color32::from_rgb(0x10, 0x12, 0x18));
    let q = bottom_r.width() / 4.0;
    p.rect_filled(Rect::from_min_max(pos2(bottom_r.min.x, bottom_r.min.y), pos2(bottom_r.min.x + q, bottom_r.max.y)), 0.0, Color32::from_rgb(0x0B, 0x2A, 0x52));
    p.rect_filled(
        Rect::from_min_max(pos2(bottom_r.min.x + q, bottom_r.min.y), pos2(bottom_r.min.x + q * 2.0, bottom_r.max.y)),
        0.0,
        Color32::from_rgb(0xE8, 0xEC, 0xF2),
    );
    p.rect_filled(
        Rect::from_min_max(pos2(bottom_r.min.x + q * 2.0, bottom_r.min.y), pos2(bottom_r.min.x + q * 3.0, bottom_r.max.y)),
        0.0,
        Color32::from_rgb(0x36, 0x14, 0x54),
    );
    let badge = Rect::from_min_size(pos2(r.min.x + 6.0, r.min.y + 6.0), vec2(40.0, 15.0));
    p.rect_filled(badge, CornerRadius::same(4), Color32::from_rgba_unmultiplied(12, 14, 20, 210));
    p.text(badge.center(), Align2::CENTER_CENTER, "1 kHz", Tokens::mono(9.5), Color32::WHITE);
    painter.rect_stroke(r, CornerRadius::same(6), Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 28)), StrokeKind::Inside);
}

fn draw_bottom_columns(app: &mut FilmcraftApp, ui: &mut egui::Ui, t: &Tokens, dark: bool, height: f32) {
    let total_w = ui.available_width();
    if total_w < 640.0 {
        draw_left_column(app, ui, t, dark, total_w, (height * 0.45).max(200.0));
        ui.add_space(COL_GAP);
        draw_media_pool(app, ui, t, dark, total_w, (height * 0.52).max(240.0));
        return;
    }

    let left_w = ((total_w - COL_GAP) * 0.60).round();
    let right_w = (total_w - COL_GAP - left_w).max(200.0);
    let row_rect = Rect::from_min_size(ui.cursor().min, vec2(total_w, height));
    let _ = ui.allocate_exact_size(vec2(total_w, height), Sense::hover());

    let left_rect = Rect::from_min_size(row_rect.min, vec2(left_w, height));
    let mut left_ui = ui.new_child(egui::UiBuilder::new().max_rect(left_rect).layout(egui::Layout::top_down(egui::Align::Min)));
    left_ui.set_clip_rect(left_rect);
    draw_left_column(app, &mut left_ui, t, dark, left_w, height);

    let right_rect = Rect::from_min_size(pos2(left_rect.max.x + COL_GAP, row_rect.min.y), vec2(right_w, height));
    let mut right_ui = ui.new_child(egui::UiBuilder::new().max_rect(right_rect).layout(egui::Layout::top_down(egui::Align::Min)));
    right_ui.set_clip_rect(right_rect);
    draw_media_pool(app, &mut right_ui, t, dark, right_w, height);
}

fn draw_left_column(app: &mut FilmcraftApp, ui: &mut egui::Ui, t: &Tokens, dark: bool, width: f32, height: f32) {
    let recent_paths: Vec<String> = app.session.prefs.general.recent_projects.iter().take(RECENT_ROWS).cloned().collect();
    let recent_h = if recent_paths.is_empty() { 0.0 } else { 32.0 + recent_paths.len() as f32 * RECENT_ROW_H + 14.0 };
    let drop_h = if recent_paths.is_empty() { height } else { (height - recent_h - 12.0).max(160.0) };

    draw_drop_zone(app, ui, t, dark, vec2(width, drop_h));
    if !recent_paths.is_empty() {
        ui.add_space(12.0);
        draw_recent_box(app, ui, t, dark, width, &recent_paths);
    }
}

fn draw_drop_zone(app: &mut FilmcraftApp, ui: &mut egui::Ui, t: &Tokens, dark: bool, size: egui::Vec2) {
    let (r, resp) = ui.allocate_exact_size(size, Sense::click());
    let resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand);
    app.auto.add("import.dropZone", r, tl!("Drag and drop files here for quick import"));

    let os_drag_hover = ui.ctx().input(|i| !i.raw.hovered_files.is_empty());
    let active = resp.hovered() || os_drag_hover;
    let fill = match (dark, active) {
        (true, true) => Color32::from_rgb(0x15, 0x1E, 0x33),
        (true, false) => Color32::from_rgb(0x12, 0x15, 0x1E),
        (false, true) => t.hover,
        (false, false) => t.field_bg,
    };
    let border_col = if active {
        ROYAL_BLUE
    } else if dark {
        Color32::from_rgba_unmultiplied(255, 255, 255, 42)
    } else {
        t.control_border
    };

    ui.painter().rect_filled(r, CornerRadius::same(10), fill);
    paint_dashed_rect(ui.painter(), r.shrink(4.0), 8.0, Stroke::new(1.5, border_col));

    let cy = r.center().y;
    let cx = r.center().x;
    // Centered icon circle
    let badge_c = pos2(cx, cy - 42.0);
    ui.painter().circle_filled(badge_c, 22.0, Color32::from_rgba_unmultiplied(0x2F, 0x6F, 0xED, if dark { 32 } else { 40 }));
    ui.painter().circle_stroke(badge_c, 22.0, Stroke::new(1.0, Color32::from_rgba_unmultiplied(0x5B, 0x95, 0xFF, 100)));
    icons::paint(ui.painter(), Rect::from_center_size(pos2(cx, badge_c.y + 1.0), vec2(18.0, 18.0)), Icon::Folder, Color32::from_rgb(0x8C, 0xB4, 0xFF));

    let max_text_w = (r.width() - 48.0).max(80.0);
    let headline = elided_line(ui, tl!("Drag and drop files here for quick import"), Tokens::semibold(15.0), t.text, max_text_w);
    ui.painter().galley(pos2(cx - headline.size().x / 2.0, cy - 6.0), headline, t.text);

    let sub = elided_line(ui, tl!("or click anywhere inside to browse your computer"), Tokens::ui(12.0), t.text_dim, max_text_w);
    ui.painter().galley(pos2(cx - sub.size().x / 2.0, cy + 16.0), sub, t.text_dim);

    // Centered compact format pills (MP4, MOV, WAV, MP3, MKV, PNG)
    let pill_w = 42.0;
    let pill_gap = 6.0;
    let total_pills_w = FORMAT_PILLS.len() as f32 * pill_w + (FORMAT_PILLS.len() - 1) as f32 * pill_gap;
    if total_pills_w <= r.width() - 24.0 {
        let mut px = cx - total_pills_w / 2.0;
        let py = cy + 44.0;
        let pill_bg = if dark { Color32::from_rgb(0x1B, 0x20, 0x2D) } else { t.control_surface };
        for ext in FORMAT_PILLS {
            let pr = Rect::from_min_size(pos2(px, py), vec2(pill_w, 20.0));
            ui.painter().rect_filled(pr, CornerRadius::same(5), pill_bg);
            ui.painter().rect_stroke(pr, CornerRadius::same(5), Stroke::new(1.0, t.control_border), StrokeKind::Inside);
            ui.painter().text(pr.center(), Align2::CENTER_CENTER, ext, Tokens::mono(10.5), t.text_dim);
            px += pill_w + pill_gap;
        }
    }

    if resp.clicked() {
        let ctx = ui.ctx().clone();
        let _ = crate::menus::invoke(app, &ctx, "file.import", json!({}));
    }
}

fn paint_dashed_rect(painter: &egui::Painter, rect: Rect, radius: f32, stroke: Stroke) {
    let r = radius.min(rect.width() * 0.4).min(rect.height() * 0.4).max(0.0);
    let dash = 8.0_f32;
    let gap = 5.0_f32;
    let draw_seg = |a: egui::Pos2, b: egui::Pos2| {
        let d = b - a;
        let len = d.length();
        if len < 1.0 {
            return;
        }
        let dir = d / len;
        let mut t = 0.0_f32;
        while t < len {
            let end = (t + dash).min(len);
            painter.line_segment([a + dir * t, a + dir * end], stroke);
            t += dash + gap;
        }
    };
    draw_seg(pos2(rect.min.x + r, rect.min.y), pos2(rect.max.x - r, rect.min.y));
    draw_seg(pos2(rect.min.x + r, rect.max.y), pos2(rect.max.x - r, rect.max.y));
    draw_seg(pos2(rect.min.x, rect.min.y + r), pos2(rect.min.x, rect.max.y - r));
    draw_seg(pos2(rect.max.x, rect.min.y + r), pos2(rect.max.x, rect.max.y - r));
    for c in [
        pos2(rect.min.x + r * 0.35, rect.min.y + r * 0.35),
        pos2(rect.max.x - r * 0.35, rect.min.y + r * 0.35),
        pos2(rect.min.x + r * 0.35, rect.max.y - r * 0.35),
        pos2(rect.max.x - r * 0.35, rect.max.y - r * 0.35),
    ] {
        painter.circle_filled(c, stroke.width * 0.65, stroke.color);
    }
}

fn draw_recent_box(app: &mut FilmcraftApp, ui: &mut egui::Ui, t: &Tokens, dark: bool, width: f32, paths: &[String]) {
    let bg = if dark { Color32::from_rgb(0x13, 0x16, 0x1F) } else { t.tl_header_bg };
    let border = if dark { Color32::from_rgba_unmultiplied(255, 255, 255, 20) } else { t.control_border };
    egui::Frame::new().fill(bg).stroke(Stroke::new(1.0, border)).corner_radius(CornerRadius::same(10)).inner_margin(egui::Margin::symmetric(12, 8)).show(
        ui,
        |ui| {
            ui.set_width((width - 24.0).max(80.0));
            ui.label(egui::RichText::new(tl!("Recent")).font(Tokens::semibold(12.5)).color(t.text));
            ui.add_space(4.0);
            let mut open = None;
            for (i, path) in paths.iter().enumerate() {
                let name = crate::menus::recent_name(path);
                let (row, resp) = ui.allocate_exact_size(vec2(ui.available_width(), RECENT_ROW_H), Sense::click());
                let resp = resp.on_hover_text(path.as_str());
                app.auto.add(&format!("import.recent.{i}"), row, &name);
                if resp.hovered() {
                    ui.painter().rect_filled(row, 4.0, t.hover);
                }
                let p = ui.painter().with_clip_rect(row);
                icons::paint(&p, Rect::from_center_size(pos2(row.min.x + 12.0, row.center().y), vec2(12.0, 12.0)), Icon::Sequence, t.text_dim);
                let name_r = p.text(pos2(row.min.x + 24.0, row.center().y), Align2::LEFT_CENTER, &name, Tokens::ui(12.0), t.text);
                if let Some(folder) = std::path::Path::new(path).parent() {
                    let avail = (row.max.x - name_r.max.x - 16.0).max(0.0);
                    if avail > 24.0 {
                        let fg = elided_line(ui, &folder.to_string_lossy(), Tokens::ui(11.0), t.text_dim, avail);
                        p.galley(pos2(name_r.max.x + 10.0, row.center().y - fg.size().y / 2.0), fg, t.text_dim);
                    }
                }
                if resp.clicked() {
                    open = Some(path.clone());
                }
            }
            if let Some(path) = open {
                let ctx = ui.ctx().clone();
                // Ask about unsaved changes first; switch to Edit once the chosen project opens.
                if crate::panels::clip_dialogs::ask_before_open(app, &path).is_some() {
                    if let Some(info) = app.ui.clip_dialog.as_mut().and_then(|d| d.info.as_object_mut()) {
                        info.insert("editAfterOpen".into(), json!(true));
                    }
                } else if crate::menus::invoke(app, &ctx, "file.open", json!({"path": path})).is_ok() {
                    app.ui.mode = Mode::Edit;
                }
            }
        },
    );
}

fn draw_media_pool(app: &mut FilmcraftApp, ui: &mut egui::Ui, t: &Tokens, dark: bool, width: f32, height: f32) {
    let bg = if dark { Color32::from_rgb(0x14, 0x17, 0x20) } else { t.tl_header_bg };
    let border = if dark { Color32::from_rgba_unmultiplied(255, 255, 255, 22) } else { t.control_border };

    egui::Frame::new().fill(bg).stroke(Stroke::new(1.0, border)).corner_radius(CornerRadius::same(10)).inner_margin(egui::Margin::same(12)).show(ui, |ui| {
        let inner_w = (width - 24.0).max(120.0);
        let inner_h = (height - 24.0).max(140.0);
        ui.set_min_size(vec2(inner_w, inner_h));
        ui.set_width(inner_w);

        let items = collect_pool_items(app);
        let total_pool = items.total_media;

        // Title row with total item count pill
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(tl!("Imported Media")).font(Tokens::semibold(14.0)).color(t.text));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let count_str = total_pool.to_string();
                let g = ui.painter().layout_no_wrap(count_str, Tokens::mono(11.0), t.text_dim);
                let badge_w = (g.size().x + 14.0).max(24.0);
                let (br, _) = ui.allocate_exact_size(vec2(badge_w, 18.0), Sense::hover());
                ui.painter().rect_filled(br, CornerRadius::same(9), if dark { Color32::from_rgb(0x1E, 0x23, 0x30) } else { t.field_bg });
                ui.painter().galley(pos2(br.center().x - g.size().x / 2.0, br.center().y - g.size().y / 2.0), g, t.text_dim);
            });
        });

        ui.add_space(8.0);
        draw_media_search_and_tabs(app, ui, t, dark, inner_w);
        ui.add_space(8.0);

        let grid_h = ui.available_height().max(80.0);
        let mut scroll_ui = ui.new_child(egui::UiBuilder::new().max_rect(Rect::from_min_size(ui.cursor().min, vec2(inner_w, grid_h))));
        egui::ScrollArea::vertical().id_salt("import-media-pool-scroll").auto_shrink([false, false]).show(&mut scroll_ui, |ui| {
            ui.set_width((inner_w - 8.0).max(100.0));
            if items.filtered.is_empty() {
                draw_media_empty_state(ui, t, total_pool == 0);
            } else {
                draw_media_grid(app, ui, t, dark, &items.filtered);
            }
        });
        ui.advance_cursor_after_rect(Rect::from_min_size(ui.cursor().min, vec2(inner_w, grid_h)));
    });
}

fn draw_media_search_and_tabs(app: &mut FilmcraftApp, ui: &mut egui::Ui, t: &Tokens, dark: bool, inner_w: f32) {
    // Search box row
    let search_h = 28.0;
    ui.horizontal(|ui| {
        let has_q = !app.ui.import_search.is_empty();
        let clear_w = if has_q { 24.0 } else { 0.0 };
        let edit_w = (inner_w - clear_w - if has_q { 4.0 } else { 0.0 }).max(60.0);
        let te = ui.add_sized(vec2(edit_w, search_h), egui::TextEdit::singleline(&mut app.ui.import_search).hint_text(tl!("Search media…")));
        app.auto.add("import.media.search", te.rect, tl!("Search media…"));
        if has_q {
            let clr = ui.add_sized(vec2(24.0, search_h), egui::Button::new("×")).on_hover_text(tl!("Clear search"));
            if clr.clicked() {
                app.ui.import_search.clear();
            }
        }
    });

    ui.add_space(6.0);
    // Filter tabs: All | Video | Audio | Images
    let tabs = [
        (ImportMediaFilter::All, "all", tl!("All")),
        (ImportMediaFilter::Video, "video", tl!("Video")),
        (ImportMediaFilter::Audio, "audio", tl!("Audio")),
        (ImportMediaFilter::Images, "images", tl!("Images")),
    ];
    let tab_gap = 6.0;
    let tab_w = ((inner_w - tab_gap * 3.0) / 4.0).max(40.0);
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = tab_gap;
        for (filter, key, label) in tabs {
            let active = app.ui.import_filter == filter;
            let (r, resp) = ui.allocate_exact_size(vec2(tab_w, 24.0), Sense::click());
            app.auto.add(&format!("import.media.filter.{key}"), r, label);
            let fill = if active {
                ROYAL_BLUE
            } else if resp.hovered() {
                t.hover
            } else if dark {
                Color32::from_rgb(0x1B, 0x1F, 0x2B)
            } else {
                t.field_bg
            };
            let border = if active { ROYAL_BLUE } else { t.control_border };
            let fg = if active { Color32::WHITE } else { t.text_dim };
            ui.painter().rect_filled(r, CornerRadius::same(6), fill);
            ui.painter().rect_stroke(r, CornerRadius::same(6), Stroke::new(1.0, border), StrokeKind::Inside);
            let g = elided_line(ui, label, if active { Tokens::semibold(11.0) } else { Tokens::ui(11.0) }, fg, (r.width() - 8.0).max(12.0));
            ui.painter().galley(pos2(r.center().x - g.size().x / 2.0, r.center().y - g.size().y / 2.0), g, fg);
            if resp.clicked() {
                app.ui.import_filter = filter;
            }
        }
    });
}

fn draw_media_empty_state(ui: &mut egui::Ui, t: &Tokens, pool_empty: bool) {
    let h = ui.available_height().max(120.0);
    let (r, _) = ui.allocate_exact_size(vec2(ui.available_width(), h), Sense::hover());
    icons::paint(ui.painter(), Rect::from_center_size(pos2(r.center().x, r.center().y - 16.0), vec2(26.0, 26.0)), Icon::Film, t.text_faint);
    let msg = if pool_empty { tl!("No media imported yet") } else { tl!("No matching media") };
    let g = elided_line(ui, msg, Tokens::ui(12.0), t.text_dim, (r.width() - 24.0).max(40.0));
    ui.painter().galley(pos2(r.center().x - g.size().x / 2.0, r.center().y + 10.0), g, t.text_dim);
}

fn draw_media_grid(app: &mut FilmcraftApp, ui: &mut egui::Ui, t: &Tokens, dark: bool, ids: &[ItemId]) {
    let avail_w = ui.available_width();
    let cols = if avail_w >= 460.0 { 3 } else { 2 };
    let gap = 10.0;
    let card_w = ((avail_w - gap * (cols - 1) as f32) / cols as f32).max(110.0);
    let thumb_h = (card_w * 9.0 / 16.0).round();
    let card_h = thumb_h + 32.0;

    let mut add_item: Option<(ItemId, String)> = None;
    let rows = ids.len().div_ceil(cols);
    for row in 0..rows {
        if row > 0 {
            ui.add_space(gap);
        }
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = gap;
            for col in 0..cols {
                let idx = row * cols + col;
                let Some(&id) = ids.get(idx) else { continue };
                if let Some((clicked_id, name)) = draw_media_card(app, ui, t, dark, vec2(card_w, card_h), thumb_h, id) {
                    add_item = Some((clicked_id, name));
                }
            }
        });
    }

    if let Some((id, name)) = add_item {
        quick_add_to_timeline(app, ui.ctx(), id, &name);
    }
}

fn draw_media_card(app: &mut FilmcraftApp, ui: &mut egui::Ui, t: &Tokens, dark: bool, size: egui::Vec2, thumb_h: f32, id: ItemId) -> Option<(ItemId, String)> {
    let it = app.session.project.item(id).cloned()?;
    let (r, resp) = ui.allocate_exact_size(size, Sense::hover());
    let bg = if dark { Color32::from_rgb(0x19, 0x1D, 0x28) } else { t.control_surface };
    let border = if resp.hovered() { Color32::from_rgba_unmultiplied(0x5B, 0x95, 0xFF, 120) } else { t.control_border };
    ui.painter().rect_filled(r, CornerRadius::same(7), bg);
    ui.painter().rect_stroke(r, CornerRadius::same(7), Stroke::new(1.0, border), StrokeKind::Inside);

    let th = Rect::from_min_size(r.min, vec2(r.width(), thumb_h));
    ui.painter().rect_filled(th, CornerRadius { nw: 7, ne: 7, sw: 0, se: 0 }, Color32::from_rgb(12, 14, 19));

    let tt = filmcraft_engine::keyboard::poster_frame(&it).unwrap_or(Tick(it.duration().0 * 3 / 10));
    let ctx = ui.ctx().clone();
    let tw = (th.width() as u32).clamp(96, 320);
    if let Some((tex, sz)) = app.thumbnail(&ctx, it.id, crate::panels::project::quantize(tt), tw) {
        ui.painter().with_clip_rect(th).image(
            tex,
            crate::panels::monitor::fit(th, sz.x, sz.y),
            Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
            Color32::WHITE,
        );
    } else {
        icons::paint(
            ui.painter(),
            Rect::from_center_size(th.center(), vec2(th.height().min(28.0) * 0.78, th.height().min(28.0) * 0.78)),
            crate::panels::project_views::item_icon(&it.kind),
            t.text_faint,
        );
    }

    // Duration / Still badge at bottom-right of thumbnail
    let badge_text = item_duration_label(&it);
    let bg_g = ui.painter().layout_no_wrap(badge_text, Tokens::mono(10.0), Color32::WHITE);
    let badge_w = bg_g.size().x + 10.0;
    let badge_r = Rect::from_min_size(pos2(th.max.x - badge_w - 5.0, th.max.y - 19.0), vec2(badge_w, 15.0));
    ui.painter().rect_filled(badge_r, CornerRadius::same(4), Color32::from_rgba_unmultiplied(10, 12, 18, 205));
    ui.painter().galley(pos2(badge_r.center().x - bg_g.size().x / 2.0, badge_r.center().y - bg_g.size().y / 2.0), bg_g, Color32::WHITE);

    app.auto.add(&format!("import.media.card.{}", id.0), r, &it.name);

    // Card footer: filename + `+` quick-add-to-timeline button
    let footer = Rect::from_min_max(pos2(r.min.x, th.max.y), r.max);
    let add_btn = Rect::from_center_size(pos2(footer.max.x - 16.0, footer.center().y), vec2(22.0, 22.0));
    let add_resp = ui.interact(add_btn, egui::Id::unique(("import-media-add", id.0)), Sense::click()).on_hover_text(tl!("Add to Timeline"));
    app.auto.add(&format!("import.media.add.{}", id.0), add_btn, tl!("Add to Timeline"));
    let add_bg = if add_resp.hovered() {
        ROYAL_BLUE_HOVER
    } else if dark {
        Color32::from_rgb(0x25, 0x33, 0x52)
    } else {
        ROYAL_BLUE
    };
    ui.painter().rect_filled(add_btn, CornerRadius::same(5), add_bg);
    ui.painter().text(add_btn.center(), Align2::CENTER_CENTER, "+", Tokens::semibold(13.5), Color32::WHITE);

    let name_max_w = (add_btn.min.x - footer.min.x - 12.0).max(20.0);
    let name_g = elided_line(ui, &it.name, Tokens::ui(11.5), t.text, name_max_w);
    let elided = name_g.elided;
    ui.painter().galley(pos2(footer.min.x + 8.0, footer.center().y - name_g.size().y / 2.0), name_g, t.text);
    if elided {
        let _ = resp.on_hover_text(&it.name);
    }

    if add_resp.clicked() { Some((id, it.name)) } else { None }
}

fn quick_add_to_timeline(app: &mut FilmcraftApp, ctx: &egui::Context, id: ItemId, name: &str) {
    let res = if let Some(seq) = app.session.active_sequence() {
        let end_at = seq.duration().0;
        crate::menus::invoke(app, ctx, "timeline.place", json!({"item": id.0, "time": end_at, "matchSequence": true}))
    } else {
        crate::menus::invoke(app, ctx, "file.newSequenceFromClip", json!({"items": [id.0]}))
    };
    match res {
        Ok(_) => app.ui.status = tlf!("Added {name} to timeline", name = name),
        Err(e) => app.ui.status = e,
    }
}

struct PoolItems {
    total_media: usize,
    filtered: Vec<ItemId>,
}

fn collect_pool_items(app: &FilmcraftApp) -> PoolItems {
    let mut raw = Vec::new();
    app.session.project.root.all_items(&mut raw);
    let mut seen = HashSet::new();
    let search = app.ui.import_search.trim().to_ascii_lowercase();
    let filter = app.ui.import_filter;

    let mut total_media = 0usize;
    let mut filtered = Vec::new();
    for id in raw {
        if !seen.insert(id) {
            continue;
        }
        let Some(it) = app.session.project.item(id) else { continue };
        if matches!(it.kind, ItemKind::Sequence(_)) || !pp::listed(&app.session.project, it) {
            continue;
        }
        total_media += 1;
        if !matches_category(it, filter) {
            continue;
        }
        let item_name = it.name.to_ascii_lowercase();
        if !search.is_empty() && !search.split_whitespace().all(|w| item_name.contains(w)) {
            continue;
        }
        filtered.push(id);
    }
    PoolItems { total_media, filtered }
}

fn matches_category(it: &ProjectItem, filter: ImportMediaFilter) -> bool {
    match filter {
        ImportMediaFilter::All => true,
        ImportMediaFilter::Images => match &it.kind {
            ItemKind::Media(m) => matches!(m.info.kind, MediaKind::Still),
            ItemKind::Graphic { .. } => true,
            _ => false,
        },
        ImportMediaFilter::Audio => match &it.kind {
            ItemKind::Media(m) => {
                matches!(m.info.kind, MediaKind::AudioOnly) || (matches!(m.info.kind, MediaKind::Synthetic) && !it.has_video() && it.has_audio())
            }
            ItemKind::Subclip { .. } => !it.has_video() && it.has_audio(),
            _ => false,
        },
        ImportMediaFilter::Video => match &it.kind {
            ItemKind::Media(m) => {
                matches!(m.info.kind, MediaKind::Movie | MediaKind::ImageSequence) || (matches!(m.info.kind, MediaKind::Synthetic) && it.has_video())
            }
            ItemKind::Subclip { .. } | ItemKind::AdjustmentLayer { .. } => it.has_video(),
            _ => false,
        },
    }
}

fn item_duration_label(it: &ProjectItem) -> String {
    if matches!(&it.kind, ItemKind::Media(m) if matches!(m.info.kind, MediaKind::Still)) {
        return tl!("Image").to_string();
    }
    let secs_f = it.duration().seconds();
    let secs = if secs_f.is_finite() { secs_f.max(0.0).round() as u64 } else { 0 };
    if secs >= 3600 { format!("{}:{:02}:{:02}", secs / 3600, (secs / 60) % 60, secs % 60) } else { format!("{:02}:{:02}", secs / 60, secs % 60) }
}

fn elided_line(ui: &egui::Ui, text: &str, font: egui::FontId, color: Color32, max_width: f32) -> std::sync::Arc<egui::Galley> {
    let mut job = egui::text::LayoutJob::single_section(text.to_owned(), egui::TextFormat::simple(font, color));
    job.wrap.max_width = max_width.max(0.0);
    job.wrap.max_rows = 1;
    job.wrap.break_anywhere = true;
    job.wrap.overflow_character = Some('…');
    ui.fonts_mut(|f| f.layout_job(job))
}
