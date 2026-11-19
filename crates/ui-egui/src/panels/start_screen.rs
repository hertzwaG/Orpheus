//! Launch Start Screen: two-column workspace chooser and recent-projects launcher.
//!
//! Displays the Orpheus branding, keyboard-accessible (`Tab` / `Enter`) **Creator** and
//! **Studio** mode cards with ambient top gradients, **New Project** / **Open Project…** primary
//! actions on the left column, a translucent **Recent Projects** panel on the right column, and
//! a subtle bottom-right studio signature (`By YBV Studios`).

use std::f32::consts::FRAC_PI_2;

use egui::epaint::{Mesh, PathShape, PathStroke, Vertex};
use egui::{Align2, Color32, CornerRadius, FontId, Key, Painter, Pos2, Rect, Sense, Stroke, StrokeKind, pos2, vec2};

use crate::FilmcraftApp;
use crate::icons::{self, Icon};
use crate::state::{AppExperience, Mode};
use crate::theme::Tokens;

const MAX_RECENT: usize = 5;
const RECENT_ROW_H: f32 = 52.0;
const RECENT_ROW_GAP: f32 = 8.0;

const ROYAL_BLUE: Color32 = Color32::from_rgb(0x2F, 0x6F, 0xED);
const ROYAL_BLUE_HOVER: Color32 = Color32::from_rgb(0x3B, 0x7A, 0xF5);
const VIOLET_ACCENT: Color32 = Color32::from_rgb(0x8B, 0x5C, 0xF6);

struct Palette {
    bg: Color32,
    card_bg: Color32,
    card_hover: Color32,
    card_border: Color32,
    panel_bg: Color32,
    panel_border: Color32,
    recent_bg: Color32,
    recent_hover: Color32,
    recent_border: Color32,
    secondary_bg: Color32,
    secondary_hover: Color32,
    secondary_border: Color32,
    title: Color32,
    text: Color32,
    muted: Color32,
    signature: Color32,
}

impl Palette {
    fn for_theme(dark: bool, t: &Tokens) -> Self {
        if dark {
            Self {
                bg: Color32::from_rgb(0x0E, 0x10, 0x15),
                card_bg: Color32::from_rgb(0x13, 0x16, 0x1E),
                card_hover: Color32::from_rgb(0x17, 0x1B, 0x25),
                card_border: Color32::from_rgba_unmultiplied(255, 255, 255, 20),
                panel_bg: Color32::from_rgb(0x13, 0x16, 0x1D),
                panel_border: Color32::from_rgba_unmultiplied(255, 255, 255, 15),
                recent_bg: Color32::from_rgb(0x18, 0x1B, 0x24),
                recent_hover: Color32::from_rgb(0x1F, 0x23, 0x2F),
                recent_border: Color32::from_rgba_unmultiplied(255, 255, 255, 15),
                secondary_bg: Color32::from_rgb(0x14, 0x17, 0x20),
                secondary_hover: Color32::from_rgb(0x1C, 0x20, 0x2C),
                secondary_border: Color32::from_rgba_unmultiplied(255, 255, 255, 30),
                title: Color32::from_rgb(0xF3, 0xF5, 0xFA),
                text: Color32::from_rgb(0xE6, 0xEA, 0xF2),
                muted: Color32::from_rgb(0x88, 0x91, 0xA4),
                signature: Color32::from_rgba_unmultiplied(255, 255, 255, 61),
            }
        } else {
            Self {
                bg: t.app_bg,
                card_bg: t.panel_bg,
                card_hover: t.hover,
                card_border: t.control_border,
                panel_bg: t.panel_bg,
                panel_border: t.control_border,
                recent_bg: t.field_bg,
                recent_hover: t.hover,
                recent_border: t.control_border,
                secondary_bg: t.panel_bg,
                secondary_hover: t.hover,
                secondary_border: t.control_border,
                title: t.text,
                text: t.text,
                muted: t.text_dim,
                signature: Color32::from_rgba_unmultiplied(0, 0, 0, 68),
            }
        }
    }
}

pub fn show(app: &mut FilmcraftApp, ui: &mut egui::Ui, full: Rect) {
    let t = app.tokens;
    let pal = Palette::for_theme(app.ui.dark, &t);
    let painter = ui.painter();
    painter.rect_filled(full, 0.0, pal.bg);

    let stacked = full.width() < 760.0;
    let footer_reserve = 52.0;
    let banner_h = crate::updates_ui::banner_height(app);
    let top_pad = if banner_h > 0.0 {
        let banner_r = Rect::from_min_size(pos2(full.min.x + 24.0, full.min.y + 14.0), vec2((full.width() - 48.0).max(240.0), banner_h));
        crate::updates_ui::show_banner(app, ui, banner_r);
        banner_h + 24.0
    } else {
        28.0
    };
    let safe_h = (full.height() - footer_reserve - top_pad).max(320.0);

    let (left_rect, right_rect) = if !stacked {
        let max_w = (full.width() - 96.0).clamp(520.0, 1080.0);
        let col_gap = (max_w * 0.04).clamp(28.0, 44.0);
        let left_w = ((max_w - col_gap) * 0.55).clamp(340.0, 576.0);
        let right_w = (max_w - col_gap - left_w).max(240.0);
        let content_h = 372.0_f32.min(safe_h);
        let total_w = left_w + col_gap + right_w;
        let x0 = full.center().x - total_w / 2.0;
        let y0 = (full.min.y + top_pad + (safe_h - content_h) / 2.0).max(full.min.y + top_pad);
        let left = Rect::from_min_size(pos2(x0, y0), vec2(left_w, content_h));
        let right = Rect::from_min_size(pos2(left.max.x + col_gap, y0), vec2(right_w, content_h));
        (left, right)
    } else {
        let w = (full.width() - 48.0).clamp(280.0, 560.0);
        let left_h = 382.0;
        let right_h = 260.0;
        let gap = 20.0;
        let total_h = left_h + gap + right_h;
        let x0 = full.center().x - w / 2.0;
        let y0 = (full.center().y - total_h / 2.0).max(full.min.y + 20.0);
        let left = Rect::from_min_size(pos2(x0, y0), vec2(w, left_h));
        let right = Rect::from_min_size(pos2(x0, left.max.y + gap), vec2(w, right_h));
        (left, right)
    };

    draw_left_column(app, ui, left_rect, &pal, &t);
    draw_recent_panel(app, ui, right_rect, &pal, &t);

    // Bottom-right signature watermark (static across all languages, 24px inset).
    ui.painter().text(pos2(full.max.x - 24.0, full.max.y - 24.0), Align2::RIGHT_BOTTOM, "By YBV Studios", Tokens::display_rounded(23.0), pal.signature);
}

fn draw_left_column(app: &mut FilmcraftApp, ui: &mut egui::Ui, area: Rect, pal: &Palette, t: &Tokens) {
    let logo_h = 68.0;
    let logo_gap = if area.height() < 350.0 { 18.0 } else { 26.0 };
    let btn_h = 46.0;
    let btn_gap = if area.height() < 350.0 { 16.0 } else { 20.0 };
    let card_h = (area.height() - logo_h - logo_gap - btn_gap - btn_h).max(162.0);

    // 1. Header / Logo: 68x68 embedded Orpheus PNG logo with subtle blue/purple ambient glow + bold title
    let badge_r = Rect::from_min_size(area.min, vec2(logo_h, logo_h));
    if app.ui.dark {
        paint_logo_glow(ui.painter(), badge_r.center());
    }
    crate::brand::paint_logo_png(ui, badge_r, app.ui.dark);
    ui.painter().text(pos2(badge_r.max.x + 18.0, badge_r.center().y), Align2::LEFT_CENTER, crate::APP_NAME, Tokens::bold(34.0), pal.title);

    // 2. Mode Cards side-by-side (Creator | Studio)
    let cards_y = area.min.y + logo_h + logo_gap;
    let card_gap = 16.0;
    let card_w = ((area.width() - card_gap) / 2.0).max(120.0);
    let creator_rect = Rect::from_min_size(pos2(area.min.x, cards_y), vec2(card_w, card_h));
    let studio_rect = Rect::from_min_size(pos2(creator_rect.max.x + card_gap, cards_y), vec2(card_w, card_h));

    let creator_id = ui.make_persistent_id("start-card-creator");
    let studio_id = ui.make_persistent_id("start-card-studio");
    if ui.memory(|m| m.focused().is_none()) && ui.input(|i| i.key_pressed(Key::Tab)) {
        ui.memory_mut(|m| m.request_focus(creator_id));
    }

    let cards = [
        (AppExperience::Creator, creator_id, creator_rect, "start.creator", "Creator", tl!("Fast editing for Shorts and TikTok."), ROYAL_BLUE),
        (AppExperience::Studio, studio_id, studio_rect, "start.studio", "Studio", tl!("Full timeline and advanced tools."), VIOLET_ACCENT),
    ];

    for (exp, id, rect, auto_id, title, desc, tint) in cards {
        let resp = ui.interact(rect, id, Sense::click());
        app.auto.add(auto_id, rect, title);

        let focused = resp.has_focus();
        let hovered = resp.hovered();
        let selected = app.ui.experience == exp;
        let fill = if hovered || focused { pal.card_hover } else { pal.card_bg };

        ui.painter().rect_filled(rect, CornerRadius::same(12), fill);

        let top_alpha = if hovered || focused {
            if app.ui.dark { 58 } else { 36 }
        } else if selected {
            if app.ui.dark { 44 } else { 26 }
        } else if app.ui.dark {
            28
        } else {
            16
        };
        paint_rounded_top_gradient(ui.painter(), rect, 12.0, rect.height() * 0.62, tint, top_alpha);

        let border = if hovered || focused {
            Stroke::new(1.5, Color32::from_rgba_unmultiplied(tint.r(), tint.g(), tint.b(), 210))
        } else if selected {
            Stroke::new(1.25, Color32::from_rgba_unmultiplied(tint.r(), tint.g(), tint.b(), 135))
        } else {
            Stroke::new(1.0, pal.card_border)
        };
        ui.painter().rect_stroke(rect, CornerRadius::same(12), border, StrokeKind::Inside);

        let pad = 24.0;
        let dot_center = pos2(rect.max.x - 18.0, rect.min.y + 18.0);
        let halo_alpha = if hovered || focused || selected { 48 } else { 28 };
        let dot_alpha = if hovered || focused || selected { 255 } else { 220 };
        ui.painter().circle_filled(dot_center, 7.0, Color32::from_rgba_unmultiplied(tint.r(), tint.g(), tint.b(), halo_alpha));
        ui.painter().circle_filled(dot_center, 3.5, Color32::from_rgba_unmultiplied(tint.r(), tint.g(), tint.b(), dot_alpha));

        let icon_box = Rect::from_min_size(pos2(rect.min.x + pad, rect.min.y + pad), vec2(74.0, 44.0));
        match exp {
            AppExperience::Creator => paint_creator_graphic(ui.painter(), icon_box, tint),
            AppExperience::Studio => paint_studio_graphic(ui.painter(), icon_box, tint),
        }

        let title_y = icon_box.max.y + 20.0;
        let title_r = ui.painter().text(pos2(rect.min.x + pad, title_y), Align2::LEFT_TOP, title, Tokens::semibold(19.5), pal.text);

        let desc_y = title_r.max.y + 9.0;
        let desc_job = egui::text::LayoutJob::simple(desc.to_string(), Tokens::ui(13.0), pal.muted, (rect.width() - pad * 2.0).max(40.0));
        let desc_galley = ui.painter().layout_job(desc_job);
        ui.painter().galley(pos2(rect.min.x + pad, desc_y), desc_galley, pal.muted);

        let enter = focused && ui.input(|i| i.key_pressed(Key::Enter) || i.key_pressed(Key::Space));
        if resp.clicked() || enter {
            if let Some(path) = app.session.path.clone() {
                app.ui.remember_project_experience(&path, exp);
            }
            app.ui.experience = exp;
            app.ui.mode = Mode::Edit;
            app.ui.start_screen = false;
        }
    }

    // 3. Action Buttons below cards (New Project primary | Open Project… secondary)
    let btns_y = cards_y + card_h + btn_gap;
    let action_gap = 12.0;
    let new_w = ((area.width() - action_gap) * 0.48).max(110.0);
    let open_w = (area.width() - action_gap - new_w).max(110.0);
    let new_rect = Rect::from_min_size(pos2(area.min.x, btns_y), vec2(new_w, btn_h));
    let open_rect = Rect::from_min_size(pos2(new_rect.max.x + action_gap, btns_y), vec2(open_w, btn_h));

    draw_new_project_button(app, ui, new_rect, t);
    draw_open_project_button(app, ui, open_rect, pal);
}

fn draw_new_project_button(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect, t: &Tokens) {
    let label = tl!("New Project");
    let id = ui.make_persistent_id("start-new-project");
    let resp = ui.interact(rect, id, Sense::click());
    app.auto.add("start.newProject", rect, label);

    let focused = resp.has_focus();
    let active = resp.hovered() || focused;
    let fill = if resp.is_pointer_button_down_on() {
        Color32::from_rgb(0x27, 0x5E, 0xCF)
    } else if active {
        ROYAL_BLUE_HOVER
    } else {
        ROYAL_BLUE
    };
    ui.painter().rect_filled(rect, CornerRadius::same(10), fill);
    let border =
        if focused { Stroke::new(1.5, Color32::WHITE) } else { Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, if active { 48 } else { 28 })) };
    ui.painter().rect_stroke(rect, CornerRadius::same(10), border, StrokeKind::Inside);

    let text_g = ui.painter().layout_no_wrap(label.to_string(), Tokens::semibold(14.0), Color32::WHITE);
    let icon_s = 14.0;
    let icon_gap = 8.0;
    let content_w = icon_s + icon_gap + text_g.size().x;
    let start_x = rect.center().x - content_w / 2.0;
    icons::paint(ui.painter(), Rect::from_center_size(pos2(start_x + icon_s / 2.0, rect.center().y), vec2(icon_s, icon_s)), Icon::Plus, Color32::WHITE);
    ui.painter().galley(pos2(start_x + icon_s + icon_gap, rect.center().y - text_g.size().y / 2.0), text_g, Color32::WHITE);

    let _ = t;
    let enter = focused && ui.input(|i| i.key_pressed(Key::Enter) || i.key_pressed(Key::Space));
    if resp.clicked() || enter {
        let ctx = ui.ctx().clone();
        if crate::menus::invoke(app, &ctx, "file.newProject", serde_json::json!({})).is_ok() {
            app.ui.mode = Mode::Edit;
            app.ui.start_screen = false;
        }
    }
}

fn draw_open_project_button(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect, pal: &Palette) {
    let label = tl!("Open Project…");
    let id = ui.make_persistent_id("start-open-project");
    let resp = ui.interact(rect, id, Sense::click());
    app.auto.add("start.openProject", rect, label);

    let focused = resp.has_focus();
    let active = resp.hovered() || focused;
    let fill = if active { pal.secondary_hover } else { pal.secondary_bg };
    let border = if focused {
        Stroke::new(1.5, ROYAL_BLUE)
    } else if active {
        Stroke::new(1.0, Color32::from_rgba_unmultiplied(pal.text.r(), pal.text.g(), pal.text.b(), 64))
    } else {
        Stroke::new(1.0, pal.secondary_border)
    };
    ui.painter().rect_filled(rect, CornerRadius::same(10), fill);
    ui.painter().rect_stroke(rect, CornerRadius::same(10), border, StrokeKind::Inside);

    let p = ui.painter().with_clip_rect(rect.shrink2(vec2(10.0, 2.0)));
    let text_g = p.layout_no_wrap(label.to_string(), Tokens::semibold(13.5), pal.text);
    let icon_s = 15.0;
    let icon_gap = 8.0;
    let content_w = icon_s + icon_gap + text_g.size().x;
    let start_x = (rect.center().x - content_w / 2.0).max(rect.min.x + 12.0);
    icons::paint(
        &p,
        Rect::from_center_size(pos2(start_x + icon_s / 2.0, rect.center().y), vec2(icon_s, icon_s)),
        Icon::Folder,
        if active { pal.text } else { pal.muted },
    );
    p.galley(pos2(start_x + icon_s + icon_gap, rect.center().y - text_g.size().y / 2.0), text_g, pal.text);

    let enter = focused && ui.input(|i| i.key_pressed(Key::Enter) || i.key_pressed(Key::Space));
    if resp.clicked() || enter {
        let ctx = ui.ctx().clone();
        let prev = app.session.path.clone();
        if crate::menus::invoke(app, &ctx, "file.open", serde_json::json!({})).is_ok() && app.session.path != prev {
            if let Some(opened) = app.session.path.as_deref().and_then(|p| app.ui.project_experience(p)) {
                app.ui.experience = opened;
            }
            app.ui.mode = Mode::Edit;
            app.ui.start_screen = false;
        }
    }
}

fn draw_recent_panel(app: &mut FilmcraftApp, ui: &mut egui::Ui, panel_rect: Rect, pal: &Palette, t: &Tokens) {
    let painter = ui.painter();
    painter.rect_filled(panel_rect, CornerRadius::same(16), pal.panel_bg);
    painter.rect_stroke(panel_rect, CornerRadius::same(16), Stroke::new(1.0, pal.panel_border), StrokeKind::Inside);

    let pad = 22.0;
    let inner = panel_rect.shrink(pad);
    painter.text(inner.min, Align2::LEFT_TOP, tl!("Recent Projects"), Tokens::semibold(16.0), pal.title);

    let list_top = inner.min.y + 34.0;
    let avail_h = (inner.max.y - list_top).max(RECENT_ROW_H);
    let max_fit = ((avail_h + RECENT_ROW_GAP) / (RECENT_ROW_H + RECENT_ROW_GAP)).floor().max(1.0) as usize;
    let count = max_fit.min(MAX_RECENT);
    let recent_paths: Vec<String> = app.session.prefs.general.recent_projects.iter().take(count).cloned().collect();

    if recent_paths.is_empty() {
        let empty_center = pos2(inner.center().x, list_top + avail_h * 0.46);
        icons::paint(
            painter,
            Rect::from_center_size(pos2(empty_center.x, empty_center.y - 16.0), vec2(26.0, 26.0)),
            Icon::Folder,
            Color32::from_rgba_unmultiplied(pal.muted.r(), pal.muted.g(), pal.muted.b(), 110),
        );
        painter.text(pos2(empty_center.x, empty_center.y + 14.0), Align2::CENTER_CENTER, tl!("No recent projects"), Tokens::ui(13.0), pal.muted);
        return;
    }

    let mut open_recent: Option<String> = None;
    for (i, path) in recent_paths.iter().enumerate() {
        let row_y = list_top + i as f32 * (RECENT_ROW_H + RECENT_ROW_GAP);
        let row = Rect::from_min_size(pos2(inner.min.x, row_y), vec2(inner.width(), RECENT_ROW_H));
        if row.max.y > panel_rect.max.y - 10.0 {
            break;
        }
        let name = crate::menus::recent_name(path);
        let id = ui.make_persistent_id(("start-recent", i));
        let resp = ui.interact(row, id, Sense::click()).on_hover_text(path.as_str());
        app.auto.add(&format!("start.recent.{i}"), row, &name);

        let focused = resp.has_focus();
        let active = resp.hovered() || focused;
        let fill = if active { pal.recent_hover } else { pal.recent_bg };
        let stroke = if active {
            Stroke::new(1.0, Color32::from_rgba_unmultiplied(ROYAL_BLUE.r(), ROYAL_BLUE.g(), ROYAL_BLUE.b(), 140))
        } else {
            Stroke::new(1.0, pal.recent_border)
        };
        ui.painter().rect_filled(row, CornerRadius::same(10), fill);
        ui.painter().rect_stroke(row, CornerRadius::same(10), stroke, StrokeKind::Inside);

        let p = ui.painter().with_clip_rect(row.shrink2(vec2(12.0, 4.0)));
        let badge = Rect::from_center_size(pos2(row.min.x + 27.0, row.center().y), vec2(32.0, 32.0));
        let badge_fill = if app.ui.dark { Color32::from_rgb(0x1E, 0x22, 0x2F) } else { t.panel_bg };
        p.rect_filled(badge, CornerRadius::same(8), badge_fill);
        icons::paint(&p, Rect::from_center_size(badge.center(), vec2(15.0, 15.0)), Icon::Sequence, Color32::from_rgb(0x6E, 0x9F, 0xFF));

        let mut text_right = row.max.x - 14.0;
        if let Some(exp) = app.ui.project_experience(path) {
            let tag = match exp {
                AppExperience::Creator => "Creator",
                AppExperience::Studio => "Studio",
            };
            let tg = p.layout_no_wrap(tag.to_string(), Tokens::ui(10.5), pal.muted);
            let tw = tg.size().x + 12.0;
            let tr = Rect::from_min_size(pos2(text_right - tw, row.center().y - 9.5), vec2(tw, 19.0));
            p.rect_filled(tr, CornerRadius::same(5), badge_fill);
            p.rect_stroke(tr, CornerRadius::same(5), Stroke::new(1.0, pal.recent_border), StrokeKind::Inside);
            p.galley(pos2(tr.min.x + 6.0, tr.center().y - tg.size().y / 2.0), tg, pal.muted);
            text_right = tr.min.x - 10.0;
        }

        let text_left = badge.max.x + 12.0;
        let text_w = (text_right - text_left).max(20.0);
        let name_shown = fit_text(&p, &name, Tokens::semibold(13.5), pal.text, text_w);
        p.text(pos2(text_left, row.center().y - 8.5), Align2::LEFT_CENTER, name_shown, Tokens::semibold(13.5), pal.text);

        if let Some(folder) = std::path::Path::new(path).parent() {
            let dir_str = folder.to_string_lossy();
            if !dir_str.is_empty() {
                let dir_shown = fit_middle_ellipsis(&p, &dir_str, Tokens::ui(11.5), pal.muted, text_w);
                p.text(pos2(text_left, row.center().y + 9.0), Align2::LEFT_CENTER, dir_shown, Tokens::ui(11.5), pal.muted);
            }
        }

        let enter = focused && ui.input(|k| k.key_pressed(Key::Enter) || k.key_pressed(Key::Space));
        if resp.clicked() || enter {
            open_recent = Some(path.clone());
        }
    }

    if let Some(path) = open_recent {
        let ctx = ui.ctx().clone();
        if crate::menus::invoke(app, &ctx, "file.open", serde_json::json!({ "path": path })).is_ok() {
            if let Some(exp) = app.ui.project_experience(&path) {
                app.ui.experience = exp;
            }
            app.ui.mode = Mode::Edit;
            app.ui.start_screen = false;
        }
    }
}

/// Subtle blue + purple radial ambient glow centered behind the 64x64 Orpheus logo badge.
fn paint_logo_glow(painter: &Painter, center: Pos2) {
    paint_radial_glow(painter, center + vec2(-12.0, -6.0), 58.0, Color32::from_rgba_unmultiplied(47, 111, 237, 56));
    paint_radial_glow(painter, center + vec2(12.0, 8.0), 56.0, Color32::from_rgba_unmultiplied(139, 92, 246, 46));
}

fn paint_radial_glow(painter: &Painter, center: Pos2, radius: f32, center_color: Color32) {
    const SEGMENTS: usize = 28;
    let mut mesh = Mesh::default();
    mesh.vertices.push(Vertex { pos: center, uv: Pos2::ZERO, color: center_color });
    let edge_color = Color32::from_rgba_unmultiplied(center_color.r(), center_color.g(), center_color.b(), 0);
    for i in 0..=SEGMENTS {
        let ang = (i as f32 / SEGMENTS as f32) * std::f32::consts::TAU;
        mesh.vertices.push(Vertex { pos: center + vec2(ang.cos() * radius, ang.sin() * radius), uv: Pos2::ZERO, color: edge_color });
    }
    for i in 0..SEGMENTS {
        mesh.indices.push(0);
        mesh.indices.push((i + 1) as u32);
        mesh.indices.push((i + 2) as u32);
    }
    painter.add(egui::Shape::mesh(mesh));
}

/// Top ambient gradient clipped cleanly to the rounded top corners (`radius`) with zero corner bleed.
fn paint_rounded_top_gradient(painter: &Painter, rect: Rect, radius: f32, grad_h: f32, tint: Color32, top_alpha: u8) {
    let r = radius.min(rect.width() * 0.5).min(rect.height() * 0.5).max(0.0);
    let h = grad_h.min(rect.height()).max(r + 1.0);
    let alpha_at = |y: f32| -> Color32 {
        let t = ((y - rect.min.y) / h).clamp(0.0, 1.0);
        let fade = (1.0 - t) * (1.0 - t);
        let a = (f32::from(top_alpha) * fade).round().clamp(0.0, 255.0) as u8;
        Color32::from_rgba_unmultiplied(tint.r(), tint.g(), tint.b(), a)
    };

    const ARC_STEPS: usize = 8;
    let mut rows: Vec<(f32, f32, f32)> = Vec::with_capacity(ARC_STEPS + 4);
    for step in 0..=ARC_STEPS {
        // Sweep from top chord (`PI/2`) down to the end of the corner radius (`0`).
        let ang = FRAC_PI_2 * (1.0 - step as f32 / ARC_STEPS as f32);
        let y = rect.min.y + r - r * ang.sin();
        let dx = r - r * ang.cos();
        rows.push((y, rect.min.x + dx, rect.max.x - dx));
    }
    let mid_y = rect.min.y + r + (h - r) * 0.45;
    if mid_y > rect.min.y + r {
        rows.push((mid_y, rect.min.x, rect.max.x));
    }
    rows.push((rect.min.y + h, rect.min.x, rect.max.x));

    let mut mesh = Mesh::default();
    for (y, xl, xr) in &rows {
        let col = alpha_at(*y);
        mesh.vertices.push(Vertex { pos: pos2(*xl, *y), uv: Pos2::ZERO, color: col });
        mesh.vertices.push(Vertex { pos: pos2(*xr, *y), uv: Pos2::ZERO, color: col });
    }
    for i in 0..rows.len().saturating_sub(1) {
        let top = (i * 2) as u32;
        let bot = ((i + 1) * 2) as u32;
        mesh.add_triangle(top, top + 1, bot + 1);
        mesh.add_triangle(top, bot + 1, bot);
    }
    painter.add(egui::Shape::mesh(mesh));
}

/// Creator card illustration: vertical smartphone frame flanked by a lightning bolt (left) and
/// beamed music/shorts notes (right).
fn paint_creator_graphic(painter: &Painter, box_r: Rect, tint: Color32) {
    let cy = box_r.center().y;
    let bright = Color32::from_rgb(0x7C, 0xB3, 0xFF);
    let soft = Color32::from_rgba_unmultiplied(tint.r(), tint.g(), tint.b(), 44);

    // Center vertical phone frame
    let phone = Rect::from_center_size(pos2(box_r.min.x + 34.0, cy), vec2(21.0, 36.0));
    painter.rect_filled(phone, CornerRadius::same(4), soft);
    painter.rect_stroke(phone, CornerRadius::same(4), Stroke::new(1.5, bright), StrokeKind::Inside);
    painter.line_segment([pos2(phone.center().x - 3.0, phone.min.y + 4.0), pos2(phone.center().x + 3.0, phone.min.y + 4.0)], Stroke::new(1.3, bright));
    painter.line_segment([pos2(phone.center().x - 4.0, phone.max.y - 4.0), pos2(phone.center().x + 4.0, phone.max.y - 4.0)], Stroke::new(1.3, bright));
    // Play triangle inside phone
    let pc = phone.center() + vec2(0.5, 0.0);
    painter.add(PathShape::convex_polygon(vec![pos2(pc.x - 3.0, pc.y - 4.0), pos2(pc.x + 4.0, pc.y), pos2(pc.x - 3.0, pc.y + 4.0)], bright, Stroke::NONE));

    // Left flank: crisp lightning bolt
    let lx = box_r.min.x + 9.0;
    let bolt = vec![
        pos2(lx + 2.5, cy - 10.5),
        pos2(lx - 4.5, cy + 0.5),
        pos2(lx + 0.5, cy + 0.5),
        pos2(lx - 2.0, cy + 10.5),
        pos2(lx + 5.5, cy - 1.5),
        pos2(lx + 0.5, cy - 1.5),
    ];
    painter.add(PathShape { points: bolt, closed: true, fill: bright, stroke: PathStroke::NONE });

    // Right flank: double-beamed music / shorts eighth notes
    let rx = box_r.min.x + 60.0;
    let n1 = pos2(rx - 5.0, cy + 6.5);
    let n2 = pos2(rx + 4.5, cy + 4.5);
    painter.circle_filled(n1, 2.8, bright);
    painter.circle_filled(n2, 2.8, bright);
    painter.line_segment([pos2(n1.x + 2.0, n1.y - 0.5), pos2(n1.x + 2.0, cy - 8.5)], Stroke::new(1.6, bright));
    painter.line_segment([pos2(n2.x + 2.0, n2.y - 0.5), pos2(n2.x + 2.0, cy - 10.5)], Stroke::new(1.6, bright));
    painter.line_segment([pos2(n1.x + 1.5, cy - 8.5), pos2(n2.x + 2.5, cy - 10.5)], Stroke::new(2.6, bright));
}

/// Studio card illustration: filmstrip frame on the left next to audio waveform bars on the right.
fn paint_studio_graphic(painter: &Painter, box_r: Rect, tint: Color32) {
    let cy = box_r.center().y;
    let bright = Color32::from_rgb(0xB8, 0x99, 0xFF);
    let dim_bar = Color32::from_rgb(0x93, 0x72, 0xF5);
    let soft = Color32::from_rgba_unmultiplied(tint.r(), tint.g(), tint.b(), 44);

    // Left: filmstrip frame with top & bottom sprocket holes
    let film = Rect::from_center_size(pos2(box_r.min.x + 18.0, cy), vec2(34.0, 28.0));
    painter.rect_filled(film, CornerRadius::same(4), soft);
    painter.rect_stroke(film, CornerRadius::same(4), Stroke::new(1.5, bright), StrokeKind::Inside);
    painter.line_segment([pos2(film.min.x + 1.5, film.min.y + 7.0), pos2(film.max.x - 1.5, film.min.y + 7.0)], Stroke::new(1.1, bright));
    painter.line_segment([pos2(film.min.x + 1.5, film.max.y - 7.0), pos2(film.max.x - 1.5, film.max.y - 7.0)], Stroke::new(1.1, bright));
    for slot in 0..3 {
        let sx = film.min.x + 5.5 + slot as f32 * 9.5;
        painter.rect_filled(Rect::from_min_size(pos2(sx, film.min.y + 2.8), vec2(4.0, 2.2)), CornerRadius::same(1), bright);
        painter.rect_filled(Rect::from_min_size(pos2(sx, film.max.y - 5.0), vec2(4.0, 2.2)), CornerRadius::same(1), bright);
    }
    let fc = film.center() + vec2(0.5, 0.0);
    painter.add(PathShape::convex_polygon(vec![pos2(fc.x - 2.5, fc.y - 3.5), pos2(fc.x + 3.5, fc.y), pos2(fc.x - 2.5, fc.y + 3.5)], bright, Stroke::NONE));

    // Right: 5 vertical rounded-cap audio waveform bars
    const HEIGHTS: [f32; 5] = [0.42, 0.86, 0.56, 1.0, 0.46];
    let bars_x0 = film.max.x + 10.0;
    let bar_w = 3.2;
    let bar_gap = 3.2;
    let max_h = 28.0;
    for (i, ratio) in HEIGHTS.iter().enumerate() {
        let h = max_h * ratio;
        let x = bars_x0 + i as f32 * (bar_w + bar_gap);
        let r = Rect::from_center_size(pos2(x + bar_w / 2.0, cy), vec2(bar_w, h));
        painter.rect_filled(r, CornerRadius::same(2), if i % 2 == 1 { bright } else { dim_bar });
    }
}

fn fit_text(painter: &Painter, text: &str, font: FontId, color: Color32, max_w: f32) -> String {
    if painter.layout_no_wrap(text.to_string(), font.clone(), color).size().x <= max_w {
        return text.to_string();
    }
    let chars: Vec<char> = text.chars().collect();
    let mut hi = chars.len();
    while hi > 0 {
        let cand: String = chars[..hi].iter().collect::<String>() + "…";
        if painter.layout_no_wrap(cand.clone(), font.clone(), color).size().x <= max_w {
            return cand;
        }
        hi -= 1;
    }
    "…".to_string()
}

fn fit_middle_ellipsis(painter: &Painter, text: &str, font: FontId, color: Color32, max_w: f32) -> String {
    if painter.layout_no_wrap(text.to_string(), font.clone(), color).size().x <= max_w {
        return text.to_string();
    }
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= 6 {
        return fit_text(painter, text, font, color, max_w);
    }
    let keep_left = (chars.len() / 3).clamp(3, 12);
    let mut keep_right = chars.len().saturating_sub(keep_left + 1);
    while keep_right > 4 {
        let mut cand: String = chars[..keep_left].iter().collect();
        cand.push('…');
        cand.extend(&chars[chars.len() - keep_right..]);
        if painter.layout_no_wrap(cand.clone(), font.clone(), color).size().x <= max_w {
            return cand;
        }
        keep_right -= 1;
    }
    fit_text(painter, text, font, color, max_w)
}
