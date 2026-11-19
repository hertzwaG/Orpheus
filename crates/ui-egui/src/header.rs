//! The header bar (Premiere 26 layout, 38 pt): Home and Import / Edit / Export on the left, the
//! document title centred, the workspace name and quick actions on the right. On macOS the menus
//! live in the native menu bar (set up by the app); elsewhere a compact in-window menu bar follows
//! the mode tabs.

use egui::{Color32, Rect, Sense, Stroke, pos2, vec2};

use crate::FilmcraftApp;
use crate::icons::{self, Icon};
use crate::state::{AppExperience, Mode};
use crate::theme::Tokens;

/// Height of the header bar. With `integrated_titlebar` it sits in the macOS title-bar strip.
pub const HEIGHT: f32 = 38.0;

/// Draws the header bar in `rect` and handles its clicks: Home, the mode tabs, the workspace name
/// and its menu (the workspaces, then Reset to Saved Layout, Save Changes to this Workspace, Save as
/// New Workspace… and Edit Workspaces…), Discord and the quick actions.
pub fn show(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect) {
    let t = app.tokens;
    let p = ui.painter().clone();
    p.rect_filled(rect, 0.0, t.header_bg);
    p.line_segment([rect.left_bottom(), rect.right_bottom()], Stroke::new(1.0, t.separator));
    let drag = ui.interact(rect, egui::Id::unique("header-drag"), Sense::click_and_drag());
    if drag.drag_started_by(egui::PointerButton::Primary) {
        ui.ctx().send_viewport_cmd(egui::ViewportCommand::StartDrag);
    }
    if drag.double_clicked() {
        let max = ui.ctx().input(|i| i.viewport().maximized.unwrap_or(false));
        ui.ctx().send_viewport_cmd(egui::ViewportCommand::Maximized(!max));
    }
    if app.ui.experience == AppExperience::Creator {
        show_creator_header(app, ui, rect, &t);
        return;
    }
    let mut x = rect.min.x + if app.integrated_titlebar { 104.0 } else { 14.0 };
    // Home
    let home = Rect::from_center_size(pos2(x + 10.0, rect.center().y), vec2(26.0, 26.0));
    let hresp = ui.interact(home, egui::Id::unique("hdr-home"), Sense::click()).on_hover_text(tl!("Home"));
    app.auto.add("header.home", home, "Home");
    if hresp.hovered() {
        p.rect_filled(home, 4.0, t.hover);
    }
    icons::paint(&p, home.shrink(5.0), Icon::Home, if hresp.hovered() { t.text } else { t.text_dim });
    if hresp.clicked() {
        app.ui.mode = Mode::Import;
    }
    x = home.max.x + 16.0;
    // Mode tabs (14 pt; active = primary text with a 2 pt underline under the label)
    for (m, label, shown) in [(Mode::Import, "Import", tl!("Import")), (Mode::Edit, "Edit", tl!("Edit")), (Mode::Export, "Export", tl!("Export"))] {
        let active = app.ui.mode == m;
        let galley = p.layout_no_wrap(shown.to_string(), Tokens::ui(14.0), if active { t.tab_text_active } else { t.tab_text });
        let r = Rect::from_min_size(pos2(x - 6.0, rect.min.y + 6.0), vec2(galley.size().x + 12.0, rect.height() - 12.0));
        let resp = ui.interact(r, egui::Id::unique(("mode", label)), Sense::click());
        app.auto.add(&format!("header.mode.{}", label.to_ascii_lowercase()), r, label);
        let col = if active || resp.hovered() { t.tab_text_active } else { t.tab_text };
        let gw = galley.size().x;
        p.galley_with_override_text_color(pos2(x, rect.center().y - galley.size().y / 2.0), galley, col);
        if active {
            p.line_segment([pos2(x, rect.min.y + 29.0), pos2(x + gw, rect.min.y + 29.0)], Stroke::new(2.0, t.tab_text_active));
        }
        if resp.clicked() {
            app.ui.mode = m;
        }
        x += gw + 24.0;
    }
    let mut left_end = x;
    // In-window menus when there is no native menu bar.
    if app.ui.show_menu_bar {
        // Vertical divider between mode tabs (Import | Edit | Export) and the menu bar (File...).
        let div1_x = x - 10.0;
        p.line_segment([pos2(div1_x, rect.min.y + 9.0), pos2(div1_x, rect.max.y - 9.0)], Stroke::new(1.0, t.separator));
        let menu_rect = Rect::from_min_max(pos2(x + 2.0, rect.min.y + 7.0), pos2(x + 520.0, rect.max.y - 7.0));
        let mut mu = ui.new_child(egui::UiBuilder::new().max_rect(menu_rect).layout(egui::Layout::left_to_right(egui::Align::Center)));
        mu.style_mut().visuals.widgets.inactive.weak_bg_fill = Color32::TRANSPARENT;
        mu.style_mut().visuals.widgets.inactive.bg_stroke = Stroke::NONE;
        mu.style_mut().visuals.override_text_color = Some(t.text_dim);
        crate::menus::menu_bar(app, &mut mu);
        left_end = mu.min_rect().max.x;
    }
    // Document title, centred
    let title = if app.session.is_dirty() { tlf!("{name} - Edited", name = app.session.project.name) } else { app.session.project.name.clone() };
    // Right cluster: Export (primary blue CTA), Creator / Studio switcher, Workspaces, Appearance, Fullscreen.
    let mut rx = rect.max.x - 14.0;
    let exp_label = tl!("Export");
    let exp_g = p.layout_no_wrap(exp_label.to_string(), Tokens::semibold(12.5), Color32::WHITE);
    let exp_w = exp_g.size().x + 32.0;
    let exp_r = Rect::from_min_size(pos2(rx - exp_w, rect.center().y - 13.0), vec2(exp_w, 26.0));
    let exp_resp = ui.interact(exp_r, egui::Id::unique("hdr-quick-export"), Sense::click()).on_hover_text(tl!("Quick Export"));
    app.auto.add("header.quickExport", exp_r, exp_label);
    let exp_bg = if app.ui.export.quick_open || exp_resp.hovered() { t.accent_hover } else { t.accent };
    p.rect_filled(exp_r, 6.0, exp_bg);
    icons::paint(&p, Rect::from_center_size(pos2(exp_r.min.x + 14.0, exp_r.center().y), vec2(13.0, 13.0)), Icon::Export, Color32::WHITE);
    p.galley(pos2(exp_r.min.x + 25.0, exp_r.center().y - exp_g.size().y / 2.0), exp_g, Color32::WHITE);
    if exp_resp.clicked() {
        app.ui.export.quick_open = !app.ui.export.quick_open;
        ui.ctx().data_mut(|d| d.insert_temp(egui::Id::unique("quick-export-toggled"), true));
    }
    crate::panels::export_mode::quick_export(app, ui.ctx(), pos2(exp_r.max.x - 340.0, rect.max.y + 4.0));
    rx = exp_r.min.x - 8.0;
    let donate_r = draw_donate_pill(app, ui, &p, &t, rx, rect.center().y);
    rx = donate_r.min.x - 6.0;

    let mut btn = |ui: &mut egui::Ui, icon: Icon, id: &str, tip: &str, app: &mut FilmcraftApp| -> egui::Response {
        let r = Rect::from_center_size(pos2(rx - 14.0, rect.center().y), vec2(28.0, 28.0));
        rx -= 34.0;
        let resp = ui.interact(r, egui::Id::unique(("hdr", id)), Sense::click()).on_hover_text(tip);
        app.auto.add(&format!("header.{id}"), r, tip);
        if resp.hovered() {
            ui.painter().rect_filled(r, 4.0, t.hover);
        }
        icons::paint(ui.painter(), r.shrink(6.0), icon, if resp.hovered() { t.text } else { t.text_dim });
        resp
    };
    if btn(ui, Icon::Fullscreen, "fullscreen", tl!("Full screen"), app).clicked() {
        let fs = ui.ctx().input(|i| i.viewport().fullscreen.unwrap_or(false));
        ui.ctx().send_viewport_cmd(egui::ViewportCommand::Fullscreen(!fs));
    }
    // Appearance Mode: click for the next one (Auto, Light, Dark); the icon shows the current one.
    let (icon, mode) = match app.session.prefs.appearance.appearance_mode.as_str() {
        "auto" => (Icon::Monitor, tl!("Sync with system")),
        "light" => (Icon::Sun, tl!("Light")),
        _ => (Icon::Moon, tl!("Dark")),
    };
    let tip = format!("{}: {mode}", tl!("Appearance Mode"));
    let appearance = btn(ui, icon, "appearance", &tip, app);
    appearance.context_menu(|ui| crate::panels::settings::appearance_context_menu(app, ui));
    if appearance.clicked() {
        crate::panels::settings::cycle_appearance(app, ui.ctx());
    }
    let ws_resp = btn(ui, Icon::Workspaces, "workspaces", tl!("Workspaces"), app);
    let ws = crate::i18n::to_uppercase(crate::i18n::t(&app.ui.workspace));
    app.auto.add("header.workspaceName", ws_resp.rect, &ws);
    {
        let sw_rect = paint_experience_switcher(app, ui, pos2(rx - 6.0, rect.center().y), true, &t);
        // Vertical divider between the menu bar (Help) and the project title.
        if app.ui.show_menu_bar && left_end + 16.0 < sw_rect.min.x {
            let div2_x = left_end + 8.0;
            p.line_segment([pos2(div2_x, rect.min.y + 9.0), pos2(div2_x, rect.max.y - 9.0)], Stroke::new(1.0, t.separator));
        }
        // Localized menus and user titles can be wider than English. Truncate the title first on narrow windows
        // so the Creator/Studio switcher and Export button stay visible.
        let left = left_end + if app.ui.show_menu_bar { 18.0 } else { 12.0 };
        let right = sw_rect.min.x - 12.0;
        if right > left + 24.0 {
            let mut job = egui::text::LayoutJob::simple_singleline(title, Tokens::ui(14.0), t.tab_text_active);
            job.wrap.max_width = (right - left).max(1.0);
            job.wrap.max_rows = 1;
            job.wrap.break_anywhere = true;
            job.wrap.overflow_character = Some('…');
            let galley = p.layout_job(job);
            let half = galley.size().x / 2.0;
            let center = if rect.center().x - half >= left && rect.center().x + half <= right { rect.center().x } else { (left + right) / 2.0 };
            p.with_clip_rect(Rect::from_min_max(pos2(left, rect.min.y), pos2(right, rect.max.y))).galley(
                pos2(center - half, rect.center().y - galley.size().y / 2.0),
                galley,
                t.tab_text_active,
            );
        }
    }
    let popup_id = egui::Id::unique("workspaces-popup");
    if ws_resp.clicked() {
        ui.ctx().data_mut(|d| d.insert_temp(popup_id, true));
    }
    let open = ui.ctx().data(|d| d.get_temp::<bool>(popup_id).unwrap_or(false));
    if open {
        let anchor = pos2((ws_resp.rect.max.x - 220.0).max(rect.min.x + 8.0), rect.max.y + 4.0);
        let area = egui::Area::new(popup_id.with("area")).order(egui::Order::Foreground).fixed_pos(anchor).show(ui.ctx(), |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.with_layout(egui::Layout::top_down_justified(egui::Align::Min), |ui| {
                    ui.set_min_width(220.0);
                    for w in crate::dock::names(&app.workspaces) {
                        let sel = app.ui.workspace == w;
                        if ui.selectable_label(sel, crate::i18n::t(&w)).clicked() {
                            app.set_workspace(&w);
                            ui.ctx().data_mut(|d| d.insert_temp(popup_id, false));
                        }
                    }
                    ui.separator();
                    // the actions of Window ▸ Workspaces, as in Premiere's workspace menu
                    let actions = [
                        ("reset", tl!("Reset to Saved Layout")),
                        ("saveChanges", tl!("Save Changes to this Workspace")),
                        ("saveAs", tl!("Save as New Workspace…")),
                        ("edit", tl!("Edit Workspaces…")),
                    ];
                    for (action, label) in actions {
                        let resp = ui.button(label);
                        app.auto.add(&format!("header.workspaces.{action}"), resp.rect, label);
                        if resp.clicked() {
                            ui.ctx().data_mut(|d| d.insert_temp(popup_id, false));
                            let ctx = ui.ctx().clone();
                            if let Err(e) = crate::menus::invoke(app, &ctx, &format!("window.workspace.{action}"), serde_json::json!({})) {
                                app.ui.status = e;
                            }
                        }
                    }
                });
            });
        });
        if area.response.clicked_elsewhere() && !ws_resp.clicked() {
            ui.ctx().data_mut(|d| d.insert_temp(popup_id, false));
        }
    }
}

fn show_creator_header(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect, t: &Tokens) {
    let p = ui.painter().clone();
    let mut x = rect.min.x + if app.integrated_titlebar { 104.0 } else { 14.0 };

    // Lyre of Time badge (click returns to Edit if in Export/Import)
    let lyre_r = Rect::from_center_size(pos2(x + 12.0, rect.center().y), vec2(24.0, 24.0));
    let lyre_resp = ui.interact(lyre_r, egui::Id::unique("hdr-creator-lyre"), Sense::click()).on_hover_text(crate::APP_NAME);
    app.auto.add("header.lyre", lyre_r, crate::APP_NAME);
    crate::brand::paint_lyre_badge(&p, lyre_r, app.ui.dark);
    if lyre_resp.clicked() {
        app.ui.mode = Mode::Edit;
    }
    x = lyre_r.max.x + 12.0;

    // Project name beside the lyre
    let title = if app.session.is_dirty() { tlf!("{name} - Edited", name = app.session.project.name) } else { app.session.project.name.clone() };
    let tg = p.layout_no_wrap(title.clone(), Tokens::semibold(13.5), t.tab_text_active);
    let max_title_w = (rect.width() * 0.28).max(80.0);
    let title_clip = Rect::from_min_max(pos2(x, rect.min.y), pos2((x + max_title_w).min(rect.center().x - 80.0), rect.max.y));
    if title_clip.width() > 20.0 {
        p.with_clip_rect(title_clip).galley(pos2(x, rect.center().y - tg.size().y / 2.0), tg, t.tab_text_active);
    }

    // Center: Creator / Studio switcher
    let _sw = paint_experience_switcher(app, ui, rect.center(), false, t);

    // Right: Export button (primary accent) + Save + Appearance
    let exp_label = tl!("Export");
    let exp_g = p.layout_no_wrap(exp_label.to_string(), Tokens::semibold(12.5), Color32::WHITE);
    let exp_w = exp_g.size().x + 32.0;
    let exp_r = Rect::from_min_size(pos2(rect.max.x - 14.0 - exp_w, rect.center().y - 13.0), vec2(exp_w, 26.0));
    let exp_resp = ui.interact(exp_r, egui::Id::unique("hdr-creator-export"), Sense::click());
    app.auto.add("header.creator.export", exp_r, exp_label);
    let exp_bg = if app.ui.mode == Mode::Export || exp_resp.hovered() { t.accent_hover } else { t.accent };
    p.rect_filled(exp_r, 6.0, exp_bg);
    icons::paint(&p, Rect::from_center_size(pos2(exp_r.min.x + 14.0, exp_r.center().y), vec2(13.0, 13.0)), Icon::Export, Color32::WHITE);
    p.galley(pos2(exp_r.min.x + 25.0, exp_r.center().y - exp_g.size().y / 2.0), exp_g, Color32::WHITE);
    if exp_resp.clicked() {
        app.ui.mode = if app.ui.mode == Mode::Export { Mode::Edit } else { Mode::Export };
    }

    // Save button to the left of Export
    let save_label = tl!("Save");
    let save_g = p.layout_no_wrap(save_label.to_string(), Tokens::ui(12.0), t.text);
    let save_w = save_g.size().x + 20.0;
    let save_r = Rect::from_min_size(pos2(exp_r.min.x - save_w - 8.0, rect.center().y - 13.0), vec2(save_w, 26.0));
    let save_resp = ui.interact(save_r, egui::Id::unique("hdr-creator-save"), Sense::click());
    app.auto.add("header.creator.save", save_r, save_label);
    p.rect_filled(save_r, 6.0, if save_resp.hovered() { t.hover } else { t.control_surface });
    p.rect_stroke(save_r, 6.0, Stroke::new(1.0, t.control_border), egui::StrokeKind::Inside);
    p.galley(pos2(save_r.min.x + 10.0, save_r.center().y - save_g.size().y / 2.0), save_g, t.text);
    if save_resp.clicked() {
        let ctx = ui.ctx().clone();
        if let Err(e) = crate::menus::invoke(app, &ctx, "file.save", serde_json::json!({})) {
            app.ui.status = e;
        }
    }
    let _ = draw_donate_pill(app, ui, &p, t, save_r.min.x - 8.0, rect.center().y);
}

fn donate_label() -> &'static str {
    let v = crate::i18n::t("header.donate");
    if v == "header.donate" { "♥ Donate" } else { v }
}

fn donate_tooltip() -> &'static str {
    tl!("Coming soon")
}

fn paint_heart_icon(p: &egui::Painter, center: egui::Pos2, size: f32, col: Color32) {
    let r = size * 0.24;
    let l = pos2(center.x - r * 0.92, center.y - r * 0.42);
    let rc = pos2(center.x + r * 0.92, center.y - r * 0.42);
    p.circle_filled(l, r, col);
    p.circle_filled(rc, r, col);
    let pts = vec![pos2(center.x - r * 1.84, center.y - r * 0.10), pos2(center.x + r * 1.84, center.y - r * 0.10), pos2(center.x, center.y + size * 0.44)];
    p.add(egui::Shape::convex_polygon(pts, col, Stroke::NONE));
}

fn draw_donate_pill(app: &mut FilmcraftApp, ui: &mut egui::Ui, p: &egui::Painter, t: &Tokens, right_x: f32, cy: f32) -> Rect {
    let raw = donate_label();
    let shown = raw.trim_start_matches(['♥', ' ']);
    let tip = donate_tooltip();
    let fg = t.text_dim;
    let heart_col = t.text_dim;
    let g = p.layout_no_wrap(shown.to_string(), Tokens::semibold(11.5), fg);
    let pill_w = g.size().x + 30.0;
    let r = Rect::from_min_size(pos2(right_x - pill_w, cy - 12.0), vec2(pill_w, 24.0));
    let _resp = ui.interact(r, egui::Id::unique("hdr-donate-revolut"), Sense::hover()).on_hover_text(tip);
    app.auto.add("header.donate", r, raw);

    let bg = if app.ui.dark { Color32::from_rgba_unmultiplied(0x93, 0x33, 0xEA, 22) } else { Color32::from_rgba_unmultiplied(0xE5, 0xE7, 0xEB, 120) };
    let border = t.control_border;
    p.rect_filled(r, 12.0, bg);
    p.rect_stroke(r, 12.0, Stroke::new(1.0, border), egui::StrokeKind::Inside);
    paint_heart_icon(p, pos2(r.min.x + 12.0, r.center().y), 11.0, heart_col);
    p.galley(pos2(r.min.x + 21.0, r.center().y - g.size().y / 2.0), g, fg);
    r
}

fn paint_experience_switcher(app: &mut FilmcraftApp, ui: &mut egui::Ui, anchor: egui::Pos2, right_aligned: bool, t: &Tokens) -> Rect {
    let p = ui.painter().clone();
    let cg = p.layout_no_wrap("Creator".to_string(), Tokens::semibold(12.0), Color32::WHITE);
    let sg = p.layout_no_wrap("Studio".to_string(), Tokens::semibold(12.0), Color32::WHITE);
    let cw = cg.size().x + 18.0;
    let sw = sg.size().x + 18.0;
    let total_w = cw + sw + 4.0;
    let min_x = if right_aligned { anchor.x - total_w } else { anchor.x - total_w / 2.0 };
    let pill = Rect::from_min_size(pos2(min_x, anchor.y - 13.0), vec2(total_w, 26.0));
    p.rect_filled(pill, 6.0, t.control_surface);
    p.rect_stroke(pill, 6.0, Stroke::new(1.0, t.control_border), egui::StrokeKind::Inside);

    let c_rect = Rect::from_min_size(pos2(pill.min.x + 2.0, pill.min.y + 2.0), vec2(cw, 22.0));
    let s_rect = Rect::from_min_size(pos2(c_rect.max.x, pill.min.y + 2.0), vec2(sw, 22.0));

    for (exp, r, g, id, label) in [(AppExperience::Creator, c_rect, cg, "creator", "Creator"), (AppExperience::Studio, s_rect, sg, "studio", "Studio")] {
        let active = app.ui.experience == exp;
        let resp = ui.interact(r, egui::Id::unique(("hdr-exp", id)), Sense::click());
        app.auto.add(&format!("header.experience.{id}"), r, label);
        if active {
            p.rect_filled(r, 4.0, t.accent);
        } else if resp.hovered() {
            p.rect_filled(r, 4.0, t.hover);
        }
        let col = if active {
            Color32::WHITE
        } else if resp.hovered() {
            t.text
        } else {
            t.text_dim
        };
        p.galley(pos2(r.center().x - g.size().x / 2.0, r.center().y - g.size().y / 2.0), g, col);
        if resp.clicked() {
            app.ui.experience = exp;
            app.ui.mode = Mode::Edit;
            if let Some(path) = app.session.path.clone() {
                app.ui.remember_project_experience(&path, exp);
            }
        }
    }
    pill
}

#[cfg(test)]
mod tests {
    #[test]
    fn workspace_name_uppercasing_is_turkish_aware() {
        crate::i18n::set_current(crate::i18n::Language::Tr);
        assert_eq!(crate::i18n::to_uppercase("Birleştirme"), "BİRLEŞTİRME");
        assert_eq!(crate::i18n::to_uppercase("Altyazılar ve Grafikler"), "ALTYAZILAR VE GRAFİKLER");
        crate::i18n::set_current(crate::i18n::Language::En);
        assert_eq!(crate::i18n::to_uppercase("Birleştirme"), "BIRLEŞTIRME");
    }
}
