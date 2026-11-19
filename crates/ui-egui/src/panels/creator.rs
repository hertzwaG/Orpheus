//! Streamlined Creator workspace for fast Shorts and TikTok editing.
//!
//! Layout:
//! - Left: compact tool rail (**Media · Text · Audio**) opening into one shared collapsible drawer
//! - Center: maximum-width 9:16-friendly Program preview (`PanelKind::Program`)
//! - Right: contextual inspector shown only when a clip or text overlay is selected
//! - Bottom: full-width Timeline with visible Split, Delete, Undo, and Redo controls

use egui::{CornerRadius, Rect, Sense, Stroke, pos2, vec2};

use crate::FilmcraftApp;
use crate::dock::PanelKind;
use crate::state::CreatorTool;
use crate::theme::Tokens;

const GAP: f32 = 4.0;
const RAIL_W: f32 = 64.0;
const DRAWER_W: f32 = 264.0;
const INSPECTOR_W: f32 = 264.0;
const HEADER_H: f32 = 30.0;
const TIMELINE_BAR_H: f32 = 32.0;

pub fn show(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect) {
    let t = app.tokens;
    ui.painter().rect_filled(rect, 0.0, t.app_bg);

    if rect.width() < 240.0 || rect.height() < 200.0 {
        return;
    }

    let top_h = (rect.height() * 0.60).round().clamp(180.0, (rect.height() - 140.0).max(180.0));
    let top_rect = Rect::from_min_max(rect.min, pos2(rect.max.x, rect.min.y + top_h));
    let bot_rect = Rect::from_min_max(pos2(rect.min.x, top_rect.max.y + GAP), rect.max);

    let rail_rect = Rect::from_min_max(top_rect.min, pos2((top_rect.min.x + RAIL_W).min(top_rect.max.x), top_rect.max.y));
    draw_tool_rail(app, ui, rail_rect, &t);

    let mut left_x = rail_rect.max.x + GAP;
    if let Some(tool) = app.ui.creator_tool {
        let drawer_w = DRAWER_W.min((top_rect.width() * 0.27).max(190.0));
        let drawer_rect = Rect::from_min_max(pos2(left_x, top_rect.min.y), pos2((left_x + drawer_w).min(top_rect.max.x), top_rect.max.y));
        draw_tool_drawer(app, ui, drawer_rect, tool, &t);
        left_x = drawer_rect.max.x + GAP;
    }

    let has_selection = !app.session.state.selection.is_empty();
    let right_x = if has_selection && (top_rect.max.x - left_x) > 420.0 {
        let insp_w = INSPECTOR_W.min((top_rect.width() * 0.25).max(200.0));
        let insp_rect = Rect::from_min_max(pos2(top_rect.max.x - insp_w, top_rect.min.y), top_rect.max);
        draw_contextual_inspector(app, ui, insp_rect, &t);
        insp_rect.min.x - GAP
    } else {
        top_rect.max.x
    };

    let preview_rect = Rect::from_min_max(pos2(left_x, top_rect.min.y), pos2(right_x.max(left_x + 80.0), top_rect.max.y));
    draw_preview_pane(app, ui, preview_rect, &t);

    draw_bottom_timeline(app, ui, bot_rect, &t);
}

fn frame_panel(ui: &egui::Ui, rect: Rect, t: &Tokens, focused: bool) {
    let border = if focused { t.accent } else { t.control_border };
    ui.painter().rect_filled(rect, CornerRadius::same(6), t.panel_bg);
    ui.painter().rect_stroke(rect, CornerRadius::same(6), Stroke::new(1.0, border), egui::StrokeKind::Inside);
}

fn draw_tool_rail(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect, t: &Tokens) {
    frame_panel(ui, rect, t, false);
    let mut y = rect.min.y + 8.0;
    let items = [
        (CreatorTool::Media, "media", "[]", tl!("Media")),
        (CreatorTool::Text, "text", "T", tl!("Text")),
        (CreatorTool::Audio, "audio", "♪", tl!("Audio")),
        (CreatorTool::Ai, "ai", "", tl!("AI")),
    ];
    for (tool, id, glyph, label) in items {
        let active = app.ui.creator_tool == Some(tool);
        let btn = Rect::from_min_size(pos2(rect.min.x + 6.0, y), vec2((rect.width() - 12.0).max(20.0), 50.0));
        let resp = ui.interact(btn, egui::Id::unique(("creator-rail", id)), Sense::click());
        app.auto.add(&format!("creator.rail.{id}"), btn, label);
        let bg = if active {
            t.control_surface
        } else if resp.hovered() {
            t.hover
        } else {
            egui::Color32::TRANSPARENT
        };
        if bg != egui::Color32::TRANSPARENT {
            ui.painter().rect_filled(btn, 6.0, bg);
        }
        if active {
            ui.painter().rect_stroke(btn, 6.0, Stroke::new(1.0, t.accent), egui::StrokeKind::Inside);
        }
        let fg = if active || resp.hovered() { t.text } else { t.text_dim };
        let icon_center = pos2(btn.center().x, btn.min.y + 17.0);
        if tool == CreatorTool::Ai {
            crate::icons::paint(ui.painter(), Rect::from_center_size(icon_center, vec2(15.0, 15.0)), crate::icons::Icon::Sparkle, fg);
        } else {
            ui.painter().text(icon_center, egui::Align2::CENTER_CENTER, glyph, Tokens::semibold(14.0), fg);
        }
        ui.painter().text(pos2(btn.center().x, btn.max.y - 12.0), egui::Align2::CENTER_CENTER, label, Tokens::ui(10.5), fg);
        if resp.clicked() {
            app.ui.creator_tool = if active { None } else { Some(tool) };
        }
        y = btn.max.y + 6.0;
    }
}

fn draw_tool_drawer(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect, tool: CreatorTool, t: &Tokens) {
    let kind = match tool {
        CreatorTool::Media => PanelKind::Project,
        CreatorTool::Text => PanelKind::Text,
        CreatorTool::Audio => PanelKind::AudioClipMixer,
        CreatorTool::Ai => PanelKind::AiTools,
    };
    if ui.input(|i| i.pointer.any_pressed() && i.pointer.interact_pos().is_some_and(|p| rect.contains(p))) {
        app.ui.focused = kind;
    }
    frame_panel(ui, rect, t, app.ui.focused == kind);

    let hdr = Rect::from_min_max(rect.min, pos2(rect.max.x, (rect.min.y + HEADER_H).min(rect.max.y)));
    ui.painter().line_segment([hdr.left_bottom(), hdr.right_bottom()], Stroke::new(1.0, t.control_border));
    let title = match tool {
        CreatorTool::Media => tl!("Media"),
        CreatorTool::Text => tl!("Text"),
        CreatorTool::Audio => tl!("Audio"),
        CreatorTool::Ai => tl!("AI Tools"),
    };

    let col_r = Rect::from_center_size(pos2(hdr.max.x - 16.0, hdr.center().y), vec2(22.0, 22.0));
    let col_resp = ui.interact(col_r, egui::Id::unique("creator-drawer-close"), Sense::click()).on_hover_text(tl!("Close Panel"));
    app.auto.add("creator.drawer.close", col_r, tl!("Close Panel"));
    if col_resp.hovered() {
        ui.painter().rect_filled(col_r, 4.0, t.hover);
    }
    ui.painter().text(col_r.center(), egui::Align2::CENTER_CENTER, "◀", Tokens::ui(10.0), t.text_dim);
    if col_resp.clicked() {
        app.ui.creator_tool = None;
    }

    let mut title_max_x = col_r.min.x - 6.0;
    match tool {
        CreatorTool::Media => {
            let imp_label = format!("+ {}", tl!("Import"));
            let imp_g = ui.painter().layout_no_wrap(imp_label.clone(), Tokens::ui(11.0), egui::Color32::WHITE);
            let imp_w = imp_g.size().x + 14.0;
            let imp_r = Rect::from_min_size(pos2(col_r.min.x - imp_w - 6.0, hdr.center().y - 11.0), vec2(imp_w, 22.0));
            title_max_x = imp_r.min.x - 6.0;
            let imp_resp = ui.interact(imp_r, egui::Id::unique("creator-import"), Sense::click());
            app.auto.add("creator.import", imp_r, &imp_label);
            ui.painter().rect_filled(imp_r, 4.0, if imp_resp.hovered() { t.accent_hover } else { t.accent });
            ui.painter().galley(pos2(imp_r.min.x + 7.0, imp_r.center().y - imp_g.size().y / 2.0), imp_g, egui::Color32::WHITE);
            if imp_resp.clicked() {
                let ctx = ui.ctx().clone();
                if let Err(e) = crate::menus::invoke(app, &ctx, "file.import", serde_json::json!({})) {
                    app.ui.status = e;
                }
            }
        }
        CreatorTool::Text => {
            let txt_label = format!("+ {}", tl!("New Text"));
            let txt_g = ui.painter().layout_no_wrap(txt_label.clone(), Tokens::ui(11.0), egui::Color32::WHITE);
            let txt_w = txt_g.size().x + 14.0;
            let txt_r = Rect::from_min_size(pos2(col_r.min.x - txt_w - 6.0, hdr.center().y - 11.0), vec2(txt_w, 22.0));
            title_max_x = txt_r.min.x - 6.0;
            let txt_resp = ui.interact(txt_r, egui::Id::unique("creator-drawer-add-text"), Sense::click());
            app.auto.add("creator.drawer.addText", txt_r, &txt_label);
            ui.painter().rect_filled(txt_r, 4.0, if txt_resp.hovered() { t.accent_hover } else { t.accent });
            ui.painter().galley(pos2(txt_r.min.x + 7.0, txt_r.center().y - txt_g.size().y / 2.0), txt_g, egui::Color32::WHITE);
            if txt_resp.clicked() {
                let ctx = ui.ctx().clone();
                if let Err(e) = crate::menus::invoke(app, &ctx, "graphics.newText", serde_json::json!({ "text": "Text" })) {
                    app.ui.status = e;
                }
            }
        }
        CreatorTool::Audio | CreatorTool::Ai => {}
    }

    let title_clip = Rect::from_min_max(pos2(hdr.min.x + 8.0, hdr.min.y), pos2(title_max_x.max(hdr.min.x + 24.0), hdr.max.y));
    ui.painter().with_clip_rect(title_clip).text(pos2(hdr.min.x + 10.0, hdr.center().y), egui::Align2::LEFT_CENTER, title, Tokens::semibold(12.0), t.text);

    let body = Rect::from_min_max(pos2(rect.min.x + 2.0, hdr.max.y + 1.0), pos2(rect.max.x - 2.0, rect.max.y - 2.0));
    if body.height() > 10.0 {
        let mut child = ui.new_child(egui::UiBuilder::new().max_rect(body).id_salt(("creator-drawer-body", kind)));
        child.set_clip_rect(body);
        super::show(app, &mut child, kind, body);
    }
}

fn draw_preview_pane(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect, t: &Tokens) {
    if ui.input(|i| i.pointer.any_pressed() && i.pointer.interact_pos().is_some_and(|p| rect.contains(p))) {
        app.ui.focused = PanelKind::Program;
    }
    frame_panel(ui, rect, t, app.ui.focused == PanelKind::Program);

    let hdr = Rect::from_min_max(rect.min, pos2(rect.max.x, (rect.min.y + HEADER_H).min(rect.max.y)));
    ui.painter().line_segment([hdr.left_bottom(), hdr.right_bottom()], Stroke::new(1.0, t.control_border));
    ui.painter().text(pos2(hdr.min.x + 12.0, hdr.center().y), egui::Align2::LEFT_CENTER, tl!("Preview"), Tokens::semibold(12.0), t.text);

    let mut rx = hdr.max.x - 8.0;
    let txt_label = format!("+ {}", tl!("New Text"));
    let txt_g = ui.painter().layout_no_wrap(txt_label.clone(), Tokens::ui(11.0), t.text);
    let txt_w = txt_g.size().x + 14.0;
    let txt_r = Rect::from_min_size(pos2(rx - txt_w, hdr.center().y - 11.0), vec2(txt_w, 22.0));
    let txt_resp = ui.interact(txt_r, egui::Id::unique("creator-add-text"), Sense::click());
    app.auto.add("creator.addText", txt_r, &txt_label);
    ui.painter().rect_filled(txt_r, 4.0, if txt_resp.hovered() { t.hover } else { t.control_surface });
    ui.painter().rect_stroke(txt_r, 4.0, Stroke::new(1.0, t.control_border), egui::StrokeKind::Inside);
    ui.painter().galley(pos2(txt_r.min.x + 7.0, txt_r.center().y - txt_g.size().y / 2.0), txt_g, t.text);
    if txt_resp.clicked() {
        let ctx = ui.ctx().clone();
        if let Err(e) = crate::menus::invoke(app, &ctx, "graphics.newText", serde_json::json!({ "text": "Text" })) {
            app.ui.status = e;
        }
    }
    rx = txt_r.min.x - 6.0;

    if app.session.state.active_sequence.is_none() {
        let seq_label = "+ 9:16";
        let seq_g = ui.painter().layout_no_wrap(seq_label.to_string(), Tokens::ui(11.0), egui::Color32::WHITE);
        let seq_w = seq_g.size().x + 14.0;
        let seq_r = Rect::from_min_size(pos2(rx - seq_w, hdr.center().y - 11.0), vec2(seq_w, 22.0));
        let seq_resp = ui.interact(seq_r, egui::Id::unique("creator-new-vertical"), Sense::click());
        app.auto.add("creator.newVertical", seq_r, seq_label);
        ui.painter().rect_filled(seq_r, 4.0, if seq_resp.hovered() { t.accent_hover } else { t.accent });
        ui.painter().galley(pos2(seq_r.min.x + 7.0, seq_r.center().y - seq_g.size().y / 2.0), seq_g, egui::Color32::WHITE);
        if seq_resp.clicked()
            && let Err(e) = app.session.execute("file.newSequence", serde_json::json!({ "name": "Shorts", "width": 1080, "height": 1920, "fps": 30.0 }))
        {
            app.ui.status = e.to_string();
        }
    }

    let body = Rect::from_min_max(pos2(rect.min.x + 2.0, hdr.max.y + 1.0), pos2(rect.max.x - 2.0, rect.max.y - 2.0));
    if body.height() > 10.0 {
        let mut child = ui.new_child(egui::UiBuilder::new().max_rect(body).id_salt("creator-program-body"));
        child.set_clip_rect(body);
        super::show(app, &mut child, PanelKind::Program, body);
    }
}

fn selected_is_graphic(app: &FilmcraftApp) -> bool {
    let Some(seq) = app.session.active_sequence() else {
        return false;
    };
    app.session.state.selection.iter().any(|&cid| seq.find_item(cid).is_some_and(|(_, clip)| clip.graphic.is_some()))
}

fn draw_contextual_inspector(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect, t: &Tokens) {
    let is_graphic = selected_is_graphic(app);
    let kind = if is_graphic { PanelKind::Text } else { PanelKind::Properties };
    if ui.input(|i| i.pointer.any_pressed() && i.pointer.interact_pos().is_some_and(|p| rect.contains(p))) {
        app.ui.focused = kind;
    }
    frame_panel(ui, rect, t, app.ui.focused == kind);

    let hdr = Rect::from_min_max(rect.min, pos2(rect.max.x, (rect.min.y + HEADER_H).min(rect.max.y)));
    ui.painter().line_segment([hdr.left_bottom(), hdr.right_bottom()], Stroke::new(1.0, t.control_border));
    let title = if is_graphic { tl!("Text") } else { tl!("Properties") };
    ui.painter().text(pos2(hdr.min.x + 10.0, hdr.center().y), egui::Align2::LEFT_CENTER, title, Tokens::semibold(12.0), t.text);

    let body = Rect::from_min_max(pos2(rect.min.x + 2.0, hdr.max.y + 1.0), pos2(rect.max.x - 2.0, rect.max.y - 2.0));
    if body.height() > 10.0 {
        let mut child = ui.new_child(egui::UiBuilder::new().max_rect(body).id_salt(("creator-insp-body", kind)));
        child.set_clip_rect(body);
        super::show(app, &mut child, kind, body);
    }
}

fn draw_bottom_timeline(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect, t: &Tokens) {
    if rect.width() < 16.0 || rect.height() < 48.0 {
        return;
    }
    if ui.input(|i| i.pointer.any_pressed() && i.pointer.interact_pos().is_some_and(|p| rect.contains(p))) {
        app.ui.focused = PanelKind::Timeline;
    }
    frame_panel(ui, rect, t, app.ui.focused == PanelKind::Timeline);

    let bar = Rect::from_min_max(pos2(rect.min.x + 2.0, rect.min.y + 2.0), pos2(rect.max.x - 2.0, (rect.min.y + TIMELINE_BAR_H).min(rect.max.y)));
    ui.painter().line_segment([bar.left_bottom(), bar.right_bottom()], Stroke::new(1.0, t.control_border));

    let delete_cmd =
        if filmcraft_engine::find_command("edit.rippleDelete").is_some_and(|c| (c.enabled)(&app.session).is_ok()) { "edit.rippleDelete" } else { "edit.clear" };
    let actions = [
        ("split", "sequence.addEdit", tl!("Split")),
        ("delete", delete_cmd, tl!("Delete")),
        ("undo", "edit.undo", tl!("Undo")),
        ("redo", "edit.redo", tl!("Redo")),
        ("removePauses", "ai.removePauses", tl!("Remove Pauses")),
    ];
    let mut x = bar.min.x + 8.0;
    for (id, cmd, label) in actions {
        let enabled = filmcraft_engine::find_command(cmd).is_some_and(|c| (c.enabled)(&app.session).is_ok());
        let fg = if enabled { t.text } else { t.text_dim };
        let g = ui.painter().layout_no_wrap(label.to_string(), Tokens::ui(11.5), fg);
        let btn = Rect::from_min_size(pos2(x, bar.center().y - 11.0), vec2(g.size().x + 16.0, 22.0));
        let sense = if enabled { Sense::click() } else { Sense::hover() };
        let resp = ui.interact(btn, egui::Id::unique(("creator-tl-act", id)), sense);
        app.auto.add(&format!("creator.action.{id}"), btn, label);
        let bg = if enabled && resp.hovered() { t.hover } else { t.control_surface };
        ui.painter().rect_filled(btn, 4.0, bg);
        ui.painter().rect_stroke(btn, 4.0, Stroke::new(1.0, t.control_border), egui::StrokeKind::Inside);
        ui.painter().galley(pos2(btn.min.x + 8.0, btn.center().y - g.size().y / 2.0), g, fg);
        if enabled && resp.clicked() {
            let ctx = ui.ctx().clone();
            if let Err(e) = crate::menus::invoke(app, &ctx, cmd, serde_json::json!({})) {
                app.ui.status = e;
            }
        }
        x = btn.max.x + 6.0;
    }

    // "AI Tools" button opening the AI Tools drawer
    let ai_open = app.ui.creator_tool == Some(CreatorTool::Ai);
    let ai_label = tl!("AI Tools");
    let ai_fg = if ai_open { egui::Color32::WHITE } else { t.text };
    let ai_g = ui.painter().layout_no_wrap(ai_label.to_string(), Tokens::semibold(11.5), ai_fg);
    let ai_btn = Rect::from_min_size(pos2(x, bar.center().y - 11.0), vec2(ai_g.size().x + 18.0, 22.0));
    let ai_resp = ui.interact(ai_btn, egui::Id::unique("creator-tl-ai-tools"), Sense::click());
    app.auto.add("creator.action.aiTools", ai_btn, ai_label);
    let ai_bg = if ai_open {
        t.accent
    } else if ai_resp.hovered() {
        t.hover
    } else {
        t.control_surface
    };
    ui.painter().rect_filled(ai_btn, 4.0, ai_bg);
    ui.painter().rect_stroke(ai_btn, 4.0, Stroke::new(1.0, if ai_open { t.accent } else { t.control_border }), egui::StrokeKind::Inside);
    ui.painter().galley(pos2(ai_btn.min.x + 9.0, ai_btn.center().y - ai_g.size().y / 2.0), ai_g, ai_fg);
    if ai_resp.clicked() {
        app.ui.creator_tool = if ai_open { None } else { Some(CreatorTool::Ai) };
    }
    x = ai_btn.max.x + 6.0;

    // Quick action button: "AI Subtitles" (runs generation directly if ready, else opens AI drawer)
    let sub_enabled = app.session.active_sequence().is_some();
    let sub_label = tl!("AI Subtitles");
    let sub_fg = if sub_enabled { t.text } else { t.text_dim };
    let sub_g = ui.painter().layout_no_wrap(sub_label.to_string(), Tokens::semibold(11.5), sub_fg);
    let sub_btn = Rect::from_min_size(pos2(x, bar.center().y - 11.0), vec2(sub_g.size().x + 18.0, 22.0));
    if sub_btn.max.x <= bar.max.x - 4.0 {
        let sub_resp = ui.interact(sub_btn, egui::Id::unique("creator-tl-ai-subtitles"), if sub_enabled { Sense::click() } else { Sense::hover() });
        app.auto.add("creator.action.subtitles", sub_btn, sub_label);
        if let Some(el) = app.auto.elements.last_mut() {
            el.enabled = Some(sub_enabled);
        }
        let sub_bg = if sub_enabled && sub_resp.hovered() { t.hover } else { t.control_surface };
        ui.painter().rect_filled(sub_btn, 4.0, sub_bg);
        ui.painter().rect_stroke(sub_btn, 4.0, Stroke::new(1.0, t.control_border), egui::StrokeKind::Inside);
        ui.painter().galley(pos2(sub_btn.min.x + 9.0, sub_btn.center().y - sub_g.size().y / 2.0), sub_g, sub_fg);
        if sub_enabled && sub_resp.clicked() {
            let ctx = ui.ctx().clone();
            if let Err(e) = crate::menus::invoke(app, &ctx, "ai.generateSubtitles", serde_json::json!({})) {
                app.ui.status = e;
                app.ui.creator_tool = Some(CreatorTool::Ai);
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
