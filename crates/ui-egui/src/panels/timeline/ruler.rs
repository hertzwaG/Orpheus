use super::layout::Layout;
use super::{
    Align2, Color32, Drag, EDGE_PX, FilmcraftApp, Icon, ItemId, Rect, Sense, Sequence, Stroke, StrokeKind, ThumbnailMode, Tick, TimeRange, Tokens, Tool, Value,
    format_time, icons, json, marker_range_rect, marker_rect, pos2, vec2, work_area_rect,
};
use crate::panels::timeline_hit::marker_end_rect;
use crate::panels::tools::{draw_tool_button, tool_tooltip};

/// The Ruler's two strides, in frames: `(tick, label)`.
///
/// `label` is the first ladder step wide enough to keep timestamps apart (110 pt); `tick` is the
/// finest divisor of `label` that still leaves ticks 12 pt apart. Deriving the tick stride as a
/// divisor puts every label on a tick and keeps the ticks between labels evenly spaced.
pub fn ruler_steps(frame_px: f64, base: i64) -> (i64, i64) {
    // `FrameRate` is plain serde data, so a step past `base` can overflow; saturate instead.
    let big = |k: i64| base.saturating_mul(k);
    let steps = [1, 2, 5, 10, base / 2, base, big(2), big(5), big(10), big(15), big(30), big(60), big(120), big(300), big(600), big(1800), big(3600)];
    let label_step = (*steps.iter().find(|s| **s as f64 * frame_px >= 110.0).unwrap_or(&big(3600))).max(1);
    // Labels are at least 110 pt apart, so a divisor no larger than 9 keeps ticks >= 12 pt.
    let minor =
        (1..=9).rev().filter(|k| label_step.rem_euclid(*k) == 0).map(|k| label_step / k).find(|m| *m as f64 * frame_px >= 12.0).unwrap_or(label_step).max(1);
    (minor, label_step)
}

/// Draw the unified horizontal timeline toolbar (timecode | tools | snapping | zoom | settings | More)
/// above the calm time ruler.
pub(super) fn draw_top(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect, seq: &Sequence, layout: &Layout, t: &Tokens, seq_id: ItemId) {
    let p = ui.painter().clone();
    let ruler = layout.ruler;
    if app.ui.experience == crate::state::AppExperience::Creator {
        // Creator Mode: clean timecode pill beside the ruler (no dense Studio 15-tool strip).
        let tc_box = Rect::from_min_max(rect.min, pos2(ruler.min.x, ruler.max.y));
        p.rect_filled(tc_box, 0.0, t.tl_header_bg);
        p.line_segment([pos2(tc_box.max.x - 0.5, tc_box.min.y), pos2(tc_box.max.x - 0.5, tc_box.max.y)], Stroke::new(1.0, t.separator));
        p.line_segment([tc_box.left_bottom(), tc_box.right_bottom()], Stroke::new(1.0, t.separator));
        let rate = seq.settings.frame_rate;
        let tc = format_time(app.session.playhead(), rate, seq.settings.drop_frame, seq.settings.display, seq.settings.sample_rate as i64);
        p.text(tc_box.center(), Align2::CENTER_CENTER, &tc, Tokens::semibold(11.5), t.timecode);
        app.auto.add("timeline.timecode", tc_box, &tc);
        draw_ruler_body(app, ui, &p, seq, layout, t, seq_id);
        return;
    }
    p.rect_filled(Rect::from_min_max(rect.min, pos2(rect.max.x, ruler.min.y)), 0.0, t.panel_bg);
    let strip = Rect::from_min_max(pos2(rect.min.x + 8.0, rect.min.y + 3.0), pos2(rect.max.x - 8.0, ruler.min.y - 3.0));
    draw_toolbar(app, ui, &p, strip, seq, t, seq_id);
    draw_ruler_body(app, ui, &p, seq, layout, t, seq_id);
}

fn draw_toolbar(app: &mut FilmcraftApp, ui: &mut egui::Ui, p: &egui::Painter, strip: Rect, seq: &Sequence, t: &Tokens, seq_id: ItemId) {
    let ctx = ui.ctx().clone();
    let rate = seq.settings.frame_rate;
    p.rect_filled(strip, 6.0, t.control_surface);
    p.rect_stroke(strip, 6.0, Stroke::new(1.0, t.separator), StrokeKind::Inside);

    let cy = strip.center().y;
    let btn_s = 24.0;
    let gap = 2.0;

    // 1. Left: editable playhead timecode
    let mut left_x = strip.min.x + 8.0;
    let tc = format_time(app.session.playhead(), rate, seq.settings.drop_frame, seq.settings.display, seq.settings.sample_rate as i64);
    let tc_w = (96.0_f32).min((strip.width() - 120.0).max(60.0));
    let tc_rect = Rect::from_min_size(pos2(left_x, cy - 11.0), vec2(tc_w, 22.0));
    let tc_id = egui::Id::unique(("timeline-tc", seq_id));
    match crate::widgets::timecode_field(
        ui,
        tc_id,
        tc_rect,
        &tc,
        app.session.playhead(),
        Tick::ZERO,
        rate,
        seq.settings.drop_frame,
        seq.settings.display,
        t.timecode,
    ) {
        Some(Ok(time)) => {
            if let Err(e) = app.session.execute("playhead.set", json!({"time": time.0})) {
                app.ui.status = e.to_string();
            }
        }
        Some(Err(e)) => app.ui.status = e,
        None => {}
    }
    app.auto.add("timeline.timecode", tc_rect, &tc);
    left_x = tc_rect.max.x + 8.0;

    // 2. Right: Settings wrench + Zoom controls (Out / Fit / In)
    let mut right_x = strip.max.x - 6.0;
    let settings_r = Rect::from_min_size(pos2(right_x - btn_s, cy - btn_s / 2.0), vec2(btn_s, btn_s));
    let sresp = ui.interact(settings_r, egui::Id::unique(("tl-toggle", "settings")), Sense::click()).on_hover_text(tl!("Timeline Display Settings"));
    app.auto.add("timeline.toggle.settings", settings_r, tl!("Timeline Display Settings"));
    if sresp.hovered() {
        p.rect_filled(settings_r, 4.0, t.hover);
    }
    icons::paint(p, settings_r.shrink(5.0), Icon::Wrench, if sresp.hovered() { t.text } else { t.text_dim });
    egui::Popup::menu(&sresp).close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside).show(|ui| timeline_settings(app, ui));
    right_x = settings_r.min.x - 4.0;

    let zoom_cmds: [(&str, &str, &str); 3] =
        [("+", "view.zoomIn", tl!("Zoom In")), ("[]", "view.zoomToSequence", tl!("Zoom to Sequence")), ("-", "view.zoomOut", tl!("Zoom Out"))];
    for (glyph, cmd, tip) in zoom_cmds {
        if right_x - btn_s <= left_x + 40.0 {
            break;
        }
        let r = Rect::from_min_size(pos2(right_x - btn_s, cy - btn_s / 2.0), vec2(btn_s, btn_s));
        let resp = ui.interact(r, egui::Id::unique(("tl-zoom", cmd)), Sense::click()).on_hover_text(tip);
        app.auto.add(&format!("timeline.zoom.{cmd}"), r, tip);
        if resp.hovered() {
            p.rect_filled(r, 4.0, t.hover);
        }
        let col = if resp.hovered() { t.text } else { t.text_dim };
        if glyph == "[]" {
            icons::paint(p, r.shrink(5.0), Icon::Fullscreen, col);
        } else {
            p.text(r.center(), Align2::CENTER_CENTER, glyph, Tokens::semibold(13.5), col);
        }
        if resp.clicked() {
            let _ = crate::menus::invoke(app, &ctx, cmd, json!({}));
        }
        right_x = r.min.x - gap;
    }
    right_x -= 4.0;
    if right_x > left_x + 32.0 {
        p.line_segment([pos2(right_x, cy - 9.0), pos2(right_x, cy + 9.0)], Stroke::new(1.0, t.separator));
        right_x -= 6.0;
    }

    draw_toolbar_middle(app, ui, p, strip, left_x, right_x, btn_s, gap, t);
}

#[allow(clippy::too_many_arguments)]
fn draw_toolbar_middle(
    app: &mut FilmcraftApp,
    ui: &mut egui::Ui,
    p: &egui::Painter,
    strip: Rect,
    mut left_x: f32,
    right_x: f32,
    btn_s: f32,
    gap: f32,
    t: &Tokens,
) {
    let cy = strip.center().y;
    let held_id = egui::Id::unique("tools-held");
    if ui.input(|i| i.pointer.any_pressed()) {
        ui.data_mut(|d| d.remove::<egui::Id>(held_id));
    }
    let groups = Tool::groups();
    let toggles: [(Icon, &str, bool, bool, &str); 5] = [
        (Icon::Magnet, "snap", app.session.state.snapping, true, tl!("Snap in Timeline (S)")),
        (Icon::Link, "linked", app.session.state.linked_selection, true, tl!("Linked Selection")),
        (Icon::Marker, "marker", false, false, tl!("Add Marker (M)")),
        (Icon::Nest, "nest", !app.session.state.sequences_as_clips, true, tl!("Insert and overwrite sequences as nests or individual clips")),
        (Icon::Captions, "captions", false, false, tl!("Caption track options")),
    ];

    if left_x + 40.0 < right_x {
        p.line_segment([pos2(left_x, cy - 9.0), pos2(left_x, cy + 9.0)], Stroke::new(1.0, t.separator));
        left_x += 6.0;
    }

    let total_needed = groups.len() as f32 * (btn_s + gap) + 12.0 + toggles.len() as f32 * (btn_s + gap);
    let avail = (right_x - left_x).max(0.0);
    let needs_more = total_needed > avail;
    let usable_right = if needs_more { (right_x - btn_s - 4.0).max(left_x) } else { right_x };

    let mut overflow_tools: Vec<Vec<Tool>> = Vec::new();
    let mut shown_tools: Vec<Vec<Tool>> = Vec::new();
    let tool_budget = if needs_more { (usable_right - left_x - 3.0 * (btn_s + gap) - 12.0).max(btn_s) } else { usable_right - left_x };
    let mut used_tool_w = 0.0;
    for group in groups {
        let is_rare = matches!(group[0], Tool::Pen | Tool::Zoom);
        if (!needs_more || !is_rare) && used_tool_w + btn_s <= tool_budget {
            shown_tools.push(group);
            used_tool_w += btn_s + gap;
        } else {
            overflow_tools.push(group);
        }
    }
    for group in shown_tools {
        let r = Rect::from_min_size(pos2(left_x, cy - btn_s / 2.0), vec2(btn_s, btn_s));
        draw_tool_button(app, ui, r, &group, held_id, 15.0);
        left_x += btn_s + gap;
    }

    if left_x + 10.0 < usable_right {
        left_x += 4.0;
        p.line_segment([pos2(left_x, cy - 9.0), pos2(left_x, cy + 9.0)], Stroke::new(1.0, t.separator));
        left_x += 6.0;
    }

    let mut overflow_toggles: Vec<(Icon, &str, bool, bool, &str)> = Vec::new();
    for (icon, key, on, toggle, tip) in toggles {
        if left_x + btn_s <= usable_right {
            let r = Rect::from_min_size(pos2(left_x, cy - btn_s / 2.0), vec2(btn_s, btn_s));
            let resp = ui.interact(r, egui::Id::unique(("tl-toggle", key)), Sense::click()).on_hover_text(tip);
            app.auto.add(&format!("timeline.toggle.{key}"), r, tip);
            if toggle && on {
                p.rect_filled(r, 4.0, t.pressed);
            } else if resp.hovered() {
                p.rect_filled(r, 4.0, t.hover);
            }
            icons::paint(p, r.shrink(5.0), icon, if toggle && on { t.text } else { t.text_dim });
            if resp.clicked() {
                run_timeline_toggle(app, key);
            }
            left_x += btn_s + gap;
        } else {
            overflow_toggles.push((icon, key, on, toggle, tip));
        }
    }

    if !overflow_tools.is_empty() || !overflow_toggles.is_empty() {
        let more_r = Rect::from_min_size(pos2(right_x - btn_s, cy - btn_s / 2.0), vec2(btn_s, btn_s));
        let resp = ui.interact(more_r, egui::Id::unique(("tl-toolbar", "more")), Sense::click()).on_hover_text(tl!("More"));
        app.auto.add_response("timeline.toolbar.more", &resp, tl!("More"));
        if resp.hovered() {
            p.rect_filled(more_r, 4.0, t.hover);
        }
        for offset in [-5.0, 0.0, 5.0] {
            p.circle_filled(more_r.center() + vec2(offset, 0.0), 1.6, t.icon);
        }
        egui::Popup::menu(&resp).show(|ui| {
            for group in &overflow_tools {
                for &tl in group {
                    let row = ui.selectable_label(app.ui.tool == tl, tool_tooltip(app, tl));
                    app.auto.add_response(&format!("tools.select.{tl:?}"), &row, tl.label());
                    if row.clicked() {
                        app.ui.tool = tl;
                        ui.close();
                    }
                }
            }
            if !overflow_tools.is_empty() && !overflow_toggles.is_empty() {
                ui.separator();
            }
            for &(_icon, key, on, toggle, tip) in &overflow_toggles {
                let row = if toggle { ui.selectable_label(on, tip) } else { ui.button(tip) };
                app.auto.add_response(&format!("timeline.toggle.{key}"), &row, tip);
                if row.clicked() {
                    run_timeline_toggle(app, key);
                    ui.close();
                }
            }
        });
    }
}

fn draw_ruler_body(app: &mut FilmcraftApp, ui: &mut egui::Ui, p: &egui::Painter, seq: &Sequence, layout: &Layout, t: &Tokens, seq_id: ItemId) {
    let rate = seq.settings.frame_rate;
    let ruler = layout.ruler;
    // ----- calm time ruler: markers row · labels · thin ticks · 2 pt render bar
    p.rect_filled(ruler, 0.0, t.tl_bg);
    p.line_segment([pos2(ruler.min.x, ruler.max.y - 0.5), pos2(ruler.max.x, ruler.max.y - 0.5)], Stroke::new(1.0, t.separator));
    let clip = p.with_clip_rect(ruler);
    let label_y = if ruler.height() >= 44.0 { ruler.min.y + 30.0 } else { ruler.min.y + 20.0 };
    let tick_base = ruler.max.y - 3.0;
    let (minor, label_step) = ruler_steps(layout.pps / rate.as_f64(), rate.timecode_base());
    // In/Out band
    if seq.mark_in.is_some() || seq.mark_out.is_some() {
        let a = layout.x_of(seq.mark_in.unwrap_or(Tick::ZERO));
        let b = layout.x_of(seq.mark_out.map(|o| o + rate.frame_duration()).unwrap_or(seq.duration()));
        clip.rect_filled(Rect::from_min_max(pos2(a, label_y + 6.0), pos2(b, tick_base)), 0.0, Color32::from_rgb(0x4a, 0x52, 0x60));
    }
    // split points: a short bracket with a V or A tag (video above, audio below)
    let sp = seq.split;
    for (t, is_in, tag, low) in
        [(sp.video_in, true, "V", false), (sp.video_out, false, "V", false), (sp.audio_in, true, "A", true), (sp.audio_out, false, "A", true)]
    {
        let Some(t) = t else { continue };
        let x = layout.x_of(if is_in { t } else { t + rate.frame_duration() });
        let y0 = if low { tick_base - 7.0 } else { label_y + 6.0 };
        let col = Color32::from_rgb(0xd0, 0xd0, 0xd0);
        let dx = if is_in { 4.0 } else { -4.0 };
        clip.line_segment([pos2(x, y0), pos2(x, y0 + 7.0)], Stroke::new(1.5, col));
        clip.line_segment([pos2(x, y0), pos2(x + dx, y0)], Stroke::new(1.5, col));
        clip.text(pos2(x + 2.0 * dx, y0 + 3.5), Align2::CENTER_CENTER, tag, Tokens::ui(8.5), col);
        let kind = if low { "audio" } else { "video" };
        let side = if is_in { "In" } else { "Out" };
        app.auto.add(&format!("timeline.split.{kind}{side}"), Rect::from_center_size(pos2(x, y0 + 3.5), vec2(12.0, 9.0)), &format!("Split {tag} {side}"));
    }
    let f0 = rate.frame_at(layout.tick_at(ruler.min.x)).max(0);
    let f1 = rate.frame_at(layout.tick_at(ruler.max.x)) + 1;
    let mut f = (f0 / minor) * minor;
    let minor_tick_col = Color32::from_rgba_unmultiplied(t.tl_ruler_tick.r(), t.tl_ruler_tick.g(), t.tl_ruler_tick.b(), 145);
    while f <= f1 {
        let x = layout.x_of(rate.tick_of(f));
        let major = f % label_step == 0;
        let h = if major { 8.0 } else { 3.5 };
        clip.line_segment([pos2(x, tick_base - h), pos2(x, tick_base)], Stroke::new(1.0, if major { t.tl_ruler_tick } else { minor_tick_col }));
        if major {
            let label = format_time(rate.tick_of(f), rate, seq.settings.drop_frame, seq.settings.display, 48000);
            clip.text(pos2(x + 1.0, label_y), Align2::CENTER_CENTER, label, Tokens::ui(10.5), t.tl_ruler_text);
        }
        f += minor;
    }
    // render bar
    // Per segment (engine `previews`): green = rendered preview, yellow = should play in real time,
    // red = needs rendering, nothing = plays natively.
    let rb = Rect::from_min_max(pos2(ruler.min.x, ruler.max.y - 2.0), ruler.max);
    let bar = app.session.previews.bar(&app.session.project, seq_id);
    for (i, span) in bar.iter().enumerate() {
        use filmcraft_engine::previews::BarState;
        let (c, what) = match span.state {
            BarState::None => continue,
            BarState::Yellow => (t.render_yellow, tl!("Unrendered: should play back in real time")),
            BarState::Red => (t.render_red, tl!("Unrendered: render to play back in real time")),
            BarState::Green => (t.render_green, tl!("Rendered preview")),
        };
        let r = Rect::from_min_max(pos2(layout.x_of(span.start), rb.min.y), pos2(layout.x_of(span.end), rb.max.y)).intersect(ruler);
        if r.width() <= 0.0 {
            continue;
        }
        clip.rect_filled(r, 0.0, c);
        let hit = r.expand2(vec2(0.0, 2.0));
        app.auto.add(&format!("timeline.renderBar.{i}"), hit, what);
        if ui.rect_contains_pointer(hit) {
            egui::Tooltip::always_open(ui.ctx().clone(), ui.layer_id(), egui::Id::unique(("rb", i)), egui::PopupAnchor::Pointer).show(|ui| {
                ui.label(what);
            });
        }
    }
    // work area bar: between the markers row and the time labels, as dragged while a drag of it runs
    let shown = match app.tl.drag {
        Some(Drag::WorkArea { start, end, .. }) => Some(TimeRange::from_bounds(start, end)),
        _ => seq.work_area,
    };
    if let Some(area) = shown {
        let r = work_area_rect(layout, area);
        clip.rect_filled(r, 1.0, t.text_dim);
        for x in [r.min.x, r.max.x] {
            clip.rect_filled(Rect::from_center_size(pos2(x, r.center().y), vec2(3.0, r.height() + 2.0)), 1.0, t.text);
        }
        app.auto.add("timeline.workArea", r.intersect(ruler), "Work Area Bar");
        let fmt = |tick: Tick| format_time(tick, rate, seq.settings.drop_frame, seq.settings.display, 48000);
        if ui.rect_contains_pointer(r.expand2(vec2(EDGE_PX, 1.0)).intersect(ruler)) {
            egui::Tooltip::always_open(ui.ctx().clone(), ui.layer_id(), egui::Id::unique("work-area-tip"), egui::PopupAnchor::Pointer).show(|ui| {
                ui.label(tlf!("Start: {tc}", tc = fmt(area.start)));
                ui.label(tlf!("End: {tc}", tc = fmt(area.end())));
                ui.label(tlf!("Duration: {tc}", tc = fmt(area.duration)));
            });
        }
    }
    // markers: 8 × 12 pt pentagons in the marker colour, on the top row
    draw_markers(app, ui, &clip, layout, seq);
    app.auto.add("timeline.ruler", ruler, "time ruler");
}

/// The ruler's marker row: one pentagon per marker in its colour, its range band behind it and,
/// on hover, its name.
fn draw_markers(app: &mut FilmcraftApp, ui: &egui::Ui, clip: &egui::Painter, layout: &Layout, seq: &Sequence) {
    let ruler = layout.ruler;
    let marker_y = ruler.min.y + 2.0;
    // Point flags stay whole; ranges split the same glyph across their endpoints.
    for m in &seq.markers {
        let x = layout.x_of(m.start);
        let c = m.custom_color.map_or_else(|| app.session.prefs.labels.rgb(m.color), |c| c.0);
        let c = Color32::from_rgb(c[0], c[1], c[2]);
        let glyph = |x| Rect::from_min_size(pos2(x - 4.0, marker_y), vec2(8.0, 12.0));
        let is_chapter = m.kind == filmcraft_project::MarkerKind::Chapter;
        let mut pill_hovered = false;
        if let Some(band) = marker_range_rect(layout, m).filter(|r| r.width() > 0.0) {
            if is_chapter {
                let pill = Rect::from_min_max(pos2(band.min.x, marker_y), pos2(band.max.x, (marker_y + 20.0).min(ruler.max.y - 2.0)));
                clip.rect_filled(pill, 4.0, c.gamma_multiply(0.78));
                clip.rect_stroke(pill, 4.0, egui::Stroke::new(1.25, c), egui::StrokeKind::Inside);
                // keep the title readable while the chapter's start is scrolled off to the left
                let visible = pill.intersect(ruler);
                pill_hovered = ui.rect_contains_pointer(visible);
                if visible.width() >= 24.0 && !m.name.is_empty() {
                    let text_clip = clip.with_clip_rect(visible.shrink2(vec2(6.0, 0.0)));
                    let text_pos = pos2(visible.min.x + 8.0, pill.center().y);
                    let font = Tokens::semibold(12.5);
                    text_clip.text(text_pos + vec2(0.0, 1.0), Align2::LEFT_CENTER, &m.name, font.clone(), Color32::from_rgba_unmultiplied(0, 0, 0, 170));
                    text_clip.text(text_pos, Align2::LEFT_CENTER, &m.name, font, Color32::WHITE);
                }
            } else {
                clip.rect_filled(band, 0.0, c.gamma_multiply(0.6));
            }
            app.auto.add(&format!("timeline.markerRange.{}", m.id.0), band, &m.name);
        }
        let end = marker_end_rect(layout, m);
        if !is_chapter {
            if let Some(end) = end {
                crate::widgets::marker_glyph(clip, glyph(x), 4.0, crate::widgets::MarkerPart::Left, c);
                crate::widgets::marker_glyph(clip, glyph(end.center().x), 4.0, crate::widgets::MarkerPart::Right, c);
                let visible = end.intersect(ruler);
                if visible.is_positive() {
                    app.auto.add(&format!("timeline.markerEnd.{}", m.id.0), visible, &m.name);
                }
            } else {
                crate::widgets::marker_glyph(clip, glyph(x), 4.0, crate::widgets::MarkerPart::Whole, c);
            }
        } else if let Some(end) = end {
            let visible = end.intersect(ruler);
            if visible.is_positive() {
                app.auto.add(&format!("timeline.markerEnd.{}", m.id.0), visible, &m.name);
            }
        }
        let mr = marker_rect(layout, m.start);
        app.auto.add(&format!("timeline.marker.{}", m.id.0), mr, &m.name);
        if (pill_hovered || ui.rect_contains_pointer(mr) || end.is_some_and(|r| ui.rect_contains_pointer(r))) && !m.name.is_empty() {
            egui::Tooltip::always_open(ui.ctx().clone(), ui.layer_id(), egui::Id::unique(("mk", m.id.0)), egui::PopupAnchor::Pointer).show(|ui| {
                ui.label(&m.name);
            });
        }
    }
}

/// The Timeline Display Settings popup from the ruler toolbar: thumbnails, waveforms, edit
/// markers and the track-height presets.
fn timeline_settings(app: &mut FilmcraftApp, ui: &mut egui::Ui) {
    let c = ui.checkbox(&mut app.ui.timeline.show_thumbnails, tl!("Show Video Thumbnails"));
    app.auto.add("timeline.settings.showThumbnails", c.rect, "Show Video Thumbnails");
    let on = app.ui.timeline.show_thumbnails;
    ui.indent("timeline.settings.thumbnails", |ui| {
        for (mode, label) in
            [(ThumbnailMode::Head, tl!("Head")), (ThumbnailMode::HeadAndTail, tl!("Head and Tail")), (ThumbnailMode::Continuous, tl!("Continuous"))]
        {
            let r = ui.add_enabled(on, egui::RadioButton::new(app.ui.timeline.thumbnail_mode == mode, label));
            app.auto.add(&format!("timeline.settings.thumbnails.{}", mode.name()), r.rect, label);
            if r.clicked() {
                app.ui.timeline.thumbnail_mode = mode;
            }
        }
    });
    ui.checkbox(&mut app.ui.timeline.show_waveforms, tl!("Show Audio Waveform"));
    let c = ui.checkbox(&mut app.ui.timeline.show_video_names, tl!("Show Video Names"));
    app.auto.add("timeline.settings.showVideoNames", c.rect, "Show Video Names");
    let c = ui.checkbox(&mut app.ui.timeline.show_audio_names, tl!("Show Audio Names"));
    app.auto.add("timeline.settings.showAudioNames", c.rect, "Show Audio Names");
    ui.separator();
    let mut te = app.session.state.show_through_edits;
    let c = ui.checkbox(&mut te, tl!("Show Through Edits"));
    app.auto.add("timeline.settings.showThroughEdits", c.rect, "Show Through Edits");
    if c.changed() {
        let _ = app.session.execute("sequence.showThroughEdits", json!({"on": te}));
    }
    let mut df = app.session.state.show_duplicate_frame_markers;
    let c = ui.checkbox(&mut df, tl!("Show Duplicate Frame Markers"));
    app.auto.add("timeline.settings.showDuplicateFrameMarkers", c.rect, "Show Duplicate Frame Markers");
    if c.changed() {
        let _ = app.session.execute("sequence.showDuplicateFrameMarkers", json!({"on": df}));
    }
    ui.separator();
    if ui.button(tl!("Expand All Tracks")).clicked() {
        app.ui.timeline.video_track_h = 64.0;
        app.ui.timeline.audio_track_h = 64.0;
        ui.close();
    }
    if ui.button(tl!("Minimize All Tracks")).clicked() {
        app.ui.timeline.video_track_h = 26.0;
        app.ui.timeline.audio_track_h = 26.0;
        ui.close();
    }
}

fn run_timeline_toggle(app: &mut FilmcraftApp, key: &str) {
    let _ = match key {
        "nest" => app.session.execute("sequence.nestSequences", json!({})),
        "snap" => app.session.execute("sequence.snap", json!({})),
        "linked" => app.session.execute("sequence.linkedSelection", json!({})),
        "marker" => app.session.execute("markers.add", json!({})),
        _ => Ok(Value::Null),
    };
}
