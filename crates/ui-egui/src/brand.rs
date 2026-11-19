//! Code-drawn "Lyre of Time" badge and wordmark for Orpheus (replaces the upstream ArtCraft
//! trademarked wordmark per AGENTS.md §1.8 and `docs/brand/LICENSE-brand.txt`).

use std::sync::{Arc, OnceLock};

use egui::{Align2, Color32, CornerRadius, Stroke, pos2, vec2};

use crate::theme::Tokens;

fn logo_image() -> Option<Arc<egui::ColorImage>> {
    static IMAGE: OnceLock<Option<Arc<egui::ColorImage>>> = OnceLock::new();
    IMAGE
        .get_or_init(|| {
            let bytes = include_bytes!("../assets/orpheus_logo.png");
            let rgba = image::load_from_memory_with_format(bytes, image::ImageFormat::Png).ok()?.into_rgba8();
            let size = [rgba.width() as usize, rgba.height() as usize];
            Some(Arc::new(egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw())))
        })
        .clone()
}

/// Draw the embedded `orpheus_logo.png` inside `r` with linear texture filtering, falling back to
/// [`paint_lyre_badge`] if texture creation is unavailable.
pub fn paint_logo_png(ui: &mut egui::Ui, r: egui::Rect, dark: bool) {
    let key = egui::Id::unique("orpheus-welcome-logo-tex");
    let handle: Option<egui::TextureHandle> = ui.ctx().data(|d| d.get_temp(key)).or_else(|| {
        let img = logo_image()?;
        let tex = ui.ctx().load_texture("orpheus-welcome-logo", img, egui::TextureOptions::LINEAR);
        ui.ctx().data_mut(|d| d.insert_temp(key, tex.clone()));
        Some(tex)
    });
    if let Some(tex) = handle {
        let uv = egui::Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));
        ui.painter().image(tex.id(), r, uv, Color32::WHITE);
    } else {
        paint_lyre_badge(ui.painter(), r, dark);
    }
}

/// Draw the Orpheus "Lyre of Time" badge inside `r`.
pub fn paint_lyre_badge(painter: &egui::Painter, r: egui::Rect, dark: bool) {
    let s = r.width().min(r.height()).max(14.0);
    // Graphite badge tile with subtle border
    let tile_fill = if dark { Color32::from_rgb(0x20, 0x24, 0x2e) } else { Color32::from_rgb(0x1c, 0x1f, 0x26) };
    let tile_border = Color32::from_rgb(0x36, 0x3c, 0x4a);
    painter.rect_filled(r, CornerRadius::same((s * 0.22) as u8), tile_fill);
    painter.rect_stroke(r, CornerRadius::same((s * 0.22) as u8), Stroke::new(1.0, tile_border), egui::StrokeKind::Inside);

    // Subtle blue orbit arc
    let orbit_color = Color32::from_rgb(0x3f, 0x7c, 0xe8);
    painter.line_segment(
        [pos2(r.min.x + s * 0.16, r.max.y - s * 0.28), pos2(r.max.x - s * 0.18, r.min.y + s * 0.26)],
        Stroke::new((s * 0.08).max(1.0), orbit_color),
    );

    // Minimalist silver lyre silhouette (crossbar + two arms + center string)
    let lyre_color = Color32::from_rgb(0xe8, 0xec, 0xf4);
    let sw = (s * 0.10).max(1.2);
    let top_y = r.min.y + s * 0.26;
    let bot_y = r.max.y - s * 0.24;
    let lx = r.min.x + s * 0.30;
    let rx = r.max.x - s * 0.30;
    let cx = r.center().x;
    painter.line_segment([pos2(lx, top_y), pos2(rx, top_y)], Stroke::new(sw, lyre_color));
    painter.line_segment([pos2(lx + s * 0.03, top_y), pos2(cx - s * 0.10, bot_y)], Stroke::new(sw, lyre_color));
    painter.line_segment([pos2(rx - s * 0.03, top_y), pos2(cx + s * 0.10, bot_y)], Stroke::new(sw, lyre_color));
    painter.line_segment([pos2(cx - s * 0.10, bot_y), pos2(cx + s * 0.10, bot_y)], Stroke::new(sw, lyre_color));
    painter.line_segment([pos2(cx, top_y), pos2(cx, bot_y)], Stroke::new((s * 0.06).max(1.0), Color32::from_rgb(0x57, 0x94, 0xec)));

    // Star point on upper-right orbit
    painter.circle_filled(pos2(r.max.x - s * 0.18, r.min.y + s * 0.24), (s * 0.08).max(1.3), Color32::from_rgb(0xe8, 0xf2, 0xff));
}

/// Draw the Orpheus "Lyre of Time" badge and wordmark `height` points tall with its left edge at
/// `left_center` (vertically centred). Returns the drawn rect.
pub fn paint_wordmark(ui: &mut egui::Ui, left_center: egui::Pos2, height: f32, dark: bool) -> Option<egui::Rect> {
    let painter = ui.painter();
    let s = height.max(14.0);
    let r = egui::Rect::from_min_size(pos2(left_center.x, left_center.y - s / 2.0), vec2(s, s));
    paint_lyre_badge(painter, r, dark);

    let text_color = if dark { Color32::from_rgb(0xe4, 0xe7, 0xee) } else { Color32::from_rgb(0x1b, 0x1d, 0x22) };
    let font = Tokens::semibold((s * 0.85).max(12.0));
    let text_pos = pos2(r.max.x + 8.0, left_center.y);
    let galley_rect = painter.text(text_pos, Align2::LEFT_CENTER, crate::APP_NAME, font, text_color);

    Some(r.union(galley_rect))
}
