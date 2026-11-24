//! Orpheus cinematic Home / Startup Screen.
//!
//! Renders the obsidian lyre-and-wave backdrop, top-left Orpheus / YBV Studio serif brand header,
//! three equal-height cards (**Creator**, **Studio**, **Recent Projects**), and the centered
//! **Quick Actions** bar (**Open Project**, **About**, **Settings**).

use std::f32::consts::{FRAC_PI_2, TAU};
use std::sync::{Arc, OnceLock};

use egui::epaint::{Mesh, PathShape, PathStroke, Vertex};
use egui::{Align2, Color32, CornerRadius, FontId, Key, Painter, Pos2, Rect, Sense, Stroke, StrokeKind, TextureHandle, TextureId, TextureOptions, pos2, vec2};

use crate::icons::{self, Icon};
use crate::state::{AppExperience, Mode};
use crate::theme::Tokens;
use crate::{Dialog, FilmcraftApp};

const MAX_RECENT_INLINE: usize = 4;
const CARD_RADIUS: f32 = 16.0;
const CARD_GAP: f32 = 22.0;

// Brand & backdrop palette
const BG_TOP: Color32 = Color32::from_rgb(0x09, 0x0C, 0x12);
const BG_BOTTOM: Color32 = Color32::from_rgb(0x0E, 0x13, 0x1F);
const LYRE_ICE_BLUE: Color32 = Color32::from_rgb(0x93, 0xC5, 0xFD);
const TEXT_WHITE: Color32 = Color32::from_rgb(0xF8, 0xFA, 0xFC);
const TEXT_SUBTITLE: Color32 = Color32::from_rgb(0xCB, 0xD5, 0xE1);
const TEXT_MUTED: Color32 = Color32::from_rgb(0x94, 0xA3, 0xB8);
const TEXT_FAINT: Color32 = Color32::from_rgb(0x64, 0x74, 0x8B);

// Creator card palette
const CREATOR_TOP: Color32 = Color32::from_rgb(0x0D, 0x18, 0x2E);
const CREATOR_BOT: Color32 = Color32::from_rgb(0x09, 0x10, 0x1E);
const CREATOR_BORDER: Color32 = Color32::from_rgb(0x25, 0x63, 0xEB);
const CREATOR_BORDER_HOVER: Color32 = Color32::from_rgb(0x3B, 0x82, 0xF6);
const CREATOR_BADGE_FILL: Color32 = Color32::from_rgb(0x12, 0x23, 0x42);
const CREATOR_BADGE_BORDER: Color32 = Color32::from_rgb(0x3B, 0x82, 0xF6);
const CREATOR_CHECK: Color32 = Color32::from_rgb(0x60, 0xA5, 0xFA);
const CREATOR_CTA_LEFT: Color32 = Color32::from_rgb(0x88, 0xBC, 0xFF);
const CREATOR_CTA_MID: Color32 = Color32::from_rgb(0x4E, 0x8E, 0xF7);
const CREATOR_CTA_RIGHT: Color32 = Color32::from_rgb(0x60, 0xA5, 0xFA);
const CREATOR_CTA_TEXT: Color32 = Color32::from_rgb(0x09, 0x10, 0x1E);

// Studio card palette
const STUDIO_TOP: Color32 = Color32::from_rgb(0x18, 0x10, 0x2B);
const STUDIO_BOT: Color32 = Color32::from_rgb(0x0E, 0x0A, 0x1A);
const STUDIO_BORDER: Color32 = Color32::from_rgb(0x7C, 0x3A, 0xED);
const STUDIO_BORDER_HOVER: Color32 = Color32::from_rgb(0x9D, 0x6B, 0xFA);
const STUDIO_BADGE_FILL: Color32 = Color32::from_rgb(0x23, 0x15, 0x3D);
const STUDIO_BADGE_BORDER: Color32 = Color32::from_rgb(0x8B, 0x5C, 0xF6);
const STUDIO_CHECK: Color32 = Color32::from_rgb(0xA7, 0x8B, 0xFA);
const STUDIO_CTA_FILL: Color32 = Color32::from_rgba_premultiplied(16, 12, 30, 232);
const STUDIO_CTA_HOVER: Color32 = Color32::from_rgba_premultiplied(30, 21, 56, 240);
const STUDIO_CTA_BORDER: Color32 = Color32::from_rgb(0x8B, 0x5C, 0xF6);

// Recent Projects card & Quick Actions palette
const RECENT_BG: Color32 = Color32::from_rgba_premultiplied(12, 14, 22, 230);
const RECENT_BORDER: Color32 = Color32::from_rgb(0x1E, 0x29, 0x3B);
const RECENT_ROW_BG: Color32 = Color32::from_rgba_premultiplied(20, 25, 38, 225);
const RECENT_ROW_HOVER: Color32 = Color32::from_rgba_premultiplied(28, 36, 54, 242);
const OPEN_BTN_BG: Color32 = Color32::from_rgb(0x15, 0x1A, 0x24);
const OPEN_BTN_HOVER: Color32 = Color32::from_rgb(0x1E, 0x26, 0x36);
const OPEN_BTN_BORDER: Color32 = Color32::from_rgb(0x2D, 0x37, 0x48);
const QUICK_TILE_BG: Color32 = Color32::from_rgba_premultiplied(11, 15, 22, 235);
const QUICK_TILE_HOVER: Color32 = Color32::from_rgba_premultiplied(18, 25, 39, 245);
const QUICK_TILE_BORDER: Color32 = Color32::from_rgb(0x1E, 0x29, 0x3B);
const QUICK_TILE_HOVER_BORDER: Color32 = Color32::from_rgb(0x3B, 0x82, 0xF6);

#[derive(Clone, Copy)]
enum HomeArt {
    Backdrop,
    Creator,
    Studio,
}

impl HomeArt {
    fn key(self) -> &'static str {
        match self {
            HomeArt::Backdrop => "orpheus-home-backdrop-tex",
            HomeArt::Creator => "orpheus-home-creator-tex",
            HomeArt::Studio => "orpheus-home-studio-tex",
        }
    }

    fn bytes(self) -> &'static [u8] {
        match self {
            HomeArt::Backdrop => include_bytes!("../../../../assets/home/home_backdrop.png"),
            HomeArt::Creator => include_bytes!("../../../../assets/home/creator_card_art.png"),
            HomeArt::Studio => include_bytes!("../../../../assets/home/studio_card_art.png"),
        }
    }

    fn decoded(self) -> Option<Arc<egui::ColorImage>> {
        static BACKDROP: OnceLock<Option<Arc<egui::ColorImage>>> = OnceLock::new();
        static CREATOR: OnceLock<Option<Arc<egui::ColorImage>>> = OnceLock::new();
        static STUDIO: OnceLock<Option<Arc<egui::ColorImage>>> = OnceLock::new();
        let slot = match self {
            HomeArt::Backdrop => &BACKDROP,
            HomeArt::Creator => &CREATOR,
            HomeArt::Studio => &STUDIO,
        };
        slot.get_or_init(|| decode_png(self.bytes())).clone()
    }

    fn texture(self, ctx: &egui::Context) -> Option<TextureHandle> {
        let id = egui::Id::unique(self.key());
        ctx.data(|d| d.get_temp::<TextureHandle>(id)).or_else(|| {
            let img = self.decoded()?;
            let tex = ctx.load_texture(self.key(), img, TextureOptions::LINEAR);
            ctx.data_mut(|d| d.insert_temp(id, tex.clone()));
            Some(tex)
        })
    }
}

fn decode_png(bytes: &[u8]) -> Option<Arc<egui::ColorImage>> {
    let rgba = image::load_from_memory_with_format(bytes, image::ImageFormat::Png).ok()?.into_rgba8();
    let size = [rgba.width() as usize, rgba.height() as usize];
    Some(Arc::new(egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw())))
}

/// Draw the full-window Orpheus Home / Startup Screen inside `full`.
pub fn show(app: &mut FilmcraftApp, ui: &mut egui::Ui, full: Rect) {
    paint_backdrop(ui, full);

    let banner_h = crate::updates_ui::banner_height(app);
    let top_reserve = if banner_h > 0.0 {
        let banner_r = Rect::from_min_size(pos2(full.min.x + 24.0, full.min.y + 12.0), vec2((full.width() - 48.0).max(240.0), banner_h));
        crate::updates_ui::show_banner(app, ui, banner_r);
        banner_h + 22.0
    } else {
        16.0
    };

    let layout = HomeLayout::compute(full, top_reserve);
    let s = layout.scale;

    draw_brand_header(ui, layout.brand, s);
    draw_creator_card(app, ui, layout.creator, s);
    draw_studio_card(app, ui, layout.studio, s);
    draw_recent_card(app, ui, layout.recent, s);
    draw_quick_actions(app, ui, layout.quick_actions, s);
    draw_all_recent_modal(app, ui.ctx());
}

struct HomeLayout {
    scale: f32,
    brand: Rect,
    creator: Rect,
    studio: Rect,
    recent: Rect,
    quick_actions: Rect,
}

impl HomeLayout {
    fn compute(full: Rect, top_reserve: f32) -> Self {
        let s = (full.width() / 1600.0).min(full.height() / 900.0).clamp(0.75, 1.35);
        let grid_w = (1180.0 * s).min((full.width() - 32.0 * s).max(480.0));
        let x0 = full.center().x - grid_w * 0.5;

        let brand_h = 84.0 * s;
        let gap_brand_cards = 22.0 * s;
        let card_h = 340.0 * s;
        let card_gap = CARD_GAP * s;
        let gap_cards_quick = 24.0 * s;
        let quick_h = 92.0 * s;

        let avail_h = (full.height() - top_reserve - 24.0 * s).max(360.0);
        let total_h = brand_h + gap_brand_cards + card_h + gap_cards_quick + quick_h;
        let y0 = (full.min.y + top_reserve + (avail_h - total_h) * 0.44).max(full.min.y + top_reserve);

        let brand = Rect::from_min_size(pos2(x0, y0), vec2(grid_w, brand_h));
        let cards_y = brand.max.y + gap_brand_cards;
        let card_w = ((grid_w - card_gap * 2.0) / 3.0).max(160.0);

        let creator = Rect::from_min_size(pos2(x0, cards_y), vec2(card_w, card_h));
        let studio = Rect::from_min_size(pos2(creator.max.x + card_gap, cards_y), vec2(card_w, card_h));
        let recent = Rect::from_min_size(pos2(studio.max.x + card_gap, cards_y), vec2(card_w, card_h));
        let quick_actions = Rect::from_min_size(pos2(x0, creator.max.y + gap_cards_quick), vec2(grid_w, quick_h));

        Self { scale: s, brand, creator, studio, recent, quick_actions }
    }
}

fn paint_backdrop(ui: &mut egui::Ui, full: Rect) {
    let p = ui.painter();
    paint_vertical_gradient(p, full, BG_TOP, BG_BOTTOM);
    if let Some(tex) = HomeArt::Backdrop.texture(ui.ctx()) {
        let uv = cover_uv_upper_right(tex.size_vec2(), full.size());
        p.image(tex.id(), full, uv, Color32::WHITE);
    }
}

fn draw_brand_header(ui: &mut egui::Ui, area: Rect, s: f32) {
    let p = ui.painter();
    let lyre_size = 56.0 * s;
    let lyre_rect = Rect::from_min_size(pos2(area.min.x + 2.0 * s, area.min.y + 4.0 * s), vec2(lyre_size, lyre_size));
    paint_ice_lyre(p, lyre_rect, LYRE_ICE_BLUE);

    // Subtle hairline accent below the lyre icon matching the welcome mockup
    let line_y = lyre_rect.max.y + 10.0 * s;
    p.line_segment(
        [pos2(lyre_rect.min.x + 4.0 * s, line_y), pos2(lyre_rect.max.x - 2.0 * s, line_y)],
        Stroke::new(1.0, Color32::from_rgba_unmultiplied(0x93, 0xC5, 0xFD, 42)),
    );

    let text_x = lyre_rect.max.x + 22.0 * s;
    let title_rect = p.text(pos2(text_x, area.min.y + 2.0 * s), Align2::LEFT_TOP, crate::APP_NAME, Tokens::serif(48.0 * s), TEXT_WHITE);

    let studio_y = title_rect.max.y + 3.0 * s;
    paint_tracked_text(p, pos2(text_x + 2.0 * s, studio_y), "YBV Studio", Tokens::ui(13.5 * s), TEXT_SUBTITLE, 3.0 * s);
}

struct ModeCardSpec<'a> {
    experience: AppExperience,
    art: HomeArt,
    card_id: &'static str,
    cta_id: &'static str,
    auto_id: &'static str,
    title: &'a str,
    subtitle: &'a str,
    bullets: [&'a str; 3],
    cta_label: &'a str,
    top_col: Color32,
    bot_col: Color32,
    left_scrim: Color32,
    border: Color32,
    border_hover: Color32,
    badge_fill: Color32,
    badge_border: Color32,
    check_col: Color32,
    filled_cta: bool,
}

fn draw_creator_card(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect, s: f32) {
    let spec = ModeCardSpec {
        experience: AppExperience::Creator,
        art: HomeArt::Creator,
        card_id: "start-card-creator",
        cta_id: "start-cta-creator",
        auto_id: "start.creator",
        title: "Creator",
        subtitle: tl!("Quick and easy editing for Shorts and TikTok."),
        bullets: [tl!("Fast timeline"), tl!("Smart tools"), tl!("Export in seconds")],
        cta_label: tl!("Start Creating"),
        top_col: CREATOR_TOP,
        bot_col: CREATOR_BOT,
        left_scrim: CREATOR_BOT,
        border: CREATOR_BORDER,
        border_hover: CREATOR_BORDER_HOVER,
        badge_fill: CREATOR_BADGE_FILL,
        badge_border: CREATOR_BADGE_BORDER,
        check_col: CREATOR_CHECK,
        filled_cta: true,
    };
    draw_mode_card(app, ui, rect, s, &spec);
}

fn draw_studio_card(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect, s: f32) {
    let spec = ModeCardSpec {
        experience: AppExperience::Studio,
        art: HomeArt::Studio,
        card_id: "start-card-studio",
        cta_id: "start-cta-studio",
        auto_id: "start.studio",
        title: "Studio",
        subtitle: tl!("Professional editing for advanced creators."),
        bullets: [tl!("Advanced timeline"), tl!("Color grading"), tl!("Pro-level tools")],
        cta_label: tl!("Open Studio"),
        top_col: STUDIO_TOP,
        bot_col: STUDIO_BOT,
        left_scrim: STUDIO_BOT,
        border: STUDIO_BORDER,
        border_hover: STUDIO_BORDER_HOVER,
        badge_fill: STUDIO_BADGE_FILL,
        badge_border: STUDIO_BADGE_BORDER,
        check_col: STUDIO_CHECK,
        filled_cta: false,
    };
    draw_mode_card(app, ui, rect, s, &spec);
}

fn draw_mode_card(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect, s: f32, spec: &ModeCardSpec<'_>) {
    let card_id = ui.make_persistent_id(spec.card_id);
    if spec.experience == AppExperience::Creator && ui.memory(|m| m.focused().is_none()) && ui.input(|i| i.key_pressed(Key::Tab)) {
        ui.memory_mut(|m| m.request_focus(card_id));
    }

    let pad = 24.0 * s;
    let cta_h = 42.0 * s;
    let radius = CARD_RADIUS * s;
    let cta_rect = Rect::from_min_max(pos2(rect.min.x + pad, rect.max.y - pad - cta_h), pos2(rect.max.x - pad, rect.max.y - pad));

    let card_resp = ui.interact(rect, card_id, Sense::click()).on_hover_cursor(egui::CursorIcon::PointingHand);
    let cta_resp = ui.interact(cta_rect, ui.make_persistent_id(spec.cta_id), Sense::click()).on_hover_cursor(egui::CursorIcon::PointingHand);
    app.auto.add(spec.auto_id, rect, spec.title);

    let hovered = card_resp.hovered() || cta_resp.hovered();
    let focused = card_resp.has_focus() || cta_resp.has_focus();

    let p = ui.painter();
    paint_rounded_vertical_gradient(p, rect, radius, spec.top_col, spec.bot_col);
    if let Some(tex) = spec.art.texture(ui.ctx()) {
        paint_rounded_textured_rect(p, rect, radius, tex.id(), cover_uv(tex.size_vec2(), rect.size()), Color32::WHITE);
    }
    paint_rounded_left_scrim(p, rect, radius, spec.left_scrim, 195);

    let border_col = if hovered || focused { spec.border_hover } else { spec.border };
    p.rect_stroke(rect, CornerRadius::same(radius.round() as u8), Stroke::new(1.5, border_col), StrokeKind::Inside);

    // Top-left rounded square icon badge
    let badge_rect = Rect::from_min_size(pos2(rect.min.x + pad, rect.min.y + 22.0 * s), vec2(42.0 * s, 42.0 * s));
    p.rect_filled(badge_rect, CornerRadius::same((10.0 * s).round() as u8), spec.badge_fill);
    p.rect_stroke(badge_rect, CornerRadius::same((10.0 * s).round() as u8), Stroke::new(1.2, spec.badge_border), StrokeKind::Inside);
    match spec.experience {
        AppExperience::Creator => paint_lightning_icon(p, badge_rect.center(), LYRE_ICE_BLUE),
        AppExperience::Studio => paint_video_frame_icon(p, badge_rect.center(), Color32::from_rgb(0xC4, 0xB5, 0xFD)),
    }

    // Serif title
    let title_y = badge_rect.max.y + 16.0 * s;
    let title_r = p.text(pos2(rect.min.x + pad, title_y), Align2::LEFT_TOP, spec.title, Tokens::serif(27.0 * s), TEXT_WHITE);

    // Subtitle
    let sub_y = title_r.max.y + 5.0 * s;
    let sub_w = (rect.width() * 0.56).clamp(110.0 * s, 210.0 * s);
    let sub_job = egui::text::LayoutJob::simple(spec.subtitle.to_string(), Tokens::ui(13.0 * s), TEXT_SUBTITLE, sub_w);
    let sub_galley = p.layout_job(sub_job);
    let sub_h = sub_galley.size().y;
    p.galley(pos2(rect.min.x + pad, sub_y), sub_galley, TEXT_SUBTITLE);

    // Three checklist items
    let list_y = sub_y + sub_h + 16.0 * s;
    let row_step = 23.0 * s;
    for (idx, item) in spec.bullets.iter().enumerate() {
        let cy = list_y + idx as f32 * row_step + 8.0 * s;
        let cx = rect.min.x + pad + 6.0 * s;
        paint_checkmark(p, pos2(cx, cy), spec.check_col);
        p.text(pos2(rect.min.x + pad + 22.0 * s, cy), Align2::LEFT_CENTER, *item, Tokens::ui(12.5 * s), TEXT_SUBTITLE);
    }

    // Bottom pill CTA button
    if spec.filled_cta {
        paint_creator_cta_button(p, cta_rect, s, spec.cta_label, cta_resp.hovered() || card_resp.hovered(), cta_resp.is_pointer_button_down_on());
    } else {
        paint_studio_cta_button(p, cta_rect, s, spec.cta_label, cta_resp.hovered() || card_resp.hovered(), cta_resp.is_pointer_button_down_on());
    }

    let enter = focused && ui.input(|i| i.key_pressed(Key::Enter) || i.key_pressed(Key::Space));
    if cta_resp.clicked() || card_resp.clicked() || enter {
        launch_mode(app, ui.ctx(), spec.experience);
    }
}

fn launch_mode(app: &mut FilmcraftApp, ctx: &egui::Context, exp: AppExperience) {
    if app.session.project.items.is_empty() && app.session.state.active_sequence.is_none() {
        let _ = crate::menus::invoke(app, ctx, "file.newProject", serde_json::json!({}));
    }
    if let Some(path) = app.session.path.clone() {
        app.ui.remember_project_experience(&path, exp);
    }
    app.ui.experience = exp;
    app.ui.mode = Mode::Edit;
    app.ui.start_screen = false;
}

fn paint_creator_cta_button(p: &Painter, rect: Rect, s: f32, label: &str, hovered: bool, pressed: bool) {
    let radius = rect.height() * 0.5;
    let (l, m, r) = if pressed {
        (Color32::from_rgb(0x52, 0x8F, 0xE8), Color32::from_rgb(0x2F, 0x6F, 0xDC), Color32::from_rgb(0x46, 0x87, 0xEB))
    } else if hovered {
        (Color32::from_rgb(0xA3, 0xCE, 0xFF), Color32::from_rgb(0x5B, 0x9B, 0xFC), Color32::from_rgb(0x78, 0xB8, 0xFF))
    } else {
        (CREATOR_CTA_LEFT, CREATOR_CTA_MID, CREATOR_CTA_RIGHT)
    };
    paint_pill_horizontal_gradient(p, rect, radius, l, m, r);
    p.rect_stroke(rect, CornerRadius::same(radius as u8), Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 72)), StrokeKind::Inside);

    let g = p.layout_no_wrap(label.to_string(), Tokens::bold(13.5 * s), CREATOR_CTA_TEXT);
    let icon_w = 11.0 * s;
    let arrow_w = 14.0 * s;
    let gap = 9.0 * s;
    let total_w = icon_w + gap + g.size().x + gap + arrow_w;
    let x0 = rect.center().x - total_w * 0.5;
    let cy = rect.center().y;

    paint_play_triangle(p, pos2(x0 + icon_w * 0.5, cy), 5.2 * s, CREATOR_CTA_TEXT);
    p.galley(pos2(x0 + icon_w + gap, cy - g.size().y * 0.5), g, CREATOR_CTA_TEXT);
    paint_right_arrow(p, pos2(x0 + total_w - arrow_w * 0.5, cy), 12.0 * s, CREATOR_CTA_TEXT, 1.7);
}

fn paint_studio_cta_button(p: &Painter, rect: Rect, s: f32, label: &str, hovered: bool, pressed: bool) {
    let radius = rect.height() * 0.5;
    let fill = if pressed {
        Color32::from_rgba_premultiplied(38, 25, 68, 245)
    } else if hovered {
        STUDIO_CTA_HOVER
    } else {
        STUDIO_CTA_FILL
    };
    let border_col = if hovered { Color32::from_rgb(0xA7, 0x8B, 0xFA) } else { STUDIO_CTA_BORDER };
    p.rect_filled(rect, CornerRadius::same(radius as u8), fill);
    p.rect_stroke(rect, CornerRadius::same(radius as u8), Stroke::new(1.5, border_col), StrokeKind::Inside);

    let g = p.layout_no_wrap(label.to_string(), Tokens::semibold(13.5 * s), TEXT_WHITE);
    let icon_w = 15.0 * s;
    let arrow_w = 14.0 * s;
    let gap = 9.0 * s;
    let total_w = icon_w + gap + g.size().x + gap + arrow_w;
    let x0 = rect.center().x - total_w * 0.5;
    let cy = rect.center().y;

    paint_folder_outline(p, Rect::from_center_size(pos2(x0 + icon_w * 0.5, cy), vec2(14.5 * s, 12.0 * s)), TEXT_WHITE, 1.4);
    p.galley(pos2(x0 + icon_w + gap, cy - g.size().y * 0.5), g, TEXT_WHITE);
    paint_right_arrow(p, pos2(x0 + total_w - arrow_w * 0.5, cy), 12.0 * s, TEXT_WHITE, 1.5);
}

fn draw_recent_card(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect, s: f32) {
    let radius = CARD_RADIUS * s;
    let p = ui.painter();
    p.rect_filled(rect, CornerRadius::same(radius.round() as u8), RECENT_BG);
    p.rect_stroke(rect, CornerRadius::same(radius.round() as u8), Stroke::new(1.0, RECENT_BORDER), StrokeKind::Inside);

    let pad = 24.0 * s;
    let inner = rect.shrink(pad);

    // Header row: clock icon + "Recent Projects" on left, "View All →" on right
    let hdr_cy = inner.min.y + 12.0 * s;
    let clock_r = Rect::from_center_size(pos2(inner.min.x + 10.0 * s, hdr_cy), vec2(20.0 * s, 20.0 * s));
    paint_clock_icon(p, clock_r.center(), 8.8 * s, TEXT_SUBTITLE);
    p.text(pos2(clock_r.max.x + 10.0 * s, hdr_cy), Align2::LEFT_CENTER, tl!("Recent Projects"), Tokens::semibold(15.5 * s), TEXT_WHITE);

    let view_all_text = tl!("View All");
    let va_galley = p.layout_no_wrap(view_all_text.to_string(), Tokens::ui(12.0 * s), TEXT_MUTED);
    let va_w = va_galley.size().x + 18.0 * s;
    let va_rect = Rect::from_min_size(pos2(inner.max.x - va_w, hdr_cy - 12.0 * s), vec2(va_w, 24.0 * s));
    let va_resp = ui.interact(va_rect, ui.make_persistent_id("start-recent-view-all"), Sense::click()).on_hover_cursor(egui::CursorIcon::PointingHand);
    app.auto.add("start.recent.viewAll", va_rect, view_all_text);
    let va_col = if va_resp.hovered() { TEXT_WHITE } else { TEXT_MUTED };
    p.galley_with_override_text_color(pos2(va_rect.min.x, hdr_cy - va_galley.size().y * 0.5), va_galley, va_col);
    paint_right_arrow(p, pos2(va_rect.max.x - 6.0 * s, hdr_cy + 0.5 * s), 10.0 * s, va_col, 1.3);
    if va_resp.clicked() {
        let modal_id = egui::Id::unique("start-recent-view-all-open");
        ui.ctx().data_mut(|d| d.insert_temp(modal_id, true));
    }

    let body_top = inner.min.y + 36.0 * s;
    let open_btn_h = 36.0 * s;
    let open_btn_w = (150.0 * s).min(inner.width());
    let recent_paths: Vec<String> = app.session.prefs.general.recent_projects.iter().take(MAX_RECENT_INLINE).cloned().collect();

    if recent_paths.is_empty() {
        // Centered empty state + centered Open Project pill directly beneath, matching reference
        let block_cy = body_top + (inner.max.y - body_top) * 0.45;
        let folder_r = Rect::from_center_size(pos2(inner.center().x, block_cy - 42.0 * s), vec2(38.0 * s, 30.0 * s));
        paint_folder_outline(p, folder_r, TEXT_FAINT, 1.6);

        p.text(pos2(inner.center().x, block_cy - 6.0 * s), Align2::CENTER_CENTER, tl!("No recent projects yet"), Tokens::semibold(13.5 * s), TEXT_SUBTITLE);
        let hint_job = egui::text::LayoutJob::simple(
            tl!("Start a new project or open an existing one.").to_string(),
            Tokens::ui(11.5 * s),
            TEXT_FAINT,
            (inner.width() - 8.0 * s).max(80.0),
        );
        let hint_g = p.layout_job(hint_job);
        p.galley(pos2(inner.center().x - hint_g.size().x * 0.5, block_cy + 10.0 * s), hint_g, TEXT_FAINT);

        let btn_rect = Rect::from_center_size(pos2(inner.center().x, block_cy + 58.0 * s), vec2(open_btn_w, open_btn_h));
        draw_recent_open_button(app, ui, btn_rect, s);
        return;
    }

    let list_bottom = inner.max.y - open_btn_h - 14.0 * s;
    let mut open_path: Option<String> = None;
    let row_h = 46.0 * s;
    let row_gap = 8.0 * s;
    for (idx, path) in recent_paths.iter().enumerate() {
        let y = body_top + idx as f32 * (row_h + row_gap);
        if y + row_h > list_bottom + 2.0 {
            break;
        }
        let row = Rect::from_min_size(pos2(inner.min.x, y), vec2(inner.width(), row_h));
        if draw_recent_row(app, ui, row, path, idx, "start-recent") {
            open_path = Some(path.clone());
        }
    }

    let btn_rect = Rect::from_center_size(pos2(inner.center().x, inner.max.y - open_btn_h * 0.5), vec2(open_btn_w, open_btn_h));
    draw_recent_open_button(app, ui, btn_rect, s);

    if let Some(path) = open_path {
        open_recent_project(app, ui.ctx(), &path);
    }
}

fn draw_recent_row(app: &mut FilmcraftApp, ui: &mut egui::Ui, row: Rect, path: &str, idx: usize, salt: &str) -> bool {
    let name = crate::menus::recent_name(path);
    let resp = ui.interact(row, ui.make_persistent_id((salt, idx)), Sense::click()).on_hover_text(path).on_hover_cursor(egui::CursorIcon::PointingHand);
    if salt == "start-recent" {
        app.auto.add(&format!("start.recent.{idx}"), row, &name);
    }

    let active = resp.hovered() || resp.has_focus();
    let p = ui.painter();
    p.rect_filled(row, CornerRadius::same(10), if active { RECENT_ROW_HOVER } else { RECENT_ROW_BG });
    let border = if active { Stroke::new(1.0, QUICK_TILE_HOVER_BORDER) } else { Stroke::new(1.0, RECENT_BORDER) };
    p.rect_stroke(row, CornerRadius::same(10), border, StrokeKind::Inside);

    let clip = p.with_clip_rect(row.shrink2(vec2(10.0, 4.0)));
    let icon_r = Rect::from_center_size(pos2(row.min.x + 22.0, row.center().y), vec2(16.0, 13.0));
    paint_folder_outline(&clip, icon_r, if active { LYRE_ICE_BLUE } else { TEXT_MUTED }, 1.4);

    let mut right_x = row.max.x - 12.0;
    if let Some(ts) = format_project_timestamp(path) {
        let tg = clip.layout_no_wrap(ts, Tokens::ui(11.0), TEXT_FAINT);
        clip.galley(pos2(right_x - tg.size().x, row.center().y - 7.5 - tg.size().y * 0.5), tg.clone(), TEXT_FAINT);
        right_x -= tg.size().x + 8.0;
    }

    let text_x = icon_r.max.x + 12.0;
    let name_w = (right_x - text_x).max(24.0);
    let shown_name = fit_text(&clip, &name, Tokens::semibold(13.0), TEXT_WHITE, name_w);
    clip.text(pos2(text_x, row.center().y - 7.5), Align2::LEFT_CENTER, shown_name, Tokens::semibold(13.0), TEXT_WHITE);

    if let Some(parent) = std::path::Path::new(path).parent() {
        let dir = parent.to_string_lossy();
        if !dir.is_empty() {
            let dir_w = (row.max.x - 12.0 - text_x).max(24.0);
            let shown_dir = fit_middle_ellipsis(&clip, &dir, Tokens::ui(11.0), TEXT_FAINT, dir_w);
            clip.text(pos2(text_x, row.center().y + 8.5), Align2::LEFT_CENTER, shown_dir, Tokens::ui(11.0), TEXT_FAINT);
        }
    }

    let enter = resp.has_focus() && ui.input(|i| i.key_pressed(Key::Enter) || i.key_pressed(Key::Space));
    resp.clicked() || enter
}

fn draw_recent_open_button(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect, s: f32) {
    let label = tl!("Open Project");
    let resp = ui.interact(rect, ui.make_persistent_id("start-open-project"), Sense::click()).on_hover_cursor(egui::CursorIcon::PointingHand);
    app.auto.add("start.openProject", rect, label);

    let active = resp.hovered() || resp.has_focus();
    let p = ui.painter();
    let radius = rect.height() * 0.5;
    p.rect_filled(rect, CornerRadius::same(radius as u8), if active { OPEN_BTN_HOVER } else { OPEN_BTN_BG });
    p.rect_stroke(rect, CornerRadius::same(radius as u8), Stroke::new(1.0, if active { QUICK_TILE_HOVER_BORDER } else { OPEN_BTN_BORDER }), StrokeKind::Inside);

    let g = p.layout_no_wrap(label.to_string(), Tokens::semibold(12.5 * s), TEXT_WHITE);
    let icon_w = 14.0 * s;
    let gap = 8.0 * s;
    let total_w = icon_w + gap + g.size().x;
    let x0 = rect.center().x - total_w * 0.5;
    paint_folder_outline(p, Rect::from_center_size(pos2(x0 + icon_w * 0.5, rect.center().y), vec2(13.5 * s, 11.0 * s)), TEXT_SUBTITLE, 1.35);
    p.galley(pos2(x0 + icon_w + gap, rect.center().y - g.size().y * 0.5), g, TEXT_WHITE);

    let enter = resp.has_focus() && ui.input(|i| i.key_pressed(Key::Enter) || i.key_pressed(Key::Space));
    if resp.clicked() || enter {
        trigger_open_project_dialog(app, ui.ctx());
    }
}

fn draw_quick_actions(app: &mut FilmcraftApp, ui: &mut egui::Ui, area: Rect, s: f32) {
    let p = ui.painter();
    p.text(pos2(area.center().x, area.min.y + 8.0 * s), Align2::CENTER_CENTER, tl!("Quick Actions"), Tokens::ui(12.5 * s), TEXT_MUTED);

    let tile_w = 110.0 * s;
    let tile_h = 62.0 * s;
    let gap = 14.0 * s;
    let total_w = tile_w * 3.0 + gap * 2.0;
    let x0 = area.center().x - total_w * 0.5;
    let y0 = area.min.y + 24.0 * s;

    let tiles = [
        ("start.quick.openProject", "start-qa-open", tl!("Open Project"), QuickAction::OpenProject),
        ("start.quick.about", "start-qa-about", tl!("About"), QuickAction::About),
        ("start.quick.settings", "start-qa-settings", tl!("Settings"), QuickAction::Settings),
    ];

    for (idx, (auto_id, widget_id, label, action)) in tiles.into_iter().enumerate() {
        let rect = Rect::from_min_size(pos2(x0 + idx as f32 * (tile_w + gap), y0), vec2(tile_w, tile_h));
        let resp = ui.interact(rect, ui.make_persistent_id(widget_id), Sense::click()).on_hover_cursor(egui::CursorIcon::PointingHand);
        app.auto.add(auto_id, rect, label);

        let active = resp.hovered() || resp.has_focus();
        let p = ui.painter();
        let tile_radius = (12.0 * s).round() as u8;
        p.rect_filled(rect, CornerRadius::same(tile_radius), if active { QUICK_TILE_HOVER } else { QUICK_TILE_BG });
        p.rect_stroke(
            rect,
            CornerRadius::same(tile_radius),
            Stroke::new(if active { 1.4 } else { 1.0 }, if active { QUICK_TILE_HOVER_BORDER } else { QUICK_TILE_BORDER }),
            StrokeKind::Inside,
        );

        let icon_center = pos2(rect.center().x, rect.min.y + 22.0 * s);
        let icon_col = if active { TEXT_WHITE } else { TEXT_SUBTITLE };
        match action {
            QuickAction::OpenProject => paint_folder_outline(p, Rect::from_center_size(icon_center, vec2(16.0 * s, 13.0 * s)), icon_col, 1.45),
            QuickAction::About => paint_info_circle_icon(p, icon_center, 8.2 * s, icon_col),
            QuickAction::Settings => icons::paint(p, Rect::from_center_size(icon_center, vec2(17.0 * s, 17.0 * s)), Icon::Gear, icon_col),
        }
        p.text(
            pos2(rect.center().x, rect.max.y - 15.0 * s),
            Align2::CENTER_CENTER,
            label,
            Tokens::ui(11.0 * s),
            if active { TEXT_WHITE } else { TEXT_SUBTITLE },
        );

        let enter = resp.has_focus() && ui.input(|i| i.key_pressed(Key::Enter) || i.key_pressed(Key::Space));
        if resp.clicked() || enter {
            match action {
                QuickAction::OpenProject => trigger_open_project_dialog(app, ui.ctx()),
                QuickAction::About => app.dialog = Some(Dialog::About),
                QuickAction::Settings => {
                    if let Err(err) = crate::panels::settings::open(app, "general") {
                        app.ui.status = err;
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
enum QuickAction {
    OpenProject,
    About,
    Settings,
}

fn draw_all_recent_modal(app: &mut FilmcraftApp, ctx: &egui::Context) {
    let modal_id = egui::Id::unique("start-recent-view-all-open");
    let open = ctx.data(|d| d.get_temp::<bool>(modal_id).unwrap_or(false));
    if !open {
        return;
    }
    let mut still_open = true;
    let mut close_clicked = false;
    let mut open_picker = false;
    let mut chosen: Option<String> = None;
    let all_recent = app.session.prefs.general.recent_projects.clone();

    egui::Window::new(tl!("Recent Projects"))
        .id(egui::Id::unique("start-all-recent-window"))
        .open(&mut still_open)
        .collapsible(false)
        .resizable(false)
        .pivot(Align2::CENTER_CENTER)
        .default_pos(crate::widgets::dialog_center(ctx))
        .show(ctx, |ui| {
            ui.set_width(460.0);
            if all_recent.is_empty() {
                ui.add_space(12.0);
                ui.label(egui::RichText::new(tl!("No recent projects yet")).color(TEXT_SUBTITLE));
                ui.label(egui::RichText::new(tl!("Start a new project or open an existing one.")).small().color(TEXT_MUTED));
                ui.add_space(12.0);
            } else {
                egui::ScrollArea::vertical().max_height(320.0).show(ui, |ui| {
                    for (idx, path) in all_recent.iter().enumerate() {
                        let (row, _) = ui.allocate_exact_size(vec2(ui.available_width(), 48.0), Sense::hover());
                        if draw_recent_row(app, ui, row, path, idx, "start-recent-modal") {
                            chosen = Some(path.clone());
                        }
                        ui.add_space(6.0);
                    }
                });
            }
            ui.separator();
            ui.horizontal(|ui| {
                if ui.button(tl!("Open Project")).clicked() {
                    close_clicked = true;
                    open_picker = true;
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button(tl!("Close")).clicked() {
                        close_clicked = true;
                    }
                });
            });
        });

    if close_clicked || crate::widgets::escape_closes(ctx) {
        still_open = false;
    }
    if open_picker {
        trigger_open_project_dialog(app, ctx);
    }
    if let Some(path) = chosen {
        still_open = false;
        open_recent_project(app, ctx, &path);
    }
    if !still_open {
        ctx.data_mut(|d| d.insert_temp(modal_id, false));
    }
}

fn trigger_open_project_dialog(app: &mut FilmcraftApp, ctx: &egui::Context) {
    let prev = app.session.path.clone();
    if crate::menus::invoke(app, ctx, "file.open", serde_json::json!({})).is_ok() && app.session.path != prev {
        if let Some(opened) = app.session.path.as_deref().and_then(|p| app.ui.project_experience(p)) {
            app.ui.experience = opened;
        }
        app.ui.mode = Mode::Edit;
        app.ui.start_screen = false;
    }
}

fn open_recent_project(app: &mut FilmcraftApp, ctx: &egui::Context, path: &str) {
    if crate::menus::invoke(app, ctx, "file.open", serde_json::json!({ "path": path })).is_ok() {
        if let Some(exp) = app.ui.project_experience(path) {
            app.ui.experience = exp;
        }
        app.ui.mode = Mode::Edit;
        app.ui.start_screen = false;
    }
}

fn format_project_timestamp(path: &str) -> Option<String> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = path;
        None
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let modified = std::fs::metadata(path).ok()?.modified().ok()?;
        if let Ok(elapsed) = modified.elapsed() {
            let secs = elapsed.as_secs();
            if secs < 60 {
                return Some("Just now".to_string());
            }
            if secs < 3600 {
                return Some(format!("{}m ago", secs / 60));
            }
            if secs < 86_400 {
                return Some(format!("{}h ago", secs / 3600));
            }
            if secs < 86_400 * 7 {
                return Some(format!("{}d ago", secs / 86_400));
            }
        }
        let dt: chrono::DateTime<chrono::Utc> = modified.into();
        Some(dt.format("%b %d, %Y").to_string())
    }
}

// ---------------------------------------------------------------------------
// Custom vector icons & graphics helpers
// ---------------------------------------------------------------------------

/// Paint tracked (wide letter-spaced) text starting at `origin` (left-top) and return its height.
fn paint_tracked_text(p: &Painter, origin: Pos2, text: &str, font: FontId, color: Color32, extra_spacing: f32) -> f32 {
    let mut x = origin.x;
    let mut max_h = font.size;
    for ch in text.chars() {
        let g = p.layout_no_wrap(ch.to_string(), font.clone(), color);
        let sz = g.size();
        max_h = max_h.max(sz.y);
        p.galley(pos2(x, origin.y), g, color);
        x += sz.x + extra_spacing;
    }
    max_h
}

/// Classical ice-blue Orpheus lyre icon (64x64 design grid, `#93C5FD`).
fn paint_ice_lyre(p: &Painter, r: Rect, color: Color32) {
    let sx = r.width() / 64.0;
    let sy = r.height() / 64.0;
    let pt = |x: f32, y: f32| pos2(r.min.x + x * sx, r.min.y + y * sy);
    let sw = (3.6 * sx.min(sy)).max(2.0);
    let thin = (1.6 * sx.min(sy)).max(1.1);

    // Subtle soft aura behind the lyre
    paint_radial_glow(p, r.center(), r.width() * 0.62, Color32::from_rgba_unmultiplied(0x93, 0xC5, 0xFD, 28));

    // Rounded soundbox base (trapezoid with curved bottom)
    let base = vec![pt(22.0, 48.0), pt(42.0, 48.0), pt(45.5, 53.5), pt(41.5, 58.0), pt(22.5, 58.0), pt(18.5, 53.5)];
    p.add(PathShape { points: base, closed: true, fill: color, stroke: PathStroke::NONE });

    // Left S-curved arm with top scroll
    let left_arm = sample_cubic_chain(&[
        (pt(21.0, 53.0), pt(8.5, 44.0), pt(8.0, 27.0), pt(16.0, 15.0)),
        (pt(16.0, 15.0), pt(19.0, 10.5), pt(15.5, 6.0), pt(11.5, 8.2)),
        (pt(11.5, 8.2), pt(8.8, 9.8), pt(10.0, 13.5), pt(13.2, 13.0)),
    ]);
    p.add(PathShape::line(left_arm, Stroke::new(sw, color)));

    // Right S-curved arm with top scroll (mirrored)
    let right_arm = sample_cubic_chain(&[
        (pt(43.0, 53.0), pt(55.5, 44.0), pt(56.0, 27.0), pt(48.0, 15.0)),
        (pt(48.0, 15.0), pt(45.0, 10.5), pt(48.5, 6.0), pt(52.5, 8.2)),
        (pt(52.5, 8.2), pt(55.2, 9.8), pt(54.0, 13.5), pt(50.8, 13.0)),
    ]);
    p.add(PathShape::line(right_arm, Stroke::new(sw, color)));

    // Gently arched top crossbar
    let crossbar = sample_cubic_chain(&[(pt(15.5, 16.0), pt(25.0, 19.5), pt(39.0, 19.5), pt(48.5, 16.0))]);
    p.add(PathShape::line(crossbar, Stroke::new((sw * 0.85).max(1.8), color)));

    // Four vertical strings
    for x in [24.0_f32, 29.3, 34.7, 40.0] {
        p.line_segment([pt(x, 18.2), pt(x, 48.5)], Stroke::new(thin, color));
    }
}

fn sample_cubic_chain(segments: &[(Pos2, Pos2, Pos2, Pos2)]) -> Vec<Pos2> {
    const STEPS: usize = 12;
    let mut pts = Vec::with_capacity(segments.len() * STEPS + 1);
    for (idx, &(p0, p1, p2, p3)) in segments.iter().enumerate() {
        let start = usize::from(idx > 0);
        for step in start..=STEPS {
            let t = step as f32 / STEPS as f32;
            let u = 1.0 - t;
            let x = u * u * u * p0.x + 3.0 * u * u * t * p1.x + 3.0 * u * t * t * p2.x + t * t * t * p3.x;
            let y = u * u * u * p0.y + 3.0 * u * u * t * p1.y + 3.0 * u * t * t * p2.y + t * t * t * p3.y;
            pts.push(pos2(x, y));
        }
    }
    pts
}

fn paint_lightning_icon(p: &Painter, c: Pos2, color: Color32) {
    let pts = vec![
        pos2(c.x + 2.0, c.y - 10.5),
        pos2(c.x - 6.5, c.y + 1.0),
        pos2(c.x - 0.5, c.y + 1.0),
        pos2(c.x - 2.5, c.y + 10.5),
        pos2(c.x + 6.5, c.y - 1.5),
        pos2(c.x + 0.5, c.y - 1.5),
    ];
    p.add(PathShape { points: pts, closed: true, fill: color, stroke: PathStroke::NONE });
}

fn paint_video_frame_icon(p: &Painter, c: Pos2, color: Color32) {
    let r = Rect::from_center_size(c, vec2(22.0, 17.0));
    p.rect_stroke(r, CornerRadius::same(3), Stroke::new(1.5, color), StrokeKind::Inside);
    p.line_segment([pos2(r.min.x + 1.0, r.min.y + 4.5), pos2(r.max.x - 1.0, r.min.y + 4.5)], Stroke::new(1.1, color));
    p.line_segment([pos2(r.min.x + 1.0, r.max.y - 4.5), pos2(r.max.x - 1.0, r.max.y - 4.5)], Stroke::new(1.1, color));
    for i in 0..3 {
        let sx = r.min.x + 4.0 + i as f32 * 5.8;
        p.rect_filled(Rect::from_min_size(pos2(sx, r.min.y + 1.8), vec2(2.6, 1.5)), CornerRadius::ZERO, color);
        p.rect_filled(Rect::from_min_size(pos2(sx, r.max.y - 3.3), vec2(2.6, 1.5)), CornerRadius::ZERO, color);
    }
    paint_play_triangle(p, c + vec2(0.5, 0.0), 3.0, color);
}

fn paint_checkmark(p: &Painter, c: Pos2, color: Color32) {
    let pts = vec![pos2(c.x - 4.5, c.y + 0.2), pos2(c.x - 1.2, c.y + 3.5), pos2(c.x + 5.2, c.y - 3.6)];
    p.add(PathShape::line(pts, Stroke::new(1.8, color)));
}

fn paint_play_triangle(p: &Painter, c: Pos2, r: f32, color: Color32) {
    let pts = vec![pos2(c.x - r * 0.65, c.y - r), pos2(c.x + r * 0.95, c.y), pos2(c.x - r * 0.65, c.y + r)];
    p.add(PathShape::convex_polygon(pts, color, Stroke::NONE));
}

fn paint_right_arrow(p: &Painter, c: Pos2, width: f32, color: Color32, stroke_w: f32) {
    let half = width * 0.5;
    let tip = pos2(c.x + half, c.y);
    p.line_segment([pos2(c.x - half, c.y), tip], Stroke::new(stroke_w, color));
    p.add(PathShape::line(vec![pos2(tip.x - 4.2, c.y - 3.8), tip, pos2(tip.x - 4.2, c.y + 3.8)], Stroke::new(stroke_w, color)));
}

fn paint_folder_outline(p: &Painter, r: Rect, color: Color32, stroke_w: f32) {
    let w = r.width();
    let h = r.height();
    let tab_w = w * 0.38;
    let tab_h = h * 0.22;
    let pts = vec![
        pos2(r.min.x, r.min.y + tab_h),
        pos2(r.min.x, r.min.y + 1.5),
        pos2(r.min.x + tab_w, r.min.y + 1.5),
        pos2(r.min.x + tab_w + h * 0.14, r.min.y + tab_h),
        pos2(r.max.x, r.min.y + tab_h),
        pos2(r.max.x, r.max.y),
        pos2(r.min.x, r.max.y),
    ];
    p.add(PathShape { points: pts, closed: true, fill: Color32::TRANSPARENT, stroke: PathStroke::new(stroke_w, color) });
}

fn paint_clock_icon(p: &Painter, c: Pos2, radius: f32, color: Color32) {
    p.circle_stroke(c, radius, Stroke::new(1.5, color));
    p.line_segment([c, pos2(c.x, c.y - radius * 0.52)], Stroke::new(1.5, color));
    p.line_segment([c, pos2(c.x + radius * 0.42, c.y + radius * 0.22)], Stroke::new(1.5, color));
}

fn paint_info_circle_icon(p: &Painter, c: Pos2, radius: f32, color: Color32) {
    p.circle_stroke(c, radius, Stroke::new(1.45, color));
    p.circle_filled(pos2(c.x, c.y - radius * 0.42), 1.15, color);
    p.line_segment([pos2(c.x, c.y - radius * 0.08), pos2(c.x, c.y + radius * 0.50)], Stroke::new(1.55, color));
}

// ---------------------------------------------------------------------------
// Gradient, mesh & UV helpers (zero corner bleed)
// ---------------------------------------------------------------------------

fn cover_uv_upper_right(tex_size: egui::Vec2, rect_size: egui::Vec2) -> Rect {
    if tex_size.x <= 0.0 || tex_size.y <= 0.0 || rect_size.x <= 0.0 || rect_size.y <= 0.0 {
        return Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));
    }
    let tex_aspect = tex_size.x / tex_size.y;
    let rect_aspect = rect_size.x / rect_size.y;
    if tex_aspect > rect_aspect {
        let keep = (rect_aspect / tex_aspect).clamp(0.01, 1.0);
        Rect::from_min_max(pos2(1.0 - keep, 0.0), pos2(1.0, 1.0))
    } else {
        let keep = (tex_aspect / rect_aspect).clamp(0.01, 1.0);
        Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, keep))
    }
}

fn cover_uv(tex_size: egui::Vec2, rect_size: egui::Vec2) -> Rect {
    if tex_size.x <= 0.0 || tex_size.y <= 0.0 || rect_size.x <= 0.0 || rect_size.y <= 0.0 {
        return Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));
    }
    let tex_aspect = tex_size.x / tex_size.y;
    let rect_aspect = rect_size.x / rect_size.y;
    if tex_aspect > rect_aspect {
        let keep = (rect_aspect / tex_aspect).clamp(0.01, 1.0);
        let pad = (1.0 - keep) * 0.5;
        Rect::from_min_max(pos2(pad, 0.0), pos2(1.0 - pad, 1.0))
    } else {
        let keep = (tex_aspect / rect_aspect).clamp(0.01, 1.0);
        let pad = (1.0 - keep) * 0.5;
        Rect::from_min_max(pos2(0.0, pad), pos2(1.0, 1.0 - pad))
    }
}

fn rounded_rect_perimeter(rect: Rect, radius: f32) -> Vec<Pos2> {
    const STEPS: usize = 8;
    let r = radius.min(rect.width() * 0.5).min(rect.height() * 0.5).max(0.0);
    let corners = [
        (pos2(rect.max.x - r, rect.min.y + r), -FRAC_PI_2, 0.0),
        (pos2(rect.max.x - r, rect.max.y - r), 0.0, FRAC_PI_2),
        (pos2(rect.min.x + r, rect.max.y - r), FRAC_PI_2, std::f32::consts::PI),
        (pos2(rect.min.x + r, rect.min.y + r), std::f32::consts::PI, std::f32::consts::PI + FRAC_PI_2),
    ];
    let mut pts = Vec::with_capacity(4 * (STEPS + 1));
    for (center, start_a, end_a) in corners {
        for step in 0..=STEPS {
            let t = step as f32 / STEPS as f32;
            let ang = start_a + (end_a - start_a) * t;
            pts.push(pos2(center.x + ang.cos() * r, center.y + ang.sin() * r));
        }
    }
    pts
}

fn paint_rounded_textured_rect(p: &Painter, rect: Rect, radius: f32, tex_id: TextureId, uv: Rect, tint: Color32) {
    let ring = rounded_rect_perimeter(rect, radius);
    if ring.len() < 3 || rect.width() <= 0.0 || rect.height() <= 0.0 {
        return;
    }
    let uv_at = |pt: Pos2| {
        let u = uv.min.x + ((pt.x - rect.min.x) / rect.width()).clamp(0.0, 1.0) * uv.width();
        let v = uv.min.y + ((pt.y - rect.min.y) / rect.height()).clamp(0.0, 1.0) * uv.height();
        pos2(u, v)
    };
    let mut mesh = Mesh::with_texture(tex_id);
    let center = rect.center();
    mesh.vertices.push(Vertex { pos: center, uv: uv_at(center), color: tint });
    for &pt in &ring {
        mesh.vertices.push(Vertex { pos: pt, uv: uv_at(pt), color: tint });
    }
    let n = ring.len() as u32;
    for i in 0..n {
        mesh.add_triangle(0, 1 + i, 1 + ((i + 1) % n));
    }
    p.add(egui::Shape::mesh(mesh));
}

fn paint_vertical_gradient(p: &Painter, rect: Rect, top: Color32, bottom: Color32) {
    let mut mesh = Mesh::default();
    mesh.vertices.push(Vertex { pos: rect.left_top(), uv: Pos2::ZERO, color: top });
    mesh.vertices.push(Vertex { pos: rect.right_top(), uv: Pos2::ZERO, color: top });
    mesh.vertices.push(Vertex { pos: rect.right_bottom(), uv: Pos2::ZERO, color: bottom });
    mesh.vertices.push(Vertex { pos: rect.left_bottom(), uv: Pos2::ZERO, color: bottom });
    mesh.add_triangle(0, 1, 2);
    mesh.add_triangle(0, 2, 3);
    p.add(egui::Shape::mesh(mesh));
}

fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let mix = |x: u8, y: u8| (f32::from(x) + (f32::from(y) - f32::from(x)) * t).round().clamp(0.0, 255.0) as u8;
    Color32::from_rgba_unmultiplied(mix(a.r(), b.r()), mix(a.g(), b.g()), mix(a.b(), b.b()), mix(a.a(), b.a()))
}

fn paint_rounded_vertical_gradient(p: &Painter, rect: Rect, radius: f32, top: Color32, bottom: Color32) {
    let ring = rounded_rect_perimeter(rect, radius);
    if ring.len() < 3 || rect.height() <= 0.0 {
        return;
    }
    let color_at = |pt: Pos2| lerp_color(top, bottom, (pt.y - rect.min.y) / rect.height());
    let mut mesh = Mesh::default();
    let center = rect.center();
    mesh.vertices.push(Vertex { pos: center, uv: Pos2::ZERO, color: color_at(center) });
    for &pt in &ring {
        mesh.vertices.push(Vertex { pos: pt, uv: Pos2::ZERO, color: color_at(pt) });
    }
    let n = ring.len() as u32;
    for i in 0..n {
        mesh.add_triangle(0, 1 + i, 1 + ((i + 1) % n));
    }
    p.add(egui::Shape::mesh(mesh));
}

fn paint_rounded_left_scrim(p: &Painter, rect: Rect, radius: f32, base: Color32, left_alpha: u8) {
    let ring = rounded_rect_perimeter(rect, radius);
    if ring.len() < 3 || rect.width() <= 0.0 {
        return;
    }
    let color_at = |pt: Pos2| {
        let t = ((pt.x - rect.min.x) / (rect.width() * 0.65)).clamp(0.0, 1.0);
        let a = (f32::from(left_alpha) * (1.0 - t) * (1.0 - t)).round().clamp(0.0, 255.0) as u8;
        Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), a)
    };
    let mut mesh = Mesh::default();
    let center = rect.center();
    mesh.vertices.push(Vertex { pos: center, uv: Pos2::ZERO, color: color_at(center) });
    for &pt in &ring {
        mesh.vertices.push(Vertex { pos: pt, uv: Pos2::ZERO, color: color_at(pt) });
    }
    let n = ring.len() as u32;
    for i in 0..n {
        mesh.add_triangle(0, 1 + i, 1 + ((i + 1) % n));
    }
    p.add(egui::Shape::mesh(mesh));
}

fn paint_pill_horizontal_gradient(p: &Painter, rect: Rect, radius: f32, left: Color32, mid: Color32, right: Color32) {
    let ring = rounded_rect_perimeter(rect, radius);
    if ring.len() < 3 || rect.width() <= 0.0 {
        return;
    }
    let color_at = |pt: Pos2| {
        let t = ((pt.x - rect.min.x) / rect.width()).clamp(0.0, 1.0);
        if t < 0.5 { lerp_color(left, mid, t * 2.0) } else { lerp_color(mid, right, (t - 0.5) * 2.0) }
    };
    let mut mesh = Mesh::default();
    let center = rect.center();
    mesh.vertices.push(Vertex { pos: center, uv: Pos2::ZERO, color: color_at(center) });
    for &pt in &ring {
        mesh.vertices.push(Vertex { pos: pt, uv: Pos2::ZERO, color: color_at(pt) });
    }
    let n = ring.len() as u32;
    for i in 0..n {
        mesh.add_triangle(0, 1 + i, 1 + ((i + 1) % n));
    }
    p.add(egui::Shape::mesh(mesh));
}

fn paint_radial_glow(p: &Painter, center: Pos2, radius: f32, center_color: Color32) {
    const SEGMENTS: usize = 24;
    let mut mesh = Mesh::default();
    mesh.vertices.push(Vertex { pos: center, uv: Pos2::ZERO, color: center_color });
    let edge = Color32::from_rgba_unmultiplied(center_color.r(), center_color.g(), center_color.b(), 0);
    for i in 0..=SEGMENTS {
        let ang = (i as f32 / SEGMENTS as f32) * TAU;
        mesh.vertices.push(Vertex { pos: center + vec2(ang.cos() * radius, ang.sin() * radius), uv: Pos2::ZERO, color: edge });
    }
    for i in 0..SEGMENTS {
        mesh.add_triangle(0, (i + 1) as u32, (i + 2) as u32);
    }
    p.add(egui::Shape::mesh(mesh));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_home_textures_decode_cleanly() {
        for art in [HomeArt::Backdrop, HomeArt::Creator, HomeArt::Studio] {
            let img = art.decoded().expect("embedded home texture PNG must decode");
            assert!(img.size[0] >= 256 && img.size[1] >= 256, "unexpected texture size {:?}", img.size);
        }
    }

    #[test]
    fn start_screen_renders_and_registers_automation_ids() {
        let ctx = egui::Context::default();
        crate::theme::install(&ctx, &Tokens::for_kind(crate::theme::ThemeKind::Dark));
        let mut app = FilmcraftApp::new(filmcraft_engine::Session::default());
        assert!(app.ui.start_screen);
        let full = Rect::from_min_size(Pos2::ZERO, vec2(1280.0, 720.0));
        let mut out = ctx.run_ui(egui::RawInput { screen_rect: Some(full), ..Default::default() }, |ui| {
            show(&mut app, ui, full);
        });
        out.textures_delta.clear();
        for id in [
            "start.creator",
            "start.studio",
            "start.openProject",
            "start.recent.viewAll",
            "start.quick.openProject",
            "start.quick.about",
            "start.quick.settings",
        ] {
            assert!(app.auto.elements.iter().any(|e| e.id == id), "missing automation id `{id}`");
        }
    }

    #[test]
    fn launch_mode_switches_experience_and_leaves_start_screen() {
        let ctx = egui::Context::default();
        let mut app = FilmcraftApp::new(filmcraft_engine::Session::default());
        launch_mode(&mut app, &ctx, AppExperience::Creator);
        assert_eq!(app.ui.experience, AppExperience::Creator);
        assert_eq!(app.ui.mode, Mode::Edit);
        assert!(!app.ui.start_screen);

        app.ui.start_screen = true;
        launch_mode(&mut app, &ctx, AppExperience::Studio);
        assert_eq!(app.ui.experience, AppExperience::Studio);
        assert_eq!(app.ui.mode, Mode::Edit);
        assert!(!app.ui.start_screen);
    }
}
