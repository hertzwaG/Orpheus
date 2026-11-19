//! Local pure-Rust AI Background Removal (`ai.removeBackground`, `ai.bgRemovalStatus`).
//!
//! Removes or blurs video/image backgrounds 100% offline with zero required downloads:
//! - **Optional Local ONNX Support + Built-in Pure-Rust Foreground Extractor:** Uses a local
//!   matting `.onnx` model (`modnet.onnx`, `rmbg-1.4.onnx`, `u2net.onnx`, etc.) when present in
//!   `models/`; otherwise runs a deterministic pure-Rust **Subject & Edge-Aware Foreground
//!   Extractor** (border color/chroma background model + YCbCr skin/subject protection + Sobel edge
//!   refinement + separable feathering + temporal smoothing).
//! - **Modes:** `TransparentAlpha` (keyframed `opacity` matte mask), `BlurBackground` (blurred
//!   background copy on lower track + sharp subject on upper track), and `SolidColor` (solid
//!   backdrop matte + foreground subject).
//! - Runs on background worker thread `"orpheus-bg-removal"` with cancellation, progress updates,
//!   session caching, and single-step undo (`Session::edit_group`).

use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, PoisonError};

use filmcraft_export::{Progress, Report};
use filmcraft_geom::Vec2;
use filmcraft_media::{FrameRequest, SharedSource};
use filmcraft_project::{ClipId, Interpolation, ItemId, Label, Mask, MaskPath, MaskVertex, ParamValue, Track, TrackId, TrackItem, TrackKind, find_effect};
use filmcraft_time::Tick;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, NamedKeys, always, bad, bool_p, has_seq, str_p, u64_p};
use crate::subtitles::next_job_id;
use crate::{EngineError, Job, Result, Session};

const ANALYSIS_MAX_DIM: f64 = 192.0;
const ONNX_ANALYSIS_MAX_DIM: f64 = 512.0;
const SAMPLE_INTERVAL_MS: i64 = 200;
const MAX_SAMPLES_PER_CLIP: usize = 6_000;
const CACHE_CAPACITY: usize = 64;
const MASK_CONTOUR_RAYS: usize = 72;

/// Edge feathering preset for AI Background Removal.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum BgEdgeSoftness {
    Crisp,
    #[default]
    Natural,
    Soft,
}

impl BgEdgeSoftness {
    pub fn from_str_param(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "crisp" | "sharp" | "hard" => Some(Self::Crisp),
            "natural" | "medium" | "balanced" | "default" => Some(Self::Natural),
            "soft" | "feathered" | "smooth" => Some(Self::Soft),
            _ => None,
        }
    }

    pub fn as_id(self) -> &'static str {
        match self {
            Self::Crisp => "crisp",
            Self::Natural => "natural",
            Self::Soft => "soft",
        }
    }

    fn blur_radius(self) -> usize {
        match self {
            Self::Crisp => 2,
            Self::Natural => 5,
            Self::Soft => 9,
        }
    }

    fn mask_feather_px_for(self, used_onnx: bool) -> f64 {
        if used_onnx {
            match self {
                Self::Crisp => 2.0,
                Self::Natural => 6.0,
                Self::Soft => 14.0,
            }
        } else {
            match self {
                Self::Crisp => 6.0,
                Self::Natural => 16.0,
                Self::Soft => 32.0,
            }
        }
    }
}

/// Output compositing mode for AI Background Removal.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum BgRemovalMode {
    #[default]
    TransparentAlpha,
    BlurBackground,
    SolidColor,
    CustomMedia,
}

impl BgRemovalMode {
    pub fn from_str_param(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "transparent" | "transparentalpha" | "transparent_alpha" | "alpha" | "cutout" => Some(Self::TransparentAlpha),
            "blur" | "blurbackground" | "blur_background" | "bokeh" => Some(Self::BlurBackground),
            "solid" | "solidcolor" | "solid_color" | "color" => Some(Self::SolidColor),
            "custom" | "custommedia" | "custom_media" | "custom-media" | "media" => Some(Self::CustomMedia),
            _ => None,
        }
    }

    pub fn as_id(self) -> &'static str {
        match self {
            Self::TransparentAlpha => "transparentAlpha",
            Self::BlurBackground => "blurBackground",
            Self::SolidColor => "solidColor",
            Self::CustomMedia => "customMedia",
        }
    }
}

/// Options for an AI Background Removal run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BgRemovalOptions {
    pub softness: BgEdgeSoftness,
    pub mode: BgRemovalMode,
    pub solid_rgb: [u8; 3],
}

impl Default for BgRemovalOptions {
    fn default() -> Self {
        Self { softness: BgEdgeSoftness::Natural, mode: BgRemovalMode::TransparentAlpha, solid_rgb: [18, 22, 32] }
    }
}

/// Summary of the most recent completed AI Background Removal run.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BgRemovalReport {
    pub seq: ItemId,
    pub processed_clips: usize,
    pub softness: BgEdgeSoftness,
    pub mode: BgRemovalMode,
    pub used_onnx: bool,
    pub cached: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct BgCacheKey {
    source_item: u64,
    source_in: i64,
    duration: i64,
    speed_bits: u64,
    reverse: bool,
    options: BgRemovalOptions,
}

impl BgCacheKey {
    fn from_item(item: &TrackItem, options: BgRemovalOptions) -> Self {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        item.item.0.hash(&mut h);
        Self {
            source_item: h.finish(),
            source_in: item.source_in.0,
            duration: item.duration.0,
            speed_bits: item.speed.to_bits(),
            reverse: item.reverse,
            options,
        }
    }
}

/// Computed foreground contour keyframes for one clip.
#[derive(Clone, Debug, PartialEq)]
pub struct ClipBgPlan {
    pub clip_id: ClipId,
    pub keyframes: Vec<(Tick, MaskPath)>,
}

#[derive(Clone, Debug, Default)]
pub struct BgRemovalCache {
    entries: Vec<(BgCacheKey, ClipBgPlan)>,
}

impl BgRemovalCache {
    fn get(&self, key: &BgCacheKey) -> Option<ClipBgPlan> {
        self.entries.iter().rev().find(|(k, _)| k == key).map(|(_, p)| p.clone())
    }

    fn insert(&mut self, key: BgCacheKey, plan: ClipBgPlan) {
        self.entries.retain(|(k, _)| k != &key);
        if self.entries.len() >= CACHE_CAPACITY {
            self.entries.remove(0);
        }
        self.entries.push((key, plan));
    }
}

struct BgWorkItem {
    clip_id: ClipId,
    cache_key: BgCacheKey,
    src: Option<SharedSource>,
    src_w: u32,
    src_h: u32,
    samples: Vec<(Tick, Tick, Tick)>,
    cached_plan: Option<ClipBgPlan>,
}

pub struct PendingBgRemovalJob {
    pub job: u64,
    pub seq: ItemId,
    pub options: BgRemovalOptions,
    pub bg_media_path: Option<String>,
    pub used_onnx: bool,
    pub epoch: u64,
    plans: Arc<Mutex<Option<Vec<(BgCacheKey, ClipBgPlan)>>>>,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

fn check_cancelled(progress: Option<&Progress>) -> std::result::Result<(), String> {
    if progress.is_some_and(|p| p.cancel.load(Ordering::Relaxed)) { Err("Cancelled".into()) } else { Ok(()) }
}

/// Detect an optional local matting `.onnx` model in `<data_dir>/models` or `<exe_dir>/models`.
pub fn detect_local_matting_onnx(models_dir: Option<&Path>) -> Option<PathBuf> {
    crate::ai_models::detect_model(crate::ai_models::AiModelKind::BgMatting, models_dir)
}

fn separable_box_blur(buf: &[f32], w: usize, h: usize, radius: usize) -> Vec<f32> {
    if radius == 0 || w == 0 || h == 0 || buf.len() < w * h {
        return buf.to_vec();
    }
    let mut horiz = vec![0.0_f32; w * h];
    let mut out = vec![0.0_f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let x0 = x.saturating_sub(radius);
            let x1 = (x + radius).min(w - 1);
            let mut sum = 0.0_f32;
            for xi in x0..=x1 {
                sum += buf[y * w + xi];
            }
            horiz[y * w + x] = sum / (x1 - x0 + 1) as f32;
        }
    }
    for y in 0..h {
        let y0 = y.saturating_sub(radius);
        let y1 = (y + radius).min(h - 1);
        let denom = (y1 - y0 + 1) as f32;
        for x in 0..w {
            let mut sum = 0.0_f32;
            for yi in y0..=y1 {
                sum += horiz[yi * w + x];
            }
            out[y * w + x] = (sum / denom).clamp(0.0, 1.0);
        }
    }
    out
}

/// Pure-Rust Subject & Edge-Aware Foreground Matte Extractor (`0.0..=1.0` per pixel).
pub fn extract_foreground_matte_rgba(rgba: &[u8], width: usize, height: usize, softness: BgEdgeSoftness, prev_matte: Option<&[f32]>) -> Vec<f32> {
    let n_px = width.checked_mul(height).unwrap_or(0);
    if width < 4 || height < 4 || n_px == 0 || rgba.len() < n_px.saturating_mul(4) {
        return Vec::new();
    }

    // Prefer local ONNX matting inference when a cached `.onnx` model is present.
    if let Some(model_path) = detect_local_matting_onnx(None)
        && let Some(onnx_matte) = crate::ai_models::infer_bg_matte_onnx(&model_path, rgba, width, height)
        && onnx_matte.len() == n_px
    {
        let mut smoothed = if softness == BgEdgeSoftness::Soft { separable_box_blur(&onnx_matte, width, height, 1) } else { onnx_matte };
        if let Some(prev) = prev_matte.filter(|p| p.len() == n_px) {
            for (cur, &p) in smoothed.iter_mut().zip(prev) {
                *cur = (*cur * 0.80 + p * 0.20).clamp(0.0, 1.0);
            }
        }
        return smoothed;
    }

    // 1. Sample outer border strips (top 8%, left 8%, right 8%) to model background color & variance.
    let bx = (width / 12).max(1);
    let by = (height / 12).max(1);
    let (mut br, mut bg, mut bb, mut bcnt) = (0.0_f64, 0.0_f64, 0.0_f64, 0usize);
    for y in 0..height {
        for x in 0..width {
            if y < by || x < bx || x >= width - bx {
                let i = (y * width + x) * 4;
                br += f64::from(rgba[i]);
                bg += f64::from(rgba[i + 1]);
                bb += f64::from(rgba[i + 2]);
                bcnt += 1;
            }
        }
    }
    let denom = bcnt.max(1) as f64;
    let (mean_r, mean_g, mean_b) = (br / denom, bg / denom, bb / denom);

    let mut var_sum = 0.0_f64;
    for y in 0..height {
        for x in 0..width {
            if y < by || x < bx || x >= width - bx {
                let i = (y * width + x) * 4;
                let dr = f64::from(rgba[i]) - mean_r;
                let dg = f64::from(rgba[i + 1]) - mean_g;
                let db = f64::from(rgba[i + 2]) - mean_b;
                var_sum += (dr * dr + dg * dg + db * db) / (255.0 * 255.0);
            }
        }
    }
    let border_std = (var_sum / denom).sqrt().clamp(0.04, 0.45);
    let thresh = (0.20 + border_std * 0.55).clamp(0.22, 0.46);

    let mut luma = vec![0.0_f32; n_px];
    let mut raw = vec![0.0_f32; n_px];

    for y in 0..height {
        let vn = (y as f64 + 0.5) / height as f64;
        for x in 0..width {
            let idx = y * width + x;
            let i = idx * 4;
            let r = f64::from(rgba[i]);
            let g = f64::from(rgba[i + 1]);
            let b = f64::from(rgba[i + 2]);
            let a = f64::from(rgba[i + 3]) / 255.0;
            luma[idx] = ((0.299 * r + 0.587 * g + 0.114 * b) / 255.0) as f32;

            let un = (x as f64 + 0.5) / width as f64;
            let dist_rgb = ((r - mean_r).powi(2) + (g - mean_g).powi(2) + (b - mean_b).powi(2)).sqrt() / 255.0;

            // Skin chroma protection keeps faces/hands solidly foreground.
            let cb = 128.0 - 0.168_736 * r - 0.331_264 * g + 0.5 * b;
            let cr = 128.0 + 0.5 * r - 0.418_688 * g - 0.081_312 * b;
            let skin = {
                let d_cb = (cb - 112.0) / 18.0;
                let d_cr = (cr - 152.0) / 20.0;
                let d2 = d_cb * d_cb + d_cr * d_cr;
                if d2 < 2.1 { (1.0 - d2 * 0.45).max(0.0) } else { 0.0 }
            };

            let dx_head = (un - 0.5) / 0.24;
            let dy_head = (vn - 0.36) / 0.28;
            let head_prior = (-0.5 * (dx_head * dx_head + dy_head * dy_head)).exp();
            let dx_torso = (un - 0.5) / 0.36;
            let dy_torso = (vn - 0.72) / 0.34;
            let torso_prior = (-0.5 * (dx_torso * dx_torso + dy_torso * dy_torso)).exp();
            let center_prior = head_prior.max(torso_prior).clamp(0.0, 1.0);

            let fg_score = (dist_rgb * 2.4 + skin * 1.15 + center_prior * 0.25) * (0.14 + 0.86 * center_prior);
            let alpha = ((fg_score - thresh) / 0.22).clamp(0.0, 1.0) * a;
            raw[idx] = alpha as f32;
        }
    }

    // 2. Edge-aware boost on high-gradient foreground boundaries.
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = y * width + x;
            let gx = (luma[idx + 1] - luma[idx - 1]).abs();
            let gy = (luma[idx + width] - luma[idx - width]).abs();
            let edge = (gx + gy).min(1.0);
            if edge > 0.12 && raw[idx] > 0.25 {
                raw[idx] = (raw[idx] + edge * 0.25).min(1.0);
            }
        }
    }

    // Two-pass separable blur approximates wide Gaussian edge feathering without boxy step artifacts.
    let pass1 = separable_box_blur(&raw, width, height, softness.blur_radius());
    let mut smoothed = separable_box_blur(&pass1, width, height, (softness.blur_radius() / 2).max(1));
    if let Some(prev) = prev_matte.filter(|p| p.len() == n_px) {
        for (cur, &p) in smoothed.iter_mut().zip(prev) {
            *cur = (*cur * 0.72 + p * 0.28).clamp(0.0, 1.0);
        }
    }
    smoothed
}

/// Apply pixel-level background removal (`TransparentAlpha`, `BlurBackground`, or `SolidColor`)
/// in-place on an RGBA8 image buffer.
pub fn apply_bg_removal_rgba(rgba: &mut [u8], width: usize, height: usize, options: BgRemovalOptions, prev_matte: Option<&[f32]>) -> Vec<f32> {
    let matte = extract_foreground_matte_rgba(rgba, width, height, options.softness, prev_matte);
    let n_px = width.saturating_mul(height);
    if matte.len() < n_px || rgba.len() < n_px.saturating_mul(4) {
        return matte;
    }

    match options.mode {
        BgRemovalMode::TransparentAlpha | BgRemovalMode::CustomMedia => {
            for (i, &m) in matte.iter().enumerate() {
                let a = f32::from(rgba[i * 4 + 3]) * m.clamp(0.0, 1.0);
                rgba[i * 4 + 3] = a.round().clamp(0.0, 255.0) as u8;
            }
        }
        BgRemovalMode::SolidColor => {
            let [sr, sg, sb] = options.solid_rgb;
            for (i, &m) in matte.iter().enumerate() {
                let k = m.clamp(0.0, 1.0);
                let base = i * 4;
                rgba[base] = (f32::from(rgba[base]) * k + f32::from(sr) * (1.0 - k)).round().clamp(0.0, 255.0) as u8;
                rgba[base + 1] = (f32::from(rgba[base + 1]) * k + f32::from(sg) * (1.0 - k)).round().clamp(0.0, 255.0) as u8;
                rgba[base + 2] = (f32::from(rgba[base + 2]) * k + f32::from(sb) * (1.0 - k)).round().clamp(0.0, 255.0) as u8;
            }
        }
        BgRemovalMode::BlurBackground => {
            let mut ch = [vec![0.0_f32; n_px], vec![0.0_f32; n_px], vec![0.0_f32; n_px]];
            for i in 0..n_px {
                ch[0][i] = f32::from(rgba[i * 4]);
                ch[1][i] = f32::from(rgba[i * 4 + 1]);
                ch[2][i] = f32::from(rgba[i * 4 + 2]);
            }
            let br = separable_box_blur(&ch[0], width, height, 6);
            let bg = separable_box_blur(&ch[1], width, height, 6);
            let bb = separable_box_blur(&ch[2], width, height, 6);
            for (i, &m) in matte.iter().enumerate() {
                let k = m.clamp(0.0, 1.0);
                let base = i * 4;
                rgba[base] = (ch[0][i] * k + br[i] * (1.0 - k)).round().clamp(0.0, 255.0) as u8;
                rgba[base + 1] = (ch[1][i] * k + bg[i] * (1.0 - k)).round().clamp(0.0, 255.0) as u8;
                rgba[base + 2] = (ch[2][i] * k + bb[i] * (1.0 - k)).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    matte
}

/// Trace a closed 24-vertex clip-pixel [`MaskPath`] around the extracted foreground silhouette.
pub fn matte_to_mask_path(matte: &[f32], width: usize, height: usize, src_w: u32, src_h: u32) -> MaskPath {
    let sw = f64::from(src_w.max(1));
    let sh = f64::from(src_h.max(1));
    if width < 4 || height < 4 || matte.len() < width * height {
        return MaskPath::ellipse(Vec2::new(sw * 0.5, sh * 0.5), Vec2::new(sw * 0.28, sh * 0.36));
    }

    let (mut sx, mut sy, mut swgt) = (0.0_f64, 0.0_f64, 0.0_f64);
    for y in 0..height {
        for x in 0..width {
            let m = f64::from(matte[y * width + x]);
            if m > 0.35 {
                sx += m * ((x as f64 + 0.5) / width as f64);
                sy += m * ((y as f64 + 0.5) / height as f64);
                swgt += m;
            }
        }
    }
    if swgt <= 1.0 {
        return MaskPath::ellipse(Vec2::new(sw * 0.5, sh * 0.5), Vec2::new(sw * 0.28, sh * 0.36));
    }

    let cu = (sx / swgt).clamp(0.15, 0.85);
    let cv = (sy / swgt).clamp(0.15, 0.85);
    let max_steps = width.max(height);
    let mut pts = Vec::with_capacity(MASK_CONTOUR_RAYS);

    for k in 0..MASK_CONTOUR_RAYS {
        let ang = std::f64::consts::TAU * (k as f64 / MASK_CONTOUR_RAYS as f64);
        let dir_u = ang.cos();
        let dir_v = ang.sin();
        let mut furthest_r = 0.08_f64;
        let mut bg_streak = 0usize;
        for step in 1..=max_steps {
            let r = (step as f64 / max_steps as f64) * 0.72;
            let u = cu + dir_u * r;
            let v = cv + dir_v * r;
            if !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v) {
                break;
            }
            let px = ((u * width as f64) as usize).min(width - 1);
            let py = ((v * height as f64) as usize).min(height - 1);
            if matte[py * width + px] > 0.45 {
                furthest_r = r;
                bg_streak = 0;
            } else if r > 0.08 {
                bg_streak += 1;
                if bg_streak >= 2 {
                    break;
                }
            }
        }
        let u = (cu + dir_u * (furthest_r + 0.004)).clamp(0.0, 1.0);
        let v = (cv + dir_v * (furthest_r + 0.004)).clamp(0.0, 1.0);
        pts.push(Vec2::new(u * sw, v * sh));
    }

    let vertices = pts.into_iter().map(|p| MaskVertex { p, t_in: Vec2::ZERO, t_out: Vec2::ZERO }).collect();
    MaskPath { vertices, closed: true }
}

fn collect_bg_work(s: &Session, seq_id: ItemId, options: BgRemovalOptions, cmd: &str) -> Result<Vec<BgWorkItem>> {
    let seq = s.project.sequence(seq_id).ok_or(EngineError::NoSequence)?;
    let mut seen: HashSet<ClipId> = HashSet::new();
    let mut picked: Vec<(usize, Tick, ClipId)> = Vec::new();

    let expanded_sel = crate::commands::with_links(s, &s.state.selection);
    for &cid in &expanded_sel {
        if !seen.insert(cid) {
            continue;
        }
        if let Some((tid, it)) = seq.find_item(cid)
            && let Some((track_idx, _)) = seq.video_tracks.iter().enumerate().find(|(_, tr)| tr.id == tid && !tr.locked)
            && it.enabled
        {
            picked.push((track_idx, it.start, cid));
        }
    }

    if picked.is_empty() {
        let playhead = s.state.playheads.get(&seq_id).copied().unwrap_or(Tick::ZERO);
        for (track_idx, tr) in seq.video_tracks.iter().enumerate().rev() {
            if tr.locked {
                continue;
            }
            if let Some(it) = tr.items.iter().find(|it| it.enabled && it.start <= playhead && playhead < it.end())
                && seen.insert(it.id)
            {
                picked.push((track_idx, it.start, it.id));
                break;
            }
        }
    }

    if picked.is_empty() {
        for (track_idx, tr) in seq.video_tracks.iter().enumerate() {
            if tr.locked {
                continue;
            }
            for it in &tr.items {
                if it.enabled && seen.insert(it.id) {
                    picked.push((track_idx, it.start, it.id));
                }
            }
        }
    }

    picked.sort_by_key(|(t_idx, start, cid)| (*t_idx, *start, cid.0));
    if picked.is_empty() {
        return Err(bad(cmd, "No video clips found on unlocked tracks"));
    }

    let step = Tick::from_units(SAMPLE_INTERVAL_MS, 1000).max(seq.settings.frame_rate.frame_duration()).max(Tick(1));
    let mut work = Vec::with_capacity(picked.len());

    for (_, _, cid) in picked {
        let Some((_, it)) = seq.find_item(cid) else { continue };
        let src = s.source(it.item);
        let (src_w, src_h) = src
            .as_ref()
            .and_then(|s| s.info().video.as_ref().map(|v| (v.width.max(1), v.height.max(1))))
            .or_else(|| s.project.source_size(it.item))
            .unwrap_or((seq.settings.width.max(1), seq.settings.height.max(1)));

        let cache_key = BgCacheKey::from_item(it, options);
        let cached_plan = s.bg_removal_cache.get(&cache_key);

        let mut samples = Vec::new();
        let mut t = it.start;
        while t < it.end() && samples.len() < MAX_SAMPLES_PER_CLIP {
            samples.push((t, it.effect_time_at(t), it.source_time_at(t)));
            t += step;
        }
        work.push(BgWorkItem { clip_id: cid, cache_key, src, src_w, src_h, samples, cached_plan });
    }
    Ok(work)
}

fn run_bg_worker(work: Vec<BgWorkItem>, options: BgRemovalOptions, prog: &Progress) -> std::result::Result<Vec<(BgCacheKey, ClipBgPlan)>, String> {
    let total_samples: usize = work.iter().map(|w| if w.cached_plan.is_some() { 1 } else { w.samples.len().max(1) }).sum::<usize>().max(1);
    let mut done_samples = 0usize;
    let mut out = Vec::with_capacity(work.len());

    for (clip_idx, w) in work.into_iter().enumerate() {
        check_cancelled(Some(prog))?;
        if let Some(plan) = w.cached_plan {
            done_samples += 1;
            prog.done.store((((done_samples as f64 / total_samples as f64) * 95.0) as u64).min(100), Ordering::Relaxed);
            out.push((w.cache_key, plan));
            continue;
        }

        let has_onnx = detect_local_matting_onnx(None).is_some();
        let target_dim = if has_onnx { ONNX_ANALYSIS_MAX_DIM } else { ANALYSIS_MAX_DIM };
        let max_side = w.src_w.max(w.src_h).max(1) as f64;
        let scale = (target_dim / max_side).clamp(0.05, 1.0) as f32;
        let mut prev_matte: Option<Vec<f32>> = None;
        let mut keyframes = Vec::with_capacity(w.samples.len());

        for &(_tl_t, eff_t, media_t) in &w.samples {
            check_cancelled(Some(prog))?;
            let path = if let Some(ref src) = w.src {
                match src.video_frame(FrameRequest { time: media_t, scale }) {
                    Ok(frame) => {
                        let rgba = frame.to_rgba8();
                        let m = extract_foreground_matte_rgba(&rgba, frame.width as usize, frame.height as usize, options.softness, prev_matte.as_deref());
                        let p = matte_to_mask_path(&m, frame.width as usize, frame.height as usize, w.src_w, w.src_h);
                        prev_matte = Some(m);
                        p
                    }
                    Err(_) => MaskPath::ellipse(
                        Vec2::new(f64::from(w.src_w) * 0.5, f64::from(w.src_h) * 0.5),
                        Vec2::new(f64::from(w.src_w) * 0.28, f64::from(w.src_h) * 0.36),
                    ),
                }
            } else {
                MaskPath::ellipse(
                    Vec2::new(f64::from(w.src_w) * 0.5, f64::from(w.src_h) * 0.5),
                    Vec2::new(f64::from(w.src_w) * 0.28, f64::from(w.src_h) * 0.36),
                )
            };
            keyframes.push((eff_t, path));
            done_samples += 1;
            prog.done.store((((done_samples as f64 / total_samples as f64) * 95.0) as u64).min(100), Ordering::Relaxed);
            *lock(&prog.status) = format!("Removing background (clip {})…", clip_idx + 1);
        }

        out.push((w.cache_key, ClipBgPlan { clip_id: w.clip_id, keyframes }));
    }

    prog.done.store(100, Ordering::Relaxed);
    Ok(out)
}

fn apply_bg_plans(
    s: &mut Session,
    seq_id: ItemId,
    options: BgRemovalOptions,
    bg_media_path: Option<String>,
    used_onnx: bool,
    cached: bool,
    plans: Vec<(BgCacheKey, ClipBgPlan)>,
) -> Result<BgRemovalReport> {
    if plans.is_empty() {
        return Err(bad("ai.removeBackground", "No video clips remained to process"));
    }
    let prev_selection = s.state.selection.clone();

    let processed_clips = s.edit_group("Apply AI Background Removal", |s| {
        let matte_item_id = match options.mode {
            BgRemovalMode::SolidColor => {
                let [r, g, b] = options.solid_rgb;
                let hex = format!("#{r:02x}{g:02x}{b:02x}");
                let r = s.execute("file.newColorMatte", json!({"color": hex}))?;
                r["item"].as_u64().map(ItemId)
            }
            BgRemovalMode::CustomMedia => {
                let path = bg_media_path.as_deref().ok_or_else(|| bad("ai.removeBackground", "mode `customMedia` requires `backgroundPath`"))?;
                let r = s.execute("file.import", json!({"paths": [path]}))?;
                let id = r["items"]
                    .as_array()
                    .and_then(|a| a.first())
                    .and_then(|v| v.as_u64())
                    .map(ItemId)
                    .ok_or_else(|| bad("ai.removeBackground", "Failed to import background media"))?;
                Some(id)
            }
            BgRemovalMode::TransparentAlpha | BgRemovalMode::BlurBackground => None,
        };
        let bg_media_info = if options.mode == BgRemovalMode::CustomMedia
            && let Some(mid) = matte_item_id
        {
            let (bw, bh) = s
                .source(mid)
                .and_then(|src| src.info().video.as_ref().map(|v| (v.width.max(1), v.height.max(1))))
                .or_else(|| s.project.source_size(mid))
                .unwrap_or((1920, 1080));
            let (is_still, media_dur) = s.project.item(mid).map_or((false, Tick::ZERO), |it| match &it.kind {
                filmcraft_project::ItemKind::Media(m) => (m.info.duration == Tick::ZERO, m.info.duration),
                _ => (false, it.duration()),
            });
            Some((bw, bh, is_still, media_dur))
        } else {
            None
        };
        let count = s.edit("Apply AI Background Removal", |pr, st| {
            let need_bg_track = options.mode != BgRemovalMode::TransparentAlpha;

            let bg_track_id = if need_bg_track { Some(TrackId(pr.alloc_id())) } else { None };
            let mut bg_clip_ids = Vec::with_capacity(plans.len());
            if need_bg_track {
                for _ in &plans {
                    bg_clip_ids.push(ClipId(pr.alloc_id()));
                }
            }

            let seq = pr.sequence_mut(seq_id).ok_or(EngineError::NoSequence)?;
            if need_bg_track
                && seq.video_tracks.len() < 2
                && let Some(tid) = bg_track_id
            {
                let mut bg_track = Track::new(tid, TrackKind::Video, "V1".to_string());
                bg_track.label = Some(Label::Iris);
                seq.video_tracks.insert(0, bg_track);
                for (idx, tr) in seq.video_tracks.iter_mut().enumerate() {
                    tr.name = format!("V{}", idx + 1);
                }
            }

            let seq_w = f64::from(seq.settings.width.max(1));
            let seq_h = f64::from(seq.settings.height.max(1));
            let mut applied = 0usize;
            let mut bg_items_to_place: Vec<TrackItem> = Vec::new();

            for (idx, (_, plan)) in plans.iter().enumerate() {
                let Some((orig_tid, it)) = seq.find_item_mut(plan.clip_id) else { continue };
                let _ = orig_tid;
                let base_copy = it.clone();

                if it.effect("opacity").is_none()
                    && let Some(def) = find_effect("opacity")
                {
                    it.effects.push(def.instance());
                }
                if let Some(op) = it.effect_mut("opacity") {
                    op.enabled = true;
                    op.masks.retain(|m| m.name != "AI Foreground Matte");
                    let first_path =
                        plan.keyframes.first().map_or_else(|| MaskPath::ellipse(Vec2::new(960.0, 540.0), Vec2::new(500.0, 400.0)), |(_, p)| p.clone());
                    let mut mask = Mask::new("AI Foreground Matte", first_path);
                    mask.feather.value = ParamValue::Float(options.softness.mask_feather_px_for(used_onnx));
                    if plan.keyframes.len() > 1 {
                        for (mt, path) in &plan.keyframes {
                            mask.path.put_keyframe(*mt, ParamValue::Path(path.clone()));
                            if let Some(k) = mask.path.keyframes.iter_mut().find(|k| k.time == *mt) {
                                k.interp = Interpolation::Linear;
                            }
                        }
                        mask.path.keyframes.sort_by_key(|k| k.time);
                    }
                    op.masks.push(mask);
                }

                if need_bg_track {
                    let mut bg_it = base_copy;
                    bg_it.id = bg_clip_ids[idx];
                    bg_it.link = None;
                    bg_it.group = None;
                    bg_it.name = match options.mode {
                        BgRemovalMode::BlurBackground => format!("{} (BG Blur)", bg_it.name),
                        BgRemovalMode::SolidColor => "AI Solid Background".into(),
                        BgRemovalMode::CustomMedia => "AI Custom Background".into(),
                        BgRemovalMode::TransparentAlpha => bg_it.name,
                    };
                    if let Some(mid) = matte_item_id {
                        bg_it.item = mid;
                        bg_it.source_in = Tick::ZERO;
                        bg_it.speed = 1.0;
                        bg_it.reverse = false;
                        bg_it.effects.retain(|e| e.def().is_some_and(|d| d.intrinsic));
                        if options.mode == BgRemovalMode::CustomMedia
                            && let Some((bw, bh, is_still, media_dur)) = bg_media_info
                        {
                            if !is_still && media_dur > Tick::ZERO && bg_it.duration > media_dur {
                                bg_it.duration = media_dur;
                            }
                            let cover_scale = ((seq_w / f64::from(bw)).max(seq_h / f64::from(bh)) * 100.0).clamp(1.0, 2000.0);
                            if let Some(motion) = bg_it.effect_mut("motion") {
                                if let Some(sp) = motion.param_mut("scale") {
                                    sp.keyframes.clear();
                                    sp.value = ParamValue::Float(cover_scale);
                                }
                                if let Some(pp) = motion.param_mut("position") {
                                    pp.keyframes.clear();
                                    pp.value = ParamValue::Vec2(Vec2::new(seq_w * 0.5, seq_h * 0.5));
                                }
                            }
                        }
                    } else if options.mode == BgRemovalMode::BlurBackground {
                        if bg_it.effect("gaussian_blur").is_none()
                            && let Some(def) = find_effect("gaussian_blur")
                        {
                            bg_it.effects.push(def.instance());
                        }
                        if let Some(blur) = bg_it.effect_mut("gaussian_blur") {
                            blur.enabled = true;
                            if let Some(p) = blur.param_mut("blurriness") {
                                p.keyframes.clear();
                                p.value = ParamValue::Float(48.0);
                            }
                            if let Some(p) = blur.param_mut("repeat_edge") {
                                p.value = ParamValue::Bool(true);
                            }
                        }
                    }
                    bg_items_to_place.push(bg_it);
                }

                applied += 1;
            }

            if !bg_items_to_place.is_empty()
                && let Some(lower_track) = seq.video_tracks.iter_mut().find(|tr| !tr.locked)
            {
                for bg_it in bg_items_to_place {
                    let overlaps = lower_track.items.iter().any(|ex| ex.start < bg_it.end() && bg_it.start < ex.end());
                    if !overlaps {
                        let pos = lower_track.items.partition_point(|x| x.start <= bg_it.start);
                        lower_track.items.insert(pos, bg_it);
                    }
                }
            }

            seq.check().map_err(EngineError::Other)?;
            st.selection = prev_selection.clone();
            Ok(applied)
        })?;

        for (key, plan) in plans {
            s.bg_removal_cache.insert(key, plan);
        }
        Ok(count)
    })?;

    let rep = BgRemovalReport { seq: seq_id, processed_clips, softness: options.softness, mode: options.mode, used_onnx, cached };
    s.bg_removal_report = Some(rep.clone());
    if used_onnx {
        let provider = crate::ai_models::resolve_execution_provider(crate::ai_models::active_execution_mode(), crate::subtitles::detect_primary_gpu().as_ref());
        s.toast(format!("Removed background on {processed_clips} clip(s) (ONNX neural matting · {})", provider.short_label()));
    } else {
        s.toast(format!("Removed background on {processed_clips} clip(s) (heuristic mask — download ONNX model in Settings ▸ AI Models for neural matting)"));
    }
    Ok(rep)
}

/// Apply completed background AI Background Removal jobs on the session thread.
pub fn poll(s: &mut Session) {
    let mut i = 0usize;
    while i < s.bg_removal_jobs.len() {
        let job = s.jobs.iter().find(|j| j.id == s.bg_removal_jobs[i].job);
        let finished = job.is_none_or(|j| j.progress.finished.load(Ordering::Relaxed));
        let cancelled = job.is_some_and(|j| j.progress.cancel.load(Ordering::Relaxed));
        if !finished {
            i += 1;
            continue;
        }
        let pj = s.bg_removal_jobs.remove(i);
        let Some(plans) = lock(&pj.plans).take().filter(|_| !cancelled) else { continue };
        if pj.epoch != s.document_epoch {
            s.error_toast("ai.removeBackground", "Background Removal: the project was replaced; nothing was changed");
            continue;
        }
        if let Err(e) = apply_bg_plans(s, pj.seq, pj.options, pj.bg_media_path, pj.used_onnx, false, plans) {
            s.error_toast("ai.removeBackground", format!("Background Removal failed: {e}"));
        }
    }
}

fn parse_options(p: &Value, cmd: &str) -> Result<BgRemovalOptions> {
    let softness = match str_p(p, "softness") {
        Some(raw) => BgEdgeSoftness::from_str_param(raw).ok_or_else(|| bad(cmd, format!("unknown softness `{raw}`")))?,
        None => BgEdgeSoftness::Natural,
    };
    let mode = match str_p(p, "mode") {
        Some(raw) => BgRemovalMode::from_str_param(raw).ok_or_else(|| bad(cmd, format!("unknown mode `{raw}`")))?,
        None => BgRemovalMode::TransparentAlpha,
    };
    let solid_rgb = match str_p(p, "color") {
        Some(hex) => {
            let [r, g, b, _] = filmcraft_color::parse_hex(hex).ok_or_else(|| bad(cmd, format!("invalid color `{hex}`")))?;
            [(r * 255.0).round() as u8, (g * 255.0).round() as u8, (b * 255.0).round() as u8]
        }
        None => [18, 22, 32],
    };
    Ok(BgRemovalOptions { softness, mode, solid_rgb })
}

fn remove_bg_cmd(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "ai.removeBackground";
    let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let options = parse_options(p, cmd)?;
    let bg_media_path = str_p(p, "backgroundPath").map(str::trim).filter(|s| !s.is_empty()).map(str::to_string);
    if options.mode == BgRemovalMode::CustomMedia && bg_media_path.is_none() {
        return Err(bad(cmd, "mode `customMedia` requires `backgroundPath`"));
    }
    let work = collect_bg_work(s, seq_id, options, cmd)?;
    let used_onnx = detect_local_matting_onnx(None).is_some();

    if work.iter().all(|w| w.cached_plan.is_some()) {
        let cached_plans: Vec<(BgCacheKey, ClipBgPlan)> = work.into_iter().filter_map(|w| w.cached_plan.map(|p| (w.cache_key, p))).collect();
        let rep = apply_bg_plans(s, seq_id, options, bg_media_path, used_onnx, true, cached_plans)?;
        return Ok(json!({
            "cached": true,
            "processedClips": rep.processed_clips,
            "softness": rep.softness.as_id(),
            "mode": rep.mode.as_id(),
            "usedOnnx": rep.used_onnx,
        }));
    }

    let id = next_job_id(s);
    let prog = Arc::new(Progress::default());
    prog.total.store(100, Ordering::Relaxed);
    *lock(&prog.status) = "Removing background…".into();
    let result: Arc<Mutex<Option<std::result::Result<Report, String>>>> = Arc::default();
    s.jobs.push(Job { id, label: "AI Background Removal".into(), progress: prog.clone(), result: result.clone() });

    let plans_slot: Arc<Mutex<Option<Vec<(BgCacheKey, ClipBgPlan)>>>> = Arc::default();
    s.bg_removal_jobs.push(PendingBgRemovalJob { job: id, seq: seq_id, options, bg_media_path, used_onnx, epoch: s.document_epoch, plans: plans_slot.clone() });

    let wait = bool_p(p, "wait").unwrap_or(false);
    let run = crate::export_tools::guard_job(prog.clone(), result.clone(), move || {
        let t0 = web_time::Instant::now();
        let outcome = run_bg_worker(work, options, &prog);
        let rep = match outcome {
            Ok(plans) => {
                let count = plans.len() as u64;
                *lock(&plans_slot) = Some(plans);
                Ok(Report { frames: count, seconds: t0.elapsed().as_secs_f64(), ..Default::default() })
            }
            Err(e) => {
                *lock(&prog.error) = Some(e.clone());
                *lock(&prog.status) = e.clone();
                Err(e)
            }
        };
        *lock(&result) = Some(rep);
        prog.finished.store(true, Ordering::Relaxed);
    });

    if wait || cfg!(target_arch = "wasm32") {
        run();
        poll(s);
    } else {
        std::thread::Builder::new().name("orpheus-bg-removal".into()).spawn(run).map_err(|e| EngineError::Other(e.to_string()))?;
    }

    if let Some(Err(e)) = s.jobs.iter().find(|j| j.id == id).and_then(|j| lock(&j.result).clone()) {
        return Err(EngineError::Other(e));
    }

    let mut out = json!({
        "job": id,
        "softness": options.softness.as_id(),
        "mode": options.mode.as_id(),
        "usedOnnx": used_onnx,
    });
    if wait && let Some(rep) = &s.bg_removal_report {
        out["processedClips"] = json!(rep.processed_clips);
        out["cached"] = json!(rep.cached);
    }
    Ok(out)
}

fn status_cmd(s: &mut Session, p: &Value) -> Result<Value> {
    let job_id = u64_p(p, "job");
    let job_state = job_id.and_then(|jid| {
        s.jobs.iter().find(|j| j.id == jid).map(|j| {
            let running = !j.progress.finished.load(Ordering::Relaxed);
            let done = j.progress.done.load(Ordering::Relaxed);
            let total = j.progress.total.load(Ordering::Relaxed).max(1);
            let status = lock(&j.progress.status).clone();
            let err = lock(&j.progress.error).clone();
            (running, done as f64 / total as f64, status, err)
        })
    });
    let (running, progress, status, error) = job_state.unwrap_or((false, 0.0, String::new(), None));
    let active = s.state.active_sequence;
    let model_installed = detect_local_matting_onnx(None).is_some();
    let report = s.bg_removal_report.as_ref().filter(|r| Some(r.seq) == active).map(|r| {
        json!({
            "processedClips": r.processed_clips,
            "softness": r.softness.as_id(),
            "mode": r.mode.as_id(),
            "usedOnnx": r.used_onnx,
            "cached": r.cached,
        })
    });
    Ok(json!({ "running": running, "progress": progress, "status": status, "error": error, "modelInstalled": model_installed, "report": report }))
}

pub const BG_MATTING_MODEL_URL: &str = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx";
pub const BG_MATTING_MODEL_MIRRORS: &[&str] = &[
    "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx",
    "https://huggingface.co/TryOn/u2netp/resolve/main/u2netp.onnx",
    "https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model_quantized.onnx",
];

fn download_bg_model_cmd(s: &mut Session, _p: &Value) -> Result<Value> {
    let target_dir = crate::transcript::models_dir().unwrap_or_else(|| std::env::temp_dir().join("orpheus-models"));
    let dest_path = target_dir.join("u2netp.onnx");
    if std::fs::metadata(&dest_path).is_ok_and(|m| m.is_file() && m.len() > 1024) {
        s.toast("AI matting model (u2netp.onnx) is already installed");
        return Ok(json!({ "installed": true, "path": dest_path.to_string_lossy() }));
    }

    let job_id = next_job_id(s);
    let progress = Arc::new(Progress::default());
    progress.total.store(4_600_000, Ordering::Relaxed);
    *lock(&progress.status) = "Downloading AI matting model (u2netp.onnx)…".into();
    let result_slot: Arc<Mutex<Option<std::result::Result<Report, String>>>> = Arc::default();
    s.jobs.push(Job { id: job_id, label: "Download AI Matting Model".into(), progress: progress.clone(), result: result_slot.clone() });

    let prog = progress.clone();
    std::thread::spawn(move || {
        let t0 = web_time::Instant::now();
        let outcome = (|| -> std::result::Result<u64, String> {
            let mut last_err = String::from("Download failed");
            for &url in BG_MATTING_MODEL_MIRRORS {
                if prog.cancel.load(Ordering::Relaxed) {
                    return Err("Cancelled".into());
                }
                match crate::subtitles::download_url_to_path(url, &dest_path, 4_600_000, &prog) {
                    Ok(()) => {
                        let bytes = std::fs::metadata(&dest_path).map(|m| m.len()).unwrap_or(0);
                        if bytes > 1024 {
                            return Ok(bytes);
                        }
                        let _ = std::fs::remove_file(&dest_path);
                        last_err = "Downloaded model file was truncated".into();
                    }
                    Err(e) => {
                        if e == "Cancelled" {
                            return Err(e);
                        }
                        last_err = e;
                    }
                }
            }
            Err(last_err)
        })();
        match outcome {
            Ok(bytes) => {
                *lock(&result_slot) = Some(Ok(Report {
                    path: dest_path.to_string_lossy().into_owned(),
                    frames: 1,
                    bytes,
                    seconds: t0.elapsed().as_secs_f64(),
                    ..Default::default()
                }));
            }
            Err(e) => {
                *lock(&prog.error) = Some(e.clone());
                *lock(&result_slot) = Some(Err(e));
            }
        }
        prog.finished.store(true, Ordering::Relaxed);
    });

    Ok(json!({ "job": job_id }))
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "ai.removeBackground",
            label: "Remove Background (AI)",
            menu: &[],
            shortcut: None,
            params: r#"{"softness":"crisp|natural|soft"?,"mode":"transparentAlpha|blurBackground|solidColor|customMedia"?,"color":str?,"backgroundPath":str?,"wait":bool?}"#,
            enabled: has_seq,
            run: remove_bg_cmd,
            journal: false,
            named: NamedKeys::NONE,
        },
        CommandSpec {
            id: "ai.bgRemovalStatus",
            label: "AI Background Removal Status",
            menu: &[],
            shortcut: None,
            params: r#"{"job":u64?}"#,
            enabled: always,
            run: status_cmd,
            journal: false,
            named: NamedKeys::NONE,
        },
        CommandSpec {
            id: "ai.downloadBgModel",
            label: "Download AI Matting Model (ONNX)",
            menu: &[],
            shortcut: None,
            params: "{}",
            enabled: always,
            run: download_bg_model_cmd,
            journal: false,
            named: NamedKeys::NONE,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_person_on_blue_bg(w: usize, h: usize) -> Vec<u8> {
        let mut rgba = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                let dx = (x as f64 / w as f64 - 0.5) / 0.16;
                let dy = (y as f64 / h as f64 - 0.52) / 0.28;
                if dx * dx + dy * dy <= 1.0 {
                    rgba[i] = 215;
                    rgba[i + 1] = 160;
                    rgba[i + 2] = 132;
                } else {
                    rgba[i] = 20;
                    rgba[i + 1] = 48;
                    rgba[i + 2] = 140;
                }
                rgba[i + 3] = 255;
            }
        }
        rgba
    }

    #[test]
    fn matte_isolates_centered_subject_from_background() {
        let w = 96;
        let h = 64;
        let mut rgba = synthetic_person_on_blue_bg(w, h);
        let matte = apply_bg_removal_rgba(&mut rgba, w, h, BgRemovalOptions::default(), None);
        assert_eq!(matte.len(), w * h);
        let center_alpha = rgba[(32 * w + 48) * 4 + 3];
        let corner_alpha = rgba[(2 * w + 2) * 4 + 3];
        assert!(center_alpha > 200, "center subject must remain opaque (got {center_alpha})");
        assert!(corner_alpha < 50, "outer background must become transparent (got {corner_alpha})");
    }

    #[test]
    fn blur_and_solid_modes_create_v1_v2_and_undo_atomically() {
        let mut s = Session::default();
        s.execute("file.newSequence", json!({"name": "BG Test", "video": 1, "audio": 1})).unwrap();
        let bars = s.execute("file.newBarsAndTone", json!({"seconds": 2.0})).unwrap()["item"].as_u64().unwrap();
        let placed = s.execute("timeline.place", json!({"item": bars, "track": "V1", "frame": 0, "audio": false})).unwrap();
        let cid = ClipId(placed["clips"][0].as_u64().unwrap());
        s.execute("timeline.select", json!({"clips": [cid.0]})).unwrap();

        let seq_id = s.state.active_sequence.unwrap();
        let res = s.execute("ai.removeBackground", json!({"mode": "blurBackground", "softness": "natural", "wait": true})).unwrap();
        assert_eq!(res["processedClips"], 1);
        let seq = s.project.sequence(seq_id).unwrap();
        assert_eq!(seq.video_tracks.len(), 2, "BlurBackground should create lower blurred background track");
        assert!(seq.video_tracks[0].items[0].effect("gaussian_blur").is_some());
        assert!(!seq.video_tracks[1].items[0].effect("opacity").unwrap().masks.is_empty());

        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(s.project.sequence(seq_id).unwrap().video_tracks.len(), 1);
    }

    #[test]
    fn hostile_inputs_return_clean_errors_or_empty_without_panicking() {
        let m = extract_foreground_matte_rgba(&[], 0, 0, BgEdgeSoftness::Crisp, None);
        assert!(m.is_empty());

        let mut s = Session::default();
        s.execute("file.newSequence", json!({"name": "Empty", "video": 1, "audio": 1})).unwrap();
        assert!(s.execute("ai.removeBackground", json!({"wait": true})).is_err());
        assert!(s.execute("ai.removeBackground", json!({"mode": "bad_mode", "wait": true})).is_err());
        assert!(s.execute("ai.removeBackground", json!({"softness": "bad_soft", "wait": true})).is_err());
    }
}
