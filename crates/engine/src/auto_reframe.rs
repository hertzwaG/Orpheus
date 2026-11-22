//! Local pure-Rust AI Auto-Reframe (`ai.autoReframe`, `ai.autoReframeStatus`).
//!
//! Converts horizontal (16:9) or arbitrary video clips into vertical (`9:16`), portrait (`4:5`), or
//! square (`1:1`) framing completely offline:
//! 1. **Optional Local ONNX Model + Built-in Pure-Rust Subject Tracker:** Checks `<data_dir>/models`
//!    for a local face-detection `.onnx` model; when none is installed, runs a deterministic 0 MB
//!    pure-Rust **Face-Region & Motion-Saliency Tracker** (combining YCbCr skin-tone clustering
//!    with an upper-frame speaker prior, Sobel facial contrast energy, and frame-to-frame motion
//!    difference).
//! 2. **Camera Trajectory Smoothing (`ReframeSmoothing`):** Filters raw subject coordinates using
//!    configurable motion damping (**Locked**, **Smooth Pan**, **Dynamic Follow**) with dead-zone
//!    hysteresis and outlier rejection so the virtual camera never jitters.
//! 3. **Undoable Timeline Application (`Session::edit_group`):** Optionally resizes the active
//!    sequence canvas to the target aspect (`1080x1920`, `1080x1350`, or `1080x1080`), scales clips
//!    to fill the frame with zero black bars, and writes `Interpolation::AutoBezier` Motion
//!    `position` keyframes in a single undoable edit (`Ctrl+Z` restores original framing and
//!    sequence resolution).

use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, PoisonError};

use filmcraft_export::{Progress, Report};
use filmcraft_geom::Vec2;
use filmcraft_media::{FrameRequest, SharedSource};
use filmcraft_project::{ClipId, Interpolation, ItemId, ParamValue, Track, TrackId, TrackItem, TrackKind, find_effect};
use filmcraft_time::Tick;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, NamedKeys, always, bad, bool_p, has_seq, str_p, u64_p};
use crate::subtitles::next_job_id;
use crate::{EngineError, Job, Result, Session};

/// Downscaled longest frame dimension (px) used during subject saliency analysis.
const ANALYSIS_MAX_DIM: f64 = 192.0;
/// Sampling interval (200 ms = 5 Hz) across each video clip.
const SAMPLE_INTERVAL_MS: i64 = 200;
/// Maximum sampled frames per clip (20 minutes at 5 Hz) to bound memory and CPU time.
const MAX_SAMPLES_PER_CLIP: usize = 6_000;
/// Maximum cached clip reframe plans per session.
const CACHE_CAPACITY: usize = 64;

/// Target aspect ratio preset for AI Auto-Reframe.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ReframeAspect {
    #[default]
    Vertical9x16,
    Portrait4x5,
    Square1x1,
}

impl ReframeAspect {
    pub fn from_str_param(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "9:16" | "9x16" | "vertical" | "vertical9x16" | "reels" | "shorts" | "tiktok" => Some(Self::Vertical9x16),
            "4:5" | "4x5" | "portrait" | "portrait4x5" => Some(Self::Portrait4x5),
            "1:1" | "1x1" | "square" | "square1x1" => Some(Self::Square1x1),
            _ => None,
        }
    }

    pub fn as_id(self) -> &'static str {
        match self {
            Self::Vertical9x16 => "9:16",
            Self::Portrait4x5 => "4:5",
            Self::Square1x1 => "1:1",
        }
    }

    pub fn dimensions(self) -> (u32, u32) {
        match self {
            Self::Vertical9x16 => (1080, 1920),
            Self::Portrait4x5 => (1080, 1350),
            Self::Square1x1 => (1080, 1080),
        }
    }

    pub fn aspect_ratio(self) -> f64 {
        let (w, h) = self.dimensions();
        f64::from(w) / f64::from(h)
    }
}

/// Camera tracking damping preset for AI Auto-Reframe.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ReframeSmoothing {
    Locked,
    #[default]
    SmoothPan,
    DynamicFollow,
}

impl ReframeSmoothing {
    pub fn from_str_param(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "locked" | "tripod" | "static" | "hold" => Some(Self::Locked),
            "smoothpan" | "smooth_pan" | "smooth pan" | "smooth" | "balanced" | "default" => Some(Self::SmoothPan),
            "dynamicfollow" | "dynamic_follow" | "dynamic follow" | "dynamic" | "fast" | "action" => Some(Self::DynamicFollow),
            _ => None,
        }
    }

    pub fn as_id(self) -> &'static str {
        match self {
            Self::Locked => "locked",
            Self::SmoothPan => "smoothPan",
            Self::DynamicFollow => "dynamicFollow",
        }
    }

    /// `(dead_zone_x, dead_zone_y, alpha_x, alpha_y, min_keyframe_delta_px)`
    fn damping_params(self) -> (f64, f64, f64, f64, f64) {
        match self {
            Self::Locked => (1.0, 1.0, 0.0, 0.0, 9999.0),
            Self::SmoothPan => (0.045, 0.055, 0.22, 0.15, 2.2),
            Self::DynamicFollow => (0.018, 0.028, 0.48, 0.32, 1.2),
        }
    }
}

/// Placement of the FaceCam panel in Gaming Split-Screen mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum FacecamPlacement {
    #[default]
    Bottom,
    Top,
}

impl FacecamPlacement {
    pub fn from_str_param(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "bottom" | "facecambottom" | "facecam_bottom" | "facecam bottom" => Some(Self::Bottom),
            "top" | "facecamtop" | "facecam_top" | "facecam top" => Some(Self::Top),
            _ => None,
        }
    }

    pub fn as_id(self) -> &'static str {
        match self {
            Self::Bottom => "bottom",
            Self::Top => "top",
        }
    }
}

/// Minimum combined border + face-oval confidence required to auto-accept a FaceCam box.
pub const MIN_FACECAM_CONFIDENCE: f64 = 0.28;

/// Source region where the streamer webcam lives in Gaming Split-Screen mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum FacecamRegion {
    #[default]
    Auto,
    LeftMiddle,
    BottomLeft,
    BottomRight,
    TopLeft,
    TopRight,
    RightMiddle,
    TopMiddle,
    BottomMiddle,
}

impl FacecamRegion {
    pub fn from_str_param(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().replace(['-', '_', ' '], "").as_str() {
            "auto" | "autodetect" => Some(Self::Auto),
            "leftmiddle" | "middleleft" | "left" => Some(Self::LeftMiddle),
            "bottomleft" => Some(Self::BottomLeft),
            "bottomright" => Some(Self::BottomRight),
            "topleft" => Some(Self::TopLeft),
            "topright" => Some(Self::TopRight),
            "rightmiddle" | "middleright" | "right" => Some(Self::RightMiddle),
            "topmiddle" | "middletop" => Some(Self::TopMiddle),
            "bottommiddle" | "middlebottom" => Some(Self::BottomMiddle),
            _ => None,
        }
    }

    pub fn as_id(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::LeftMiddle => "leftMiddle",
            Self::BottomLeft => "bottomLeft",
            Self::BottomRight => "bottomRight",
            Self::TopLeft => "topLeft",
            Self::TopRight => "topRight",
            Self::RightMiddle => "rightMiddle",
            Self::TopMiddle => "topMiddle",
            Self::BottomMiddle => "bottomMiddle",
        }
    }

    /// Return the locked crop box for a manually selected webcam zone, or `None` for `Auto`.
    pub fn manual_box(self) -> Option<CornerFacecamBox> {
        match self {
            Self::Auto => None,
            Self::TopLeft => Some(CornerFacecamBox { corner_idx: 0, left_pct: 1.0, top_pct: 2.0, right_pct: 73.0, bottom_pct: 66.0, score: 1.0 }),
            Self::TopRight => Some(CornerFacecamBox { corner_idx: 1, left_pct: 73.0, top_pct: 2.0, right_pct: 1.0, bottom_pct: 66.0, score: 1.0 }),
            Self::BottomLeft => Some(CornerFacecamBox { corner_idx: 2, left_pct: 1.0, top_pct: 66.0, right_pct: 73.0, bottom_pct: 2.0, score: 1.0 }),
            Self::BottomRight => Some(CornerFacecamBox { corner_idx: 3, left_pct: 73.0, top_pct: 66.0, right_pct: 1.0, bottom_pct: 2.0, score: 1.0 }),
            Self::LeftMiddle => Some(CornerFacecamBox { corner_idx: 4, left_pct: 0.0, top_pct: 25.0, right_pct: 75.0, bottom_pct: 25.0, score: 1.0 }),
            Self::RightMiddle => Some(CornerFacecamBox { corner_idx: 5, left_pct: 75.0, top_pct: 25.0, right_pct: 0.0, bottom_pct: 25.0, score: 1.0 }),
            Self::TopMiddle => Some(CornerFacecamBox { corner_idx: 6, left_pct: 37.0, top_pct: 2.0, right_pct: 37.0, bottom_pct: 66.0, score: 1.0 }),
            Self::BottomMiddle => Some(CornerFacecamBox { corner_idx: 7, left_pct: 37.0, top_pct: 66.0, right_pct: 37.0, bottom_pct: 2.0, score: 1.0 }),
        }
    }
}

/// Configuration for an AI Auto-Reframe pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AutoReframeOptions {
    pub aspect: ReframeAspect,
    pub smoothing: ReframeSmoothing,
    pub resize_sequence: bool,
    pub split_screen: bool,
    pub facecam_placement: FacecamPlacement,
    #[serde(default)]
    pub facecam_region: FacecamRegion,
}

impl Default for AutoReframeOptions {
    fn default() -> Self {
        Self {
            aspect: ReframeAspect::Vertical9x16,
            smoothing: ReframeSmoothing::SmoothPan,
            resize_sequence: true,
            split_screen: false,
            facecam_placement: FacecamPlacement::Bottom,
            facecam_region: FacecamRegion::Auto,
        }
    }
}

/// Detected corner FaceCam overlay rectangle expressed as crop percentages (`0..100`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CornerFacecamBox {
    pub corner_idx: u8,
    pub left_pct: f64,
    pub top_pct: f64,
    pub right_pct: f64,
    pub bottom_pct: f64,
    pub score: f64,
}

/// Detected subject center in normalized source coordinates (`0.0..=1.0`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FrameSubject {
    pub x: f64,
    pub y: f64,
    pub confidence: f64,
}

impl Default for FrameSubject {
    fn default() -> Self {
        Self { x: 0.5, y: 0.42, confidence: 0.0 }
    }
}

/// Computed scale and Motion `position` keyframes for one timeline video clip.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClipReframePlan {
    pub clip_id: ClipId,
    pub scale_pct: f64,
    /// `(media_time, sequence_position_px)`
    pub keyframes: Vec<(Tick, Vec2)>,
    pub split_screen: Option<SplitScreenOverlayPlan>,
}

/// Companion FaceCam overlay crop + scale/position for Gaming Split-Screen mode.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SplitScreenOverlayPlan {
    pub facecam_box: CornerFacecamBox,
    pub facecam_scale_pct: f64,
    pub facecam_anchor: Vec2,
    pub facecam_pos: Vec2,
    pub gameplay_crop_left_pct: f64,
    pub gameplay_crop_top_pct: f64,
    pub gameplay_crop_right_pct: f64,
    pub gameplay_crop_bottom_pct: f64,
    pub gameplay_anchor: Vec2,
}

/// Summary of the most recent completed AI Auto-Reframe operation on a sequence.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AutoReframeReport {
    pub seq: ItemId,
    pub reframed_clips: usize,
    pub total_keyframes: usize,
    pub split_screen_clips: usize,
    pub aspect: ReframeAspect,
    pub smoothing: ReframeSmoothing,
    pub resize_sequence: bool,
    pub split_screen: bool,
    pub used_onnx: bool,
    pub cached: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ReframeCacheKey {
    source_item: u64,
    source_in: i64,
    duration: i64,
    speed_bits: u64,
    reverse: bool,
    target_w: u32,
    target_h: u32,
    options: AutoReframeOptions,
}

impl ReframeCacheKey {
    fn from_item(item: &TrackItem, target_w: u32, target_h: u32, options: AutoReframeOptions) -> Self {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        item.item.0.hash(&mut h);
        Self {
            source_item: h.finish(),
            source_in: item.source_in.0,
            duration: item.duration.0,
            speed_bits: item.speed.to_bits(),
            reverse: item.reverse,
            target_w,
            target_h,
            options,
        }
    }
}

/// Bounded session cache storing computed [`ClipReframePlan`]s by clip signature and options.
#[derive(Clone, Debug, Default)]
pub struct AutoReframeCache {
    entries: Vec<(ReframeCacheKey, ClipReframePlan)>,
}

impl AutoReframeCache {
    fn get(&self, key: &ReframeCacheKey) -> Option<ClipReframePlan> {
        self.entries.iter().rev().find(|(k, _)| k == key).map(|(_, p)| p.clone())
    }

    fn insert(&mut self, key: ReframeCacheKey, plan: ClipReframePlan) {
        self.entries.retain(|(k, _)| k != &key);
        if self.entries.len() >= CACHE_CAPACITY {
            self.entries.remove(0);
        }
        self.entries.push((key, plan));
    }
}

struct ClipReframeWork {
    clip_id: ClipId,
    cache_key: ReframeCacheKey,
    src: Option<SharedSource>,
    src_w: u32,
    src_h: u32,
    target_w: u32,
    target_h: u32,
    /// `(timeline_tick, effect_time_tick, source_media_tick)`
    samples: Vec<(Tick, Tick, Tick)>,
    cached_plan: Option<ClipReframePlan>,
}

pub struct PendingAutoReframeJob {
    pub job: u64,
    pub seq: ItemId,
    pub options: AutoReframeOptions,
    pub used_onnx: bool,
    pub epoch: u64,
    plans: Arc<Mutex<Option<Vec<(ReframeCacheKey, ClipReframePlan)>>>>,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

fn check_cancelled(progress: Option<&Progress>) -> std::result::Result<(), String> {
    if progress.is_some_and(|p| p.cancel.load(Ordering::Relaxed)) { Err("Cancelled".into()) } else { Ok(()) }
}

/// Check if an optional local face-detection `.onnx` model exists in `<data_dir>/models` or next
/// to the executable. No network requests are ever made.
pub fn detect_local_face_onnx(models_dir: Option<&Path>) -> Option<PathBuf> {
    crate::ai_models::detect_model(crate::ai_models::AiModelKind::FaceDetect, models_dir)
}

/// Pure-Rust face-region, skin-chroma, spatial-contrast, and temporal-motion saliency tracker.
///
/// Evaluates an RGBA8 frame (`width * height * 4` bytes) and optional previous-frame luma buffer
/// (`width * height` floats in `0.0..=1.0`), returning the primary subject centroid `(x, y)` in
/// normalized `[0.0, 1.0]` coordinates and the current frame's luma buffer.
pub fn detect_frame_subject_rgba(rgba: &[u8], width: usize, height: usize, prev_luma: Option<&[f32]>) -> (FrameSubject, Vec<f32>) {
    let n_px = width.checked_mul(height).unwrap_or(0);
    if width < 4 || height < 4 || n_px == 0 || rgba.len() < n_px.saturating_mul(4) {
        return (FrameSubject::default(), Vec::new());
    }

    if let Some(model_path) = crate::ai_models::detect_model(crate::ai_models::AiModelKind::FaceDetect, None)
        && let Some((cx, cy, conf)) = crate::ai_models::infer_face_bbox_onnx(&model_path, rgba, width, height)
    {
        let mut luma = vec![0.0_f32; n_px];
        for i in 0..n_px {
            let base = i * 4;
            let r = f32::from(rgba[base]);
            let g = f32::from(rgba[base + 1]);
            let b = f32::from(rgba[base + 2]);
            let a = f32::from(rgba[base + 3]) / 255.0;
            luma[i] = ((0.299 * r + 0.587 * g + 0.114 * b) / 255.0) * a;
        }
        return (FrameSubject { x: cx, y: cy, confidence: conf.clamp(0.0, 1.0) }, luma);
    }

    let mut luma = vec![0.0_f32; n_px];
    let mut skin_weight = vec![0.0_f32; n_px];

    for i in 0..n_px {
        let base = i * 4;
        let r = f32::from(rgba[base]);
        let g = f32::from(rgba[base + 1]);
        let b = f32::from(rgba[base + 2]);
        let a = f32::from(rgba[base + 3]) / 255.0;

        let y_val = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
        luma[i] = y_val * a;

        // ITU-R BT.601 YCbCr chroma for skin-tone likelihood across diverse skin tones.
        let cb = 128.0 - 0.168_736 * r - 0.331_264 * g + 0.5 * b;
        let cr = 128.0 + 0.5 * r - 0.418_688 * g - 0.081_312 * b;
        if (0.12..=0.93).contains(&y_val) && a > 0.2 {
            let d_cb = (cb - 112.0) / 18.0;
            let d_cr = (cr - 152.0) / 20.0;
            let dist_sq = d_cb * d_cb + d_cr * d_cr;
            if dist_sq < 2.2 {
                skin_weight[i] = (1.0 - dist_sq * 0.45).max(0.0) * a;
            }
        }
    }

    let use_prev = prev_luma.filter(|p| p.len() == n_px);
    let mut heat = vec![0.0_f32; n_px];
    let mut sum_x = 0.0_f64;
    let mut sum_y = 0.0_f64;
    let mut sum_w = 0.0_f64;

    for y in 1..height - 1 {
        let v_norm = (y as f64 + 0.5) / height as f64;
        // Upper-third head/speaker prior peaks around y = 0.32 and tapers toward bottom edge.
        let head_prior = (1.0 - ((v_norm - 0.34) / 0.48).powi(2)).clamp(0.25, 1.0) as f32;
        for x in 1..width - 1 {
            let idx = y * width + x;
            let u_norm = (x as f64 + 0.5) / width as f64;
            let center_bias = (1.0 - ((u_norm - 0.5).powi(2) * 0.55 + (v_norm - 0.45).powi(2) * 0.45)).clamp(0.35, 1.0) as f32;

            let gx = luma[idx + 1] - luma[idx - 1];
            let gy = luma[idx + width] - luma[idx - width];
            let grad = (gx * gx + gy * gy).sqrt();

            let motion = use_prev.map_or(0.0_f32, |prev| {
                let d = (luma[idx] - prev[idx]).abs();
                if d > 0.025 { d - 0.025 } else { 0.0 }
            });

            let skin = skin_weight[idx] * head_prior;
            // Combine facial skin + local detail, active motion, and general contrast saliency.
            let w = (skin * (0.35 + 2.2 * grad) * 2.4 + motion * 1.8 + grad * grad * 0.85) * center_bias;
            heat[idx] = w;
            let wf = f64::from(w);
            if wf > 1e-5 {
                sum_x += wf * u_norm;
                sum_y += wf * v_norm;
                sum_w += wf;
            }
        }
    }

    if sum_w <= 1e-6 {
        return (FrameSubject::default(), luma);
    }

    let coarse_x = (sum_x / sum_w).clamp(0.05, 0.95);
    let coarse_y = (sum_y / sum_w).clamp(0.05, 0.95);

    // Local mean-shift refinement around the primary saliency cluster so a dominant speaker wins
    // cleanly over secondary edges on the opposite side of a 16:9 frame.
    let mut ref_x = 0.0_f64;
    let mut ref_y = 0.0_f64;
    let mut ref_w = 0.0_f64;
    let radius_x = 0.30_f64;
    let radius_y = 0.36_f64;

    for y in 1..height - 1 {
        let v_norm = (y as f64 + 0.5) / height as f64;
        let dy = (v_norm - coarse_y) / radius_y;
        if dy.abs() > 1.0 {
            continue;
        }
        for x in 1..width - 1 {
            let u_norm = (x as f64 + 0.5) / width as f64;
            let dx = (u_norm - coarse_x) / radius_x;
            let r2 = dx * dx + dy * dy;
            if r2 <= 1.0 {
                let kernel = 1.0 - r2 * 0.65;
                let w = f64::from(heat[y * width + x]) * kernel;
                ref_x += w * u_norm;
                ref_y += w * v_norm;
                ref_w += w;
            }
        }
    }

    let (final_x, final_y) = if ref_w > 1e-6 { ((ref_x / ref_w).clamp(0.05, 0.95), (ref_y / ref_w).clamp(0.08, 0.92)) } else { (coarse_x, coarse_y) };
    let mean_heat = sum_w / (n_px as f64);
    let confidence = (mean_heat * 18.0).clamp(0.05, 1.0);
    (FrameSubject { x: final_x, y: final_y, confidence }, luma)
}

fn avg_row_step(rgba: &[u8], width: usize, height: usize, y: usize, x_start: usize, x_end: usize) -> f64 {
    if y == 0 || y + 1 >= height || x_end <= x_start {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    let mut cnt = 0usize;
    for x in x_start..x_end {
        let up = ((y - 1) * width + x) * 4;
        let dn = ((y + 1) * width + x) * 4;
        let dr = (f64::from(rgba[dn]) - f64::from(rgba[up])).abs();
        let dg = (f64::from(rgba[dn + 1]) - f64::from(rgba[up + 1])).abs();
        let db = (f64::from(rgba[dn + 2]) - f64::from(rgba[up + 2])).abs();
        sum += (dr + dg + db) / (3.0 * 255.0);
        cnt += 1;
    }
    sum / cnt.max(1) as f64
}

fn avg_col_step(rgba: &[u8], width: usize, x: usize, y_start: usize, y_end: usize) -> f64 {
    if x == 0 || x + 1 >= width || y_end <= y_start {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    let mut cnt = 0usize;
    for y in y_start..y_end {
        let lf = (y * width + (x - 1)) * 4;
        let rt = (y * width + (x + 1)) * 4;
        let dr = (f64::from(rgba[rt]) - f64::from(rgba[lf])).abs();
        let dg = (f64::from(rgba[rt + 1]) - f64::from(rgba[lf + 1])).abs();
        let db = (f64::from(rgba[rt + 2]) - f64::from(rgba[lf + 2])).abs();
        sum += (dr + dg + db) / (3.0 * 255.0);
        cnt += 1;
    }
    sum / cnt.max(1) as f64
}

fn snap_row_border(rgba: &[u8], width: usize, height: usize, y_start: usize, y_end: usize, x_start: usize, x_end: usize) -> Option<f64> {
    if y_end <= y_start + 2 {
        return None;
    }
    let mut best = (y_start, 0.0_f64);
    for y in (y_start + 1)..y_end {
        let g = avg_row_step(rgba, width, height, y, x_start, x_end);
        if g > best.1 {
            best = (y, g);
        }
    }
    (best.1 > 0.05).then(|| best.0 as f64 / height as f64)
}

fn snap_col_border(rgba: &[u8], width: usize, x_start: usize, x_end: usize, y_start: usize, y_end: usize) -> Option<f64> {
    if x_end <= x_start + 2 {
        return None;
    }
    let mut best = (x_start, 0.0_f64);
    for x in (x_start + 1)..x_end {
        let g = avg_col_step(rgba, width, x, y_start, y_end);
        if g > best.1 {
            best = (x, g);
        }
    }
    (best.1 > 0.05).then(|| best.0 as f64 / width as f64)
}

fn row_mean_luma(rgba: &[u8], width: usize, y: usize, x_start: usize, x_end: usize) -> f64 {
    let mut sum = 0.0;
    let mut cnt = 0usize;
    for x in x_start..x_end {
        let i = (y * width + x) * 4;
        let luma = 0.299 * f64::from(rgba[i]) + 0.587 * f64::from(rgba[i + 1]) + 0.114 * f64::from(rgba[i + 2]);
        sum += luma;
        cnt += 1;
    }
    sum / cnt.max(1) as f64
}

fn col_mean_luma(rgba: &[u8], width: usize, x: usize, y_start: usize, y_end: usize) -> f64 {
    let mut sum = 0.0;
    let mut cnt = 0usize;
    for y in y_start..y_end {
        let i = (y * width + x) * 4;
        let luma = 0.299 * f64::from(rgba[i]) + 0.587 * f64::from(rgba[i + 1]) + 0.114 * f64::from(rgba[i + 2]);
        sum += luma;
        cnt += 1;
    }
    sum / cnt.max(1) as f64
}

const DARK_BORDER_LUMA: f64 = 45.0;

fn scan_outward_dark_top(rgba: &[u8], width: usize, height: usize, fy0_px: usize, min_y: usize, x_start: usize, x_end: usize) -> Option<f64> {
    let mut seen_lit = false;
    let mut dark_run = 0usize;
    let mut y = fy0_px;
    while y > min_y {
        y -= 1;
        let luma = row_mean_luma(rgba, width, y, x_start, x_end);
        if luma >= DARK_BORDER_LUMA {
            seen_lit = true;
            dark_run = 0;
        } else if seen_lit {
            dark_run += 1;
            if dark_run >= 2 {
                return Some((y + dark_run) as f64 / height as f64);
            }
        }
    }
    None
}

fn scan_outward_dark_bottom(rgba: &[u8], width: usize, height: usize, fy1_px: usize, max_y: usize, x_start: usize, x_end: usize) -> Option<f64> {
    let mut seen_lit = false;
    let mut dark_run = 0usize;
    for y in fy1_px..max_y {
        let luma = row_mean_luma(rgba, width, y, x_start, x_end);
        if luma >= DARK_BORDER_LUMA {
            seen_lit = true;
            dark_run = 0;
        } else if seen_lit {
            dark_run += 1;
            if dark_run >= 2 {
                return Some((y + 1 - dark_run) as f64 / height as f64);
            }
        }
    }
    None
}

fn scan_outward_dark_left(rgba: &[u8], width: usize, fx0_px: usize, min_x: usize, y_start: usize, y_end: usize) -> Option<f64> {
    let mut seen_lit = false;
    let mut dark_run = 0usize;
    let mut x = fx0_px;
    while x > min_x {
        x -= 1;
        let luma = col_mean_luma(rgba, width, x, y_start, y_end);
        if luma >= DARK_BORDER_LUMA {
            seen_lit = true;
            dark_run = 0;
        } else if seen_lit {
            dark_run += 1;
            if dark_run >= 2 {
                return Some((x + dark_run) as f64 / width as f64);
            }
        }
    }
    None
}

fn scan_outward_dark_right(rgba: &[u8], width: usize, fx1_px: usize, max_x: usize, y_start: usize, y_end: usize) -> Option<f64> {
    let mut seen_lit = false;
    let mut dark_run = 0usize;
    for x in fx1_px..max_x {
        let luma = col_mean_luma(rgba, width, x, y_start, y_end);
        if luma >= DARK_BORDER_LUMA {
            seen_lit = true;
            dark_run = 0;
        } else if seen_lit {
            dark_run += 1;
            if dark_run >= 2 {
                return Some((x + 1 - dark_run) as f64 / width as f64);
            }
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn refine_pip_box_around_face(
    rgba: &[u8],
    width: usize,
    height: usize,
    face: crate::ai_models::OnnxFaceBox,
    zone_idx: u8,
    zx0: f64,
    zy0: f64,
    zx1: f64,
    zy1: f64,
) -> CornerFacecamBox {
    let fw = (face.x1 - face.x0).max(0.06);
    let fh = (face.y1 - face.y0).max(0.08);
    let win_w = (fw * 2.2).clamp(0.10, (zx1 - zx0).max(0.10));
    let win_h = (fh * 1.7).clamp(0.10, (zy1 - zy0).max(0.10));

    let wx0 = (face.cx - win_w * 0.5).clamp(zx0, (zx1 - 0.06).max(zx0));
    let wx1 = (face.cx + win_w * 0.5).clamp(wx0 + 0.06, zx1);
    let wy0 = (face.cy - win_h * 0.45).clamp(zy0, (zy1 - 0.06).max(zy0));
    let wy1 = (face.cy + win_h * 0.55).clamp(wy0 + 0.06, zy1);

    let fx0_px = ((face.x0 * width as f64).round() as usize).clamp(1, width.saturating_sub(2));
    let fx1_px = ((face.x1 * width as f64).round() as usize).clamp(fx0_px + 1, width.saturating_sub(1));
    let fy0_px = ((face.y0 * height as f64).round() as usize).clamp(1, height.saturating_sub(2));
    let fy1_px = ((face.y1 * height as f64).round() as usize).clamp(fy0_px + 1, height.saturating_sub(1));

    let wx0_px = ((wx0 * width as f64).round() as usize).clamp(0, fx0_px);
    let wx1_px = ((wx1 * width as f64).round() as usize).clamp(fx1_px, width.saturating_sub(1));
    let wy0_px = ((wy0 * height as f64).round() as usize).clamp(0, fy0_px);
    let wy1_px = ((wy1 * height as f64).round() as usize).clamp(fy1_px, height.saturating_sub(1));

    // Scan outward from the ONNX face box to stop at the dark gap/border (< 45 luma) before
    // any outer HUD counters or adjacent gameplay strip, falling back to max-step border snap.
    let snap_top = scan_outward_dark_top(rgba, width, height, fy0_px, wy0_px, fx0_px, fx1_px)
        .or_else(|| snap_row_border(rgba, width, height, wy0_px, fy0_px, fx0_px, fx1_px))
        .unwrap_or(wy0);
    let snap_bot = scan_outward_dark_bottom(rgba, width, height, fy1_px, wy1_px, fx0_px, fx1_px)
        .or_else(|| snap_row_border(rgba, width, height, fy1_px, wy1_px, fx0_px, fx1_px))
        .unwrap_or(wy1);
    let snap_left = scan_outward_dark_left(rgba, width, fx0_px, wx0_px, fy0_px, fy1_px)
        .or_else(|| snap_col_border(rgba, width, wx0_px, fx0_px, fy0_px, fy1_px))
        .unwrap_or(wx0);
    let snap_right = scan_outward_dark_right(rgba, width, fx1_px, wx1_px, fy0_px, fy1_px)
        .or_else(|| snap_col_border(rgba, width, fx1_px, wx1_px, fy0_px, fy1_px))
        .unwrap_or(wx1);

    // Apply a 1.5% inner safety inset on non-flush edges so outer HUD counters or gameplay bleed stay out.
    let bw = (snap_right - snap_left).max(0.06);
    let bh = (snap_bot - snap_top).max(0.06);
    let inset_x = bw * 0.015;
    let inset_y = bh * 0.015;

    let top_bound = (snap_top + inset_y).min(face.y0);
    let bot_bound = (snap_bot - inset_y).max(face.y1);
    let mut y0 = top_bound;
    let mut y1 = bot_bound;
    if y1 - y0 > 0.27 {
        let center_y = face.cy - fh * 0.05;
        y0 = (center_y - 0.135).clamp(top_bound, (bot_bound - 0.27).max(top_bound));
        y1 = (y0 + 0.27).min(bot_bound);
    }

    let (x0, x1) = if matches!(zone_idx, 0 | 2 | 4) {
        let right_bound = (snap_right - inset_x).max(face.x1);
        (0.0, right_bound.clamp(0.06, 0.28))
    } else if matches!(zone_idx, 1 | 3 | 5) {
        let left_bound = (snap_left + inset_x).min(face.x0);
        (left_bound.clamp(0.72, 0.94), 1.0)
    } else {
        let l = if snap_left > 0.005 { (snap_left + inset_x).min(face.x0) } else { snap_left };
        let r = if snap_right < 0.995 { (snap_right - inset_x).max(face.x1) } else { snap_right };
        (l, r)
    };

    CornerFacecamBox {
        corner_idx: zone_idx,
        left_pct: (x0 * 100.0).clamp(0.0, 90.0),
        top_pct: (y0 * 100.0).clamp(0.0, 90.0),
        right_pct: ((1.0 - x1) * 100.0).clamp(0.0, 90.0),
        bottom_pct: ((1.0 - y1) * 100.0).clamp(0.0, 90.0),
        score: (0.65 + face.conf * 0.35).max(MIN_FACECAM_CONFIDENCE),
    }
}

/// Scan all 8 edge/middle zones (`Top-Left`, `Top-Right`, `Bottom-Left`, `Bottom-Right`,
/// `Left-Middle`, `Right-Middle`, `Top-Middle`, `Bottom-Middle`) of an RGBA8 frame for a streamer
/// FaceCam PiP overlay using BOTH sharp rectangular border gradients and concentrated face-oval
/// skin-tone clustering so HUD numbers never outscore an actual webcam.
pub fn detect_corner_facecam_rgba(rgba: &[u8], width: usize, height: usize) -> Option<CornerFacecamBox> {
    let n_px = width.checked_mul(height).unwrap_or(0);
    if width < 16 || height < 16 || n_px == 0 || rgba.len() < n_px.saturating_mul(4) {
        return None;
    }

    // (zone_idx, x0, y0, x1, y1, inner_x_edges, inner_y_edges)
    // 0: Top-Left, 1: Top-Right, 2: Bottom-Left, 3: Bottom-Right,
    // 4: Left-Middle (x: 0..25%, y: 25..75%), 5: Right-Middle, 6: Top-Middle, 7: Bottom-Middle.
    struct Zone {
        idx: u8,
        x0: f64,
        y0: f64,
        x1: f64,
        y1: f64,
        vx_edges: &'static [f64],
        hy_edges: &'static [f64],
    }
    const ZONES: [Zone; 8] = [
        Zone { idx: 0, x0: 0.01, y0: 0.02, x1: 0.27, y1: 0.34, vx_edges: &[0.27], hy_edges: &[0.34] },
        Zone { idx: 1, x0: 0.73, y0: 0.02, x1: 0.99, y1: 0.34, vx_edges: &[0.73], hy_edges: &[0.34] },
        Zone { idx: 2, x0: 0.01, y0: 0.66, x1: 0.27, y1: 0.98, vx_edges: &[0.27], hy_edges: &[0.66] },
        Zone { idx: 3, x0: 0.73, y0: 0.66, x1: 0.99, y1: 0.98, vx_edges: &[0.73], hy_edges: &[0.66] },
        Zone { idx: 4, x0: 0.00, y0: 0.25, x1: 0.25, y1: 0.75, vx_edges: &[0.25], hy_edges: &[0.25, 0.75] },
        Zone { idx: 5, x0: 0.75, y0: 0.25, x1: 1.00, y1: 0.75, vx_edges: &[0.75], hy_edges: &[0.25, 0.75] },
        Zone { idx: 6, x0: 0.37, y0: 0.02, x1: 0.63, y1: 0.34, vx_edges: &[0.37, 0.63], hy_edges: &[0.34] },
        Zone { idx: 7, x0: 0.37, y0: 0.66, x1: 0.63, y1: 0.98, vx_edges: &[0.37, 0.63], hy_edges: &[0.66] },
    ];

    if let Some(model_path) = detect_local_face_onnx(None)
        && let Some(face) = crate::ai_models::infer_face_box_onnx(&model_path, rgba, width, height)
        && let Some(z) = ZONES.iter().find(|z| (z.x0..=z.x1).contains(&face.cx) && (z.y0..=z.y1).contains(&face.cy))
    {
        return Some(refine_pip_box_around_face(rgba, width, height, face, z.idx, z.x0, z.y0, z.x1, z.y1));
    }

    let luma_at = |x: usize, y: usize| -> f64 {
        let i = (y * width + x) * 4;
        (0.299 * f64::from(rgba[i]) + 0.587 * f64::from(rgba[i + 1]) + 0.114 * f64::from(rgba[i + 2])) / 255.0
    };

    let mut best: Option<CornerFacecamBox> = None;

    for z in &ZONES {
        let x0 = ((z.x0 * width as f64).round() as usize).clamp(0, width - 3);
        let y0 = ((z.y0 * height as f64).round() as usize).clamp(0, height - 3);
        let x1 = ((z.x1 * width as f64).round() as usize).clamp(x0 + 2, width - 1);
        let y1 = ((z.y1 * height as f64).round() as usize).clamp(y0 + 2, height - 1);

        let bw = (x1 - x0).max(1) as f64;
        let bh = (y1 - y0).max(1) as f64;
        let mut skin_sum = 0.0_f64;
        let mut oval_skin_sum = 0.0_f64;
        let mut oval_cnt = 0usize;
        let mut count = 0usize;

        for y in y0..y1 {
            let ny = ((y - y0) as f64 + 0.5) / bh - 0.45;
            for x in x0..x1 {
                let nx = ((x - x0) as f64 + 0.5) / bw - 0.50;
                let i = (y * width + x) * 4;
                let r = f64::from(rgba[i]);
                let g = f64::from(rgba[i + 1]);
                let b = f64::from(rgba[i + 2]);
                let yv = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
                let cb = 128.0 - 0.168_736 * r - 0.331_264 * g + 0.5 * b;
                let cr = 128.0 + 0.5 * r - 0.418_688 * g - 0.081_312 * b;
                let mut skin = 0.0_f64;
                if (0.14..=0.92).contains(&yv) {
                    let d_cb = (cb - 112.0) / 18.0;
                    let d_cr = (cr - 152.0) / 20.0;
                    let dist_sq = d_cb * d_cb + d_cr * d_cr;
                    if dist_sq < 2.2 {
                        skin = (1.0 - dist_sq * 0.45).max(0.0);
                    }
                }
                skin_sum += skin;
                // Face-oval cluster in the upper-center of the webcam frame.
                let in_oval = (nx / 0.36).powi(2) + (ny / 0.38).powi(2) <= 1.0;
                if in_oval {
                    oval_skin_sum += skin;
                    oval_cnt += 1;
                }
                count += 1;
            }
        }
        let skin_density = skin_sum / count.max(1) as f64;
        let oval_skin_density = oval_skin_sum / oval_cnt.max(1) as f64;

        // Require concentrated face-oval skin clustering so HUD numbers/minimaps never win.
        if oval_skin_density < 0.06 && skin_density < 0.05 {
            continue;
        }

        // Measure sharp rectangular PiP border gradients along BOTH vertical and horizontal inner
        // edges (scanning a small +-3 px band so slightly inset streamer webcams still register).
        let mut v_edge = 0.0_f64;
        for &ex_f in z.vx_edges {
            let base = ((ex_f * width as f64).round() as isize).clamp(1, width as isize - 2);
            let mut best_line = 0.0_f64;
            for delta in -3isize..=3isize {
                let ex = (base + delta).clamp(1, width as isize - 2) as usize;
                let mut sum = 0.0_f64;
                let mut cnt = 0usize;
                for y in y0..y1 {
                    sum += (luma_at(ex + 1, y) - luma_at(ex - 1, y)).abs();
                    cnt += 1;
                }
                best_line = best_line.max(sum / cnt.max(1) as f64);
            }
            v_edge += best_line;
        }
        v_edge /= z.vx_edges.len().max(1) as f64;

        let mut h_edge = 0.0_f64;
        for &ey_f in z.hy_edges {
            let base = ((ey_f * height as f64).round() as isize).clamp(1, height as isize - 2);
            let mut best_line = 0.0_f64;
            for delta in -3isize..=3isize {
                let ey = (base + delta).clamp(1, height as isize - 2) as usize;
                let mut sum = 0.0_f64;
                let mut cnt = 0usize;
                for x in x0..x1 {
                    sum += (luma_at(x, ey + 1) - luma_at(x, ey - 1)).abs();
                    cnt += 1;
                }
                best_line = best_line.max(sum / cnt.max(1) as f64);
            }
            h_edge += best_line;
        }
        h_edge /= z.hy_edges.len().max(1) as f64;
        let rect_border = (v_edge * h_edge).sqrt() * 0.65 + (v_edge + h_edge) * 0.175;
        if rect_border < 0.04 {
            continue;
        }

        let score = oval_skin_density * 2.4 + skin_density * 1.1 + rect_border * 1.8;
        if score >= MIN_FACECAM_CONFIDENCE && best.is_none_or(|b| score > b.score) {
            best = Some(CornerFacecamBox {
                corner_idx: z.idx,
                left_pct: (z.x0 * 100.0).clamp(0.0, 90.0),
                top_pct: (z.y0 * 100.0).clamp(0.0, 90.0),
                right_pct: ((1.0 - z.x1) * 100.0).clamp(0.0, 90.0),
                bottom_pct: ((1.0 - z.y1) * 100.0).clamp(0.0, 90.0),
                score,
            });
        }
    }

    best
}

/// Smooth raw per-frame subject detections into a jitter-free normalized camera path `(u, v)`.
pub fn smooth_subject_trajectory(samples: &[(Tick, FrameSubject)], smoothing: ReframeSmoothing) -> Vec<(Tick, (f64, f64))> {
    if samples.is_empty() {
        return Vec::new();
    }
    if samples.len() == 1 || smoothing == ReframeSmoothing::Locked {
        let mut wx = 0.0_f64;
        let mut wy = 0.0_f64;
        let mut wsum = 0.0_f64;
        for (_, s) in samples {
            let w = s.confidence.max(0.1);
            wx += s.x * w;
            wy += s.y * w;
            wsum += w;
        }
        let cx = if wsum > 0.0 { (wx / wsum).clamp(0.05, 0.95) } else { 0.5 };
        let cy = if wsum > 0.0 { (wy / wsum).clamp(0.08, 0.92) } else { 0.45 };
        return vec![(samples[0].0, (cx, cy))];
    }

    // 3-tap median/weighted pre-filter rejects single-frame lighting/motion spikes.
    let n = samples.len();
    let mut pre = Vec::with_capacity(n);
    for i in 0..n {
        let prev = samples[i.saturating_sub(1)].1;
        let cur = samples[i].1;
        let next = samples[(i + 1).min(n - 1)].1;
        let x = 0.25 * prev.x + 0.50 * cur.x + 0.25 * next.x;
        let y = 0.25 * prev.y + 0.50 * cur.y + 0.25 * next.y;
        pre.push((samples[i].0, x.clamp(0.05, 0.95), y.clamp(0.08, 0.92)));
    }

    let (dead_x, dead_y, alpha_x, alpha_y, _) = smoothing.damping_params();
    let mut cam_x = pre[0].1;
    let mut cam_y = pre[0].2;
    let mut out = Vec::with_capacity(n);

    for &(t, tx, ty) in &pre {
        let err_x = tx - cam_x;
        if err_x.abs() > dead_x {
            let step_x = err_x - err_x.signum() * dead_x * 0.5;
            cam_x = (cam_x + alpha_x * step_x).clamp(0.05, 0.95);
        }
        let err_y = ty - cam_y;
        if err_y.abs() > dead_y {
            let step_y = err_y - err_y.signum() * dead_y * 0.5;
            cam_y = (cam_y + alpha_y * step_y).clamp(0.08, 0.92);
        }
        out.push((t, (cam_x, cam_y)));
    }
    out
}

/// Convert a smoothed normalized subject trajectory `(u, v)` into sequence Motion `scale` (%) and
/// clamped `position` keyframes `(media_time, Vec2)` that guarantee zero black bars.
pub fn build_clip_reframe_plan(
    clip_id: ClipId,
    src_size: (u32, u32),
    target_seq_size: (u32, u32),
    resize_sequence: bool,
    smoothing: ReframeSmoothing,
    smoothed_path: &[(Tick, (f64, f64))],
) -> ClipReframePlan {
    let src_w = f64::from(src_size.0.max(1));
    let src_h = f64::from(src_size.1.max(1));
    let sw = f64::from(target_seq_size.0.max(1));
    let sh = f64::from(target_seq_size.1.max(1));

    // Scale to completely cover the target canvas with a tiny 0.1% subpixel guard against seams.
    let cover_scale = (sw / src_w).max(sh / src_h);
    let min_reframe_scale = if !resize_sequence && ((sw / src_w) - (sh / src_h)).abs() < 0.02 { cover_scale * 1.22 } else { cover_scale };
    let scale_factor = min_reframe_scale.max(0.1) * 1.001;
    let scale_pct = ((scale_factor * 100.0) * 100.0).round() / 100.0;
    let eff_scale = scale_pct / 100.0;

    let scaled_w = src_w * eff_scale;
    let scaled_h = src_h * eff_scale;
    let max_shift_x = ((scaled_w - sw) * 0.5).max(0.0);
    let max_shift_y = ((scaled_h - sh) * 0.5).max(0.0);
    let center_x = sw * 0.5;
    let center_y = sh * 0.5;

    let to_seq_pos = |(u, v): (f64, f64)| -> Vec2 {
        let dx = (u.clamp(0.0, 1.0) - 0.5) * scaled_w;
        let dy = (v.clamp(0.0, 1.0) - 0.5) * scaled_h;
        let px = (center_x - dx).clamp(center_x - max_shift_x, center_x + max_shift_x);
        let py = (center_y - dy).clamp(center_y - max_shift_y, center_y + max_shift_y);
        Vec2::new((px * 10.0).round() / 10.0, (py * 10.0).round() / 10.0)
    };

    if smoothed_path.is_empty() {
        return ClipReframePlan { clip_id, scale_pct, keyframes: vec![(Tick::ZERO, Vec2::new(center_x, center_y))], split_screen: None };
    }

    if smoothing == ReframeSmoothing::Locked || smoothed_path.len() == 1 {
        let pos = to_seq_pos(smoothed_path[0].1);
        return ClipReframePlan { clip_id, scale_pct, keyframes: vec![(smoothed_path[0].0, pos)], split_screen: None };
    }

    let (_, _, _, _, min_delta_px) = smoothing.damping_params();
    let mut keyframes: Vec<(Tick, Vec2)> = Vec::new();

    for (idx, &(mt, uv)) in smoothed_path.iter().enumerate() {
        let pos = to_seq_pos(uv);
        let is_first = idx == 0;
        let is_last = idx + 1 == smoothed_path.len();
        let should_emit = if is_first || is_last {
            true
        } else if let Some(&(_, prev_pos)) = keyframes.last() {
            let dist = ((pos.x - prev_pos.x).powi(2) + (pos.y - prev_pos.y).powi(2)).sqrt();
            dist >= min_delta_px
        } else {
            true
        };
        if should_emit {
            if let Some(last) = keyframes.last_mut()
                && last.0 == mt
            {
                last.1 = pos;
                continue;
            }
            keyframes.push((mt, pos));
        }
    }

    keyframes.sort_by_key(|(t, _)| *t);
    ClipReframePlan { clip_id, scale_pct, keyframes, split_screen: None }
}

pub fn build_split_screen_plan(
    clip_id: ClipId,
    src_size: (u32, u32),
    target_seq_size: (u32, u32),
    placement: FacecamPlacement,
    facecam_box: CornerFacecamBox,
    _smoothing: ReframeSmoothing,
    smoothed_path: &[(Tick, (f64, f64))],
) -> ClipReframePlan {
    let src_w = f64::from(src_size.0.max(1));
    let src_h = f64::from(src_size.1.max(1));
    let sw = f64::from(target_seq_size.0.max(1));
    let sh = f64::from(target_seq_size.1.max(1));

    // Gameplay fills 55% of the vertical frame; FaceCam fills the remaining 45%.
    let game_h = sh * 0.55;
    let cam_h = sh * 0.45;
    let (game_cy, cam_cy) = match placement {
        FacecamPlacement::Bottom => (game_h * 0.5, game_h + cam_h * 0.5),
        FacecamPlacement::Top => (cam_h + game_h * 0.5, cam_h * 0.5),
    };
    let center_x = sw * 0.5;

    let raw_x0 = (facecam_box.left_pct / 100.0) * src_w;
    let raw_x1 = ((100.0 - facecam_box.right_pct) / 100.0) * src_w;
    let raw_y0 = (facecam_box.top_pct / 100.0) * src_h;
    let raw_y1 = ((100.0 - facecam_box.bottom_pct) / 100.0) * src_h;
    let raw_w = (raw_x1 - raw_x0).max(16.0);
    let raw_h = (raw_y1 - raw_y0).max(16.0);

    // 1. Strictly exclude the entire webcam horizontal span `[raw_x0..raw_x1]` from the top
    // gameplay crop window so the FaceCam never appears twice.
    let cam_is_left = matches!(facecam_box.corner_idx, 0 | 2 | 4) || ((raw_x0 + raw_x1) * 0.5 < src_w * 0.5 && !matches!(facecam_box.corner_idx, 1 | 3 | 5));
    let (avail_x0, avail_x1) = if cam_is_left {
        let x0 = (raw_x1 + src_w * 0.004).min(src_w * 0.70).max(raw_x1);
        (x0, src_w)
    } else {
        let x1 = (raw_x0 - src_w * 0.004).max(src_w * 0.30).min(raw_x0);
        (0.0, x1)
    };
    let avail_w = (avail_x1 - avail_x0).max(16.0);
    let game_pane_ar = sw / game_h.max(1.0);
    let avail_ar = avail_w / src_h;
    let (crop_gw, crop_gh) =
        if avail_ar >= game_pane_ar { ((src_h * game_pane_ar).min(avail_w), src_h) } else { (avail_w, (avail_w / game_pane_ar).min(src_h)) };

    let mean_u = if smoothed_path.is_empty() {
        ((avail_x0 + avail_x1) * 0.5) / src_w
    } else {
        smoothed_path.iter().map(|(_, (u, _))| *u).sum::<f64>() / (smoothed_path.len() as f64)
    };
    let target_cx = (mean_u * src_w).clamp(avail_x0 + avail_w * 0.35, avail_x1 - avail_w * 0.35);
    let gx0 = (target_cx - crop_gw * 0.5).clamp(avail_x0, (avail_x1 - crop_gw).max(avail_x0));
    let gx1 = (gx0 + crop_gw).min(avail_x1);
    let gy0 = ((src_h - crop_gh) * 0.5).max(0.0);
    let gy1 = (gy0 + crop_gh).min(src_h);

    let game_crop_left = (((gx0 / src_w) * 100.0 * 100.0).ceil() / 100.0).clamp(0.0, 95.0);
    let game_crop_right = ((((src_w - gx1) / src_w) * 100.0 * 100.0).ceil() / 100.0).clamp(0.0, 95.0);
    let game_crop_top = (((gy0 / src_h) * 100.0 * 100.0).round() / 100.0).clamp(0.0, 95.0);
    let game_crop_bot = ((((src_h - gy1) / src_h) * 100.0 * 100.0).round() / 100.0).clamp(0.0, 95.0);

    let act_gx0 = (game_crop_left / 100.0) * src_w;
    let act_gx1 = ((100.0 - game_crop_right) / 100.0) * src_w;
    let act_gy0 = (game_crop_top / 100.0) * src_h;
    let act_gy1 = ((100.0 - game_crop_bot) / 100.0) * src_h;
    let act_gw = (act_gx1 - act_gx0).max(16.0);
    let act_gh = (act_gy1 - act_gy0).max(16.0);
    let game_scale = (((sw / act_gw).max(game_h / act_gh) * 100.0 * 1.004) * 100.0).round() / 100.0;
    let gameplay_anchor = Vec2::new(((act_gx0 + act_gx1) * 0.5 * 10.0).round() / 10.0, ((act_gy0 + act_gy1) * 0.5 * 10.0).round() / 10.0);
    let gameplay_pos = Vec2::new((center_x * 10.0).round() / 10.0, (game_cy * 10.0).round() / 10.0);
    let game_keyframes = vec![(Tick::ZERO, gameplay_pos)];

    // 2. Tighten the FaceCam box to its inner PiP frame (trimming HUD text/counters above and
    // below tall edge zones and shaving the outer border stroke), then aspect-match and scale
    // to fill 100% of `sw x cam_h` (bottom 45% pane) edge-to-edge.
    let cx = (raw_x0 + raw_x1) * 0.5;
    let cy = (raw_y0 + raw_y1) * 0.5;
    let inner_w = (raw_w * 0.96).max(16.0);
    let inner_h = (raw_h.min(raw_w * 0.72) * 0.96).max(16.0);
    let pane_ar = sw / cam_h.max(1.0);
    let box_ar = inner_w / inner_h;
    let (adj_w, adj_h) = if box_ar > pane_ar { (inner_h * pane_ar, inner_h) } else { (inner_w, inner_w / pane_ar) };
    let x0 = (cx - adj_w * 0.5).clamp(raw_x0, (raw_x1 - 4.0).max(raw_x0));
    let x1 = (cx + adj_w * 0.5).clamp(x0 + 4.0, raw_x1);
    let y0 = (cy - adj_h * 0.5).clamp(raw_y0, (raw_y1 - 4.0).max(raw_y0));
    let y1 = (cy + adj_h * 0.5).clamp(y0 + 4.0, raw_y1);
    let fitted_box = CornerFacecamBox {
        corner_idx: facecam_box.corner_idx,
        left_pct: (((x0 / src_w) * 100.0 * 100.0).round() / 100.0).clamp(0.0, 95.0),
        top_pct: (((y0 / src_h) * 100.0 * 100.0).round() / 100.0).clamp(0.0, 95.0),
        right_pct: ((((src_w - x1) / src_w) * 100.0 * 100.0).round() / 100.0).clamp(0.0, 95.0),
        bottom_pct: ((((src_h - y1) / src_h) * 100.0 * 100.0).round() / 100.0).clamp(0.0, 95.0),
        score: facecam_box.score,
    };
    let fit_x0 = (fitted_box.left_pct / 100.0) * src_w;
    let fit_x1 = ((100.0 - fitted_box.right_pct) / 100.0) * src_w;
    let fit_y0 = (fitted_box.top_pct / 100.0) * src_h;
    let fit_y1 = ((100.0 - fitted_box.bottom_pct) / 100.0) * src_h;
    let box_w = (fit_x1 - fit_x0).max(16.0);
    let box_h = (fit_y1 - fit_y0).max(16.0);
    let cam_scale = (((sw / box_w).max(cam_h / box_h) * 100.0 * 1.004) * 100.0).round() / 100.0;

    let overlay = SplitScreenOverlayPlan {
        facecam_box: fitted_box,
        facecam_scale_pct: cam_scale,
        facecam_anchor: Vec2::new(((fit_x0 + fit_x1) * 0.5 * 10.0).round() / 10.0, ((fit_y0 + fit_y1) * 0.5 * 10.0).round() / 10.0),
        facecam_pos: Vec2::new((center_x * 10.0).round() / 10.0, (cam_cy * 10.0).round() / 10.0),
        gameplay_crop_left_pct: game_crop_left,
        gameplay_crop_top_pct: game_crop_top,
        gameplay_crop_right_pct: game_crop_right,
        gameplay_crop_bottom_pct: game_crop_bot,
        gameplay_anchor,
    };

    ClipReframePlan { clip_id, scale_pct: game_scale, keyframes: game_keyframes, split_screen: Some(overlay) }
}

fn collect_reframe_work(s: &Session, seq_id: ItemId, options: AutoReframeOptions, cmd: &str) -> Result<Vec<ClipReframeWork>> {
    let seq = s.project.sequence(seq_id).ok_or(EngineError::NoSequence)?;
    let (target_w, target_h) = if options.resize_sequence { options.aspect.dimensions() } else { (seq.settings.width.max(1), seq.settings.height.max(1)) };

    let mut seen: HashSet<ClipId> = HashSet::new();
    let mut picked: Vec<(usize, Tick, ClipId)> = Vec::new();

    for &cid in &s.state.selection {
        if !seen.insert(cid) {
            continue;
        }
        if let Some((tid, it)) = seq.find_item(cid)
            && let Some((track_idx, tr)) = seq.video_tracks.iter().enumerate().find(|(_, tr)| tr.id == tid && !tr.locked)
            && it.enabled
            && !it.name.ends_with(" (FaceCam)")
        {
            let _ = tr;
            picked.push((track_idx, it.start, cid));
        }
    }

    if picked.is_empty() {
        for (track_idx, tr) in seq.video_tracks.iter().enumerate() {
            if tr.locked {
                continue;
            }
            for it in &tr.items {
                if it.name.ends_with(" (FaceCam)") {
                    continue;
                }
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

        let cache_key = ReframeCacheKey::from_item(it, target_w, target_h, options);
        let cached_plan = s.auto_reframe_cache.get(&cache_key);

        let mut samples = Vec::new();
        let mut t = it.start;
        while t < it.end() && samples.len() < MAX_SAMPLES_PER_CLIP {
            samples.push((t, it.effect_time_at(t), it.source_time_at(t)));
            t += step;
        }
        let end_t = (it.end() - Tick(1)).max(it.start);
        if samples.last().is_none_or(|(last_t, _, _)| *last_t != end_t) && samples.len() < MAX_SAMPLES_PER_CLIP {
            samples.push((end_t, it.effect_time_at(end_t), it.source_time_at(end_t)));
        }

        work.push(ClipReframeWork { clip_id: cid, cache_key, src, src_w, src_h, target_w, target_h, samples, cached_plan });
    }

    Ok(work)
}

fn run_reframe_worker(
    work: Vec<ClipReframeWork>,
    options: AutoReframeOptions,
    prog: &Progress,
) -> std::result::Result<Vec<(ReframeCacheKey, ClipReframePlan)>, String> {
    let total_samples: usize = work.iter().map(|w| if w.cached_plan.is_some() { 1 } else { w.samples.len().max(1) }).sum::<usize>().max(1);
    let mut done_samples = 0usize;
    let mut out = Vec::with_capacity(work.len());

    for (clip_idx, w) in work.into_iter().enumerate() {
        check_cancelled(Some(prog))?;
        if let Some(plan) = w.cached_plan {
            done_samples += 1;
            let pct = ((done_samples as f64 / total_samples as f64) * 95.0) as u64;
            prog.done.store(pct.min(100), Ordering::Relaxed);
            out.push((w.cache_key, plan));
            continue;
        }

        let max_side = w.src_w.max(w.src_h).max(1) as f64;
        let analysis_scale = (ANALYSIS_MAX_DIM / max_side).clamp(0.05, 1.0) as f32;
        let mut prev_luma: Option<Vec<f32>> = None;
        let mut raw_detections: Vec<(Tick, FrameSubject)> = Vec::with_capacity(w.samples.len());
        let mut zone_votes = [0.0_f64; 8];
        let mut best_boxes: [Option<CornerFacecamBox>; 8] = [None; 8];
        let manual_facecam = options.facecam_region.manual_box();

        for &(_tl_t, eff_t, media_t) in &w.samples {
            check_cancelled(Some(prog))?;
            let subject = if let Some(ref src) = w.src {
                match src.video_frame(FrameRequest { time: media_t, scale: analysis_scale }) {
                    Ok(frame) => {
                        let rgba = frame.to_rgba8();
                        let fw = frame.width as usize;
                        let fh = frame.height as usize;
                        if options.split_screen
                            && manual_facecam.is_none()
                            && let Some(cbox) = detect_corner_facecam_rgba(&rgba, fw, fh)
                        {
                            let c = usize::from(cbox.corner_idx.min(7));
                            zone_votes[c] += cbox.score;
                            if best_boxes[c].is_none_or(|b| cbox.score > b.score) {
                                best_boxes[c] = Some(cbox);
                            }
                        }
                        let (subj, luma) = detect_frame_subject_rgba(&rgba, fw, fh, prev_luma.as_deref());
                        prev_luma = Some(luma);
                        subj
                    }
                    Err(_) => FrameSubject::default(),
                }
            } else {
                FrameSubject::default()
            };
            raw_detections.push((eff_t, subject));
            done_samples += 1;
            let pct = ((done_samples as f64 / total_samples as f64) * 95.0) as u64;
            prog.done.store(pct.min(100), Ordering::Relaxed);
            *lock(&prog.status) = format!("Tracking subject (clip {})…", clip_idx + 1);
        }

        let smoothed = smooth_subject_trajectory(&raw_detections, options.smoothing);
        let chosen_facecam = if options.split_screen {
            manual_facecam.or_else(|| {
                let best_zone = (0..8).max_by(|&a, &b| zone_votes[a].total_cmp(&zone_votes[b]))?;
                best_boxes[best_zone].filter(|b| b.score >= MIN_FACECAM_CONFIDENCE)
            })
        } else {
            None
        };
        let plan = if let Some(facecam_box) = chosen_facecam {
            build_split_screen_plan(
                w.clip_id,
                (w.src_w, w.src_h),
                (w.target_w, w.target_h),
                options.facecam_placement,
                facecam_box,
                options.smoothing,
                &smoothed,
            )
        } else {
            build_clip_reframe_plan(w.clip_id, (w.src_w, w.src_h), (w.target_w, w.target_h), options.resize_sequence, options.smoothing, &smoothed)
        };
        out.push((w.cache_key, plan));
    }

    prog.done.store(100, Ordering::Relaxed);
    Ok(out)
}

fn apply_crop_pct(item: &mut TrackItem, left: f64, top: f64, right: f64, bottom: f64) {
    if item.effect("crop").is_none()
        && let Some(def) = find_effect("crop")
    {
        item.effects.push(def.instance());
    }
    if let Some(crop) = item.effect_mut("crop") {
        crop.enabled = true;
        for (pid, val) in [("left", left), ("top", top), ("right", right), ("bottom", bottom), ("feather", 0.0)] {
            if let Some(p) = crop.param_mut(pid) {
                p.keyframes.clear();
                p.value = ParamValue::Float(val);
            }
        }
    }
}

fn clear_transform_keyframes(item: &mut TrackItem, anchor: Vec2) {
    if item.effect("motion").is_none()
        && let Some(def) = find_effect("motion")
    {
        item.effects.insert(0, def.instance());
    }
    if let Some(motion) = item.effect_mut("motion") {
        motion.enabled = true;
        if let Some(u) = motion.param_mut("uniform_scale") {
            u.keyframes.clear();
            u.value = ParamValue::Bool(true);
        }
        if let Some(sc) = motion.param_mut("scale") {
            sc.keyframes.clear();
        }
        if let Some(sw) = motion.param_mut("scale_width") {
            sw.keyframes.clear();
            sw.value = ParamValue::Float(100.0);
        }
        if let Some(pos) = motion.param_mut("position") {
            pos.keyframes.clear();
        }
        if let Some(anc) = motion.param_mut("anchor") {
            anc.keyframes.clear();
            anc.value = ParamValue::Vec2(anchor);
        }
        if let Some(rot) = motion.param_mut("rotation") {
            rot.keyframes.clear();
            rot.value = ParamValue::Float(0.0);
        }
    }
    if let Some(crop) = item.effect_mut("crop") {
        crop.enabled = false;
        for pid in ["left", "top", "right", "bottom", "feather"] {
            if let Some(p) = crop.param_mut(pid) {
                p.keyframes.clear();
                p.value = ParamValue::Float(0.0);
            }
        }
    }
}

fn build_facecam_overlay_item(mut cam_it: TrackItem, cam_cid: ClipId, split: SplitScreenOverlayPlan) -> TrackItem {
    cam_it.id = cam_cid;
    cam_it.name = format!("{} (FaceCam)", cam_it.name);
    cam_it.link = None;
    cam_it.group = None;
    cam_it.scale_to_frame = false;
    clear_transform_keyframes(&mut cam_it, split.facecam_anchor);
    apply_crop_pct(&mut cam_it, split.facecam_box.left_pct, split.facecam_box.top_pct, split.facecam_box.right_pct, split.facecam_box.bottom_pct);
    if let Some(cam_motion) = cam_it.effect_mut("motion") {
        if let Some(sc) = cam_motion.param_mut("scale") {
            sc.value = ParamValue::Float(split.facecam_scale_pct);
        }
        if let Some(pos) = cam_motion.param_mut("position") {
            pos.value = ParamValue::Vec2(split.facecam_pos);
        }
    }
    cam_it
}

fn remove_overlapping_facecam_overlays(seq: &mut filmcraft_project::Sequence, plans: &[(ReframeCacheKey, ClipReframePlan)]) {
    let mut spans = Vec::with_capacity(plans.len());
    for (_, plan) in plans {
        if let Some((_, it)) = seq.find_item(plan.clip_id) {
            spans.push((it.start, it.end()));
        }
    }
    if spans.is_empty() {
        return;
    }
    for tr in &mut seq.video_tracks {
        if tr.locked {
            continue;
        }
        tr.items.retain(|it| !it.name.ends_with(" (FaceCam)") || !spans.iter().any(|&(s, e)| it.start < e && s < it.end()));
    }
}

fn apply_clip_plan_to_item(it: &mut TrackItem, plan: &ClipReframePlan, target_w: u32, target_h: u32) -> usize {
    it.scale_to_frame = false;
    let anchor = plan.split_screen.map_or_else(|| Vec2::new(f64::NAN, f64::NAN), |s| s.gameplay_anchor);
    clear_transform_keyframes(it, anchor);
    let mut kf_done = 0usize;
    if let Some(motion) = it.effect_mut("motion") {
        if let Some(sc) = motion.param_mut("scale") {
            sc.value = ParamValue::Float(plan.scale_pct);
        }
        if let Some(pos_param) = motion.param_mut("position") {
            let first_pos = plan.keyframes.first().map_or_else(|| Vec2::new(f64::from(target_w) * 0.5, f64::from(target_h) * 0.5), |(_, v)| *v);
            pos_param.value = ParamValue::Vec2(first_pos);
            if plan.keyframes.len() > 1 {
                for &(mt, pos) in &plan.keyframes {
                    pos_param.put_keyframe(mt, ParamValue::Vec2(pos));
                    if let Some(k) = pos_param.keyframes.iter_mut().find(|k| k.time == mt) {
                        k.interp = Interpolation::AutoBezier;
                    }
                }
                pos_param.keyframes.sort_by_key(|k| k.time);
            }
            kf_done += plan.keyframes.len().max(1);
        }
    }
    if let Some(split) = plan.split_screen {
        apply_crop_pct(it, split.gameplay_crop_left_pct, split.gameplay_crop_top_pct, split.gameplay_crop_right_pct, split.gameplay_crop_bottom_pct);
    }
    kf_done
}

fn place_facecam_overlays(seq: &mut filmcraft_project::Sequence, overlays: Vec<(usize, TrackItem)>, mut fallback_tid: Option<TrackId>) {
    let mut created_idx: Option<usize> = None;
    for (src_track_idx, cam_it) in overlays {
        let target_idx = seq
            .video_tracks
            .iter()
            .enumerate()
            .skip(src_track_idx + 1)
            .find(|(_, tr)| !tr.locked && !tr.items.iter().any(|existing| existing.start < cam_it.end() && cam_it.start < existing.end()))
            .map(|(idx, _)| idx);
        let dest_idx = if let Some(idx) = target_idx {
            idx
        } else if let Some(idx) = created_idx {
            idx
        } else if let Some(tid) = fallback_tid.take() {
            let name = format!("V{}", seq.video_tracks.len() + 1);
            seq.video_tracks.push(Track::new(tid, TrackKind::Video, name));
            let idx = seq.video_tracks.len() - 1;
            created_idx = Some(idx);
            idx
        } else {
            continue;
        };
        let upper = &mut seq.video_tracks[dest_idx];
        if !upper.items.iter().any(|existing| existing.start < cam_it.end() && cam_it.start < existing.end()) {
            let pos = upper.items.partition_point(|x| x.start <= cam_it.start);
            upper.items.insert(pos, cam_it);
        }
    }
}

fn apply_reframe_plans(
    s: &mut Session,
    seq_id: ItemId,
    options: AutoReframeOptions,
    used_onnx: bool,
    cached: bool,
    plans: Vec<(ReframeCacheKey, ClipReframePlan)>,
) -> Result<AutoReframeReport> {
    if plans.is_empty() {
        return Err(bad("ai.autoReframe", "No video clips remained to reframe"));
    }
    let prev_selection = s.state.selection.clone();
    let (target_w, target_h) = options.aspect.dimensions();

    let (reframed_clips, total_keyframes, split_screen_clips) = s.edit_group("Apply AI Auto-Reframe", |s| {
        let stats = s.edit("Apply AI Auto-Reframe", |pr, st| {
            let need_split = plans.iter().any(|(_, p)| p.split_screen.is_some());
            let new_track_id = if need_split { Some(TrackId(pr.alloc_id())) } else { None };
            let mut facecam_alloc_ids: Vec<(ClipId, ClipId)> = Vec::new();
            if need_split {
                for (_, p) in &plans {
                    if p.split_screen.is_some() {
                        facecam_alloc_ids.push((p.clip_id, ClipId(pr.alloc_id())));
                    }
                }
            }

            let seq = pr.sequence_mut(seq_id).ok_or(EngineError::NoSequence)?;
            if options.resize_sequence {
                seq.settings.width = target_w;
                seq.settings.height = target_h;
                seq.settings.par = (1, 1);
            }

            remove_overlapping_facecam_overlays(seq, &plans);

            let mut clips_done = 0usize;
            let mut kf_done = 0usize;
            let mut split_done = 0usize;
            let mut overlay_clips: Vec<(usize, TrackItem)> = Vec::new();

            for (_, plan) in &plans {
                let Some((src_tid, _)) = seq.find_item(plan.clip_id) else { continue };
                let src_track_idx = seq.video_tracks.iter().position(|tr| tr.id == src_tid).unwrap_or(0);
                let Some((_, it)) = seq.find_item_mut(plan.clip_id) else { continue };
                kf_done += apply_clip_plan_to_item(it, plan, target_w, target_h);
                let base_snapshot = it.clone();

                if let Some(split) = plan.split_screen
                    && let Some(&(_, cam_cid)) = facecam_alloc_ids.iter().find(|&&(c, _)| c == plan.clip_id)
                {
                    overlay_clips.push((src_track_idx, build_facecam_overlay_item(base_snapshot, cam_cid, split)));
                    split_done += 1;
                }
                clips_done += 1;
            }

            if !overlay_clips.is_empty() {
                place_facecam_overlays(seq, overlay_clips, new_track_id);
                crate::subtitles::reposition_subtitles_for_game_stream(seq);
            }

            seq.check().map_err(EngineError::Other)?;
            st.selection = prev_selection.clone();
            Ok((clips_done, kf_done, split_done))
        })?;

        for (key, plan) in plans {
            s.auto_reframe_cache.insert(key, plan);
        }
        Ok(stats)
    })?;

    let rep = AutoReframeReport {
        seq: seq_id,
        reframed_clips,
        total_keyframes,
        split_screen_clips,
        aspect: options.aspect,
        smoothing: options.smoothing,
        resize_sequence: options.resize_sequence,
        split_screen: options.split_screen,
        used_onnx,
        cached,
    };
    s.auto_reframe_report = Some(rep.clone());
    if options.split_screen && split_screen_clips == 0 {
        s.toast(format!(
            "No webcam detected — fell back to single-frame {} Auto-Reframe (select a manual FaceCam Position to force split-screen)",
            options.aspect.as_id()
        ));
    } else {
        s.toast(format!("Reframed {reframed_clips} clip(s) to {} ({total_keyframes} keyframes)", options.aspect.as_id()));
    }
    Ok(rep)
}

/// Apply completed background AI Auto-Reframe jobs cleanly on the session thread.
pub fn poll(s: &mut Session) {
    let mut i = 0usize;
    while i < s.auto_reframe_jobs.len() {
        let job = s.jobs.iter().find(|j| j.id == s.auto_reframe_jobs[i].job);
        let finished = job.is_none_or(|j| j.progress.finished.load(Ordering::Relaxed));
        let cancelled = job.is_some_and(|j| j.progress.cancel.load(Ordering::Relaxed));
        if !finished {
            i += 1;
            continue;
        }
        let pj = s.auto_reframe_jobs.remove(i);
        let Some(plans) = lock(&pj.plans).take().filter(|_| !cancelled) else { continue };
        if pj.epoch != s.document_epoch {
            s.error_toast("ai.autoReframe", "Auto-Reframe: the project was replaced; nothing was changed");
            continue;
        }
        if let Err(e) = apply_reframe_plans(s, pj.seq, pj.options, pj.used_onnx, false, plans) {
            s.error_toast("ai.autoReframe", format!("Auto-Reframe failed: {e}"));
        }
    }
}

fn parse_options(p: &Value, cmd: &str) -> Result<AutoReframeOptions> {
    let aspect = match str_p(p, "aspect") {
        Some(raw) => ReframeAspect::from_str_param(raw).ok_or_else(|| bad(cmd, format!("unknown aspect `{raw}`")))?,
        None => ReframeAspect::Vertical9x16,
    };
    let smoothing = match str_p(p, "smoothing") {
        Some(raw) => ReframeSmoothing::from_str_param(raw).ok_or_else(|| bad(cmd, format!("unknown smoothing `{raw}`")))?,
        None => ReframeSmoothing::SmoothPan,
    };
    let resize_sequence = bool_p(p, "resizeSequence").unwrap_or(true);
    let split_screen = bool_p(p, "splitScreen").unwrap_or(false);
    let facecam_placement = match str_p(p, "facecamPlacement") {
        Some(raw) => FacecamPlacement::from_str_param(raw).ok_or_else(|| bad(cmd, format!("unknown facecamPlacement `{raw}`")))?,
        None => FacecamPlacement::Bottom,
    };
    let facecam_region = match str_p(p, "facecamRegion").or_else(|| str_p(p, "facecamPosition")) {
        Some(raw) => FacecamRegion::from_str_param(raw).ok_or_else(|| bad(cmd, format!("unknown facecamRegion `{raw}`")))?,
        None => FacecamRegion::Auto,
    };
    Ok(AutoReframeOptions { aspect, smoothing, resize_sequence, split_screen, facecam_placement, facecam_region })
}

fn is_already_reframed(s: &Session, seq_id: ItemId, options: AutoReframeOptions, plans: &[(ReframeCacheKey, ClipReframePlan)]) -> bool {
    let Some(seq) = s.project.sequence(seq_id) else { return false };
    if options.resize_sequence {
        let (tw, th) = options.aspect.dimensions();
        if seq.settings.width != tw || seq.settings.height != th {
            return false;
        }
    }
    for (_, plan) in plans {
        let Some((_, it)) = seq.find_item(plan.clip_id) else { return false };
        let Some(motion) = it.effect("motion").filter(|m| m.enabled) else { return false };
        let sc = motion.f64_at("scale", Tick::ZERO);
        if (sc - plan.scale_pct).abs() > 0.05 {
            return false;
        }
        if motion.param("scale").is_some_and(|p| !p.keyframes.is_empty()) {
            return false;
        }
        let expected_pos_kf = if plan.keyframes.len() <= 1 { 0 } else { plan.keyframes.len() };
        if motion.param("position").is_none_or(|p| p.keyframes.len() != expected_pos_kf) {
            return false;
        }
        match plan.split_screen {
            Some(split) => {
                let Some(crop) = it.effect("crop").filter(|c| c.enabled) else { return false };
                if (crop.f64_at("left", Tick::ZERO) - split.gameplay_crop_left_pct).abs() > 0.05
                    || (crop.f64_at("right", Tick::ZERO) - split.gameplay_crop_right_pct).abs() > 0.05
                {
                    return false;
                }
            }
            None => {
                if it.effect("crop").is_some_and(|c| c.enabled) {
                    return false;
                }
            }
        }
    }
    true
}

fn auto_reframe_cmd(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "ai.autoReframe";
    let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let options = parse_options(p, cmd)?;
    let work = collect_reframe_work(s, seq_id, options, cmd)?;
    let used_onnx = detect_local_face_onnx(None).is_some();

    if work.iter().all(|w| w.cached_plan.is_some()) {
        let cached_plans: Vec<(ReframeCacheKey, ClipReframePlan)> = work.into_iter().filter_map(|w| w.cached_plan.map(|p| (w.cache_key, p))).collect();
        let rep = if is_already_reframed(s, seq_id, options, &cached_plans) {
            let reframed_clips = cached_plans.len();
            let total_keyframes: usize = cached_plans.iter().map(|(_, p)| p.keyframes.len().max(1)).sum();
            let split_screen_clips = cached_plans.iter().filter(|(_, p)| p.split_screen.is_some()).count();
            let r = AutoReframeReport {
                seq: seq_id,
                reframed_clips,
                total_keyframes,
                split_screen_clips,
                aspect: options.aspect,
                smoothing: options.smoothing,
                resize_sequence: options.resize_sequence,
                split_screen: options.split_screen,
                used_onnx,
                cached: true,
            };
            s.auto_reframe_report = Some(r.clone());
            s.toast(format!("Reframed {reframed_clips} clip(s) to {} ({total_keyframes} keyframes)", options.aspect.as_id()));
            r
        } else {
            apply_reframe_plans(s, seq_id, options, used_onnx, true, cached_plans)?
        };
        return Ok(json!({
            "cached": true,
            "reframedClips": rep.reframed_clips,
            "totalKeyframes": rep.total_keyframes,
            "splitScreenClips": rep.split_screen_clips,
            "aspect": rep.aspect.as_id(),
            "smoothing": rep.smoothing.as_id(),
            "resizeSequence": rep.resize_sequence,
            "splitScreen": rep.split_screen,
            "usedOnnx": rep.used_onnx,
        }));
    }

    let id = next_job_id(s);
    let prog = Arc::new(Progress::default());
    prog.total.store(100, Ordering::Relaxed);
    *lock(&prog.status) = "Tracking subject for Auto-Reframe…".into();
    let result: Arc<Mutex<Option<std::result::Result<Report, String>>>> = Arc::default();
    s.jobs.push(Job { id, label: "AI Auto-Reframe".into(), progress: prog.clone(), result: result.clone() });

    let plans_slot: Arc<Mutex<Option<Vec<(ReframeCacheKey, ClipReframePlan)>>>> = Arc::default();
    s.auto_reframe_jobs.push(PendingAutoReframeJob { job: id, seq: seq_id, options, used_onnx, epoch: s.document_epoch, plans: plans_slot.clone() });

    let wait = bool_p(p, "wait").unwrap_or(false);
    let run = crate::export_tools::guard_job(prog.clone(), result.clone(), move || {
        let t0 = web_time::Instant::now();
        let outcome = run_reframe_worker(work, options, &prog);
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
        std::thread::Builder::new().name("orpheus-auto-reframe".into()).spawn(run).map_err(|e| EngineError::Other(e.to_string()))?;
    }

    if let Some(Err(e)) = s.jobs.iter().find(|j| j.id == id).and_then(|j| lock(&j.result).clone()) {
        return Err(EngineError::Other(e));
    }

    let mut out = json!({
        "job": id,
        "aspect": options.aspect.as_id(),
        "smoothing": options.smoothing.as_id(),
        "resizeSequence": options.resize_sequence,
        "usedOnnx": used_onnx,
    });
    if wait && let Some(rep) = &s.auto_reframe_report {
        out["reframedClips"] = json!(rep.reframed_clips);
        out["totalKeyframes"] = json!(rep.total_keyframes);
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
    let report = s.auto_reframe_report.as_ref().filter(|r| Some(r.seq) == active).map(|r| {
        json!({
            "reframedClips": r.reframed_clips,
            "totalKeyframes": r.total_keyframes,
            "aspect": r.aspect.as_id(),
            "smoothing": r.smoothing.as_id(),
            "resizeSequence": r.resize_sequence,
            "usedOnnx": r.used_onnx,
            "cached": r.cached,
        })
    });
    Ok(json!({
        "running": running,
        "progress": progress,
        "status": status,
        "error": error,
        "report": report,
    }))
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "ai.autoReframe",
            label: "Auto-Reframe Video (AI)",
            menu: &[],
            shortcut: None,
            params: r#"{"aspect":"9:16|4:5|1:1"?,"smoothing":"locked|smoothPan|dynamicFollow"?,"resizeSequence":bool?,"splitScreen":bool?,"facecamPlacement":"bottom|top"?,"facecamRegion":"auto|leftMiddle|bottomLeft|bottomRight|topLeft|topRight|rightMiddle|topMiddle|bottomMiddle"?,"wait":bool?}"#,
            enabled: has_seq,
            run: auto_reframe_cmd,
            journal: false,
            named: NamedKeys::NONE,
        },
        CommandSpec {
            id: "ai.autoReframeStatus",
            label: "AI Auto-Reframe Status",
            menu: &[],
            shortcut: None,
            params: r#"{"job":u64?}"#,
            enabled: always,
            run: status_cmd,
            journal: false,
            named: NamedKeys::NONE,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paint_synthetic_speaker_frame(width: usize, height: usize, subj_u: f64, subj_v: f64) -> Vec<u8> {
        let mut rgba = vec![0u8; width * height * 4];
        let cx = subj_u * width as f64;
        let cy = subj_v * height as f64;
        let rx = (width as f64 * 0.09).max(4.0);
        let ry = (height as f64 * 0.14).max(4.0);

        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) * 4;
                let dx = (x as f64 - cx) / rx;
                let dy = (y as f64 - cy) / ry;
                if dx * dx + dy * dy <= 1.0 {
                    // Warm skin-tone oval with high-contrast eye/mouth horizontal features.
                    let feature_band = ((y as i32 - cy as i32).abs() == 3 || (x as i32 - cx as i32).abs() == 2) as u8;
                    if feature_band == 1 {
                        rgba[i] = 45;
                        rgba[i + 1] = 32;
                        rgba[i + 2] = 28;
                    } else {
                        rgba[i] = 218;
                        rgba[i + 1] = 164;
                        rgba[i + 2] = 138;
                    }
                    rgba[i + 3] = 255;
                } else {
                    // Cool dark slate background.
                    rgba[i] = 24;
                    rgba[i + 1] = 30;
                    rgba[i + 2] = 42;
                    rgba[i + 3] = 255;
                }
            }
        }
        rgba
    }

    #[test]
    fn tracker_locates_left_and_right_subjects_accurately() {
        let w = 160;
        let h = 90;
        let left_frame = paint_synthetic_speaker_frame(w, h, 0.25, 0.35);
        let (subj_left, luma1) = detect_frame_subject_rgba(&left_frame, w, h, None);
        assert!(subj_left.x < 0.38, "expected left subject x < 0.38, got {:.3}", subj_left.x);
        assert!((subj_left.y - 0.35).abs() < 0.15, "expected subject y near 0.35, got {:.3}", subj_left.y);

        let right_frame = paint_synthetic_speaker_frame(w, h, 0.76, 0.35);
        let (subj_right, _) = detect_frame_subject_rgba(&right_frame, w, h, Some(&luma1));
        assert!(subj_right.x > 0.62, "expected right subject x > 0.62, got {:.3}", subj_right.x);
    }

    #[test]
    fn smoothing_rejects_micro_jitter_and_locked_produces_single_target() {
        let samples: Vec<(Tick, FrameSubject)> = (0..12)
            .map(|i| {
                let jitter = if i % 2 == 0 { 0.012 } else { -0.012 };
                (Tick(i * 1_000_000), FrameSubject { x: 0.35 + jitter, y: 0.38, confidence: 0.8 })
            })
            .collect();

        let locked = smooth_subject_trajectory(&samples, ReframeSmoothing::Locked);
        assert_eq!(locked.len(), 1);
        assert!((locked[0].1.0 - 0.35).abs() < 0.02);

        let smooth = smooth_subject_trajectory(&samples, ReframeSmoothing::SmoothPan);
        assert_eq!(smooth.len(), samples.len());
        let min_x = smooth.iter().map(|s| s.1.0).fold(f64::INFINITY, f64::min);
        let max_x = smooth.iter().map(|s| s.1.0).fold(f64::NEG_INFINITY, f64::max);
        assert!(max_x - min_x < 0.005, "dead-zone hysteresis must suppress micro-jitter (span={:.4})", max_x - min_x);
    }

    #[test]
    fn reframe_plan_scales_16x9_to_fill_9x16_without_black_bars() {
        let path = vec![(Tick::ZERO, (0.02, 0.5)), (Tick(10_000), (0.5, 0.5)), (Tick(20_000), (0.98, 0.5))];
        let plan = build_clip_reframe_plan(ClipId(1), (1920, 1080), (1080, 1920), true, ReframeSmoothing::DynamicFollow, &path);

        assert!(plan.scale_pct >= 177.77, "1920x1080 -> 1080x1920 requires >= 177.78% scale, got {}", plan.scale_pct);
        let scaled_w = 1920.0 * (plan.scale_pct / 100.0);
        for &(_, pos) in &plan.keyframes {
            let left_edge = pos.x - scaled_w * 0.5;
            let right_edge = pos.x + scaled_w * 0.5;
            assert!(left_edge <= 0.5, "left edge must cover x=0 without black bar (got {left_edge:.2})");
            assert!(right_edge >= 1079.5, "right edge must cover x=1080 without black bar (got {right_edge:.2})");
        }
    }

    #[test]
    fn command_reframes_sequence_caches_and_undoes_atomically() {
        let mut s = Session::default();
        s.execute("file.newSequence", json!({"name": "Reframe Seq", "video": 1, "audio": 1})).unwrap();
        let bars = s.execute("file.newBarsAndTone", json!({"seconds": 2.0})).unwrap()["item"].as_u64().unwrap();
        let placed = s.execute("timeline.place", json!({"item": bars, "track": "V1", "frame": 0, "audio": false})).unwrap();
        let cid = ClipId(placed["clips"][0].as_u64().unwrap());
        s.execute("timeline.select", json!({"clips": [cid.0]})).unwrap();

        let seq_id = s.state.active_sequence.unwrap();
        let orig_w = s.project.sequence(seq_id).unwrap().settings.width;
        let orig_h = s.project.sequence(seq_id).unwrap().settings.height;

        let res = s.execute("ai.autoReframe", json!({"aspect": "9:16", "smoothing": "smoothPan", "resizeSequence": true, "wait": true})).unwrap();
        assert_eq!(res["reframedClips"], 1);
        assert_eq!(s.project.sequence(seq_id).unwrap().settings.width, 1080);
        assert_eq!(s.project.sequence(seq_id).unwrap().settings.height, 1920);

        let clip_after = s.active_sequence().unwrap().find_item(cid).unwrap().1;
        let scale_after = clip_after.effect("motion").unwrap().f64_at("scale", Tick::ZERO);
        assert!(scale_after > 170.0, "expected vertical fill scale > 170%, got {scale_after}");

        // Second run with identical settings hits cache immediately.
        let cached = s.execute("ai.autoReframe", json!({"aspect": "9:16", "smoothing": "smoothPan", "resizeSequence": true, "wait": true})).unwrap();
        assert_eq!(cached["cached"], true);

        // Single undo restores sequence dimensions and clip framing atomically.
        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(s.project.sequence(seq_id).unwrap().settings.width, orig_w);
        assert_eq!(s.project.sequence(seq_id).unwrap().settings.height, orig_h);
    }

    #[test]
    fn hostile_and_invalid_inputs_return_errors_without_panicking() {
        let (subj, luma) = detect_frame_subject_rgba(&[], 0, 0, None);
        assert_eq!(subj, FrameSubject::default());
        assert!(luma.is_empty());

        let mut s = Session::default();
        s.execute("file.newSequence", json!({"name": "Empty", "video": 1, "audio": 1})).unwrap();
        assert!(s.execute("ai.autoReframe", json!({"wait": true})).is_err(), "empty sequence must return error");
        assert!(s.execute("ai.autoReframe", json!({"aspect": "bad_aspect", "wait": true})).is_err());
        assert!(s.execute("ai.autoReframe", json!({"smoothing": "bad_mode", "wait": true})).is_err());
    }

    #[test]
    fn corner_facecam_detector_finds_pip_box() {
        let w = 160;
        let h = 90;
        // Put a bright bordered FaceCam skin box in the top-right corner (x=0.73..0.99, y=0.02..0.34).
        let mut rgba = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                let u = x as f64 / w as f64;
                let v = y as f64 / h as f64;
                if (0.74..=0.98).contains(&u) && (0.03..=0.33).contains(&v) {
                    rgba[i] = 220;
                    rgba[i + 1] = 165;
                    rgba[i + 2] = 135;
                } else {
                    rgba[i] = 20;
                    rgba[i + 1] = 24;
                    rgba[i + 2] = 35;
                }
                rgba[i + 3] = 255;
            }
        }
        let detected = detect_corner_facecam_rgba(&rgba, w, h).expect("should detect top-right PiP corner");
        assert_eq!(detected.corner_idx, 1, "expected top-right corner (1)");
    }

    #[test]
    fn left_middle_webcam_beats_corner_hud_and_manual_override_locks_region() {
        let w = 160;
        let h = 90;
        // Paint a Left-Middle FaceCam PiP box (x: 0..25%, y: 25..75%) with a clear skin oval,
        // plus high-contrast white/cyan HUD text/boxes in the Bottom-Right corner (no skin).
        let mut rgba = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                let u = x as f64 / w as f64;
                let v = y as f64 / h as f64;
                if u <= 0.25 && (0.25..=0.75).contains(&v) {
                    // Warm facecam backdrop + centered skin oval
                    let nx = (u - 0.125) / 0.08;
                    let ny = (v - 0.48) / 0.15;
                    if nx * nx + ny * ny <= 1.0 {
                        rgba[i] = 224;
                        rgba[i + 1] = 168;
                        rgba[i + 2] = 138;
                    } else {
                        rgba[i] = 150;
                        rgba[i + 1] = 140;
                        rgba[i + 2] = 145;
                    }
                } else if u >= 0.74 && v >= 0.67 {
                    // Bright non-skin HUD numbers/bars in Bottom-Right corner
                    let bright = (x + y) % 3 == 0;
                    rgba[i] = if bright { 245 } else { 15 };
                    rgba[i + 1] = if bright { 250 } else { 20 };
                    rgba[i + 2] = if bright { 255 } else { 30 };
                } else {
                    rgba[i] = 18;
                    rgba[i + 1] = 22;
                    rgba[i + 2] = 32;
                }
                rgba[i + 3] = 255;
            }
        }
        let detected = detect_corner_facecam_rgba(&rgba, w, h).expect("should detect Left-Middle webcam");
        assert_eq!(detected.corner_idx, 4, "Left-Middle (4) must beat corner HUD noise");
        assert!(detected.score >= MIN_FACECAM_CONFIDENCE);
        assert!((detected.left_pct - 0.0).abs() < 0.1);
        assert!((detected.top_pct - 25.0).abs() < 0.1);

        // Pure HUD / no-webcam frame returns None so Auto-Reframe falls back cleanly.
        let mut hud_only = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                let bright = x > 120 && y > 60 && (x % 2 == 0);
                hud_only[i] = if bright { 240 } else { 20 };
                hud_only[i + 1] = if bright { 240 } else { 25 };
                hud_only[i + 2] = if bright { 245 } else { 35 };
                hud_only[i + 3] = 255;
            }
        }
        assert!(detect_corner_facecam_rgba(&hud_only, w, h).is_none(), "HUD-only frame must not trigger false FaceCam");
    }

    #[test]
    fn gaming_split_screen_creates_v2_facecam_and_undoes_cleanly() {
        let mut s = Session::default();
        s.execute("file.newSequence", json!({"name": "Gaming", "video": 1, "audio": 1})).unwrap();
        let bars = s.execute("file.newBarsAndTone", json!({"seconds": 2.0})).unwrap()["item"].as_u64().unwrap();
        s.execute("timeline.place", json!({"item": bars, "track": "V1", "frame": 0, "audio": false})).unwrap();

        let seq_id = s.state.active_sequence.unwrap();
        assert_eq!(s.project.sequence(seq_id).unwrap().video_tracks.len(), 1);

        // Manual FaceCam region override ("leftMiddle") locks the crop box and creates V2 FaceCam.
        let res = s
            .execute(
                "ai.autoReframe",
                json!({
                    "aspect": "9:16",
                    "smoothing": "smoothPan",
                    "resizeSequence": true,
                    "splitScreen": true,
                    "facecamPlacement": "bottom",
                    "facecamRegion": "leftMiddle",
                    "wait": true
                }),
            )
            .unwrap();
        assert_eq!(res["reframedClips"], 1);

        let seq_after = s.project.sequence(seq_id).unwrap();
        assert_eq!(seq_after.video_tracks.len(), 2, "Gaming Split-Screen should create V2 FaceCam track");
        assert_eq!(seq_after.video_tracks[1].items.len(), 1, "V2 should hold the cropped FaceCam clip");
        let cam_crop = seq_after.video_tracks[1].items[0].effect("crop").expect("FaceCam crop effect");
        assert!(cam_crop.f64_at("left", Tick::ZERO) < 5.0, "Left-Middle crop should stay anchored on left edge");
        assert!(cam_crop.f64_at("right", Tick::ZERO) >= 70.0, "Left-Middle crop should trim right 75%");

        // Verify 55% top gameplay pane (center y = 528) and 45% bottom FaceCam pane (center y = 1488).
        let game_pos = seq_after.video_tracks[0].items[0].effect("motion").unwrap().vec2_at("position", Tick::ZERO);
        let cam_pos = seq_after.video_tracks[1].items[0].effect("motion").unwrap().vec2_at("position", Tick::ZERO);
        assert!((game_pos.y - 1920.0 * 0.55 * 0.5).abs() < 1.0, "gameplay should center in top 55% pane, got y={}", game_pos.y);
        assert!((cam_pos.y - 1920.0 * (0.55 + 0.45 * 0.5)).abs() < 1.0, "facecam should center in bottom 45% pane, got y={}", cam_pos.y);

        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(s.project.sequence(seq_id).unwrap().video_tracks.len(), 1);
    }

    #[test]
    fn split_screen_geometry_excludes_webcam_fills_panes_and_clears_keyframes() {
        let (src_w, src_h) = (1920.0_f64, 1080.0_f64);
        let (seq_w, seq_h) = (1080.0_f64, 1920.0_f64);
        let top_pane_h = seq_h * 0.55;
        let bot_pane_h = seq_h * 0.45;

        // Check both Left-Middle (left edge) and Bottom-Right (right edge) webcams.
        for region in [FacecamRegion::LeftMiddle, FacecamRegion::BottomLeft, FacecamRegion::RightMiddle, FacecamRegion::TopRight] {
            let raw_box = region.manual_box().unwrap();
            let plan = build_split_screen_plan(
                ClipId(1),
                (1920, 1080),
                (1080, 1920),
                FacecamPlacement::Bottom,
                raw_box,
                ReframeSmoothing::SmoothPan,
                &[(Tick::ZERO, (0.5, 0.5)), (Tick(1_000_000), (0.2, 0.4))],
            );
            let split = plan.split_screen.expect("split_screen plan");

            // (a) Top gameplay crop X-range must have ZERO overlap with the webcam X-range.
            let game_x0 = (split.gameplay_crop_left_pct / 100.0) * src_w;
            let game_x1 = ((100.0 - split.gameplay_crop_right_pct) / 100.0) * src_w;
            let raw_cam_x0 = (raw_box.left_pct / 100.0) * src_w;
            let raw_cam_x1 = ((100.0 - raw_box.right_pct) / 100.0) * src_w;
            let fit_cam_x0 = (split.facecam_box.left_pct / 100.0) * src_w;
            let fit_cam_x1 = ((100.0 - split.facecam_box.right_pct) / 100.0) * src_w;

            if matches!(region, FacecamRegion::LeftMiddle | FacecamRegion::BottomLeft) {
                assert!(
                    game_x0 >= raw_cam_x1 - 1e-3 && game_x0 >= fit_cam_x1 - 1e-3,
                    "left webcam ({region:?}): gameplay left edge ({game_x0:.1}) must be >= cam right edge ({raw_cam_x1:.1})"
                );
            } else {
                assert!(
                    game_x1 <= raw_cam_x0 + 1e-3 && game_x1 <= fit_cam_x0 + 1e-3,
                    "right webcam ({region:?}): gameplay right edge ({game_x1:.1}) must be <= cam left edge ({raw_cam_x0:.1})"
                );
            }

            // (b) Both panes must fill 100% of their target rect (`seq_w x top_pane_h` and `seq_w x bot_pane_h`) edge-to-edge.
            let game_y0 = (split.gameplay_crop_top_pct / 100.0) * src_h;
            let game_y1 = ((100.0 - split.gameplay_crop_bottom_pct) / 100.0) * src_h;
            let rendered_game_w = (game_x1 - game_x0) * (plan.scale_pct / 100.0);
            let rendered_game_h = (game_y1 - game_y0) * (plan.scale_pct / 100.0);
            assert!(rendered_game_w >= seq_w, "top gameplay width {rendered_game_w:.1} must fill {seq_w}");
            assert!(rendered_game_h >= top_pane_h, "top gameplay height {rendered_game_h:.1} must fill {top_pane_h}");

            let fit_cam_y0 = (split.facecam_box.top_pct / 100.0) * src_h;
            let fit_cam_y1 = ((100.0 - split.facecam_box.bottom_pct) / 100.0) * src_h;
            let rendered_cam_w = (fit_cam_x1 - fit_cam_x0) * (split.facecam_scale_pct / 100.0);
            let rendered_cam_h = (fit_cam_y1 - fit_cam_y0) * (split.facecam_scale_pct / 100.0);
            assert!(rendered_cam_w >= seq_w, "bottom facecam width {rendered_cam_w:.1} must fill {seq_w}");
            assert!(rendered_cam_h >= bot_pane_h, "bottom facecam height {rendered_cam_h:.1} must fill {bot_pane_h}");
        }

        // (c) Re-running Auto-Reframe clears existing scale, position, anchor, rotation, and crop keyframes.
        let mut s = Session::default();
        s.execute("file.newSequence", json!({"name": "KF Clear", "video": 1, "audio": 1})).unwrap();
        let bars = s.execute("file.newBarsAndTone", json!({"seconds": 2.0})).unwrap()["item"].as_u64().unwrap();
        let placed = s.execute("timeline.place", json!({"item": bars, "track": "V1", "frame": 0, "audio": false})).unwrap();
        let cid = ClipId(placed["clips"][0].as_u64().unwrap());
        let seq_id = s.state.active_sequence.unwrap();

        {
            let seq = Arc::make_mut(&mut s.project).sequence_mut(seq_id).unwrap();
            let (_, it) = seq.find_item_mut(cid).unwrap();
            if it.effect("motion").is_none()
                && let Some(def) = find_effect("motion")
            {
                it.effects.insert(0, def.instance());
            }
            let motion = it.effect_mut("motion").unwrap();
            motion.param_mut("position").unwrap().put_keyframe(Tick(100_000), ParamValue::Vec2(Vec2::new(100.0, 200.0)));
            motion.param_mut("position").unwrap().put_keyframe(Tick(500_000), ParamValue::Vec2(Vec2::new(400.0, 600.0)));
            motion.param_mut("scale").unwrap().put_keyframe(Tick(100_000), ParamValue::Float(120.0));
            if let Some(def) = find_effect("crop") {
                it.effects.push(def.instance());
            }
            it.effect_mut("crop").unwrap().param_mut("left").unwrap().put_keyframe(Tick(100_000), ParamValue::Float(15.0));
        }

        s.execute(
            "ai.autoReframe",
            json!({
                "aspect": "9:16",
                "smoothing": "smoothPan",
                "resizeSequence": true,
                "splitScreen": true,
                "facecamPlacement": "bottom",
                "facecamRegion": "leftMiddle",
                "wait": true
            }),
        )
        .unwrap();

        let seq_after = s.project.sequence(seq_id).unwrap();
        let v1_clip = &seq_after.video_tracks[0].items[0];
        let v2_clip = &seq_after.video_tracks[1].items[0];
        assert!(v1_clip.effect("motion").unwrap().param("position").unwrap().keyframes.is_empty(), "V1 position keyframes must be cleared");
        assert!(v1_clip.effect("motion").unwrap().param("scale").unwrap().keyframes.is_empty(), "V1 scale keyframes must be cleared");
        assert!(v1_clip.effect("crop").unwrap().param("left").unwrap().keyframes.is_empty(), "V1 crop keyframes must be cleared");
        assert!(v2_clip.effect("motion").unwrap().param("position").unwrap().keyframes.is_empty(), "V2 position keyframes must be cleared");
        assert!(v2_clip.effect("crop").unwrap().param("left").unwrap().keyframes.is_empty(), "V2 crop keyframes must be cleared");
    }

    #[test]
    fn refine_pip_box_snaps_to_inner_borders_and_excludes_hud() {
        let (w, h) = (320usize, 180usize);
        // Webcam PiP box occupies x: 0.00..0.20, y: 0.34..0.66 on a dark gameplay background,
        // with a bright HUD counter bar just above (y: 0.27..0.31) and bright gameplay at x: 0.22.
        let mut rgba = vec![0u8; w * h * 4];
        for y in 0..h {
            let v = y as f64 / h as f64;
            for x in 0..w {
                let u = x as f64 / w as f64;
                let i = (y * w + x) * 4;
                if u <= 0.20 && (0.34..=0.66).contains(&v) {
                    rgba[i] = 210;
                    rgba[i + 1] = 175;
                    rgba[i + 2] = 155;
                } else if u <= 0.18 && (0.27..=0.31).contains(&v) {
                    rgba[i] = 240;
                    rgba[i + 1] = 240;
                    rgba[i + 2] = 240;
                } else {
                    rgba[i] = 16;
                    rgba[i + 1] = 20;
                    rgba[i + 2] = 28;
                }
                rgba[i + 3] = 255;
            }
        }
        let face = crate::ai_models::OnnxFaceBox { x0: 0.06, y0: 0.40, x1: 0.14, y1: 0.56, cx: 0.10, cy: 0.48, conf: 0.95 };
        let snapped = refine_pip_box_around_face(&rgba, w, h, face, 4, 0.00, 0.25, 0.25, 0.75);
        assert!(snapped.top_pct >= 34.0, "snapped top ({:.2}%) must stay below HUD at 31%", snapped.top_pct);
        assert!(100.0 - snapped.right_pct <= 20.5, "snapped right ({:.2}%) must not spill past 20%", 100.0 - snapped.right_pct);
        assert!(100.0 - snapped.bottom_pct <= 66.5, "snapped bottom ({:.2}%) must stay inside 66%", 100.0 - snapped.bottom_pct);
    }

    #[test]
    fn outward_luma_scan_stops_at_dark_gap_excluding_hud_and_gameplay() {
        let (w, h) = (320usize, 180usize);
        let mut rgba = vec![0u8; w * h * 4];
        for y in 0..h {
            let v = y as f64 / h as f64;
            for x in 0..w {
                let u = x as f64 / w as f64;
                let i = (y * w + x) * 4;
                if u <= 0.22 && (0.35..=0.60).contains(&v) {
                    // Lit webcam interior
                    rgba[i] = 180;
                    rgba[i + 1] = 150;
                    rgba[i + 2] = 135;
                } else if u <= 0.20 && (0.26..=0.31).contains(&v) {
                    // Bright HUD counter above dark gap (e.g. 140984)
                    rgba[i] = 245;
                    rgba[i + 1] = 245;
                    rgba[i + 2] = 245;
                } else if u >= 0.24 {
                    // Bright gameplay strip to the right of dark vertical border
                    rgba[i] = 160;
                    rgba[i + 1] = 190;
                    rgba[i + 2] = 220;
                } else {
                    // Dark gap / border (< 45 luma)
                    rgba[i] = 18;
                    rgba[i + 1] = 20;
                    rgba[i + 2] = 24;
                }
                rgba[i + 3] = 255;
            }
        }
        let face = crate::ai_models::OnnxFaceBox { x0: 0.06, y0: 0.41, x1: 0.15, y1: 0.54, cx: 0.105, cy: 0.475, conf: 0.95 };
        let snapped = refine_pip_box_around_face(&rgba, w, h, face, 4, 0.00, 0.25, 0.28, 0.75);
        assert!(snapped.top_pct >= 35.0, "top ({:.2}%) must exclude HUD at 26..31%", snapped.top_pct);
        assert!(100.0 - snapped.right_pct <= 22.0, "right ({:.2}%) must exclude gameplay at >= 24%", 100.0 - snapped.right_pct);
        assert!(100.0 - snapped.bottom_pct <= 60.5, "bottom ({:.2}%) must stay inside webcam", 100.0 - snapped.bottom_pct);
    }

    #[test]
    fn game_stream_mode_positions_subtitles_above_bottom_facecam_in_both_orders() {
        use crate::subtitles::{SubtitleCue, SubtitleLengthMode, detect_game_stream_split_y, insert_subtitle_track};

        let sample_cues = vec![SubtitleCue { start: Tick::ZERO, duration: Tick::from_seconds_f64(1.0), text: "CLUTCH ROUND".into() }];

        // Order A: Auto Reframe (Game Stream Mode) -> Generate Subtitles
        let mut s_a = Session::default();
        s_a.execute("file.newSequence", json!({"name": "OrderA", "video": 1, "audio": 1})).unwrap();
        let bars_a = s_a.execute("file.newBarsAndTone", json!({"seconds": 2.0})).unwrap()["item"].as_u64().unwrap();
        s_a.execute("timeline.place", json!({"item": bars_a, "track": "V1", "frame": 0, "audio": false})).unwrap();
        let seq_a = s_a.state.active_sequence.unwrap();

        s_a.execute(
            "ai.autoReframe",
            json!({
                "aspect": "9:16",
                "resizeSequence": true,
                "splitScreen": true,
                "facecamPlacement": "bottom",
                "facecamRegion": "leftMiddle",
                "wait": true
            }),
        )
        .unwrap();

        assert_eq!(detect_game_stream_split_y(s_a.project.sequence(seq_a).unwrap()), Some(0.55));
        insert_subtitle_track(&mut s_a, seq_a, "en", SubtitleLengthMode::ReelsShort, &sample_cues).unwrap();
        let track_a = &s_a.project.sequence(seq_a).unwrap().caption_tracks[0];
        assert!((track_a.style.margin - 0.45).abs() < 1e-4, "Order A subtitles should sit above 45% bottom facecam, got margin={}", track_a.style.margin);

        // Order B: Generate Subtitles (and bottom text clip) -> Auto Reframe (Game Stream Mode)
        let mut s_b = Session::default();
        s_b.execute("file.newSequence", json!({"name": "OrderB", "video": 1, "audio": 1})).unwrap();
        let bars_b = s_b.execute("file.newBarsAndTone", json!({"seconds": 2.0})).unwrap()["item"].as_u64().unwrap();
        s_b.execute("timeline.place", json!({"item": bars_b, "track": "V1", "frame": 0, "audio": false})).unwrap();
        let seq_b = s_b.state.active_sequence.unwrap();

        insert_subtitle_track(&mut s_b, seq_b, "en", SubtitleLengthMode::ReelsShort, &sample_cues).unwrap();
        assert!((s_b.project.sequence(seq_b).unwrap().caption_tracks[0].style.margin - 0.14).abs() < 1e-4);

        // Single-frame Auto-Reframe (non-Game Stream Mode) must keep standard bottom margin (0.14).
        s_b.execute("ai.autoReframe", json!({"aspect": "9:16", "resizeSequence": true, "splitScreen": false, "wait": true})).unwrap();
        assert!(detect_game_stream_split_y(s_b.project.sequence(seq_b).unwrap()).is_none());
        assert!((s_b.project.sequence(seq_b).unwrap().caption_tracks[0].style.margin - 0.14).abs() < 1e-4);

        // Switching the sequence into Game Stream Mode automatically repositions existing bottom captions above the facecam.
        s_b.execute(
            "ai.autoReframe",
            json!({
                "aspect": "9:16",
                "resizeSequence": true,
                "splitScreen": true,
                "facecamPlacement": "bottom",
                "facecamRegion": "leftMiddle",
                "wait": true
            }),
        )
        .unwrap();
        let track_b = &s_b.project.sequence(seq_b).unwrap().caption_tracks[0];
        assert!((track_b.style.margin - 0.45).abs() < 1e-4, "Order B subtitles should move above 45% bottom facecam, got margin={}", track_b.style.margin);
    }
}
