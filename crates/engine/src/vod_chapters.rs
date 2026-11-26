//! Local pure-Rust Stream VOD Chapterizer (`ai.vodChapters`, `ai.vodChapters.youtubeText`).
//!
//! Splits long stream VODs (including 10+ hour recordings) into titled, color-coded timeline
//! chapters without decoding every frame into RAM:
//!
//! 1. **Sparse Keyframe / Interval Sampling:** Samples frames every 10 seconds (bounded to
//!    `MAX_SAMPLES` frames so 10+ hour VODs stay fast and constant-memory), downscaling each
//!    sampled frame to <= 160 px and immediately dropping the pixel buffer after computing a
//!    compact `FrameSignature`.
//! 2. **Sustained Layout Transition Detection:** Requires >= 45 seconds of sustained new visual
//!    layout/HUD/face-cam state before confirming a chapter boundary so brief Alt+Tabs or loading
//!    screens do not create false cuts.
//! 3. **Topic & Title Classification:** Combines visual layout heuristics (`TalkingHead`,
//!    `GamingHud`, `DesktopReact`), local Whisper/sequence transcript keywords from the first
//!    45 seconds of each segment, and optional local Ollama summarization.
//! 4. **Undoable Timeline Application:** Splits unlocked video/audio tracks at each chapter
//!    boundary via `filmcraft_edit::razor` and attaches color-coded `MarkerKind::Chapter`
//!    markers in one `Session::edit_group("Stream VOD Chapterizer", ...)` edit.

use std::io::Read;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use filmcraft_export::{Progress, Report};
use filmcraft_media::{FrameRequest, SharedSource};
use filmcraft_project::{ItemId, Label, Marker, MarkerId, MarkerKind, Sequence, TrackKind};
use filmcraft_time::Tick;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, NamedKeys, always, bad, bool_p, has_seq, str_p};
use crate::subtitles::next_job_id;
use crate::{EngineError, Job, Result, Session};

/// Default sparse sampling interval in seconds for VOD chapter detection.
pub const DEFAULT_SAMPLE_INTERVAL_SEC: f64 = 10.0;
/// Minimum sustained duration in seconds required to confirm a new chapter boundary.
pub const DEFAULT_MIN_CHAPTER_SEC: f64 = 45.0;
/// Downscaled longest frame dimension (px) for constant-memory layout analysis.
const ANALYSIS_MAX_DIM: f64 = 160.0;
/// Maximum sampled frames per run (20 hours at 10s intervals) to bound CPU and memory.
const MAX_SAMPLES: usize = 7_200;

const CHAPTER_COLORS: [Label; 6] = [Label::Cerulean, Label::Green, Label::Mango, Label::Violet, Label::Rose, Label::Teal];

/// Visual layout classification for a sampled VOD frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum VisualLayoutKind {
    TalkingHead,
    GamingHud,
    DesktopReact,
}

impl VisualLayoutKind {
    pub fn default_title(self) -> &'static str {
        match self {
            Self::TalkingHead => "Just Chatting",
            Self::GamingHud => "Gameplay",
            Self::DesktopReact => "Reacting",
        }
    }
}

/// Compact per-sample visual signature (no raw pixels retained in RAM).
#[derive(Clone, Debug, PartialEq)]
pub struct FrameSignature {
    pub time_sec: f64,
    pub layout: VisualLayoutKind,
    pub luma_grid: [f32; 16],
    pub color_hist: [f32; 12],
    pub center_skin_ratio: f32,
    pub edge_hud_energy: f32,
    pub center_action_energy: f32,
}

fn neutral_signature(time_sec: f64) -> FrameSignature {
    FrameSignature {
        time_sec,
        layout: VisualLayoutKind::DesktopReact,
        luma_grid: [0.0; 16],
        color_hist: [0.0; 12],
        center_skin_ratio: 0.0,
        edge_hud_energy: 0.0,
        center_action_energy: 0.0,
    }
}

/// A detected or applied stream VOD chapter segment.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChapterMarker {
    pub start_ms: u64,
    pub end_ms: u64,
    pub title: String,
    pub color: String,
}

/// Summary report returned by `ai.vodChapters`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VodChapterReport {
    pub seq: ItemId,
    pub chapters_created: usize,
    pub splits_applied: usize,
    pub used_ollama: bool,
    pub chapters: Vec<ChapterMarker>,
}

/// Pending background chapterizer job polled by `Session::poll`.
pub struct PendingVodChapterJob {
    pub job: u64,
    pub seq: ItemId,
    pub split_clips: bool,
    pub epoch: u64,
    pub result: Arc<Mutex<Option<(Vec<ChapterMarker>, bool)>>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn check_cancelled(prog: Option<&Progress>) -> Result<()> {
    if prog.is_some_and(|p| p.cancel.load(Ordering::Relaxed)) {
        return Err(EngineError::Other("Cancelled".into()));
    }
    Ok(())
}

/// Format milliseconds as `HH:MM:SS` for YouTube chapter timestamps.
pub fn format_hms_ms(ms: u64) -> String {
    let total_secs = ms / 1000;
    let h = total_secs / 3600;
    let m = (total_secs % 3600) / 60;
    let s = total_secs % 60;
    format!("{h:02}:{m:02}:{s:02}")
}

/// Format all chapter markers (or all markers if no `MarkerKind::Chapter` markers exist)
/// into YouTube timestamp lines (`HH:MM:SS Title`).
pub fn format_youtube_chapters(seq: &Sequence) -> String {
    let mut markers: Vec<&Marker> = seq.markers.iter().filter(|m| m.kind == MarkerKind::Chapter).collect();
    if markers.is_empty() {
        markers = seq.markers.iter().collect();
    }
    markers.sort_by_key(|m| m.start);
    let mut lines = Vec::with_capacity(markers.len());
    for (idx, m) in markers.iter().enumerate() {
        let ms = (m.start.seconds().max(0.0).round() * 1000.0) as u64;
        let title = if m.name.trim().is_empty() { format!("Chapter {}", idx + 1) } else { m.name.trim().to_string() };
        lines.push(format!("{} {title}", format_hms_ms(ms)));
    }
    lines.join("\n")
}

/// Largest chapter count the chapterizer attaches (bounds markers, splits and batch exports).
pub const MAX_CHAPTERS: usize = 500;

/// A Windows-safe, unique, numbered file name for chapter `index` (0-based) of `total`, such as
/// `01_Just_Chatting.mp4`.
///
/// The zero-padded running number keeps two chapters with the same title (or titles that sanitize
/// to the same text) from overwriting each other. Every character that is not a letter or digit
/// (including the Windows-invalid `<>:"/\|?*`, control characters and dots) becomes `_`; Unicode
/// letters such as Cyrillic are kept.
pub fn chapter_export_filename(index: usize, total: usize, title: &str, ext: &str) -> String {
    let width = total.max(index.saturating_add(1)).to_string().len().max(2);
    let mut slug = String::new();
    let mut last_was_underscore = true;
    for c in title.trim().chars() {
        if c.is_alphanumeric() {
            slug.push(c);
            last_was_underscore = false;
        } else if !last_was_underscore {
            slug.push('_');
            last_was_underscore = true;
        }
    }
    let slug: String = slug.trim_matches('_').chars().take(40).collect();
    let slug = slug.trim_matches('_');
    let slug = if slug.is_empty() { format!("Chapter_{}", index.saturating_add(1)) } else { slug.to_string() };
    let ext: String = ext.chars().filter(char::is_ascii_alphanumeric).collect();
    let ext = if ext.is_empty() { "mp4".to_string() } else { ext };
    format!("{:0width$}_{slug}.{ext}", index.saturating_add(1))
}

/// `(title, start_seconds, end_seconds)` for every chapter marker, ready for batch export.
///
/// A chapter without its own duration (a hand-made marker) runs to the next chapter or the end of
/// the sequence; ranges are clamped to `seq_duration`, chapters that start past the end or are
/// shorter than 0.1 s are dropped, and at most [`MAX_CHAPTERS`] are returned.
pub fn chapter_export_ranges(markers: &[Marker], seq_duration: Tick) -> Vec<(String, f64, f64)> {
    let dur = seq_duration.seconds();
    if dur.is_nan() || dur <= 0.0 {
        return Vec::new();
    }
    let mut chapters: Vec<&Marker> = markers.iter().filter(|m| m.kind == MarkerKind::Chapter).collect();
    chapters.sort_by_key(|m| m.start);
    chapters.truncate(MAX_CHAPTERS);
    let mut out = Vec::with_capacity(chapters.len());
    for (i, m) in chapters.iter().enumerate() {
        let start = m.start.seconds().max(0.0);
        if start >= dur {
            continue;
        }
        let own_end = (m.duration > Tick::ZERO).then(|| (m.start + m.duration).seconds());
        let next_start = chapters.get(i + 1).map(|n| n.start.seconds());
        let end = own_end.or(next_start).unwrap_or(dur).min(dur);
        if end - start < 0.1 {
            continue;
        }
        out.push((m.name.clone(), start, end));
    }
    out
}

/// Compute a compact `FrameSignature` from an RGBA8 buffer and drop pixel dependence.
pub fn classify_frame_rgba(rgba: &[u8], width: usize, height: usize, time_sec: f64) -> FrameSignature {
    let time_sec = if time_sec.is_finite() { time_sec.max(0.0) } else { 0.0 };
    let neutral = neutral_signature(time_sec);
    let Some(expected_len) = width.checked_mul(height).and_then(|n| n.checked_mul(4)) else {
        return neutral;
    };
    if width < 8 || height < 8 || rgba.len() < expected_len {
        return neutral;
    }

    let mut grid_sum = [0.0_f64; 16];
    let mut grid_cnt = [0_usize; 16];
    let mut quad_rgb = [0.0_f64; 12];
    let mut quad_cnt = [0_usize; 4];
    let mut center_skin = 0_usize;
    let mut center_total = 0_usize;
    let mut center_grad_sum = 0.0_f64;
    let mut center_grad_cnt = 0_usize;
    let mut edge_grad_sum = 0.0_f64;
    let mut edge_total = 0_usize;
    let mut bright_ui_pixels = 0_usize;
    let mut total_pixels = 0_usize;

    let luma_at = |x: usize, y: usize| -> f32 {
        let idx = y.saturating_mul(width).saturating_add(x).saturating_mul(4);
        let r = f32::from(rgba.get(idx).copied().unwrap_or(0));
        let g = f32::from(idx.checked_add(1).and_then(|i| rgba.get(i)).copied().unwrap_or(0));
        let b = f32::from(idx.checked_add(2).and_then(|i| rgba.get(i)).copied().unwrap_or(0));
        (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    };

    for y in 0..height {
        let gy = (y.saturating_mul(4) / height).min(3);
        let qy = usize::from(y >= height / 2);
        let in_center_y = y >= height / 5 && y <= height.saturating_mul(4) / 5;
        let in_edge_band = y < height / 6 || y >= height.saturating_mul(5) / 6;
        for x in 0..width {
            let idx = y.saturating_mul(width).saturating_add(x).saturating_mul(4);
            let r = f32::from(rgba.get(idx).copied().unwrap_or(0));
            let g = f32::from(idx.checked_add(1).and_then(|i| rgba.get(i)).copied().unwrap_or(0));
            let b = f32::from(idx.checked_add(2).and_then(|i| rgba.get(i)).copied().unwrap_or(0));
            let luma = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;

            let gx = (x.saturating_mul(4) / width).min(3);
            let cell = gy * 4 + gx;
            grid_sum[cell] += luma as f64;
            grid_cnt[cell] += 1;
            total_pixels += 1;

            let qx = usize::from(x >= width / 2);
            let quad = qy * 2 + qx;
            quad_rgb[quad * 3] += f64::from(r) / 255.0;
            quad_rgb[quad * 3 + 1] += f64::from(g) / 255.0;
            quad_rgb[quad * 3 + 2] += f64::from(b) / 255.0;
            quad_cnt[quad] += 1;

            if luma > 0.82 && (r - g).abs() < 18.0 && (g - b).abs() < 18.0 {
                bright_ui_pixels += 1;
            }

            let in_center_x = x >= width / 4 && x <= width.saturating_mul(3) / 4;
            if in_center_x && in_center_y {
                center_total += 1;
                // Simple YCbCr skin-chroma heuristic for webcam/talking-head framing.
                let cb = 128.0 - 0.168_736 * r - 0.331_264 * g + 0.5 * b;
                let cr = 128.0 + 0.5 * r - 0.418_688 * g - 0.081_312 * b;
                if (77.0..=127.0).contains(&cb) && (133.0..=173.0).contains(&cr) && r > 60.0 {
                    center_skin += 1;
                }
                if x + 1 < width && y + 1 < height {
                    let dx = (luma_at(x + 1, y) - luma).abs();
                    let dy = (luma_at(x, y + 1) - luma).abs();
                    center_grad_sum += f64::from(dx + dy);
                    center_grad_cnt += 1;
                }
            }

            let in_edge_x = x < width / 6 || x >= width.saturating_mul(5) / 6;
            if (in_edge_band || in_edge_x) && x + 1 < width && y + 1 < height {
                let dx = (luma_at(x + 1, y) - luma).abs();
                let dy = (luma_at(x, y + 1) - luma).abs();
                edge_grad_sum += (dx + dy) as f64;
                edge_total += 1;
            }
        }
    }

    let mut luma_grid = [0.0_f32; 16];
    for i in 0..16 {
        if grid_cnt[i] > 0 {
            luma_grid[i] = (grid_sum[i] / grid_cnt[i] as f64) as f32;
        }
    }
    let mut color_hist = [0.0_f32; 12];
    for q in 0..4 {
        if quad_cnt[q] > 0 {
            let denom = quad_cnt[q] as f64;
            color_hist[q * 3] = (quad_rgb[q * 3] / denom) as f32;
            color_hist[q * 3 + 1] = (quad_rgb[q * 3 + 1] / denom) as f32;
            color_hist[q * 3 + 2] = (quad_rgb[q * 3 + 2] / denom) as f32;
        }
    }

    let center_skin_ratio = if center_total > 0 { center_skin as f32 / center_total as f32 } else { 0.0 };
    let edge_hud_energy = if edge_total > 0 { (edge_grad_sum / edge_total as f64) as f32 } else { 0.0 };
    let center_action_energy = if center_grad_cnt > 0 { (center_grad_sum / center_grad_cnt as f64) as f32 } else { 0.0 };
    let bright_ratio = if total_pixels > 0 { bright_ui_pixels as f32 / total_pixels as f32 } else { 0.0 };

    // Compare edge HUD detail vs center action vs centered skin framing so colorful gameplay
    // with corner webcams is not misclassified as full-screen Just Chatting.
    let layout = if center_skin_ratio >= 0.16 && edge_hud_energy < 0.09 {
        VisualLayoutKind::TalkingHead
    } else if bright_ratio >= 0.28 && edge_hud_energy >= 0.045 {
        VisualLayoutKind::DesktopReact
    } else if edge_hud_energy >= 0.065 || (center_action_energy >= 0.075 && center_skin_ratio < 0.16) {
        VisualLayoutKind::GamingHud
    } else if center_skin_ratio >= 0.11 {
        VisualLayoutKind::TalkingHead
    } else {
        VisualLayoutKind::DesktopReact
    };

    FrameSignature { time_sec, layout, luma_grid, color_hist, center_skin_ratio, edge_hud_energy, center_action_energy }
}

fn signature_l1_distance(a: &FrameSignature, b: &FrameSignature) -> f32 {
    let mut luma_sum = 0.0_f32;
    for i in 0..16 {
        luma_sum += (a.luma_grid[i] - b.luma_grid[i]).abs();
    }
    let mut color_sum = 0.0_f32;
    for i in 0..12 {
        color_sum += (a.color_hist[i] - b.color_hist[i]).abs();
    }
    let energy_diff = (a.edge_hud_energy - b.edge_hud_energy).abs() + (a.center_action_energy - b.center_action_energy).abs();
    (luma_sum / 16.0) * 0.45 + (color_sum / 12.0) * 0.45 + energy_diff * 0.25
}

/// Detect sustained chapter boundaries from sparse `FrameSignature` samples.
///
/// A new chapter is confirmed only when the new layout/visual state persists for at least
/// `min_chapter_sec` (default 45s), preventing brief Alt+Tabs or loading screens from creating
/// false splits.
pub fn detect_sustained_chapters(signatures: &[FrameSignature], total_duration_sec: f64, min_chapter_sec: f64) -> Vec<(f64, f64, VisualLayoutKind)> {
    detect_sustained_chapters_with_hints(signatures, total_duration_sec, min_chapter_sec, &[])
}

fn detect_sustained_chapters_with_hints(
    signatures: &[FrameSignature],
    total_duration_sec: f64,
    min_chapter_sec: f64,
    clip_boundaries_sec: &[f64],
) -> Vec<(f64, f64, VisualLayoutKind)> {
    let min_dur = if min_chapter_sec.is_finite() { min_chapter_sec.max(10.0) } else { DEFAULT_MIN_CHAPTER_SEC };
    if !total_duration_sec.is_finite() || total_duration_sec <= 0.1 {
        return Vec::new();
    }
    let Some(first) = signatures.first() else {
        return vec![(0.0, total_duration_sec, VisualLayoutKind::TalkingHead)];
    };

    let mut chapters: Vec<(f64, f64, VisualLayoutKind)> = Vec::new();
    let mut current_start = 0.0_f64;
    let mut current_layout = first.layout;
    let mut anchor_sig = first.clone();

    let mut candidate_start: Option<f64> = None;
    let mut candidate_layout: Option<VisualLayoutKind> = None;
    let mut candidate_sig: Option<FrameSignature> = None;
    let mut prev_t = first.time_sec;

    for sig in signatures.iter().skip(1) {
        let min_b = current_start + min_dur;
        let idx = if prev_t >= min_b { clip_boundaries_sec.partition_point(|&b| b <= prev_t) } else { clip_boundaries_sec.partition_point(|&b| b < min_b) };
        let crossed_clip = clip_boundaries_sec.get(idx).copied().filter(|&b| b > prev_t && b <= sig.time_sec);
        prev_t = sig.time_sec;
        let dist = signature_l1_distance(&anchor_sig, sig);
        let layout_changed = sig.layout != current_layout;
        let major_visual_shift = dist >= 0.18 || (crossed_clip.is_some() && dist >= 0.09);

        if let Some(boundary) = crossed_clip
            && (layout_changed || dist >= 0.12)
            && total_duration_sec - boundary >= min_dur * 0.5
        {
            chapters.push((current_start, boundary, current_layout));
            current_start = boundary;
            current_layout = sig.layout;
            anchor_sig = sig.clone();
            candidate_start = None;
            candidate_layout = None;
            candidate_sig = None;
            continue;
        }

        if !layout_changed && !major_visual_shift {
            candidate_start = None;
            candidate_layout = None;
            candidate_sig = None;
            continue;
        }

        match (candidate_start, candidate_layout) {
            (Some(cs), Some(cl)) if cl == sig.layout || signature_l1_distance(candidate_sig.as_ref().unwrap_or(sig), sig) < 0.15 => {
                // Candidate state is holding steady. Confirm once it has been sustained >= min_dur
                // and the previous chapter also meets a sensible minimum span.
                if sig.time_sec - cs >= min_dur && cs - current_start >= min_dur {
                    chapters.push((current_start, cs, current_layout));
                    current_start = cs;
                    current_layout = sig.layout;
                    anchor_sig = candidate_sig.take().unwrap_or_else(|| sig.clone());
                    candidate_start = None;
                    candidate_layout = None;
                }
            }
            _ => {
                candidate_start = Some(sig.time_sec);
                candidate_layout = Some(sig.layout);
                candidate_sig = Some(sig.clone());
            }
        }
    }

    if total_duration_sec - current_start >= 1.0 || chapters.is_empty() {
        chapters.push((current_start, total_duration_sec, current_layout));
    } else if let Some(last) = chapters.last_mut() {
        last.1 = total_duration_sec;
    }

    chapters
}

fn extract_transcript_topic_title(transcript_snippet: &str) -> Option<String> {
    const STOPWORDS: &[&str] = &[
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "from",
        "have",
        "just",
        "like",
        "what",
        "when",
        "your",
        "about",
        "they",
        "them",
        "been",
        "were",
        "will",
        "would",
        "there",
        "their",
        "here",
        "then",
        "into",
        "some",
        "more",
        "very",
        "also",
        "really",
        "actually",
        "going",
        "gonna",
        "yeah",
        "okay",
        "right",
        "well",
        "guys",
        "today",
        "let",
        "lets",
        "we're",
        "you're",
        "don't",
        "can't",
        "it's",
        "that's",
        "това",
        "което",
        "като",
        "защото",
        "много",
        "само",
        "сега",
        "тук",
        "така",
        "след",
        "преди",
        "между",
        "около",
        "дори",
        "вече",
        "още",
        "всички",
        "някои",
        "може",
        "трябва",
        "искам",
        "днес",
        "здравейте",
        "добре",
        "нещо",
    ];
    let mut counts: std::collections::HashMap<String, (usize, usize)> = std::collections::HashMap::new();
    for (pos, raw) in transcript_snippet.split_whitespace().enumerate() {
        let cleaned: String = raw.chars().filter(|c| c.is_alphanumeric() || *c == '-').collect();
        let lower = cleaned.to_lowercase();
        if lower.chars().count() < 4 || STOPWORDS.contains(&lower.as_str()) || lower.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        counts.entry(lower).and_modify(|(cnt, _)| *cnt += 1).or_insert((1, pos));
    }
    if counts.is_empty() {
        return None;
    }
    let mut ranked: Vec<(String, usize, usize)> = counts.into_iter().map(|(w, (c, p))| (w, c, p)).collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.2.cmp(&b.2)));
    let top: Vec<String> = ranked
        .into_iter()
        .take(3)
        .map(|(w, _, _)| {
            let mut chars = w.chars();
            match chars.next() {
                Some(first) => format!("{}{}", first.to_uppercase(), chars.as_str()),
                None => w,
            }
        })
        .collect();
    if top.is_empty() { None } else { Some(top.join(" ")) }
}

/// Match game/topic keywords from transcript text in the first 45 seconds of a chapter,
/// falling back to extracted content keywords before generic visual layout labels.
pub fn classify_chapter_title(layout: VisualLayoutKind, transcript_snippet: &str) -> String {
    let lower = transcript_snippet.to_ascii_lowercase();
    let keywords: [(&[&str], &str); 14] = [
        (&["escape from tarkov", "tarkov", "scav", "pmc"], "Escape from Tarkov"),
        (&["counter-strike 2", "counter strike", "cs2", "csgo", "mirage", "inferno"], "Counter-Strike 2"),
        (&["minecraft", "creeper", "nether", "redstone", "ender dragon"], "Minecraft"),
        (&["valorant", "radiant", "vandal", "spike"], "Valorant"),
        (&["fortnite", "battle royale", "tilted"], "Fortnite"),
        (&["elden ring", "malenia", "tarnished", "soulslike"], "Elden Ring"),
        (&["apex legends", "apex", "kings canyon"], "Apex Legends"),
        (&["league of legends", "summoner", "baron"], "League of Legends"),
        (&["gta", "grand theft auto", "los santos"], "GTA V"),
        (&["just chatting", "q&a", "catching up", "ama"], "Just Chatting"),
        (&["reacting", "reaction", "watch this video", "tier list"], "Reacting"),
        (&["podcast", "interview", "guest"], "Podcast & Discussion"),
        (&["coding", "programming", "rust", "refactor"], "Live Coding"),
        (&["intro", "starting soon", "welcome back"], "Stream Intro"),
    ];

    for (patterns, title) in keywords {
        if patterns.iter().any(|p| lower.contains(p)) {
            return title.to_string();
        }
    }
    if let Some(topic) = extract_transcript_topic_title(transcript_snippet) {
        return topic;
    }
    layout.default_title().to_string()
}

/// Parse LLM-returned `(start_ms, title)` topic boundaries, snap any boundary within 15 seconds
/// of a visual layout cut point, and build contiguous `ChapterMarker` spans covering `0..total_ms`.
pub fn parse_and_snap_llm_chapters(
    mut llm_boundaries: Vec<(u64, String)>,
    visual_boundaries_ms: &[u64],
    total_duration_ms: u64,
    min_gap_ms: u64,
) -> Vec<ChapterMarker> {
    if total_duration_ms == 0 || llm_boundaries.is_empty() {
        return Vec::new();
    }
    const SNAP_WINDOW_MS: u64 = 15_000;
    let min_gap = min_gap_ms.clamp(5_000, 30_000);

    for (ms, _) in &mut llm_boundaries {
        if *ms > 0
            && let Some(&snap) = visual_boundaries_ms.iter().filter(|&&vb| vb > 0).min_by_key(|&&vb| vb.abs_diff(*ms))
            && snap.abs_diff(*ms) <= SNAP_WINDOW_MS
        {
            *ms = snap;
        }
        *ms = (*ms).min(total_duration_ms.saturating_sub(1_000));
    }

    llm_boundaries.sort_by_key(|(ms, _)| *ms);
    let mut deduped: Vec<(u64, String)> = Vec::new();
    for (ms, raw_title) in llm_boundaries {
        let title = raw_title.trim();
        let clean_title = if title.is_empty() { format!("Chapter {}", deduped.len() + 1) } else { title.chars().take(48).collect() };
        if let Some(last) = deduped.last_mut() {
            if ms.saturating_sub(last.0) < min_gap {
                continue;
            }
        } else if ms > 0 {
            deduped.push((0, "Stream Intro".into()));
        }
        deduped.push((ms, clean_title));
    }
    if deduped.is_empty() {
        return Vec::new();
    }
    deduped.truncate(MAX_CHAPTERS);
    if deduped[0].0 != 0 {
        deduped[0].0 = 0;
    }

    let mut out = Vec::with_capacity(deduped.len());
    for i in 0..deduped.len() {
        let start_ms = deduped[i].0;
        let end_ms = deduped.get(i + 1).map_or(total_duration_ms, |(next_ms, _)| *next_ms).max(start_ms.saturating_add(1_000)).min(total_duration_ms);
        let color = CHAPTER_COLORS[i % CHAPTER_COLORS.len()];
        out.push(ChapterMarker { start_ms, end_ms, title: deduped[i].1.clone(), color: color.name().to_string() });
    }
    out
}

fn build_transcript_chunks(transcript: &[(f64, f64, String)]) -> Vec<Value> {
    if transcript.is_empty() {
        return Vec::new();
    }
    let total_end = transcript.last().map_or(0.0, |(_, e, _)| *e);
    let bucket_sec = (total_end / 100.0).max(30.0);
    let mut chunks = Vec::new();
    let mut cur_start = transcript[0].0;
    let mut cur_buf = String::new();

    for (s, _, text) in transcript {
        if *s - cur_start >= bucket_sec && !cur_buf.is_empty() {
            chunks.push(json!({
                "startMs": (cur_start.max(0.0) * 1000.0).round() as u64,
                "text": cur_buf.trim().chars().take(240).collect::<String>(),
            }));
            cur_start = *s;
            cur_buf.clear();
            if chunks.len() >= 120 {
                break;
            }
        }
        if !cur_buf.is_empty() {
            cur_buf.push(' ');
        }
        cur_buf.push_str(text);
    }
    if !cur_buf.is_empty() && chunks.len() < 120 {
        chunks.push(json!({
            "startMs": (cur_start.max(0.0) * 1000.0).round() as u64,
            "text": cur_buf.trim().chars().take(240).collect::<String>(),
        }));
    }
    chunks
}

fn try_detect_chapters_with_ollama(
    endpoint: &str,
    transcript: &[(f64, f64, String)],
    visual_boundaries_ms: &[u64],
    total_duration_ms: u64,
) -> Option<Vec<ChapterMarker>> {
    if transcript.is_empty() {
        return None;
    }
    let trimmed = endpoint.trim().trim_end_matches('/');
    if trimmed.is_empty() || !(trimmed.starts_with("http://127.0.0.1") || trimmed.starts_with("http://localhost") || trimmed.starts_with("http://[::1]")) {
        return None;
    }
    let chunks = build_transcript_chunks(transcript);
    if chunks.is_empty() {
        return None;
    }

    let url = format!("{trimmed}/api/generate");
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .https_only(false)
        .timeout_connect(Some(Duration::from_millis(1_500)))
        .timeout_recv_response(Some(Duration::from_secs(12)))
        .user_agent("Orpheus/1.0")
        .build()
        .new_agent();

    let prompt = format!(
        "Analyze this timestamped stream VOD transcript and visual cut hints (visualBoundariesMs: {visual_boundaries_ms:?}). \
        Identify both topic shifts inside Just Chatting / discussion segments and game or activity transitions. \
        Return JSON {{\"chapters\": [{{\"startMs\": 0, \"title\": \"Concise 2-5 Word Title\"}}]}} ordered by startMs. \
        Transcript chunks: {}",
        serde_json::to_string(&chunks).unwrap_or_default()
    );
    let body = json!({ "model": "llama3.2", "prompt": prompt, "stream": false, "format": "json" });
    let body_bytes = serde_json::to_vec(&body).unwrap_or_default();
    let resp = agent.post(&url).header("Content-Type", "application/json").send(&body_bytes).ok()?;
    let mut text = String::new();
    resp.into_body().into_reader().take(131_072).read_to_string(&mut text).ok()?;
    let outer = serde_json::from_str::<Value>(&text).ok()?;
    let inner_str = outer.get("response").and_then(Value::as_str)?;
    let parsed = serde_json::from_str::<Value>(inner_str).ok()?;
    let arr = parsed.get("chapters").and_then(Value::as_array)?;

    let mut boundaries = Vec::with_capacity(arr.len());
    for entry in arr {
        let Some(ms) = entry.get("startMs").and_then(Value::as_u64) else { continue };
        let Some(title) = entry.get("title").and_then(Value::as_str) else { continue };
        boundaries.push((ms, title.to_string()));
    }
    let chapters = parse_and_snap_llm_chapters(boundaries, visual_boundaries_ms, total_duration_ms, 20_000);
    if chapters.is_empty() { None } else { Some(chapters) }
}

fn try_refine_titles_with_ollama(endpoint: &str, drafts: &mut [(f64, f64, VisualLayoutKind, String, String)]) -> bool {
    let trimmed = endpoint.trim().trim_end_matches('/');
    if trimmed.is_empty() || !(trimmed.starts_with("http://127.0.0.1") || trimmed.starts_with("http://localhost") || trimmed.starts_with("http://[::1]")) {
        return false;
    }
    let url = format!("{trimmed}/api/generate");
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .https_only(false)
        .timeout_connect(Some(Duration::from_millis(1_500)))
        .timeout_recv_response(Some(Duration::from_secs(5)))
        .user_agent("Orpheus/1.0")
        .build()
        .new_agent();

    let summary_rows: Vec<Value> = drafts
        .iter()
        .enumerate()
        .map(|(i, (s, e, layout, title, snippet))| {
            let short_snip: String = snippet.chars().take(180).collect();
            json!({
                "index": i,
                "startSec": s.round(),
                "endSec": e.round(),
                "layout": layout,
                "heuristicTitle": title,
                "transcript": short_snip,
            })
        })
        .collect();

    let prompt = format!(
        "Given these stream VOD chapters, return JSON {{\"titles\": [\"...\"]}} with concise 2-5 word titles in the same order: {}",
        serde_json::to_string(&summary_rows).unwrap_or_default()
    );
    let body = json!({ "model": "llama3.2", "prompt": prompt, "stream": false, "format": "json" });
    let body_bytes = serde_json::to_vec(&body).unwrap_or_default();
    let Ok(resp) = agent.post(&url).header("Content-Type", "application/json").send(&body_bytes) else {
        return false;
    };
    let mut text = String::new();
    if resp.into_body().into_reader().take(65_536).read_to_string(&mut text).is_err() {
        return false;
    }
    let Ok(outer) = serde_json::from_str::<Value>(&text) else {
        return false;
    };
    let Some(inner_str) = outer.get("response").and_then(Value::as_str) else {
        return false;
    };
    let Ok(parsed) = serde_json::from_str::<Value>(inner_str) else {
        return false;
    };
    let Some(titles) = parsed.get("titles").and_then(Value::as_array) else {
        return false;
    };

    let mut applied = false;
    for (idx, item) in drafts.iter_mut().enumerate() {
        if let Some(t) = titles.get(idx).and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty()) {
            item.3 = t.chars().take(48).collect();
            applied = true;
        }
    }
    applied
}

fn collect_transcript_snippets(seq: &Sequence) -> Vec<(f64, f64, String)> {
    let mut out = Vec::new();
    for ct in &seq.caption_tracks {
        for cap in &ct.captions {
            let s = cap.start.seconds();
            let e = cap.end().seconds();
            let t = cap.text.trim();
            if !t.is_empty() {
                out.push((s, e, t.to_string()));
            }
        }
    }
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    out
}

fn snippet_for_window(transcript: &[(f64, f64, String)], max_seg_dur: f64, start_sec: f64, window_sec: f64) -> String {
    if transcript.is_empty() {
        return String::new();
    }
    let end_sec = start_sec + window_sec;
    // `transcript` is sorted by segment start time; bound both ends with binary search so
    // 12-hour timelines with 15,000+ captions stay O(log N + K) per sample.
    let hi = transcript.partition_point(|(s, _, _)| *s <= end_sec);
    let lookback = start_sec - max_seg_dur.max(0.0);
    let lo = transcript[..hi].partition_point(|(s, _, _)| *s < lookback);
    transcript[lo..hi].iter().filter(|(s, e, _)| *e >= start_sec && *s <= end_sec).map(|(_, _, t)| t.as_str()).collect::<Vec<_>>().join(" ")
}

struct SampleSourceClip {
    timeline_start: f64,
    timeline_end: f64,
    source_in: f64,
    speed: f64,
    src_w: u32,
    src_h: u32,
    src: Option<SharedSource>,
}

fn find_active_clip(tracks: &[Vec<SampleSourceClip>], t: f64) -> Option<&SampleSourceClip> {
    for tr in tracks {
        let pos = tr.partition_point(|c| c.timeline_start <= t);
        if let Some(c) = pos.checked_sub(1).and_then(|i| tr.get(i))
            && c.timeline_end > c.timeline_start
            && t >= c.timeline_start
            && t < c.timeline_end
        {
            return Some(c);
        }
    }
    None
}

fn analyze_vod_chapters(
    tracks: Vec<Vec<SampleSourceClip>>,
    total_duration_sec: f64,
    transcript: Vec<(f64, f64, String)>,
    sample_interval_sec: f64,
    min_chapter_sec: f64,
    ollama_url: String,
    prog: Option<&Progress>,
) -> Result<(Vec<ChapterMarker>, bool)> {
    check_cancelled(prog)?;
    if !total_duration_sec.is_finite() || total_duration_sec <= 0.1 || tracks.iter().all(Vec::is_empty) {
        return Err(bad("ai.vodChapters", "Add a video clip to the timeline before chapterizing"));
    }

    let max_seg_dur = transcript.iter().map(|(s, e, _)| (e - s).max(0.0)).fold(0.0_f64, f64::max);
    let safe_interval = if sample_interval_sec.is_finite() { sample_interval_sec.max(2.0) } else { DEFAULT_SAMPLE_INTERVAL_SEC };
    let safe_min_chapter = if min_chapter_sec.is_finite() { min_chapter_sec.max(10.0) } else { DEFAULT_MIN_CHAPTER_SEC };
    let step = (total_duration_sec / MAX_SAMPLES as f64).max(safe_interval);
    let total_steps = ((total_duration_sec / step).ceil() as usize).clamp(1, MAX_SAMPLES);
    let mut signatures = Vec::with_capacity(total_steps);

    let mut t = 0.0_f64;
    let mut idx = 0usize;
    while t < total_duration_sec && idx < MAX_SAMPLES {
        check_cancelled(prog)?;
        let active_clip = find_active_clip(&tracks, t);
        let sig = if let Some(c) = active_clip
            && let Some(ref src) = c.src
        {
            let speed = if c.speed.is_finite() { c.speed.abs().max(0.01) } else { 1.0 };
            let source_in = if c.source_in.is_finite() { c.source_in.max(0.0) } else { 0.0 };
            let media_sec = (source_in + (t - c.timeline_start).max(0.0) * speed).max(0.0);
            let max_side = c.src_w.max(c.src_h).max(1) as f64;
            let min_side = c.src_w.min(c.src_h).max(1) as f64;
            let scale = (ANALYSIS_MAX_DIM / max_side).max(8.0 / min_side).clamp(0.04, 1.0) as f32;
            let sampled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let frame = src.video_frame(FrameRequest { time: Tick::from_seconds_f64(media_sec), scale }).ok()?;
                if frame.width == 0 || frame.height == 0 {
                    return None;
                }
                let rgba = frame.to_rgba8();
                Some(classify_frame_rgba(&rgba, frame.width as usize, frame.height as usize, t))
            }));
            match sampled {
                Ok(Some(sig)) => sig,
                _ => neutral_signature(t),
            }
        } else {
            neutral_signature(t)
        };
        let mut sig = sig;
        // Refine borderline layout with speech-density cues when captions/transcript exist:
        // continuous dense speech hints TalkingHead, sparse speech with active visual edges hints GamingHud.
        if !transcript.is_empty() {
            let win_words = snippet_for_window(&transcript, max_seg_dur, t, step.max(10.0)).split_whitespace().count();
            if win_words >= 18 && sig.edge_hud_energy < 0.08 && sig.center_skin_ratio >= 0.08 {
                sig.layout = VisualLayoutKind::TalkingHead;
            } else if win_words <= 4 && (sig.edge_hud_energy >= 0.05 || sig.center_action_energy >= 0.06) {
                sig.layout = VisualLayoutKind::GamingHud;
            }
        }
        signatures.push(sig);
        idx += 1;
        t += step;
        if let Some(p) = prog {
            let pct = ((idx as f64 / total_steps as f64) * 85.0) as u64;
            p.done.store(pct.min(85), Ordering::Relaxed);
            *lock(&p.status) = format!("Scanning VOD layout ({}/{total_steps})…", idx);
        }
    }

    let mut clip_boundaries_sec: Vec<f64> =
        tracks.iter().flat_map(|tr| tr.iter()).map(|c| c.timeline_start).filter(|&s| s > 1.0 && s < total_duration_sec - 1.0).collect();
    clip_boundaries_sec.sort_by(f64::total_cmp);
    clip_boundaries_sec.dedup_by(|a, b| (*a - *b).abs() < 1.0);

    let raw_chapters = detect_sustained_chapters_with_hints(&signatures, total_duration_sec, safe_min_chapter, &clip_boundaries_sec);
    let total_duration_ms = (total_duration_sec.max(0.0) * 1000.0).round() as u64;
    let visual_boundaries_ms: Vec<u64> = raw_chapters.iter().map(|(s, _, _)| (s.max(0.0) * 1000.0).round() as u64).collect();

    if let Some(p) = prog {
        p.done.store(90, Ordering::Relaxed);
        *lock(&p.status) = "Classifying stream chapters…".into();
    }

    if let Some(llm_chapters) = try_detect_chapters_with_ollama(&ollama_url, &transcript, &visual_boundaries_ms, total_duration_ms) {
        if let Some(p) = prog {
            p.done.store(100, Ordering::Relaxed);
        }
        return Ok((llm_chapters, true));
    }

    let mut drafts: Vec<(f64, f64, VisualLayoutKind, String, String)> = raw_chapters
        .into_iter()
        .map(|(s, e, layout)| {
            let snip = snippet_for_window(&transcript, max_seg_dur, s, 45.0);
            let title = classify_chapter_title(layout, &snip);
            (s, e, layout, title, snip)
        })
        .collect();

    let used_ollama = try_refine_titles_with_ollama(&ollama_url, &mut drafts);

    let chapters = drafts
        .into_iter()
        .enumerate()
        .map(|(i, (s, e, _, title, _))| {
            let color = CHAPTER_COLORS[i % CHAPTER_COLORS.len()];
            ChapterMarker { start_ms: (s.max(0.0) * 1000.0).round() as u64, end_ms: (e.max(s) * 1000.0).round() as u64, title, color: color.name().to_string() }
        })
        .collect();

    if let Some(p) = prog {
        p.done.store(100, Ordering::Relaxed);
    }
    Ok((chapters, used_ollama))
}

fn apply_vod_chapters(s: &mut Session, seq_id: ItemId, split_clips: bool, chapters: Vec<ChapterMarker>, used_ollama: bool) -> Result<VodChapterReport> {
    // Bound and clamp whatever the detector (or an LLM) returned: chapters starting past the end
    // of the sequence are dropped, ends are clamped, and the count is capped.
    let dur_ms = s.project.sequence(seq_id).map(|q| (q.duration().seconds().max(0.0) * 1000.0).round() as u64).ok_or(EngineError::NoSequence)?;
    let mut chapters: Vec<ChapterMarker> = chapters
        .into_iter()
        .filter(|c| dur_ms == 0 || c.start_ms < dur_ms)
        .map(|mut c| {
            if dur_ms > 0 {
                c.end_ms = c.end_ms.min(dur_ms);
            }
            c.end_ms = c.end_ms.max(c.start_ms);
            c
        })
        .collect();
    chapters.sort_by_key(|c| c.start_ms);
    chapters.truncate(MAX_CHAPTERS);
    if chapters.is_empty() {
        return Err(bad("ai.vodChapters", "No stream chapters were detected"));
    }

    let splits_applied = s.edit_group("Stream VOD Chapterizer", |s| {
        s.edit_sequence("Stream VOD Chapterizer", |seq, ctx, _| {
            let rate = seq.settings.frame_rate;

            // Replace prior Chapter markers while keeping manual user Comment/WebLink markers intact.
            seq.markers.retain(|m| m.kind != MarkerKind::Chapter);

            let mut split_count = 0usize;
            if split_clips {
                for ch in chapters.iter().skip(1) {
                    let t = rate.snap_nearest(Tick::from_seconds_f64(ch.start_ms as f64 / 1000.0));
                    if t <= Tick::ZERO {
                        continue;
                    }
                    let splittable_tracks: Vec<filmcraft_project::TrackId> = seq
                        .all_tracks()
                        .filter(|tr| {
                            !tr.locked
                                && tr.items.iter().any(|it| {
                                    it.duration > Tick::ZERO && t > it.start && t < it.end() && (t - it.start) > Tick::ZERO && (it.end() - t) > Tick::ZERO
                                })
                        })
                        .map(|tr| tr.id)
                        .collect();
                    if !splittable_tracks.is_empty() {
                        split_count += filmcraft_edit::razor(seq, &splittable_tracks, t, ctx).len();
                    }
                }
            }

            for (idx, ch) in chapters.iter().enumerate() {
                let start_t = rate.snap_nearest(Tick::from_seconds_f64(ch.start_ms as f64 / 1000.0));
                let end_t = rate.snap_nearest(Tick::from_seconds_f64(ch.end_ms as f64 / 1000.0)).max(start_t);
                let color = CHAPTER_COLORS[idx % CHAPTER_COLORS.len()];
                seq.markers.push(Marker {
                    url: String::new(),
                    author: String::new(),
                    custom_color: None,
                    id: MarkerId(ctx.alloc()),
                    start: start_t,
                    duration: end_t - start_t,
                    name: ch.title.clone(),
                    comment: format!("{} - {}", format_hms_ms(ch.start_ms), format_hms_ms(ch.end_ms)),
                    kind: MarkerKind::Chapter,
                    color,
                });
            }
            seq.markers.sort_by_key(|m| m.start);
            Ok(split_count)
        })
    })?;

    let rep = VodChapterReport { seq: seq_id, chapters_created: chapters.len(), splits_applied, used_ollama, chapters };
    s.toast(format!("Chapterized {} stream segments ({splits_applied} clip splits)", rep.chapters_created));
    Ok(rep)
}

fn vod_chapters_cmd(s: &mut Session, p: &Value) -> Result<Value> {
    let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let seq = s.project.sequence(seq_id).ok_or(EngineError::NoSequence)?;
    let total_dur = seq.duration().seconds();
    if total_dur <= 0.1 {
        return Err(bad("ai.vodChapters", "Add a video clip to the timeline before chapterizing"));
    }

    let split_clips = bool_p(p, "splitClips").unwrap_or(true);
    let sample_interval_sec = p.get("sampleIntervalSeconds").and_then(Value::as_f64).unwrap_or(DEFAULT_SAMPLE_INTERVAL_SEC);
    let min_chapter_sec = p.get("minChapterSeconds").and_then(Value::as_f64).unwrap_or(DEFAULT_MIN_CHAPTER_SEC);
    let ollama_url = str_p(p, "ollamaUrl").unwrap_or("").to_string();
    let wait = bool_p(p, "wait").unwrap_or(false);

    let provider = s.media.provider(s.project.clone(), s.services.clone()).own_sources();
    let mut track_clips: Vec<Vec<SampleSourceClip>> = Vec::new();
    for tr in seq.video_tracks.iter().rev() {
        if tr.kind != TrackKind::Video || tr.locked {
            continue;
        }
        let mut tr_items = Vec::new();
        for it in &tr.items {
            if !it.enabled || it.duration <= Tick::ZERO {
                continue;
            }
            let src = filmcraft_render::SourceProvider::source(&provider, it.item);
            let (src_w, src_h) = src
                .as_ref()
                .and_then(|s| s.info().video.as_ref().map(|v| (v.width.max(1), v.height.max(1))))
                .unwrap_or((seq.settings.width.max(1), seq.settings.height.max(1)));
            tr_items.push(SampleSourceClip {
                timeline_start: it.start.seconds(),
                timeline_end: it.end().seconds(),
                source_in: it.source_in.seconds(),
                speed: it.speed,
                src_w,
                src_h,
                src,
            });
        }
        if !tr_items.is_empty() {
            tr_items.sort_by(|a, b| a.timeline_start.total_cmp(&b.timeline_start));
            track_clips.push(tr_items);
        }
    }
    if track_clips.is_empty() {
        return Err(bad("ai.vodChapters", "No unlocked video clips found on the timeline"));
    }
    let mut transcript = collect_transcript_snippets(seq);
    // Transcribing a long VOD blocks, so only a synchronous (`wait`) call does it; the UI path
    // uses the captions already on the timeline and never freezes the session thread.
    if wait
        && transcript.is_empty()
        && {
            let st = crate::subtitles::status();
            st.cli_found && st.models.iter().any(|m| m.installed)
        }
        && s.execute("ai.generateSubtitles", json!({"wait": true})).is_ok()
        && let Some(updated_seq) = s.project.sequence(seq_id)
    {
        transcript = collect_transcript_snippets(updated_seq);
    }

    if wait {
        let (chapters, used_ollama) = analyze_vod_chapters(track_clips, total_dur, transcript, sample_interval_sec, min_chapter_sec, ollama_url, None)?;
        let rep = apply_vod_chapters(s, seq_id, split_clips, chapters, used_ollama)?;
        return Ok(serde_json::to_value(rep).unwrap_or(Value::Null));
    }

    let id = next_job_id(s);
    let prog = Arc::new(Progress::default());
    prog.total.store(100, Ordering::Relaxed);
    *lock(&prog.status) = "Scanning stream VOD chapters…".into();
    let result: Arc<Mutex<Option<std::result::Result<Report, String>>>> = Arc::default();
    let slot: Arc<Mutex<Option<(Vec<ChapterMarker>, bool)>>> = Arc::new(Mutex::new(None));

    // The job is registered only once its worker thread exists: a failed spawn must not leave a
    // never-finishing job (and a stuck spinner) behind.
    let epoch = s.document_epoch;
    let (reg_prog, reg_result, reg_slot) = (prog.clone(), result.clone(), slot.clone());
    let register = move |s: &mut Session| {
        s.jobs.push(Job { id, label: "Stream VOD Chapterizer".into(), progress: reg_prog, result: reg_result });
        s.vod_chapters_jobs.push(PendingVodChapterJob { job: id, seq: seq_id, split_clips, epoch, result: reg_slot });
    };

    let prog_bg = prog.clone();
    let result_bg = result.clone();
    let run = crate::export_tools::guard_job(prog_bg.clone(), result_bg.clone(), move || {
        let t0 = web_time::Instant::now();
        let rep = match analyze_vod_chapters(track_clips, total_dur, transcript, sample_interval_sec, min_chapter_sec, ollama_url, Some(&prog_bg)) {
            Ok((chapters, used_ollama)) => {
                let n = chapters.len() as u64;
                *lock(&slot) = Some((chapters, used_ollama));
                Ok(Report { frames: n, seconds: t0.elapsed().as_secs_f64(), ..Default::default() })
            }
            Err(e) => {
                let msg = e.to_string();
                *lock(&prog_bg.error) = Some(msg.clone());
                *lock(&prog_bg.status) = msg.clone();
                Err(msg)
            }
        };
        *lock(&result_bg) = Some(rep);
        prog_bg.finished.store(true, Ordering::Relaxed);
    });

    if cfg!(target_arch = "wasm32") {
        register(s);
        run();
        poll(s);
        return Ok(json!({ "job": id }));
    }

    std::thread::Builder::new()
        .name(format!("orpheus-vod-chapters-{id}"))
        .spawn(run)
        .map_err(|e| EngineError::Other(format!("failed to start VOD chapterizer thread: {e}")))?;
    register(s);

    Ok(json!({ "job": id }))
}

/// Apply completed background VOD chapterizer jobs on the session thread.
pub fn poll(s: &mut Session) {
    let mut i = 0usize;
    while i < s.vod_chapters_jobs.len() {
        let job = s.jobs.iter().find(|j| j.id == s.vod_chapters_jobs[i].job);
        let finished = job.is_none_or(|j| j.progress.finished.load(Ordering::Relaxed));
        let cancelled = job.is_some_and(|j| j.progress.cancel.load(Ordering::Relaxed));
        if !finished {
            i += 1;
            continue;
        }
        let pj = s.vod_chapters_jobs.remove(i);
        let Some((chapters, used_ollama)) = lock(&pj.result).take().filter(|_| !cancelled) else { continue };
        if pj.epoch != s.document_epoch {
            s.error_toast("ai.vodChapters", "Stream VOD Chapterizer: the project was replaced; nothing was changed");
            continue;
        }
        if let Err(e) = apply_vod_chapters(s, pj.seq, pj.split_clips, chapters, used_ollama) {
            s.error_toast("ai.vodChapters", e.to_string());
        }
    }
}

pub(crate) fn commands() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "ai.vodChapters",
            label: "Stream VOD Chapterizer",
            menu: &[],
            shortcut: None,
            params: r#"{"splitClips":bool?,"sampleIntervalSeconds":f64?,"minChapterSeconds":f64?,"ollamaUrl":str?,"wait":bool?}"#,
            enabled: has_seq,
            run: vod_chapters_cmd,
            journal: false,
            named: NamedKeys::NONE,
        },
        CommandSpec {
            id: "ai.vodChapters.youtubeText",
            label: "Copy YouTube Chapters",
            menu: &[],
            shortcut: None,
            params: r#"{}"#,
            enabled: always,
            run: |s, _| {
                let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
                let text = format_youtube_chapters(seq);
                Ok(json!({ "text": text, "count": text.lines().filter(|l| !l.is_empty()).count() }))
            },
            journal: false,
            named: NamedKeys::NONE,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use filmcraft_project::{Caption, CaptionTrack};
    use filmcraft_time::TimeRange;

    fn solid_sig(time_sec: f64, layout: VisualLayoutKind, base_luma: f32) -> FrameSignature {
        FrameSignature {
            time_sec,
            layout,
            luma_grid: [base_luma; 16],
            color_hist: [base_luma; 12],
            center_skin_ratio: if layout == VisualLayoutKind::TalkingHead { 0.28 } else { 0.02 },
            edge_hud_energy: if layout == VisualLayoutKind::GamingHud { 0.14 } else { 0.03 },
            center_action_energy: if layout == VisualLayoutKind::GamingHud { 0.12 } else { 0.02 },
        }
    }

    #[test]
    fn sustained_hysteresis_ignores_brief_alt_tab_and_splits_real_transitions() {
        let mut sigs = Vec::new();
        for step in 0..30 {
            let t = step as f64 * 10.0;
            let sig = if (60.0..=70.0).contains(&t) {
                // Brief 20s Alt+Tab to desktop must NOT trigger a chapter split.
                solid_sig(t, VisualLayoutKind::DesktopReact, 0.85)
            } else if t < 120.0 {
                solid_sig(t, VisualLayoutKind::TalkingHead, 0.35)
            } else {
                // Sustained 180s gaming segment >= 45s MUST create a chapter boundary at 120s.
                solid_sig(t, VisualLayoutKind::GamingHud, 0.25)
            };
            sigs.push(sig);
        }

        let chapters = detect_sustained_chapters(&sigs, 300.0, 45.0);
        assert_eq!(chapters.len(), 2, "brief alt-tab must be ignored, leaving 2 chapters: {chapters:?}");
        assert_eq!(chapters[0].2, VisualLayoutKind::TalkingHead);
        assert_eq!(chapters[1].2, VisualLayoutKind::GamingHud);
        assert!((chapters[1].0 - 120.0).abs() < 1e-6);
    }

    #[test]
    fn transcript_keywords_classify_games_and_topics() {
        assert_eq!(classify_chapter_title(VisualLayoutKind::GamingHud, "let's drop into escape from tarkov labs"), "Escape from Tarkov");
        assert_eq!(classify_chapter_title(VisualLayoutKind::GamingHud, "playing counter-strike 2 mirage"), "Counter-Strike 2");
        assert_eq!(classify_chapter_title(VisualLayoutKind::GamingHud, "building a redstone farm in minecraft"), "Minecraft");
        assert_eq!(classify_chapter_title(VisualLayoutKind::TalkingHead, "welcome to just chatting q&a"), "Just Chatting");
        assert_eq!(classify_chapter_title(VisualLayoutKind::DesktopReact, "today we are reacting to trailers"), "Reacting");
        // Non-game transcript extracts real topic keywords instead of falling back to "Just Chatting".
        let topic = classify_chapter_title(VisualLayoutKind::TalkingHead, "Building a custom mechanical keyboard with tactile switches and custom keycaps");
        assert_ne!(topic, "Just Chatting");
        assert!(topic.contains("Custom"), "expected extracted keyword in `{topic}`");
    }

    #[test]
    fn vod_chapters_splits_tracks_attaches_markers_formats_youtube_and_undoes() {
        let mut s = Session::default();
        s.execute("file.newSequence", json!({"name": "StreamVOD", "video": 1, "audio": 1})).unwrap();
        let seq_id = s.state.active_sequence.unwrap();

        let item_val = s.execute("file.newBarsAndTone", json!({"seconds": 180.0})).unwrap();
        let item_id = ItemId(item_val["item"].as_u64().unwrap());

        s.edit("Place VOD on timeline", |pr, _| {
            let seq = pr.sequence_mut(seq_id).unwrap();
            let rate = seq.settings.frame_rate;
            let dur = Tick::from_seconds_f64(180.0);
            let v = pr.make_track_item(item_id, TrackKind::Video, Tick::ZERO, TimeRange::new(Tick::ZERO, dur), rate).unwrap();
            let a = pr.make_track_item(item_id, TrackKind::Audio, Tick::ZERO, TimeRange::new(Tick::ZERO, dur), rate).unwrap();
            let mut ct = CaptionTrack::new(filmcraft_project::TrackId(pr.alloc_id()), "C1".into(), filmcraft_project::CaptionFormat::Subtitle);
            ct.captions.push(Caption {
                id: filmcraft_project::ClipId(pr.alloc_id()),
                start: Tick::from_seconds_f64(5.0),
                duration: Tick::from_seconds_f64(10.0),
                text: "Welcome to Just Chatting!".into(),
                speaker: None,
                cue_id: None,
                settings: String::new(),
            });
            ct.captions.push(Caption {
                id: filmcraft_project::ClipId(pr.alloc_id()),
                start: Tick::from_seconds_f64(70.0),
                duration: Tick::from_seconds_f64(10.0),
                text: "Booting up Minecraft survival now".into(),
                speaker: None,
                cue_id: None,
                settings: String::new(),
            });
            let seq_mut = pr.sequence_mut(seq_id).unwrap();
            seq_mut.video_tracks[0].items.push(v);
            seq_mut.audio_tracks[0].items.push(a);
            seq_mut.caption_tracks.push(ct);
            Ok(())
        })
        .unwrap();

        let chapters = vec![
            ChapterMarker { start_ms: 0, end_ms: 65_000, title: "Just Chatting".into(), color: "Cerulean".into() },
            ChapterMarker { start_ms: 65_000, end_ms: 180_000, title: "Minecraft".into(), color: "Emerald".into() },
        ];
        let rep = apply_vod_chapters(&mut s, seq_id, true, chapters, false).unwrap();
        assert_eq!(rep.chapters_created, 2);
        assert_eq!(rep.splits_applied, 2);

        let seq = s.project.sequence(seq_id).unwrap();
        assert_eq!(seq.video_tracks[0].items.len(), 2);
        assert_eq!(seq.audio_tracks[0].items.len(), 2);
        assert_eq!(seq.markers.len(), 2);
        assert_eq!(seq.markers[0].kind, MarkerKind::Chapter);

        let yt = format_youtube_chapters(seq);
        assert_eq!(yt, "00:00:00 Just Chatting\n00:01:05 Minecraft");

        s.execute("edit.undo", json!({})).unwrap();
        let seq_after = s.project.sequence(seq_id).unwrap();
        assert_eq!(seq_after.video_tracks[0].items.len(), 1);
        assert_eq!(seq_after.markers.len(), 0);
    }

    #[test]
    fn llm_topic_boundaries_snap_to_nearby_visual_cuts_and_keep_just_chatting_splits() {
        let llm = vec![(0, "Stream Intro & News".into()), (95_000, "Tier List Debate".into()), (236_000, "Counter-Strike 2 Ranked".into())];
        // Visual boundary at 240s is within 15s of 236s -> snaps to 240_000ms.
        // 95_000ms has no nearby visual cut -> preserved as an in-segment Just Chatting topic split.
        let chapters = parse_and_snap_llm_chapters(llm, &[0, 240_000], 420_000, 20_000);
        assert_eq!(chapters.len(), 3);
        assert_eq!(chapters[0].start_ms, 0);
        assert_eq!(chapters[1].start_ms, 95_000);
        assert_eq!(chapters[1].title, "Tier List Debate");
        assert_eq!(chapters[2].start_ms, 240_000);
        assert_eq!(chapters[2].end_ms, 420_000);
    }

    fn chapter_marker(id: u64, start_s: f64, dur_s: f64, name: &str) -> Marker {
        Marker {
            url: String::new(),
            author: String::new(),
            custom_color: None,
            id: MarkerId(id),
            start: Tick::from_seconds_f64(start_s),
            duration: Tick::from_seconds_f64(dur_s),
            name: name.into(),
            comment: String::new(),
            kind: MarkerKind::Chapter,
            color: Label::Green,
        }
    }

    #[test]
    fn batch_export_file_names_are_unique_windows_safe_and_keep_unicode() {
        // Two chapters with the same title never overwrite each other: the running number differs.
        let a = chapter_export_filename(0, 3, "Just Chatting", "mp4");
        let b = chapter_export_filename(2, 3, "Just Chatting", "mp4");
        assert_eq!((a.as_str(), b.as_str()), ("01_Just_Chatting.mp4", "03_Just_Chatting.mp4"));
        // Every Windows-invalid character is replaced.
        assert_eq!(chapter_export_filename(0, 1, r#"a<b>c:d"e/f\g|h?i*j"#, "mp4"), "01_a_b_c_d_e_f_g_h_i_j.mp4");
        // Blank titles, Cyrillic titles, reserved device names and trailing dots.
        assert_eq!(chapter_export_filename(3, 5, "   ", "mov"), "04_Chapter_4.mov");
        assert_eq!(chapter_export_filename(0, 2, "Игра: Тарков", "mp4"), "01_Игра_Тарков.mp4");
        assert_eq!(chapter_export_filename(0, 2, "CON.", "mp4"), "01_CON.mp4");
        // More than 99 chapters widen the number so names still sort.
        assert_eq!(chapter_export_filename(4, 120, "x", "mp4"), "005_x.mp4");
        // A hostile extension and a huge title stay bounded and path-free.
        let f = chapter_export_filename(0, 1, &"я".repeat(500), "m/p4");
        assert_eq!(f.chars().count(), "01_".len() + 40 + ".mp4".len());
        assert!(f.ends_with(".mp4") && !f.contains('/'));
        // Hostile index/total never overflow.
        let _ = chapter_export_filename(usize::MAX, usize::MAX, "x", "mp4");
    }

    #[test]
    fn export_ranges_clamp_skip_and_fill_handmade_chapters() {
        let mut other = chapter_marker(9, 5.0, 0.0, "just a comment");
        other.kind = MarkerKind::Comment;
        let markers = vec![
            chapter_marker(1, 0.0, 0.0, "Intro"),           // hand-made: runs to the next chapter
            chapter_marker(2, 60.0, 0.0, "Chat"),           // runs to the next chapter (100 s)
            chapter_marker(3, 100.0, 500.0, "Game"),        // longer than the sequence: clamped to 180 s
            chapter_marker(5, 179.95, 0.0, "Sliver"),       // shorter than 0.1 s: dropped
            chapter_marker(4, 400.0, 10.0, "Past the end"), // starts past the end: dropped
            other,
        ];
        let ranges = chapter_export_ranges(&markers, Tick::from_seconds_f64(180.0));
        let names: Vec<&str> = ranges.iter().map(|r| r.0.as_str()).collect();
        assert_eq!(names, ["Intro", "Chat", "Game"]);
        assert_eq!((ranges[0].1, ranges[0].2), (0.0, 60.0));
        assert_eq!((ranges[1].1, ranges[1].2), (60.0, 100.0));
        assert_eq!((ranges[2].1, ranges[2].2), (100.0, 180.0));
        // An empty list or a zero-length sequence has nothing to export.
        assert!(chapter_export_ranges(&[], Tick::from_seconds_f64(180.0)).is_empty());
        assert!(chapter_export_ranges(&markers, Tick::ZERO).is_empty());
    }

    #[test]
    fn hostile_llm_boundaries_are_bounded_and_never_overflow() {
        let hostile = parse_and_snap_llm_chapters(vec![(u64::MAX, "x".into()), (0, String::new())], &[], 10_000, u64::MAX);
        assert_eq!(hostile.len(), 1);
        assert_eq!(hostile[0].title, "Chapter 1");
        assert!(hostile[0].end_ms <= 10_000);
        assert!(parse_and_snap_llm_chapters(vec![(0, "x".into())], &[], 0, 20_000).is_empty());
        // A 100-hour VOD with 2000 proposed chapters is capped, with no overflow in the ms math.
        let many: Vec<(u64, String)> = (0..2000u64).map(|i| (i * 30_000, format!("Topic {i}"))).collect();
        let capped = parse_and_snap_llm_chapters(many, &[u64::MAX], 360_000_000, 20_000);
        assert_eq!(capped.len(), MAX_CHAPTERS);
        assert!(capped.windows(2).all(|w| w[0].end_ms <= w[1].start_ms));
    }

    fn vod_session() -> (Session, ItemId) {
        let mut s = Session::default();
        s.execute("file.newSequence", json!({"name": "StreamVOD", "video": 1, "audio": 1})).unwrap();
        let seq_id = s.state.active_sequence.unwrap();
        let item_val = s.execute("file.newBarsAndTone", json!({"seconds": 180.0})).unwrap();
        let item_id = ItemId(item_val["item"].as_u64().unwrap());
        s.edit("Place VOD on timeline", |pr, _| {
            let rate = pr.sequence_mut(seq_id).unwrap().settings.frame_rate;
            let dur = Tick::from_seconds_f64(180.0);
            let v = pr.make_track_item(item_id, TrackKind::Video, Tick::ZERO, TimeRange::new(Tick::ZERO, dur), rate).unwrap();
            let a = pr.make_track_item(item_id, TrackKind::Audio, Tick::ZERO, TimeRange::new(Tick::ZERO, dur), rate).unwrap();
            let seq = pr.sequence_mut(seq_id).unwrap();
            seq.video_tracks[0].items.push(v);
            seq.audio_tracks[0].items.push(a);
            Ok(())
        })
        .unwrap();
        (s, seq_id)
    }

    fn chapter(start_ms: u64, end_ms: u64, title: &str) -> ChapterMarker {
        ChapterMarker { start_ms, end_ms, title: title.into(), color: "Green".into() }
    }

    #[test]
    fn out_of_range_chapters_are_dropped_and_an_empty_result_changes_nothing() {
        let (mut s, seq_id) = vod_session();
        let rep = apply_vod_chapters(
            &mut s,
            seq_id,
            true,
            vec![chapter(65_000, 600_000_000, "Late"), chapter(0, 65_000, "Early"), chapter(500_000_000, 600_000_000, "Past the end")],
            false,
        )
        .unwrap();
        assert_eq!(rep.chapters_created, 2, "the chapter past the end is dropped");
        assert_eq!(rep.chapters[0].title, "Early", "chapters are sorted by start");
        let seq = s.project.sequence(seq_id).unwrap();
        let seq_dur_ms = (seq.duration().seconds() * 1000.0).round() as u64;
        assert_eq!(rep.chapters[1].end_ms, seq_dur_ms, "an end past the sequence is clamped");
        assert_eq!(seq.markers.len(), 2);
        assert_eq!(seq.video_tracks[0].items.len(), 2, "one split at 65 s");

        let (mut s, seq_id) = vod_session();
        assert!(apply_vod_chapters(&mut s, seq_id, true, vec![chapter(900_000_000, 950_000_000, "Nowhere")], false).is_err());
        assert!(apply_vod_chapters(&mut s, seq_id, true, Vec::new(), false).is_err());
        let seq = s.project.sequence(seq_id).unwrap();
        assert_eq!((seq.markers.len(), seq.video_tracks[0].items.len()), (0, 1));
    }

    #[test]
    fn an_empty_timeline_is_a_clean_error_and_leaves_no_job_behind() {
        let mut s = Session::default();
        s.execute("file.newSequence", json!({"name": "Empty", "video": 1, "audio": 1})).unwrap();
        assert!(s.execute("ai.vodChapters", json!({"wait": true})).is_err());
        assert!(s.execute("ai.vodChapters", json!({})).is_err());
        assert!(s.vod_chapters_jobs.is_empty());
        assert!(s.jobs.is_empty());
    }

    #[test]
    fn multiple_distinct_clips_with_gaps_do_not_panic() {
        // 1. Variable-resolution, zero-dimension, truncated, and overflowing frames never panic.
        let _ = classify_frame_rgba(&[], 0, 0, f64::NAN);
        let _ = classify_frame_rgba(&[255; 16], 0, 100, -5.0);
        let _ = classify_frame_rgba(&[255; 16], 100, 0, f64::INFINITY);
        let _ = classify_frame_rgba(&[255; 16], usize::MAX, 4, 1.0);
        let _ = classify_frame_rgba(&[128; 10], 160, 90, 2.0);
        let sig_16_9 = classify_frame_rgba(&vec![180u8; 160 * 90 * 4], 160, 90, 10.0);
        let sig_9_16 = classify_frame_rgba(&vec![40u8; 90 * 160 * 4], 90, 160, 20.0);
        assert_eq!(sig_16_9.time_sec, 10.0);
        assert_eq!(sig_9_16.time_sec, 20.0);

        // 2. Multi-clip analysis with mixed dimensions, speeds, and gaps succeeds cleanly.
        let clips = vec![
            SampleSourceClip { timeline_start: 0.0, timeline_end: 60.0, source_in: 0.0, speed: 1.0, src_w: 1920, src_h: 1080, src: None },
            // Gap: 60.0..80.0
            SampleSourceClip { timeline_start: 80.0, timeline_end: 150.0, source_in: 0.0, speed: f64::NAN, src_w: 1080, src_h: 1920, src: None },
            // Gap: 150.0..170.0
            SampleSourceClip { timeline_start: 170.0, timeline_end: 240.0, source_in: 0.0, speed: 0.0, src_w: 0, src_h: 0, src: None },
        ];
        let (detected, used_ollama) = analyze_vod_chapters(vec![clips], 240.0, Vec::new(), 10.0, 45.0, String::new(), None).unwrap();
        assert!(!used_ollama);
        assert!(!detected.is_empty());

        // 3. Applying chapter boundaries on clip edges, gaps, and interior only splits strictly inside clips.
        let mut s = Session::default();
        s.execute("file.newSequence", json!({"name": "MultiClipGaps", "video": 1, "audio": 1})).unwrap();
        let seq_id = s.state.active_sequence.unwrap();
        let item_a = ItemId(s.execute("file.newBarsAndTone", json!({"seconds": 60.0})).unwrap()["item"].as_u64().unwrap());
        let item_b = ItemId(s.execute("file.newColorMatte", json!({"seconds": 100.0, "color": [30, 120, 200]})).unwrap()["item"].as_u64().unwrap());

        s.edit("Place distinct clips with gap", |pr, _| {
            let rate = pr.sequence_mut(seq_id).unwrap().settings.frame_rate;
            let d1 = Tick::from_seconds_f64(60.0);
            let d2 = Tick::from_seconds_f64(100.0);
            let g2 = Tick::from_seconds_f64(80.0);
            let v1 = pr.make_track_item(item_a, TrackKind::Video, Tick::ZERO, TimeRange::new(Tick::ZERO, d1), rate).unwrap();
            let a1 = pr.make_track_item(item_a, TrackKind::Audio, Tick::ZERO, TimeRange::new(Tick::ZERO, d1), rate).unwrap();
            let v2 = pr.make_track_item(item_b, TrackKind::Video, g2, TimeRange::new(Tick::ZERO, d2), rate).unwrap();
            let a2 = pr.make_track_item(item_a, TrackKind::Audio, g2, TimeRange::new(Tick::ZERO, d2), rate).unwrap();
            let seq = pr.sequence_mut(seq_id).unwrap();
            seq.video_tracks[0].items.extend([v1, v2]);
            seq.audio_tracks[0].items.extend([a1, a2]);
            Ok(())
        })
        .unwrap();

        let chapters = vec![
            chapter(0, 60_000, "Clip 1"),
            chapter(60_000, 70_000, "Exact End Edge"),
            chapter(70_000, 80_000, "Inside Gap"),
            chapter(80_000, 130_000, "Exact Start Edge"),
            chapter(130_000, 180_000, "Interior Split"),
        ];
        let rep = apply_vod_chapters(&mut s, seq_id, true, chapters, false).unwrap();
        assert_eq!(rep.chapters_created, 5);
        assert_eq!(rep.splits_applied, 2, "only the 130s interior cut should split V1 and A1");

        let seq = s.project.sequence(seq_id).unwrap();
        assert_eq!(seq.video_tracks[0].items.len(), 3);
        assert_eq!(seq.audio_tracks[0].items.len(), 3);
        for tr in seq.all_tracks() {
            for it in &tr.items {
                assert!(it.duration > Tick::ZERO, "no empty clips after chapter splitting");
            }
        }
    }
}
