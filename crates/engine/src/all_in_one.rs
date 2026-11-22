//! Orpheus All-In-One Orchestrator & Procedural 2.5D Corkboard/Documentary Motion Engine
//! (`ai.allInOne`, `ai.allInOneStatus`).
//!
//! Combines:
//! 1. **Natural Prompt & Optional Local Ollama Planner** — optionally queries a local Ollama
//!    instance (`http://127.0.0.1:11434`) with automatic fallback to a 100% offline deterministic
//!    Rust planner that recognizes both 2.5D documentary/corkboard requests and chained AI tool
//!    instructions (`"clean audio"`, `"make 9:16"`, `"split screen"`, `"remove background"`,
//!    `"beat sync"`, `"viral clips"`, `"smart triggers"`, `"subtitles"`).
//! 2. **Local File Attachments & Optional Wikimedia Commons Image Fetch** — attaches user images,
//!    videos, logos, and audio files; optionally searches free public-domain images via
//!    `commons.wikimedia.org/w/api.php` when enabled and online.
//! 3. **Procedural 2.5D Corkboard & Dynamic Shadow Engine** — generates textured corkboard,
//!    Polaroid photo frames (embedding attached/fetched portraits or procedural documentary art),
//!    yellow sticky notes, torn paper title banners, line charts, red connecting strings, and
//!    soft radial drop shadows in pure Rust (`image` crate PNGs). Places multi-track layers with
//!    `0.6x` background parallax, `135% -> 100%` drop-in scale curves, coupled dynamic shadows
//!    (`shadow_offset = base_offset - camera_delta * depth_height`), and `AutoBezier` camera
//!    panning in a single undoable `Session::edit_group` transaction.

use std::io::Read;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use filmcraft_export::{Progress, Report};
use filmcraft_geom::Vec2;
use filmcraft_project::{Caption, CaptionFormat, CaptionTrack, ClipId, Interpolation, ItemId, ParamValue, Track, TrackId, TrackItem, TrackKind, find_effect};
use filmcraft_time::{FrameRate, Tick, TimeRange};
use image::codecs::png::PngEncoder;
use image::{ColorType, ImageEncoder, Rgba, RgbaImage};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, NamedKeys, always, bad, bool_p, str_p, u64_p};
use crate::subtitles::{SubtitleLengthMode, build_tiktok_center_pop_cues_from_phrases, next_job_id, subtitle_track_style};
use crate::voiceover::{media_dir, unique_wav_path};
use crate::{EngineError, Job, Result, Session};

const MAX_ATTACHMENTS: usize = 16;
const MAX_ATTACHMENT_BYTES: u64 = 64 * 1024 * 1024;
const MAX_WIKIMEDIA_IMAGES: usize = 3;
const MAX_WIKIMEDIA_BYTES: usize = 8 * 1024 * 1024;

/// Visual theme & layout style for generated All-In-One scenes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SceneStyle {
    #[default]
    ModernTech,
    MinimalEditorial,
    CinematicTimeline,
    Corkboard,
}

/// Parsed execution plan produced by Ollama or the built-in offline deterministic planner.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AllInOnePlan {
    pub build_corkboard: bool,
    #[serde(default)]
    pub build_scene: bool,
    #[serde(default)]
    pub scene_style: SceneStyle,
    pub build_tiktok_split: bool,
    pub tiktok_article_highlight: bool,
    pub corkboard_topic: String,
    pub run_remove_pauses: bool,
    pub run_audio_enhance: bool,
    pub run_auto_reframe: bool,
    pub reframe_split_screen: bool,
    pub run_bg_removal: bool,
    pub run_beat_sync: bool,
    pub run_viral_clips: bool,
    pub run_smart_triggers: bool,
    pub run_subtitles: bool,
    pub run_vod_chapters: bool,
    #[serde(default)]
    pub run_transitions: bool,
    #[serde(default)]
    pub run_color_grade: bool,
    #[serde(default)]
    pub run_text_overlay: bool,
    pub used_ollama: bool,
}

/// Summary report of the most recent completed All-In-One run.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AllInOneReport {
    pub seq: Option<ItemId>,
    pub steps_applied: Vec<String>,
    pub generated_layers: usize,
    pub attachments_imported: usize,
    pub wikimedia_fetched: usize,
    pub used_ollama: bool,
}

struct GeneratedAsset {
    base_name: String,
    png_bytes: Arc<[u8]>,
}

struct PreparedAttachment {
    orig_path: String,
    bytes: Arc<[u8]>,
}

struct CorkboardElementSpec {
    asset_idx: usize,
    shadow_asset_idx: usize,
    name: String,
    base_pos: (f64, f64),
    depth_height: f64,
    tilt_deg: f64,
    start_sec: f64,
    dur_sec: f64,
}

struct PreparedAllInOneOutput {
    plan: AllInOnePlan,
    dir: String,
    attachments: Vec<PreparedAttachment>,
    corkboard_assets: Vec<GeneratedAsset>,
    corkboard_elements: Vec<CorkboardElementSpec>,
    tiktok_broll_assets: Vec<GeneratedAsset>,
    tiktok_highlight_bar: Option<GeneratedAsset>,
    tiktok_talking_head: Option<GeneratedAsset>,
    transcript_spans: Vec<(f64, f64, String)>,
    wikimedia_count: usize,
}

pub struct PendingAllInOneJob {
    pub job: u64,
    pub epoch: u64,
    pub prompt: String,
    output: Arc<Mutex<Option<PreparedAllInOneOutput>>>,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

fn check_cancelled(progress: Option<&Progress>) -> std::result::Result<(), String> {
    if progress.is_some_and(|p| p.cancel.load(Ordering::Relaxed)) { Err("Cancelled".into()) } else { Ok(()) }
}

/// 100% offline deterministic prompt planner mapping natural language prompts into scene/tool actions.
pub fn plan_from_prompt(prompt: &str, attachment_count: usize) -> AllInOnePlan {
    let low = prompt.to_ascii_lowercase();
    let has_corkboard_kw = ["corkboard", "detective", "evidence wall", "evidence board", "investigation", "polaroid", "sticky note", "2.5d", "dynamic shadow"]
        .iter()
        .any(|k| low.contains(k));
    let has_timeline_kw = ["timeline", "chronolog", "history", "roadmap", "milestone", "documentary", "map"].iter().any(|k| low.contains(k));
    let has_editorial_kw = ["minimal", "editorial", "typography", "kinetic", "lower third", "title card", "clean"].iter().any(|k| low.contains(k));
    let scene_style = if has_corkboard_kw {
        SceneStyle::Corkboard
    } else if has_timeline_kw {
        SceneStyle::CinematicTimeline
    } else if has_editorial_kw {
        SceneStyle::MinimalEditorial
    } else {
        SceneStyle::ModernTech
    };

    let build_tiktok_split = ["tiktok split", "talking head", "50/50", "news breakdown", "article highlight", "split b-roll"].iter().any(|k| low.contains(k));
    let tiktok_article_highlight = ["article highlight", "news breakdown", "headline", "article"].iter().any(|k| low.contains(k));

    let run_remove_pauses = ["auto-cut", "auto cut", "remove pause", "cut silence", "silence", "polish"].iter().any(|k| low.contains(k));
    let run_audio_enhance =
        ["clean audio", "audio enhance", "enhance audio", "denoise", "clean voice", "loudness", "studio voice", "polish"].iter().any(|k| low.contains(k));
    let reframe_split_screen = ["split screen", "split-screen", "facecam", "gaming"].iter().any(|k| low.contains(k));
    let run_auto_reframe =
        !build_tiktok_split && (reframe_split_screen || ["9:16", "vertical", "reframe", "reels", "shorts", "tiktok"].iter().any(|k| low.contains(k)));
    let run_bg_removal = ["remove background", "bg removal", "transparent", "cutout", "green screen"].iter().any(|k| low.contains(k));
    let run_beat_sync = ["beat sync", "beats", "cut to beat", "beat marker"].iter().any(|k| low.contains(k));
    let run_viral_clips = ["viral", "highlights", "short clips", "find clips"].iter().any(|k| low.contains(k));
    let run_smart_triggers = ["smart trigger", "zoom in", "punch in", "sfx", "callout"].iter().any(|k| low.contains(k));
    let run_subtitles = ["subtitle", "caption", "transcribe", "polish"].iter().any(|k| low.contains(k));
    let run_vod_chapters = ["split stream", "vod chapters", "stream segments", "separate stages"].iter().any(|k| low.contains(k));
    let run_transitions = ["transition", "cross dissolve", "dissolve", "fade between", "dip to black"].iter().any(|k| low.contains(k));
    let run_color_grade = ["color grad", "cinematic look", "lut", "warm look", "teal and orange", "contrast", "saturat", "black and white", "monochrome"]
        .iter()
        .any(|k| low.contains(k));
    let run_text_overlay = ["title overlay", "lower third", "text overlay", "intro text"].iter().any(|k| low.contains(k));

    let any_tool = build_tiktok_split
        || run_remove_pauses
        || run_audio_enhance
        || run_auto_reframe
        || run_bg_removal
        || run_beat_sync
        || run_viral_clips
        || run_smart_triggers
        || run_subtitles
        || run_vod_chapters
        || run_transitions
        || run_color_grade
        || run_text_overlay;
    let build_corkboard = has_corkboard_kw;
    let build_scene = build_corkboard || (!any_tool && (attachment_count > 0 || !prompt.trim().is_empty()));

    let topic = prompt.trim().chars().take(80).collect::<String>();
    let corkboard_topic = if topic.is_empty() { "Orpheus Scene".to_string() } else { topic };

    AllInOnePlan {
        build_corkboard,
        build_scene,
        scene_style,
        build_tiktok_split,
        tiktok_article_highlight,
        corkboard_topic,
        run_remove_pauses,
        run_audio_enhance,
        run_auto_reframe,
        reframe_split_screen,
        run_bg_removal,
        run_beat_sync,
        run_viral_clips,
        run_smart_triggers,
        run_subtitles,
        run_vod_chapters,
        run_transitions,
        run_color_grade,
        run_text_overlay,
        used_ollama: false,
    }
}

fn try_plan_with_ollama(endpoint: &str, prompt: &str, base: &mut AllInOnePlan) -> bool {
    let trimmed = endpoint.trim().trim_end_matches('/');
    if trimmed.is_empty() || !(trimmed.starts_with("http://127.0.0.1") || trimmed.starts_with("http://localhost") || trimmed.starts_with("http://[::1]")) {
        return false;
    }
    let url = format!("{trimmed}/api/generate");
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .https_only(false)
        .timeout_connect(Some(Duration::from_secs(2)))
        .timeout_recv_response(Some(Duration::from_secs(6)))
        .user_agent("Orpheus/1.0")
        .build()
        .new_agent();

    let sys_prompt = format!(
        "Classify this video edit request as compact JSON with boolean keys \
        build_corkboard, build_tiktok_split, tiktok_article_highlight, run_audio_enhance, run_auto_reframe, reframe_split_screen, \
        run_bg_removal, run_beat_sync, run_viral_clips, run_smart_triggers, run_subtitles, \
        and string key topic. Prompt: {prompt}"
    );
    let body = json!({ "model": "llama3.2", "prompt": sys_prompt, "stream": false, "format": "json" });
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
    if let Some(b) = parsed.get("build_corkboard").and_then(Value::as_bool) {
        base.build_corkboard |= b;
    }
    if let Some(b) = parsed.get("build_tiktok_split").and_then(Value::as_bool) {
        base.build_tiktok_split |= b;
    }
    if let Some(b) = parsed.get("tiktok_article_highlight").and_then(Value::as_bool) {
        base.tiktok_article_highlight |= b;
    }
    if let Some(b) = parsed.get("run_audio_enhance").and_then(Value::as_bool) {
        base.run_audio_enhance |= b;
    }
    if let Some(b) = parsed.get("run_auto_reframe").and_then(Value::as_bool) {
        base.run_auto_reframe |= b;
    }
    if let Some(b) = parsed.get("reframe_split_screen").and_then(Value::as_bool) {
        base.reframe_split_screen |= b;
    }
    if let Some(b) = parsed.get("run_bg_removal").and_then(Value::as_bool) {
        base.run_bg_removal |= b;
    }
    if let Some(b) = parsed.get("run_beat_sync").and_then(Value::as_bool) {
        base.run_beat_sync |= b;
    }
    if let Some(b) = parsed.get("run_viral_clips").and_then(Value::as_bool) {
        base.run_viral_clips |= b;
    }
    if let Some(b) = parsed.get("run_smart_triggers").and_then(Value::as_bool) {
        base.run_smart_triggers |= b;
    }
    if let Some(b) = parsed.get("run_subtitles").and_then(Value::as_bool) {
        base.run_subtitles |= b;
    }
    if let Some(t) = parsed.get("topic").and_then(Value::as_str).filter(|s| !s.trim().is_empty()) {
        base.corkboard_topic = t.trim().chars().take(60).collect();
    }
    base.used_ollama = true;
    true
}

const PROMPT_STOP_WORDS: &[&str] = &[
    "create",
    "build",
    "make",
    "generate",
    "please",
    "a",
    "an",
    "the",
    "of",
    "on",
    "in",
    "for",
    "to",
    "and",
    "or",
    "with",
    "about",
    "showing",
    "2.5d",
    "25d",
    "5d",
    "3d",
    "corkboard",
    "documentary",
    "dynamic",
    "shadow",
    "shadows",
    "polaroid",
    "sticky",
    "note",
    "notes",
    "parallax",
    "visual",
    "story",
    "evidence",
    "scene",
    "video",
    "clip",
];

fn extract_subject_keywords(topic: &str, max_words: usize) -> Vec<String> {
    let mut out = Vec::new();
    for raw in topic.split(|c: char| !c.is_alphanumeric() && c != '-' && c != '.') {
        let trimmed = raw.trim_matches(['-', '.']);
        if trimmed.len() < 2 {
            continue;
        }
        let low = trimmed.to_ascii_lowercase();
        if PROMPT_STOP_WORDS.contains(&low.as_str()) {
            continue;
        }
        out.push(trimmed.to_string());
        if out.len() >= max_words {
            break;
        }
    }
    out
}

fn extract_clean_headline(topic: &str) -> String {
    let words = extract_subject_keywords(topic, 6);
    if words.is_empty() {
        let fallback: String = topic
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == ' ' || *c == '-')
            .collect::<String>()
            .split_whitespace()
            .take(5)
            .collect::<Vec<_>>()
            .join(" ")
            .to_ascii_uppercase();
        if fallback.is_empty() { "CLASSIFIED ARCHIVE DOSSIER".to_string() } else { fallback }
    } else {
        words.join(" ").to_ascii_uppercase()
    }
}

fn fetch_wikimedia_images(topic: &str, prog: Option<&Progress>) -> Vec<RgbaImage> {
    let subject_words = extract_subject_keywords(topic, 3);
    if subject_words.is_empty() {
        return Vec::new();
    }
    let query = subject_words.join(" ");
    let encoded: String = query.bytes().map(|b| if b.is_ascii_alphanumeric() { (b as char).to_string() } else { format!("%{b:02X}") }).collect();

    let tls = ureq::tls::TlsConfig::builder().provider(ureq::tls::TlsProvider::NativeTls).build();
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .tls_config(tls)
        .https_only(true)
        .timeout_connect(Some(Duration::from_secs(4)))
        .timeout_recv_response(Some(Duration::from_secs(6)))
        .user_agent("Orpheus/1.0.1 (https://github.com/hertzwaG/Orpheus)")
        .build()
        .new_agent();

    let api_url = format!(
        "https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrnamespace=6&gsrsearch={encoded}&gsrlimit=4&prop=imageinfo&iiprop=url|mime&iiurlwidth=640&format=json"
    );
    let Ok(resp) = agent.get(&api_url).call() else {
        return Vec::new();
    };
    let mut body_str = String::new();
    if resp.into_body().into_reader().take(262_144).read_to_string(&mut body_str).is_err() {
        return Vec::new();
    }
    let Ok(json_val) = serde_json::from_str::<Value>(&body_str) else {
        return Vec::new();
    };
    let Some(pages) = json_val.pointer("/query/pages").and_then(Value::as_object) else {
        return Vec::new();
    };

    let mut out = Vec::new();
    for page in pages.values() {
        if check_cancelled(prog).is_err() || out.len() >= MAX_WIKIMEDIA_IMAGES {
            break;
        }
        let Some(info) = page.get("imageinfo").and_then(Value::as_array).and_then(|a| a.first()) else { continue };
        let mime = info.get("mime").and_then(Value::as_str).unwrap_or("");
        if !matches!(mime, "image/jpeg" | "image/png" | "image/webp") {
            continue;
        }
        let Some(img_url) = info.get("thumburl").or_else(|| info.get("url")).and_then(Value::as_str) else { continue };
        let title_low = page.get("title").and_then(Value::as_str).unwrap_or("").to_ascii_lowercase();
        if ["icon", "logo", "flag", "coat_of_arms", "coat of arms", "signature", "emblem", "symbol"].iter().any(|bad| title_low.contains(bad)) {
            continue;
        }
        if !img_url.starts_with("https://") {
            continue;
        }
        let Ok(img_resp) = agent.get(img_url).call() else { continue };
        let mut raw = Vec::new();
        if img_resp.into_body().into_reader().take(MAX_WIKIMEDIA_BYTES as u64).read_to_end(&mut raw).is_err() {
            continue;
        }
        if let Ok(dyn_img) = image::load_from_memory(&raw) {
            out.push(dyn_img.to_rgba8());
        }
    }
    out
}

fn encode_png(img: &RgbaImage) -> std::result::Result<Arc<[u8]>, String> {
    let mut buf = Vec::new();
    PngEncoder::new(&mut buf).write_image(img.as_raw(), img.width(), img.height(), ColorType::Rgba8.into()).map_err(|e| format!("PNG encode failed: {e}"))?;
    Ok(Arc::from(buf))
}

#[inline]
fn hash2d(x: u32, y: u32) -> f32 {
    let mut h = x.wrapping_mul(374_761_393).wrapping_add(y.wrapping_mul(668_265_263));
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    ((h ^ (h >> 16)) & 0xFFFF) as f32 / 65535.0
}

fn generate_corkboard_backdrop(w: u32, h: u32) -> RgbaImage {
    let mut img = RgbaImage::new(w, h);
    let (cx, cy) = (w as f32 * 0.5, h as f32 * 0.5);
    let max_d = (cx * cx + cy * cy).sqrt().max(1.0);
    for (x, y, px) in img.enumerate_pixels_mut() {
        let g1 = hash2d(x / 3, y / 3);
        let g2 = hash2d(x, y);
        let grain = g1 * 0.65 + g2 * 0.35;
        let dx = x as f32 - cx;
        let dy = y as f32 - cy;
        let dist = (dx * dx + dy * dy).sqrt() / max_d;
        // Radial warm studio spotlight + dark vignette around corkboard edges.
        let spot = (1.12 - dist * 0.62).clamp(0.45, 1.08);
        let r = ((166.0 + grain * 36.0) * spot).clamp(0.0, 255.0) as u8;
        let g = ((124.0 + grain * 28.0) * spot).clamp(0.0, 255.0) as u8;
        let b = ((82.0 + grain * 18.0) * spot).clamp(0.0, 255.0) as u8;
        *px = Rgba([r, g, b, 255]);
    }
    img
}

fn glyph_5x7(ch: char) -> [u8; 7] {
    match ch.to_ascii_uppercase() {
        'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'C' => [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
        'D' => [0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100],
        'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' => [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110],
        'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'J' => [0b00111, 0b00010, 0b00010, 0b00010, 0b10010, 0b10010, 0b01100],
        'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        'N' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'Q' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
        'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
        'X' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        'Y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        'Z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111],
        '3' => [0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b01110, 0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110],
        ':' => [0b00000, 0b00100, 0b00100, 0b00000, 0b00100, 0b00100, 0b00000],
        '-' => [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        '/' => [0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000],
        '.' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00110, 0b00110],
        '%' => [0b11001, 0b11010, 0b00010, 0b00100, 0b01000, 0b01011, 0b10011],
        '+' => [0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000],
        '#' => [0b01010, 0b11111, 0b01010, 0b01010, 0b11111, 0b01010, 0b00000],
        _ => [0; 7],
    }
}

fn draw_bitmap_text(img: &mut RgbaImage, text: &str, x0: u32, y0: u32, scale: u32, col: Rgba<u8>) {
    let sc = scale.max(1);
    let advance = 6 * sc;
    let mut cursor_x = x0;
    for ch in text.chars() {
        if cursor_x + 5 * sc >= img.width() {
            break;
        }
        let rows = glyph_5x7(ch);
        for (ry, &bits) in rows.iter().enumerate() {
            for rx in 0..5u32 {
                if (bits >> (4 - rx)) & 1 == 1 {
                    for dy in 0..sc {
                        for dx in 0..sc {
                            let px = cursor_x + rx * sc + dx;
                            let py = y0 + (ry as u32) * sc + dy;
                            if px < img.width() && py < img.height() {
                                img.put_pixel(px, py, col);
                            }
                        }
                    }
                }
            }
        }
        cursor_x += advance;
    }
}

fn draw_bitmap_text_centered(img: &mut RgbaImage, text: &str, cy: u32, scale: u32, col: Rgba<u8>) {
    let sc = scale.max(1);
    let max_chars = ((img.width().saturating_sub(24)) / (6 * sc)).max(1) as usize;
    let clipped: String = text.chars().take(max_chars).collect();
    let char_count = clipped.chars().count() as u32;
    let text_w = char_count.saturating_mul(6 * sc).saturating_sub(sc);
    let x0 = img.width().saturating_sub(text_w) / 2;
    let y0 = cy.saturating_sub((7 * sc) / 2);
    draw_bitmap_text(img, &clipped, x0, y0, sc, col);
}

fn paint_push_pin(img: &mut RgbaImage, pin_cx: f32, pin_cy: f32, pin_r: f32) {
    let w = img.width();
    let h = img.height();
    for y in 0..h {
        for x in 0..w {
            let d = ((x as f32 - pin_cx).powi(2) + (y as f32 - pin_cy).powi(2)).sqrt();
            if d <= pin_r {
                let shade = 1.0 - (d / pin_r) * 0.35;
                img.put_pixel(x, y, Rgba([(218.0 * shade) as u8, (34.0 * shade) as u8, (38.0 * shade) as u8, 255]));
            }
        }
    }
}

fn paint_newspaper_clipping_inset(img: &mut RgbaImage, margin: u32, inner_w: u32, inner_h: u32, headline: &str) {
    for dy in 0..inner_h {
        for dx in 0..inner_w {
            let g = (hash2d(dx, dy) - 0.5) * 10.0;
            let col = if dy < 44 {
                Rgba([30, 34, 42, 255])
            } else if dy < 52 {
                Rgba([192, 42, 42, 255])
            } else {
                Rgba([(236.0 + g) as u8, (229.0 + g) as u8, (214.0 + g) as u8, 255])
            };
            img.put_pixel(margin + dx, margin + dy, col);
        }
    }
    let header_tag = if headline.is_empty() { "FIELD REPORT" } else { headline };
    let header_clip: String = header_tag.chars().take(22).collect();
    draw_bitmap_text(img, &header_clip, margin + 16, margin + 15, 2, Rgba([244, 240, 230, 255]));
    let short_hl: String = headline.chars().take(18).collect();
    draw_bitmap_text(img, &short_hl, margin + 16, margin + 68, 2, Rgba([24, 26, 32, 255]));

    let col_w = (inner_w.saturating_sub(44)) / 2;
    for row in 0..11u32 {
        let y0 = margin + 104 + row * 22;
        if y0 + 12 >= margin + inner_h {
            break;
        }
        let highlight = row == 2 || row == 3;
        let line_h = if highlight { 8u32 } else { 6u32 };
        let line_col = if highlight { Rgba([245, 212, 66, 255]) } else { Rgba([62, 64, 70, 255]) };
        for dx in 0..col_w {
            for dy in 0..line_h {
                img.put_pixel(margin + 16 + dx, y0 + dy, line_col);
            }
        }
        let rx0 = margin + 28 + col_w;
        let (bar_w, bar_h, bar_col) = if row < 6 {
            let len = ((row * 37 + 55) % col_w.max(1)).max(col_w / 3).min(col_w);
            let col = if row == 4 { Rgba([196, 44, 48, 255]) } else { Rgba([52, 78, 112, 255]) };
            (len, 12u32, col)
        } else {
            (col_w * 4 / 5, 6u32, Rgba([82, 84, 90, 255]))
        };
        for dx in 0..bar_w {
            for dy in 0..bar_h {
                img.put_pixel(rx0 + dx, y0 + dy, bar_col);
            }
        }
    }
}

fn paint_blueprint_dossier_inset(img: &mut RgbaImage, margin: u32, inner_w: u32, inner_h: u32) {
    let w = img.width();
    let h = img.height();
    for dy in 0..inner_h {
        for dx in 0..inner_w {
            let grid = dx % 28 == 0 || dy % 28 == 0;
            let c = if dy < 40 {
                Rgba([16, 22, 34, 255])
            } else if grid {
                Rgba([34, 52, 78, 255])
            } else {
                Rgba([22, 34, 52, 255])
            };
            img.put_pixel(margin + dx, margin + dy, c);
        }
    }
    draw_bitmap_text(img, "SYS.DOSSIER // NODE MAP", margin + 14, margin + 13, 2, Rgba([102, 217, 239, 255]));
    let nodes = [(0.22_f32, 0.34_f32), (0.52, 0.26), (0.78, 0.38), (0.28, 0.72), (0.58, 0.66), (0.82, 0.76)];
    let edges = [(0usize, 1usize), (1, 2), (0, 3), (1, 4), (2, 5), (3, 4), (4, 5)];
    for &(a, b) in &edges {
        let (x0, y0) = (margin as f32 + nodes[a].0 * inner_w as f32, margin as f32 + nodes[a].1 * inner_h as f32);
        let (x1, y1) = (margin as f32 + nodes[b].0 * inner_w as f32, margin as f32 + nodes[b].1 * inner_h as f32);
        for s in 0..=160 {
            let t = s as f32 / 160.0;
            let px = (x0 + (x1 - x0) * t) as u32;
            let py = (y0 + (y1 - y0) * t) as u32;
            if px + 1 < w && py + 1 < h {
                img.put_pixel(px, py, Rgba([72, 168, 212, 255]));
                img.put_pixel(px + 1, py + 1, Rgba([72, 168, 212, 255]));
            }
        }
    }
    for (idx, &(nx, ny)) in nodes.iter().enumerate() {
        let cx = margin as f32 + nx * inner_w as f32;
        let cy = margin as f32 + ny * inner_h as f32;
        let col = if idx == 1 || idx == 4 { Rgba([245, 194, 66, 255]) } else { Rgba([88, 214, 242, 255]) };
        for dy in -12i32..=12 {
            for dx in -12i32..=12 {
                let d = ((dx * dx + dy * dy) as f32).sqrt();
                let px = (cx as i32 + dx).max(0) as u32;
                let py = (cy as i32 + dy).max(0) as u32;
                if px < w && py < h && (d <= 6.5 || (9.5..=11.5).contains(&d)) {
                    img.put_pixel(px, py, col);
                }
            }
        }
    }
}

fn generate_polaroid_frame(w: u32, h: u32, photo: Option<&RgbaImage>, variant: usize, headline: &str) -> RgbaImage {
    let mut img = RgbaImage::from_pixel(w, h, Rgba([244, 240, 230, 255]));
    let margin = (w / 14).max(12);
    let bottom_pad = (h / 5).max(44);
    let inner_w = w.saturating_sub(margin * 2).max(16);
    let inner_h = h.saturating_sub(margin + bottom_pad).max(16);

    for y in 0..h {
        for x in 0..w {
            if x < 2 || y < 2 || x + 2 >= w || y + 2 >= h {
                img.put_pixel(x, y, Rgba([186, 178, 162, 255]));
            } else if x < margin || x >= margin + inner_w || y < margin || y >= margin + inner_h {
                let n = (hash2d(x, y) - 0.5) * 8.0;
                img.put_pixel(x, y, Rgba([(244.0 + n) as u8, (240.0 + n) as u8, (230.0 + n) as u8, 255]));
            }
        }
    }

    if let Some(src) = photo {
        let resized = image::imageops::resize(src, inner_w, inner_h, image::imageops::FilterType::Triangle);
        for dy in 0..inner_h {
            for dx in 0..inner_w {
                let mut px = *resized.get_pixel(dx, dy);
                px[3] = 255;
                img.put_pixel(margin + dx, margin + dy, px);
            }
        }
    } else if variant == 0 {
        paint_newspaper_clipping_inset(&mut img, margin, inner_w, inner_h, headline);
    } else {
        paint_blueprint_dossier_inset(&mut img, margin, inner_w, inner_h);
    }

    // Crisp uppercase caption on the Polaroid bottom border derived from prompt keywords.
    let kws = extract_subject_keywords(headline, 2);
    let caption = if variant == 0 {
        format!("{} // OVERVIEW", kws.first().map_or("PRIMARY", String::as_str).to_ascii_uppercase())
    } else {
        format!("{} // DETAILS", kws.get(1).or(kws.first()).map_or("CONTEXT", String::as_str).to_ascii_uppercase())
    };
    draw_bitmap_text_centered(&mut img, &caption, h.saturating_sub(bottom_pad / 2), 2, Rgba([42, 44, 52, 255]));

    // Red push-pin at top center.
    paint_push_pin(&mut img, w as f32 * 0.5, margin as f32 * 0.52, (w as f32 * 0.032).max(6.0));
    img
}

fn extract_stat_or_phrase(topic: &str, transcript_spans: &[(f64, f64, String)]) -> (String, String, String) {
    let combined = if transcript_spans.is_empty() {
        topic.to_string()
    } else {
        let joined: String = transcript_spans.iter().take(4).map(|(_, _, t)| t.as_str()).collect::<Vec<_>>().join(" ");
        format!("{topic} {joined}")
    };
    let words = extract_subject_keywords(&combined, 6);
    let stat_token = combined
        .split_whitespace()
        .find(|w| w.chars().any(|c| c.is_ascii_digit()))
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric() && c != '%' && c != '+' && c != '-').to_ascii_uppercase())
        .filter(|w| !w.is_empty())
        .unwrap_or_else(|| words.first().map_or_else(|| "HIGHLIGHT".to_string(), |w| w.to_ascii_uppercase()));

    let sub_kicker =
        if words.len() >= 2 { format!("{} // {}", words[0].to_ascii_uppercase(), words[1].to_ascii_uppercase()) } else { extract_clean_headline(topic) };
    let note_phrase = transcript_spans
        .first()
        .map(|(_, _, s)| s.trim().to_ascii_uppercase())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| words.into_iter().skip(2).take(3).collect::<Vec<_>>().join(" ").to_ascii_uppercase());
    let note_phrase = if note_phrase.is_empty() { extract_clean_headline(topic) } else { note_phrase };
    (stat_token, sub_kicker, note_phrase)
}

fn generate_sticky_note(w: u32, h: u32, headline: &str, stat: &str, note_phrase: &str) -> RgbaImage {
    let mut img = RgbaImage::new(w, h);
    for y in 0..h {
        let v = y as f32 / h as f32;
        for x in 0..w {
            let u = x as f32 / w as f32;
            let fold = u > 0.85 && v > 0.85 && (u + v) > 1.78;
            if fold {
                img.put_pixel(x, y, Rgba([0, 0, 0, 0]));
                continue;
            }
            let top_strip = v < 0.14;
            let ruled = y > h / 5 && y % (h / 6).max(16) < 2 && u > 0.08 && u < 0.90;
            let c = if ruled {
                Rgba([218, 198, 92, 255])
            } else if top_strip {
                Rgba([242, 214, 82, 255])
            } else {
                Rgba([252, 232, 118, 255])
            };
            img.put_pixel(x, y, c);
        }
    }
    let kw: Vec<&str> = headline.split_whitespace().collect();
    let line1 = kw.first().copied().unwrap_or("TOPIC");
    let line2 = kw.get(1).copied().unwrap_or("SUMMARY");
    let phrase_clip: String = note_phrase.chars().take(18).collect();
    draw_bitmap_text(&mut img, stat, 28, 64, 3, Rgba([184, 34, 34, 255]));
    draw_bitmap_text(&mut img, line1, 28, 118, 3, Rgba([34, 34, 38, 255]));
    draw_bitmap_text(&mut img, line2, 28, 166, 3, Rgba([34, 34, 38, 255]));
    draw_bitmap_text(&mut img, &phrase_clip, 28, 224, 2, Rgba([52, 54, 62, 255]));
    paint_push_pin(&mut img, w as f32 * 0.5, 22.0, 9.5);
    img
}

fn generate_torn_banner(w: u32, h: u32, headline: &str, sub_kicker: &str) -> RgbaImage {
    let mut img = RgbaImage::new(w, h);
    for x in 0..w {
        let wave_top = ((x as f32 * 0.09).sin() * 4.0 + hash2d(x / 4, 1) * 5.0) as u32;
        let wave_bot = ((x as f32 * 0.11).cos() * 4.0 + hash2d(x / 4, 2) * 5.0) as u32;
        for y in 0..h {
            if y < wave_top + 4 || y + wave_bot + 4 >= h {
                img.put_pixel(x, y, Rgba([0, 0, 0, 0]));
            } else {
                let n = (hash2d(x, y) - 0.5) * 9.0;
                let border = (y - wave_top < 7) || (h - wave_bot - y < 7);
                let rule = (y - wave_top == 14) || (h - wave_bot - y == 14);
                let col = if border {
                    Rgba([208, 198, 178, 255])
                } else if rule {
                    Rgba([182, 42, 42, 255])
                } else {
                    Rgba([(242.0 + n) as u8, (236.0 + n) as u8, (220.0 + n) as u8, 255])
                };
                img.put_pixel(x, y, col);
            }
        }
    }
    draw_bitmap_text_centered(&mut img, sub_kicker, 34, 2, Rgba([186, 38, 38, 255]));
    draw_bitmap_text_centered(&mut img, headline, 82, 4, Rgba([26, 28, 34, 255]));
    paint_push_pin(&mut img, 34.0, h as f32 * 0.5, 9.0);
    paint_push_pin(&mut img, w as f32 - 34.0, h as f32 * 0.5, 9.0);
    img
}

fn generate_chart_card(w: u32, h: u32, headline: &str, stat: &str) -> RgbaImage {
    let mut img = RgbaImage::from_pixel(w, h, Rgba([248, 245, 236, 255]));
    for y in 0..h {
        for x in 0..w {
            if x < 3 || y < 3 || x + 3 >= w || y + 3 >= h {
                img.put_pixel(x, y, Rgba([180, 174, 160, 255]));
            } else if y < 44 {
                img.put_pixel(x, y, Rgba([32, 38, 48, 255]));
            } else if x % 32 == 0 || y % 32 == 0 {
                img.put_pixel(x, y, Rgba([226, 222, 212, 255]));
            }
        }
    }
    let short_hl: String = headline.chars().take(18).collect();
    draw_bitmap_text(&mut img, &short_hl, 18, 15, 2, Rgba([244, 240, 230, 255]));
    draw_bitmap_text(&mut img, stat, w.saturating_sub(175), 62, 2, Rgba([196, 44, 48, 255]));
    // Bold crimson ascending documentary trend curve.
    let pts = [(0.10, 0.82), (0.30, 0.66), (0.52, 0.70), (0.72, 0.38), (0.90, 0.22)];
    for seg in pts.windows(2) {
        let (x0, y0) = (seg[0].0 * w as f32, seg[0].1 * h as f32);
        let (x1, y1) = (seg[1].0 * w as f32, seg[1].1 * h as f32);
        let steps = (w as usize).max(1);
        for s in 0..=steps {
            let t = s as f32 / steps as f32;
            let px = (x0 + (x1 - x0) * t) as i32;
            let py = (y0 + (y1 - y0) * t) as i32;
            for dy in -3..=3 {
                for dx in -3..=3 {
                    let nx = px + dx;
                    let ny = py + dy;
                    if nx >= 0 && ny >= 0 && (nx as u32) < w && (ny as u32) < h {
                        img.put_pixel(nx as u32, ny as u32, Rgba([196, 44, 48, 255]));
                    }
                }
            }
        }
    }
    paint_push_pin(&mut img, w as f32 * 0.5, 18.0, 9.0);
    img
}

fn generate_red_string(w: u32, h: u32) -> RgbaImage {
    let mut img = RgbaImage::new(w, h);
    let cy = h as f32 * 0.5;
    let x_start = 18u32;
    let x_end = w.saturating_sub(18).max(x_start + 1);
    for x in x_start..=x_end {
        let u = (x - x_start) as f32 / (x_end - x_start).max(1) as f32;
        let curve_y = cy + (u * std::f32::consts::PI).sin() * (h as f32 * 0.22);
        for y in 0..h {
            let dist = (y as f32 - curve_y).abs();
            if dist <= 3.5 {
                let a = ((1.0 - (dist / 3.8).powi(2)) * 245.0).clamp(0.0, 255.0) as u8;
                img.put_pixel(x, y, Rgba([212, 34, 38, a]));
            }
        }
    }
    paint_push_pin(&mut img, x_start as f32, cy, 8.5);
    paint_push_pin(&mut img, x_end as f32, cy, 8.5);
    img
}

fn generate_soft_shadow(w: u32, h: u32) -> RgbaImage {
    let mut img = RgbaImage::new(w, h);
    let pad = 28.0_f32;
    let x0 = pad;
    let y0 = pad;
    let x1 = (w as f32 - pad).max(x0 + 1.0);
    let y1 = (h as f32 - pad).max(y0 + 1.0);
    for y in 0..h {
        for x in 0..w {
            let dx = (x0 - x as f32).max(0.0).max(x as f32 - x1);
            let dy = (y0 - y as f32).max(0.0).max(y as f32 - y1);
            let dist = (dx * dx + dy * dy).sqrt();
            let alpha = if dist <= 0.0 {
                135.0
            } else if dist < pad {
                135.0 * (1.0 - dist / pad).powi(2)
            } else {
                0.0
            };
            img.put_pixel(x, y, Rgba([10, 10, 14, alpha.round().clamp(0.0, 255.0) as u8]));
        }
    }
    img
}

/// Camera displacement keyframes `(time_sec, cam_delta_x, cam_delta_y)` driving 2.5D parallax &
/// opposing dynamic shadow shifts.
fn corkboard_camera_keyframes() -> [(f64, f64, f64); 4] {
    [(0.0, 0.0, 0.0), (2.6, -150.0, -35.0), (5.6, 155.0, 25.0), (8.6, 0.0, -45.0)]
}

fn prepare_corkboard_assets(
    topic: &str,
    attachments: &[PreparedAttachment],
    wikimedia_imgs: &[RgbaImage],
    transcript_spans: &[(f64, f64, String)],
) -> std::result::Result<(Vec<GeneratedAsset>, Vec<CorkboardElementSpec>), String> {
    let headline = extract_clean_headline(topic);
    let (stat, sub_kicker, note_phrase) = extract_stat_or_phrase(topic, transcript_spans);
    let mut decoded_photos: Vec<RgbaImage> = Vec::new();
    for att in attachments {
        if decoded_photos.len() >= 2 {
            break;
        }
        if let Ok(dyn_img) = image::load_from_memory(&att.bytes) {
            decoded_photos.push(dyn_img.to_rgba8());
        }
    }
    for wimg in wikimedia_imgs {
        if decoded_photos.len() >= 2 {
            break;
        }
        decoded_photos.push(wimg.clone());
    }

    let assets = vec![
        GeneratedAsset { base_name: "Corkboard Backdrop".into(), png_bytes: encode_png(&generate_corkboard_backdrop(2400, 1350))? },
        GeneratedAsset { base_name: "Card Shadow".into(), png_bytes: encode_png(&generate_soft_shadow(560, 640))? },
        GeneratedAsset {
            base_name: "Evidence Polaroid 1".into(),
            png_bytes: encode_png(&generate_polaroid_frame(480, 540, decoded_photos.first(), 0, &headline))?,
        },
        GeneratedAsset {
            base_name: "Evidence Polaroid 2".into(),
            png_bytes: encode_png(&generate_polaroid_frame(460, 520, decoded_photos.get(1), 1, &headline))?,
        },
        GeneratedAsset { base_name: "Sticky Note".into(), png_bytes: encode_png(&generate_sticky_note(340, 340, &headline, &stat, &note_phrase))? },
        GeneratedAsset { base_name: "Chart Card".into(), png_bytes: encode_png(&generate_chart_card(520, 340, &headline, &stat))? },
        GeneratedAsset { base_name: "Title Banner".into(), png_bytes: encode_png(&generate_torn_banner(880, 140, &headline, &sub_kicker))? },
        GeneratedAsset { base_name: "Red String".into(), png_bytes: encode_png(&generate_red_string(740, 130))? },
    ];

    let total_dur = transcript_spans.last().map(|(_, e, _)| *e).filter(|&e| e > 4.0).unwrap_or(9.0);
    let t1 = transcript_spans.first().map(|(s, _, _)| (*s).max(0.15)).unwrap_or(0.25);
    let t2 = transcript_spans.get(1).map(|(s, _, _)| *s).unwrap_or(1.20);
    let t3 = transcript_spans.get(2).map(|(s, _, _)| *s).unwrap_or(2.30);
    let t4 = transcript_spans.get(3).map(|(s, _, _)| *s).unwrap_or(3.10);

    // Distinct non-overlapping 4-quadrant + top-banner layout grid paced by spoken transitions.
    let elements = vec![
        CorkboardElementSpec {
            asset_idx: 2,
            shadow_asset_idx: 1,
            name: "Evidence Polaroid 1".into(),
            base_pos: (420.0, 455.0),
            depth_height: 1.4,
            tilt_deg: -2.8,
            start_sec: t1,
            dur_sec: (total_dur - t1).max(2.0),
        },
        CorkboardElementSpec {
            asset_idx: 5,
            shadow_asset_idx: 1,
            name: "Trend Chart Card".into(),
            base_pos: (1470.0, 420.0),
            depth_height: 1.2,
            tilt_deg: 2.2,
            start_sec: t2,
            dur_sec: (total_dur - t2).max(2.0),
        },
        CorkboardElementSpec {
            asset_idx: 3,
            shadow_asset_idx: 1,
            name: "Evidence Polaroid 2".into(),
            base_pos: (1380.0, 815.0),
            depth_height: 1.6,
            tilt_deg: 2.5,
            start_sec: t3,
            dur_sec: (total_dur - t3).max(2.0),
        },
        CorkboardElementSpec {
            asset_idx: 4,
            shadow_asset_idx: 1,
            name: "Field Sticky Note".into(),
            base_pos: (520.0, 845.0),
            depth_height: 1.0,
            tilt_deg: -2.0,
            start_sec: t4,
            dur_sec: (total_dur - t4).max(2.0),
        },
        CorkboardElementSpec {
            asset_idx: 6,
            shadow_asset_idx: 1,
            name: "Documentary Title Strip".into(),
            base_pos: (960.0, 130.0),
            depth_height: 0.9,
            tilt_deg: -0.6,
            start_sec: 0.0,
            dur_sec: total_dur,
        },
        CorkboardElementSpec {
            asset_idx: 7,
            shadow_asset_idx: 1,
            name: "Red Evidence String".into(),
            base_pos: (945.0, 440.0),
            depth_height: 0.7,
            tilt_deg: -1.5,
            start_sec: t2,
            dur_sec: (total_dur - t2).max(2.0),
        },
    ];

    Ok((assets, elements))
}

fn extract_prompt_bullets(topic: &str) -> Vec<String> {
    let mut parts: Vec<String> = topic
        .split([',', ';', '.', '\n', ':'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.chars().take(28).collect::<String>().to_ascii_uppercase())
        .collect();
    if parts.is_empty() {
        parts.push(extract_clean_headline(topic));
    }
    if parts.len() == 1 {
        let words: Vec<&str> = topic.split_whitespace().collect();
        if words.len() >= 4 {
            let mid = words.len() / 2;
            parts = vec![
                words[..mid].join(" ").chars().take(26).collect::<String>().to_ascii_uppercase(),
                words[mid..].join(" ").chars().take(26).collect::<String>().to_ascii_uppercase(),
            ];
        }
    }
    parts
}

fn generate_styled_backdrop(w: u32, h: u32, style: SceneStyle) -> RgbaImage {
    let mut img = RgbaImage::new(w, h);
    for y in 0..h {
        let v = y as f32 / h.max(1) as f32;
        for x in 0..w {
            let u = x as f32 / w.max(1) as f32;
            let glow = (1.0 - ((u - 0.5).powi(2) + (v - 0.42).powi(2)).sqrt() * 1.35).max(0.0);
            let px = match style {
                SceneStyle::ModernTech => {
                    let dot = (x % 48 == 0 && y % 48 == 0) as u8 as f32 * 14.0;
                    Rgba([(11.0 + glow * 16.0 + dot) as u8, (15.0 + glow * 28.0 + dot) as u8, (24.0 + glow * 52.0 + dot) as u8, 255])
                }
                SceneStyle::MinimalEditorial => Rgba([(18.0 + glow * 12.0) as u8, (18.0 + glow * 12.0) as u8, (22.0 + glow * 15.0) as u8, 255]),
                SceneStyle::CinematicTimeline => {
                    let grid = (x % 80 == 0 || y % 80 == 0) as u8 as f32 * 8.0;
                    let axis = (y as i32 - (h as i32 * 56 / 100)).abs() <= 2 && x > 140 && x + 140 < w;
                    if axis {
                        Rgba([68, 132, 224, 255])
                    } else {
                        Rgba([(12.0 + glow * 14.0 + grid) as u8, (18.0 + glow * 22.0 + grid) as u8, (29.0 + glow * 38.0 + grid) as u8, 255])
                    }
                }
                SceneStyle::Corkboard => Rgba([18, 20, 26, 255]),
            };
            img.put_pixel(x, y, px);
        }
    }
    img
}

fn generate_tech_or_editorial_card(w: u32, h: u32, style: SceneStyle, title: &str, body_line: &str, badge: &str, accent: Rgba<u8>) -> RgbaImage {
    let mut img = RgbaImage::new(w, h);
    let (bg, border) = match style {
        SceneStyle::MinimalEditorial => (Rgba([24, 24, 28, 242]), Rgba([220, 220, 228, 80])),
        _ => (Rgba([17, 23, 36, 242]), Rgba([accent[0], accent[1], accent[2], 140])),
    };
    for y in 0..h {
        for x in 0..w {
            let is_border = x < 2 || y < 2 || x + 2 >= w || y + 2 >= h;
            let top_bar = y >= 2 && y < 7;
            img.put_pixel(
                x,
                y,
                if top_bar {
                    accent
                } else if is_border {
                    border
                } else {
                    bg
                },
            );
        }
    }
    if !badge.is_empty() {
        draw_bitmap_text(&mut img, badge, 24, 24, 2, accent);
    }
    let title_clip: String = title.chars().take(24).collect();
    draw_bitmap_text(&mut img, &title_clip, 24, 62, 3, Rgba([244, 246, 252, 255]));
    if !body_line.is_empty() {
        let body_clip: String = body_line.chars().take(30).collect();
        draw_bitmap_text(&mut img, &body_clip, 24, 122, 2, Rgba([168, 180, 202, 255]));
    }
    img
}

fn generate_styled_hero_banner(w: u32, h: u32, style: SceneStyle, headline: &str, kicker: &str) -> RgbaImage {
    let mut img = RgbaImage::new(w, h);
    let accent = match style {
        SceneStyle::MinimalEditorial => Rgba([238, 238, 244, 255]),
        SceneStyle::CinematicTimeline => Rgba([96, 165, 250, 255]),
        _ => Rgba([56, 189, 248, 255]),
    };
    for y in 0..h {
        for x in 0..w {
            let bar = y + 4 >= h && x > 40 && x + 40 < w;
            if bar {
                img.put_pixel(x, y, accent);
            } else {
                img.put_pixel(x, y, Rgba([14, 18, 28, 220]));
            }
        }
    }
    draw_bitmap_text_centered(&mut img, kicker, 28, 2, accent);
    draw_bitmap_text_centered(&mut img, headline, 80, 4, Rgba([248, 250, 252, 255]));
    img
}

fn prepare_styled_scene_assets(
    style: SceneStyle,
    topic: &str,
    attachments: &[PreparedAttachment],
    wikimedia_imgs: &[RgbaImage],
    transcript_spans: &[(f64, f64, String)],
) -> std::result::Result<(Vec<GeneratedAsset>, Vec<CorkboardElementSpec>), String> {
    let headline = extract_clean_headline(topic);
    let (stat, sub_kicker, note_phrase) = extract_stat_or_phrase(topic, transcript_spans);
    let bullets = extract_prompt_bullets(topic);
    let b1 = bullets.first().cloned().unwrap_or_else(|| headline.clone());
    let b2 = bullets.get(1).cloned().unwrap_or_else(|| note_phrase.clone());
    let b3 = bullets.get(2).cloned().unwrap_or_else(|| sub_kicker.clone());

    let mut photo_img: Option<RgbaImage> = None;
    for att in attachments {
        if let Ok(dyn_img) = image::load_from_memory(&att.bytes) {
            photo_img = Some(dyn_img.to_rgba8());
            break;
        }
    }
    if photo_img.is_none() {
        photo_img = wikimedia_imgs.first().cloned();
    }

    let accent_a = Rgba([56, 189, 248, 255]);
    let accent_b = Rgba([167, 139, 250, 255]);
    let accent_c = Rgba([52, 211, 153, 255]);

    let card1 = if let Some(src) = photo_img.as_ref() {
        image::imageops::resize(src, 620, 360, image::imageops::FilterType::Triangle)
    } else {
        generate_tech_or_editorial_card(620, 260, style, &b1, &sub_kicker, &stat, accent_a)
    };
    let card2 = generate_tech_or_editorial_card(620, 260, style, &b2, &b3, "02", accent_b);
    let card3 = generate_tech_or_editorial_card(840, 210, style, &headline, &b1, "03", accent_c);

    let bg_name = match style {
        SceneStyle::MinimalEditorial => "Minimal Editorial Backdrop",
        SceneStyle::CinematicTimeline => "Cinematic Timeline Backdrop",
        _ => "Modern Tech Backdrop",
    };
    let assets = vec![
        GeneratedAsset { base_name: bg_name.into(), png_bytes: encode_png(&generate_styled_backdrop(2400, 1350, style))? },
        GeneratedAsset { base_name: "Card Shadow".into(), png_bytes: encode_png(&generate_soft_shadow(660, 320))? },
        GeneratedAsset { base_name: "Hero Title".into(), png_bytes: encode_png(&generate_styled_hero_banner(1120, 150, style, &headline, &sub_kicker))? },
        GeneratedAsset { base_name: "Primary Card".into(), png_bytes: encode_png(&card1)? },
        GeneratedAsset { base_name: "Secondary Card".into(), png_bytes: encode_png(&card2)? },
        GeneratedAsset { base_name: "Summary Card".into(), png_bytes: encode_png(&card3)? },
    ];

    let total_dur = transcript_spans.last().map(|(_, e, _)| *e).filter(|&e| e > 4.0).unwrap_or(9.0);
    let t1 = transcript_spans.first().map(|(s, _, _)| (*s).max(0.15)).unwrap_or(0.25);
    let t2 = transcript_spans.get(1).map(|(s, _, _)| *s).unwrap_or(1.10);
    let t3 = transcript_spans.get(2).map(|(s, _, _)| *s).unwrap_or(2.10);

    let elements = match style {
        SceneStyle::MinimalEditorial => vec![
            CorkboardElementSpec {
                asset_idx: 2,
                shadow_asset_idx: 1,
                name: "Editorial Title".into(),
                base_pos: (960.0, 280.0),
                depth_height: 0.7,
                tilt_deg: 0.0,
                start_sec: 0.0,
                dur_sec: total_dur,
            },
            CorkboardElementSpec {
                asset_idx: 5,
                shadow_asset_idx: 1,
                name: "Lower Third Summary".into(),
                base_pos: (960.0, 790.0),
                depth_height: 0.9,
                tilt_deg: 0.0,
                start_sec: t1,
                dur_sec: (total_dur - t1).max(2.0),
            },
        ],
        SceneStyle::CinematicTimeline => vec![
            CorkboardElementSpec {
                asset_idx: 2,
                shadow_asset_idx: 1,
                name: "Timeline Header".into(),
                base_pos: (960.0, 150.0),
                depth_height: 0.7,
                tilt_deg: 0.0,
                start_sec: 0.0,
                dur_sec: total_dur,
            },
            CorkboardElementSpec {
                asset_idx: 3,
                shadow_asset_idx: 1,
                name: "Timeline Node 1".into(),
                base_pos: (560.0, 440.0),
                depth_height: 1.0,
                tilt_deg: 0.0,
                start_sec: t1,
                dur_sec: (total_dur - t1).max(2.0),
            },
            CorkboardElementSpec {
                asset_idx: 4,
                shadow_asset_idx: 1,
                name: "Timeline Node 2".into(),
                base_pos: (1360.0, 440.0),
                depth_height: 1.1,
                tilt_deg: 0.0,
                start_sec: t2,
                dur_sec: (total_dur - t2).max(2.0),
            },
            CorkboardElementSpec {
                asset_idx: 5,
                shadow_asset_idx: 1,
                name: "Timeline Milestone".into(),
                base_pos: (960.0, 845.0),
                depth_height: 0.9,
                tilt_deg: 0.0,
                start_sec: t3,
                dur_sec: (total_dur - t3).max(2.0),
            },
        ],
        _ => vec![
            CorkboardElementSpec {
                asset_idx: 2,
                shadow_asset_idx: 1,
                name: "Hero Title".into(),
                base_pos: (960.0, 150.0),
                depth_height: 0.7,
                tilt_deg: 0.0,
                start_sec: 0.0,
                dur_sec: total_dur,
            },
            CorkboardElementSpec {
                asset_idx: 3,
                shadow_asset_idx: 1,
                name: "Feature Card 1".into(),
                base_pos: (560.0, 470.0),
                depth_height: 1.0,
                tilt_deg: 0.0,
                start_sec: t1,
                dur_sec: (total_dur - t1).max(2.0),
            },
            CorkboardElementSpec {
                asset_idx: 4,
                shadow_asset_idx: 1,
                name: "Feature Card 2".into(),
                base_pos: (1360.0, 470.0),
                depth_height: 1.1,
                tilt_deg: 0.0,
                start_sec: t2,
                dur_sec: (total_dur - t2).max(2.0),
            },
            CorkboardElementSpec {
                asset_idx: 5,
                shadow_asset_idx: 1,
                name: "Summary Card".into(),
                base_pos: (960.0, 840.0),
                depth_height: 0.9,
                tilt_deg: 0.0,
                start_sec: t3,
                dur_sec: (total_dur - t3).max(2.0),
            },
        ],
    };

    Ok((assets, elements))
}

fn generate_tiktok_broll_card(w: u32, h: u32, card_idx: usize, article_style: bool) -> RgbaImage {
    let mut img = RgbaImage::new(w, h);
    let palette: [(u8, u8, u8); 3] = [(18, 24, 38), (24, 34, 48), (28, 22, 36)];
    let base = palette[card_idx % palette.len()];
    for y in 0..h {
        let v = y as f32 / h.max(1) as f32;
        for x in 0..w {
            let u = x as f32 / w.max(1) as f32;
            let grid = (x % 64 < 1 || y % 64 < 1) as u8;
            let glow = ((1.0 - ((u - 0.5).powi(2) + (v - 0.45).powi(2)).sqrt() * 1.4).max(0.0)) * 26.0;
            let r = (f32::from(base.0) + glow + f32::from(grid) * 8.0).clamp(0.0, 255.0) as u8;
            let g = (f32::from(base.1) + glow + f32::from(grid) * 10.0).clamp(0.0, 255.0) as u8;
            let b = (f32::from(base.2) + glow * 1.15 + f32::from(grid) * 14.0).clamp(0.0, 255.0) as u8;
            img.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }

    // Editorial article / B-roll card inset with clean border and headline bars.
    let pad_x = w / 12;
    let pad_y = h / 8;
    let card_w = w.saturating_sub(pad_x * 2);
    let card_h = h.saturating_sub(pad_y * 2);
    for dy in 0..card_h {
        for dx in 0..card_w {
            let border = dx < 4 || dy < 4 || dx + 4 >= card_w || dy + 4 >= card_h;
            let col = if border {
                Rgba([220, 226, 238, 255])
            } else if article_style {
                Rgba([244, 246, 250, 255])
            } else {
                Rgba([30, 38, 54, 255])
            };
            img.put_pixel(pad_x + dx, pad_y + dy, col);
        }
    }

    // Headline and body bars inside the card.
    let ink = if article_style { Rgba([28, 32, 42, 255]) } else { Rgba([230, 236, 248, 255]) };
    let muted = if article_style { Rgba([164, 172, 186, 255]) } else { Rgba([92, 108, 134, 255]) };
    let accent = Rgba([230, 44, 44, 255]);
    let bar_x0 = pad_x + card_w / 10;
    let bar_w_full = card_w * 4 / 5;
    for row in 0..6 {
        let y0 = pad_y + card_h / 6 + (row as u32) * (card_h / 9);
        let bh = if row == 0 { 28 } else { 16 };
        let bw = match row {
            0 => bar_w_full * 9 / 10,
            1 => bar_w_full * 7 / 10,
            5 => bar_w_full * 3 / 5,
            _ => bar_w_full,
        };
        let col = if row == 0 {
            accent
        } else if row <= 2 {
            ink
        } else {
            muted
        };
        for dy in 0..bh {
            for dx in 0..bw {
                if bar_x0 + dx < w && y0 + dy < h {
                    img.put_pixel(bar_x0 + dx, y0 + dy, col);
                }
            }
        }
    }
    img
}

fn generate_highlight_underline_bar(w: u32, h: u32) -> RgbaImage {
    let mut img = RgbaImage::new(w, h);
    for y in 0..h {
        let dy = (y as f32 - h as f32 * 0.5).abs() / (h.max(1) as f32 * 0.5);
        let alpha = ((1.0 - dy.powi(2)).clamp(0.0, 1.0) * 245.0) as u8;
        for x in 0..w {
            img.put_pixel(x, y, Rgba([230, 40, 40, alpha]));
        }
    }
    img
}

fn generate_talking_head_placeholder(w: u32, h: u32) -> RgbaImage {
    let mut img = RgbaImage::new(w, h);
    let cx = w as f32 * 0.5;
    let head_cy = h as f32 * 0.36;
    let head_r = w as f32 * 0.12;
    let torso_cy = h as f32 * 0.88;
    let torso_rx = w as f32 * 0.28;
    let torso_ry = h as f32 * 0.42;
    for y in 0..h {
        let v = y as f32 / h.max(1) as f32;
        for x in 0..w {
            let u = x as f32 / w.max(1) as f32;
            let bg = Rgba([(24.0 + 18.0 * (1.0 - v)) as u8, (28.0 + 22.0 * (1.0 - v)) as u8, (42.0 + 30.0 * (1.0 - u)) as u8, 255]);
            let d_head = ((x as f32 - cx).powi(2) + (y as f32 - head_cy).powi(2)).sqrt();
            let d_torso = ((x as f32 - cx) / torso_rx).powi(2) + ((y as f32 - torso_cy) / torso_ry).powi(2);
            let col = if d_head <= head_r {
                Rgba([226, 184, 152, 255])
            } else if d_torso <= 1.0 {
                Rgba([56, 74, 108, 255])
            } else {
                bg
            };
            img.put_pixel(x, y, col);
        }
    }
    img
}

fn prepare_tiktok_assets(
    plan: &AllInOnePlan,
    attachments: &[PreparedAttachment],
    wikimedia_imgs: &[RgbaImage],
) -> std::result::Result<(Vec<GeneratedAsset>, Option<GeneratedAsset>, GeneratedAsset), String> {
    let mut broll_assets = Vec::new();
    for (idx, img) in wikimedia_imgs.iter().take(2).enumerate() {
        let resized = image::imageops::resize(img, 1080, 960, image::imageops::FilterType::Triangle);
        broll_assets.push(GeneratedAsset { base_name: format!("tiktok_wiki_{}", idx + 1), png_bytes: encode_png(&resized)? });
    }
    let need_cards = 3usize.saturating_sub(broll_assets.len() + attachments.len());
    for i in 0..need_cards {
        let card = generate_tiktok_broll_card(1080, 960, i, plan.tiktok_article_highlight);
        broll_assets.push(GeneratedAsset { base_name: format!("tiktok_broll_{}", i + 1), png_bytes: encode_png(&card)? });
    }
    if broll_assets.is_empty() && attachments.is_empty() {
        let card = generate_tiktok_broll_card(1080, 960, 0, plan.tiktok_article_highlight);
        broll_assets.push(GeneratedAsset { base_name: "tiktok_broll_1".into(), png_bytes: encode_png(&card)? });
    }
    let highlight_bar = if plan.tiktok_article_highlight || !attachments.is_empty() {
        let bar = generate_highlight_underline_bar(820, 16);
        Some(GeneratedAsset { base_name: "tiktok_highlight_bar".into(), png_bytes: encode_png(&bar)? })
    } else {
        None
    };
    let head = generate_talking_head_placeholder(1080, 960);
    let talking_head = GeneratedAsset { base_name: "tiktok_talking_head".into(), png_bytes: encode_png(&head)? };
    Ok((broll_assets, highlight_bar, talking_head))
}

fn configure_motion(it: &mut TrackItem, scale_keyframes: &[(Tick, f64)], pos_keyframes: &[(Tick, Vec2)], rotation_deg: f64) {
    if it.effect("motion").is_none()
        && let Some(def) = find_effect("motion")
    {
        it.effects.insert(0, def.instance());
    }
    let Some(m) = it.effect_mut("motion") else { return };
    m.enabled = true;
    if let Some(u) = m.param_mut("uniform_scale") {
        u.keyframes.clear();
        u.value = ParamValue::Bool(true);
    }
    if let Some(rot) = m.param_mut("rotation") {
        rot.keyframes.clear();
        rot.value = ParamValue::Float(rotation_deg);
    }
    if let Some(sc) = m.param_mut("scale") {
        sc.keyframes.clear();
        if let Some(&(_, first)) = scale_keyframes.first() {
            sc.value = ParamValue::Float(first);
        }
        if scale_keyframes.len() > 1 {
            for &(t, v) in scale_keyframes {
                sc.put_keyframe(t, ParamValue::Float(v));
                if let Some(k) = sc.keyframes.iter_mut().find(|k| k.time == t) {
                    k.interp = Interpolation::AutoBezier;
                }
            }
            sc.keyframes.sort_by_key(|k| k.time);
        }
    }
    if let Some(pos) = m.param_mut("position") {
        pos.keyframes.clear();
        if let Some(&(_, first)) = pos_keyframes.first() {
            pos.value = ParamValue::Vec2(first);
        }
        if pos_keyframes.len() > 1 {
            for &(t, p) in pos_keyframes {
                pos.put_keyframe(t, ParamValue::Vec2(p));
                if let Some(k) = pos.keyframes.iter_mut().find(|k| k.time == t) {
                    k.interp = Interpolation::AutoBezier;
                }
            }
            pos.keyframes.sort_by_key(|k| k.time);
        }
    }
}

fn apply_tiktok_split_scene(s: &mut Session, sid: ItemId, out: &PreparedAllInOneOutput, attachment_ids: &[ItemId]) -> Result<usize> {
    if !cfg!(target_arch = "wasm32") {
        std::fs::create_dir_all(&out.dir).map_err(|e| EngineError::Other(format!("{}: {e}", out.dir)))?;
    }
    let write_and_import = |s: &mut Session, asset: &GeneratedAsset| -> Result<ItemId> {
        let path = unique_wav_path(s, &out.dir, &asset.base_name)?.replace(".wav", ".png");
        s.services.write_file(&path, &asset.png_bytes).map_err(|e| EngineError::Other(format!("{path}: {e}")))?;
        crate::commands::import_bytes(s, &path, asset.png_bytes.clone(), None)
    };

    let mut broll_ids: Vec<ItemId> = attachment_ids.to_vec();
    for asset in &out.tiktok_broll_assets {
        broll_ids.push(write_and_import(s, asset)?);
    }
    let highlight_id = match &out.tiktok_highlight_bar {
        Some(bar) => Some(write_and_import(s, bar)?),
        None => None,
    };
    let talking_head_fallback_id = match &out.tiktok_talking_head {
        Some(head) => Some(write_and_import(s, head)?),
        None => None,
    };

    let rate = s.project.sequence(sid).map(|q| q.settings.frame_rate).unwrap_or(FrameRate::FPS_24);
    let topic = out.plan.corkboard_topic.clone();

    s.edit("Build TikTok 50/50 Split Scene", |pr, st| {
        let existing_tracks = pr.sequence(sid).ok_or(EngineError::NoSequence)?.video_tracks.len();
        let needed_tracks = 3usize.saturating_sub(existing_tracks);
        let extra_tids: Vec<TrackId> = (0..needed_tracks).map(|_| TrackId(pr.alloc_id())).collect();
        let new_cap_tid = TrackId(pr.alloc_id());

        // Find any pre-existing talking-head clip on V1 or V2 before moving/placing split layers.
        let mut talking_head_clip: Option<TrackItem> = None;
        {
            let seq = pr.sequence_mut(sid).ok_or(EngineError::NoSequence)?;
            seq.settings.width = 1080;
            seq.settings.height = 1920;
            for tid in extra_tids {
                let name = format!("V{}", seq.video_tracks.len() + 1);
                seq.video_tracks.push(Track::new(tid, TrackKind::Video, name));
            }
            if let Some(v2) = seq.video_tracks.get_mut(1)
                && !v2.items.is_empty()
            {
                talking_head_clip = Some(v2.items.remove(0));
            } else if let Some(v1) = seq.video_tracks.get_mut(0)
                && !v1.items.is_empty()
            {
                talking_head_clip = Some(v1.items.remove(0));
            }
        }

        let mut placed = 0usize;
        let total_dur = talking_head_clip.as_ref().map(|c| c.duration).unwrap_or_else(|| Tick::from_seconds_f64(9.0)).max(Tick::from_seconds_f64(3.0));
        let total_sec = total_dur.seconds();

        // Top 50% Panel (V2): Talking-Head centered at (540, 480) with periodic AutoBezier punch-in accents (100% -> 108% -> 100%).
        let punch_scale = [
            (Tick::ZERO, 100.0),
            (Tick::from_seconds_f64(total_sec * 0.28), 100.0),
            (Tick::from_seconds_f64(total_sec * 0.34), 108.0),
            (Tick::from_seconds_f64(total_sec * 0.45), 100.0),
            (Tick::from_seconds_f64(total_sec * 0.68), 100.0),
            (Tick::from_seconds_f64(total_sec * 0.74), 108.0),
            (Tick::from_seconds_f64(total_sec * 0.85), 100.0),
        ];
        let top_pos = [(Tick::ZERO, Vec2::new(540.0, 480.0))];

        if let Some(mut head_it) = talking_head_clip {
            configure_motion(&mut head_it, &punch_scale, &top_pos, 0.0);
            if let Some(v2) = pr.sequence_mut(sid).and_then(|q| q.video_tracks.get_mut(1)) {
                v2.items.push(head_it);
                placed += 1;
            }
        } else if let Some(head_id) = talking_head_fallback_id
            && let Some(mut head_it) = pr.make_track_item(head_id, TrackKind::Video, Tick::ZERO, TimeRange::new(Tick::ZERO, total_dur), rate)
        {
            head_it.name = "Talking-Head (Top 50%)".into();
            head_it.duration = total_dur;
            configure_motion(&mut head_it, &punch_scale, &top_pos, 0.0);
            if let Some(v2) = pr.sequence_mut(sid).and_then(|q| q.video_tracks.get_mut(1)) {
                v2.items.push(head_it);
                placed += 1;
            }
        }

        // Bottom 50% Panel (V1): Sequence B-roll at spoken sentence boundaries (when transcript spans exist) or even intervals.
        let broll_count = broll_ids.len().max(1);
        let seg_sec = total_sec / broll_count as f64;
        for (idx, bid) in broll_ids.into_iter().enumerate() {
            let (s_sec, e_sec) = if idx < out.transcript_spans.len() {
                let st_s = out.transcript_spans[idx].0.clamp(0.0, total_sec);
                let en_s = if idx + 1 == broll_count {
                    total_sec
                } else {
                    out.transcript_spans.get(idx + 1).map(|x| x.0).unwrap_or(out.transcript_spans[idx].1).clamp(st_s + 0.5, total_sec)
                };
                (st_s, en_s)
            } else {
                (idx as f64 * seg_sec, if idx + 1 == broll_count { total_sec } else { (idx + 1) as f64 * seg_sec })
            };
            let start = rate.snap(Tick::from_seconds_f64(s_sec));
            let end = if idx + 1 == broll_count { total_dur } else { rate.snap(Tick::from_seconds_f64(e_sec)) };
            let dur = (end - start).max(rate.frame_duration().max(Tick(1)));
            if let Some(mut b_it) = pr.make_track_item(bid, TrackKind::Video, start, TimeRange::new(Tick::ZERO, dur), rate) {
                b_it.name = format!("B-Roll Panel {} (Bottom 50%)", idx + 1);
                b_it.duration = dur;
                let ken_burns = [(Tick::ZERO, 104.0), (dur, 100.0)];
                let bot_pos = [(Tick::ZERO, Vec2::new(540.0, 1440.0))];
                configure_motion(&mut b_it, &ken_burns, &bot_pos, 0.0);
                if let Some(v1) = pr.sequence_mut(sid).and_then(|q| q.video_tracks.get_mut(0))
                    && !v1.items.iter().any(|x| x.start < b_it.end() && b_it.start < x.end())
                {
                    v1.items.push(b_it);
                    placed += 1;
                }
            }
        }

        // V3: Animated Red Highlight Underline Bar sweeping across the bottom article/B-roll panel.
        if let Some(hl_id) = highlight_id {
            let hl_dur = Tick::from_seconds_f64((total_sec * 0.45).max(1.5));
            let hl_start = rate.snap(Tick::from_seconds_f64(0.6));
            if let Some(mut hl_it) = pr.make_track_item(hl_id, TrackKind::Video, hl_start, TimeRange::new(Tick::ZERO, hl_dur), rate) {
                hl_it.name = "Article Highlight Underline".into();
                hl_it.duration = hl_dur;
                let hl_pos = [(Tick::ZERO, Vec2::new(360.0, 1480.0)), (hl_dur, Vec2::new(540.0, 1480.0))];
                configure_motion(&mut hl_it, &[(Tick::ZERO, 100.0)], &hl_pos, 0.0);
                if let Some(v3) = pr.sequence_mut(sid).and_then(|q| q.video_tracks.get_mut(2)) {
                    v3.items.push(hl_it);
                    placed += 1;
                }
            }
        }

        // C1: 1–2 Word Center Pop Subtitles at exact 50% split line (CaptionAnchor::Middle).
        let has_captions = pr.sequence(sid).is_some_and(|q| !q.caption_tracks.is_empty());
        if !has_captions {
            let default_phrases = [topic.as_str(), "KEY TAKEAWAY", "WATCH THIS"];
            let cues = build_tiktok_center_pop_cues_from_phrases(&default_phrases, 200, 420, Tick::ZERO, rate);
            if !cues.is_empty() {
                let mut captions = Vec::with_capacity(cues.len());
                for cue in &cues {
                    captions.push(Caption {
                        id: ClipId(pr.alloc_id()),
                        start: cue.start,
                        duration: cue.duration,
                        text: cue.text.clone(),
                        speaker: None,
                        cue_id: None,
                        settings: String::new(),
                    });
                }
                let first_cap = captions.first().map(|c| c.id);
                let mut track = CaptionTrack::new(new_cap_tid, "AI Subtitles (TikTok Center Pop)".into(), CaptionFormat::Subtitle);
                track.style = subtitle_track_style(SubtitleLengthMode::TikTokCenterPop);
                track.captions = captions;
                if let Some(seq) = pr.sequence_mut(sid) {
                    seq.caption_tracks.insert(0, track);
                }
                st.caption_selection = first_cap.into_iter().collect();
                placed += 1;
            }
        }

        pr.sequence_mut(sid).ok_or(EngineError::NoSequence)?.check().map_err(EngineError::Other)?;
        Ok(placed)
    })
}

fn apply_all_in_one_to_session(s: &mut Session, out: PreparedAllInOneOutput) -> Result<AllInOneReport> {
    let mut steps_applied = Vec::new();
    let mut generated_layers = 0usize;
    let mut attachments_imported = 0usize;

    let need_seq =
        s.state.active_sequence.is_none() && (out.plan.build_scene || out.plan.build_corkboard || out.plan.build_tiktok_split || !out.attachments.is_empty());
    if need_seq {
        s.execute("file.newSequence", json!({"name": "Orpheus AI Scene", "video": 1, "audio": 1}))?;
    }

    let seq_id = s.state.active_sequence;
    if !out.attachments.is_empty() || out.plan.build_scene || out.plan.build_corkboard || out.plan.build_tiktok_split {
        s.edit_group("Orpheus All-In-One", |s| {
            // 1. Import any user-attached local media files.
            let mut attachment_ids: Vec<ItemId> = Vec::with_capacity(out.attachments.len());
            // A corrupt or unsupported file is skipped instead of failing the whole edit.
            let mut attachments_skipped = 0usize;
            for att in &out.attachments {
                match crate::commands::import_bytes(s, &att.orig_path, att.bytes.clone(), None) {
                    Ok(imported) => {
                        attachment_ids.push(imported);
                        attachments_imported += 1;
                    }
                    Err(_) => attachments_skipped += 1,
                }
            }
            if attachments_imported > 0 {
                steps_applied.push(format!("Imported {attachments_imported} attachment(s)"));
            }
            if attachments_skipped > 0 {
                steps_applied.push(format!("Skipped {attachments_skipped} unreadable attachment(s)"));
            }

            // 2a. Build TikTok/Reels 50/50 Talking-Head + B-Roll + 1–2 Word Center Pop Subtitles.
            if out.plan.build_tiktok_split
                && let Some(sid) = seq_id
            {
                let layers_placed = apply_tiktok_split_scene(s, sid, &out, &attachment_ids)?;
                let _ = crate::sfx_pack::insert_sfx_at(s, crate::sfx_pack::CreatorSfx::Whoosh, Some(Tick::from_seconds_f64(3.0)), Some(&out.dir));
                generated_layers += layers_placed;
                steps_applied.push(format!("Built TikTok 50/50 Split Scene ({layers_placed} layers)"));
            }

            // 2. Build prompt-driven layered scene (Modern Tech, Minimal Editorial, Cinematic Timeline, or 2.5D Corkboard).
            if (out.plan.build_scene || out.plan.build_corkboard)
                && let Some(sid) = seq_id
            {
                if !cfg!(target_arch = "wasm32") {
                    std::fs::create_dir_all(&out.dir).map_err(|e| EngineError::Other(format!("{}: {e}", out.dir)))?;
                }
                let mut imported_assets: Vec<ItemId> = Vec::with_capacity(out.corkboard_assets.len());
                for asset in &out.corkboard_assets {
                    let path = unique_wav_path(s, &out.dir, &asset.base_name)?.replace(".wav", ".png");
                    s.services.write_file(&path, &asset.png_bytes).map_err(|e| EngineError::Other(format!("{path}: {e}")))?;
                    let id = crate::commands::import_bytes(s, &path, asset.png_bytes.clone(), None)?;
                    imported_assets.push(id);
                }

                let rate = s.project.sequence(sid).map(|q| q.settings.frame_rate).unwrap_or(FrameRate::FPS_24);
                let (seq_w, seq_h) = s.project.sequence(sid).map(|q| (q.settings.width.max(1), q.settings.height.max(1))).unwrap_or((1920, 1080));
                let insert_start = rate.snap(s.state.playheads.get(&sid).copied().unwrap_or(Tick::ZERO).max(Tick::ZERO));
                let cam_kfs = corkboard_camera_keyframes();

                let layers_placed = s.edit("Build 2.5D Corkboard Scene", |pr, _| {
                    let scene_dur = Tick::from_seconds_f64(9.0);
                    let scene_end = insert_start + scene_dur;
                    // Place the opaque corkboard backdrop on V1 if empty, or on the first track ABOVE any
                    // existing talking-head/footage tracks so cutouts never float over a talking-head face.
                    let (existing_tracks, base_track) = {
                        let seq = pr.sequence(sid).ok_or(EngineError::NoSequence)?;
                        let highest_occupied = seq
                            .video_tracks
                            .iter()
                            .enumerate()
                            .rfind(|(_, tr)| tr.items.iter().any(|x| x.start < scene_end && insert_start < x.end()))
                            .map(|(idx, _)| idx + 1)
                            .unwrap_or(0);
                        (seq.video_tracks.len(), highest_occupied)
                    };
                    let target_tracks = base_track + 1 + out.corkboard_elements.len() * 2;
                    let needed_tracks = target_tracks.saturating_sub(existing_tracks);
                    let extra_tids: Vec<TrackId> = (0..needed_tracks).map(|_| TrackId(pr.alloc_id())).collect();
                    let seq = pr.sequence_mut(sid).ok_or(EngineError::NoSequence)?;
                    for tid in extra_tids {
                        let name = format!("V{}", seq.video_tracks.len() + 1);
                        seq.video_tracks.push(Track::new(tid, TrackKind::Video, name));
                    }

                    let mut placed = 0usize;
                    let sx = f64::from(seq_w) / 1920.0;
                    let sy = f64::from(seq_h) / 1080.0;
                    let scale_adj = sx.min(sy);
                    // Opaque backdrop uses max(seq_w/2400, seq_h/1350) * 135% so 100% of any 16:9 or 9:16 canvas is covered.
                    let bg_cover_scale = (f64::from(seq_w) / 2400.0).max(f64::from(seq_h) / 1350.0) * 135.0;

                    // Base track: 100% Opaque 2.5D Parallax Corkboard Backdrop (0.6x virtual camera motion).
                    if let Some(&bg_id) = imported_assets.first()
                        && let Some(mut bg_it) = pr.make_track_item(bg_id, TrackKind::Video, insert_start, TimeRange::new(Tick::ZERO, scene_dur), rate)
                    {
                        bg_it.name = "Corkboard Backdrop (2.5D Parallax)".into();
                        bg_it.duration = scene_dur;
                        let pos_kfs: Vec<(Tick, Vec2)> = cam_kfs
                            .iter()
                            .map(|&(ts, dx, dy)| {
                                (Tick::from_seconds_f64(ts), Vec2::new(f64::from(seq_w) * 0.5 + dx * 0.6 * sx, f64::from(seq_h) * 0.5 + dy * 0.6 * sy))
                            })
                            .collect();
                        configure_motion(&mut bg_it, &[(Tick::ZERO, bg_cover_scale)], &pos_kfs, 0.0);
                        if let Some(bg_tr) = pr.sequence_mut(sid).and_then(|q| q.video_tracks.get_mut(base_track)) {
                            bg_tr.items.push(bg_it);
                            placed += 1;
                        }
                    }

                    // Elements + Mathematically Coupled Dynamic Drop Shadows.
                    for (elem_idx, elem) in out.corkboard_elements.iter().enumerate() {
                        let Some(&item_id) = imported_assets.get(elem.asset_idx) else { continue };
                        let Some(&shadow_id) = imported_assets.get(elem.shadow_asset_idx) else { continue };
                        let start = insert_start + Tick::from_seconds_f64(elem.start_sec);
                        let dur = Tick::from_seconds_f64(elem.dur_sec);
                        let sh_track = base_track + 1 + elem_idx * 2;
                        let card_track = base_track + 2 + elem_idx * 2;

                        // Coupled shadow position moves opposite to camera pan scaled by depth_height.
                        let shadow_pos_kfs: Vec<(Tick, Vec2)> = cam_kfs
                            .iter()
                            .map(|&(ts, dx, dy)| {
                                let t_local = Tick::from_seconds_f64((ts - elem.start_sec).max(0.0));
                                let x = elem.base_pos.0 * sx + dx * sx + (18.0 - dx * 0.08 * elem.depth_height) * sx;
                                let y = elem.base_pos.1 * sy + dy * sy + (22.0 - dy * 0.08 * elem.depth_height) * sy;
                                (t_local, Vec2::new(x, y))
                            })
                            .collect();
                        let card_pos_kfs: Vec<(Tick, Vec2)> = cam_kfs
                            .iter()
                            .map(|&(ts, dx, dy)| {
                                let t_local = Tick::from_seconds_f64((ts - elem.start_sec).max(0.0));
                                (t_local, Vec2::new((elem.base_pos.0 + dx) * sx, (elem.base_pos.1 + dy) * sy))
                            })
                            .collect();

                        let drop_in_scale = [(Tick::ZERO, 135.0 * scale_adj), (Tick::from_seconds_f64(0.45), 100.0 * scale_adj)];

                        // Lower track: Dynamic Shadow Sprite
                        if let Some(mut sh_it) = pr.make_track_item(shadow_id, TrackKind::Video, start, TimeRange::new(Tick::ZERO, dur), rate) {
                            sh_it.name = format!("{} (Dynamic Shadow)", elem.name);
                            sh_it.duration = dur;
                            configure_motion(&mut sh_it, &drop_in_scale, &shadow_pos_kfs, elem.tilt_deg);
                            if let Some(tr) = pr.sequence_mut(sid).and_then(|q| q.video_tracks.get_mut(sh_track))
                                && !tr.items.iter().any(|x| x.start < sh_it.end() && sh_it.start < x.end())
                            {
                                tr.items.push(sh_it);
                                placed += 1;
                            }
                        }

                        // Upper track: Foreground Documentary Element
                        if let Some(mut card_it) = pr.make_track_item(item_id, TrackKind::Video, start, TimeRange::new(Tick::ZERO, dur), rate) {
                            card_it.name = elem.name.clone();
                            card_it.duration = dur;
                            configure_motion(&mut card_it, &drop_in_scale, &card_pos_kfs, elem.tilt_deg);
                            if let Some(tr) = pr.sequence_mut(sid).and_then(|q| q.video_tracks.get_mut(card_track))
                                && !tr.items.iter().any(|x| x.start < card_it.end() && card_it.start < x.end())
                            {
                                tr.items.push(card_it);
                                placed += 1;
                            }
                        }
                    }

                    pr.sequence_mut(sid).ok_or(EngineError::NoSequence)?.check().map_err(EngineError::Other)?;
                    Ok(placed)
                })?;

                generated_layers = layers_placed;
                let style_label = match out.plan.scene_style {
                    SceneStyle::Corkboard => format!("Built 2.5D Corkboard ({layers_placed} layers)"),
                    SceneStyle::ModernTech => format!("Built Modern Tech Scene ({layers_placed} layers)"),
                    SceneStyle::MinimalEditorial => format!("Built Minimal Editorial Scene ({layers_placed} layers)"),
                    SceneStyle::CinematicTimeline => format!("Built Cinematic Timeline ({layers_placed} layers)"),
                };
                steps_applied.push(style_label);
            }
            Ok(())
        })?;
    }

    // 3. Unified Tool Orchestration (run synchronously or dispatch cleanly when sequence has clips).
    if seq_id.is_some() {
        if out.plan.run_remove_pauses && s.execute("ai.removePauses", json!({})).is_ok() {
            steps_applied.push("Auto-Cut Silence".into());
        }
        if out.plan.run_audio_enhance && s.execute("ai.enhanceAudio", json!({"wait": true})).is_ok() {
            steps_applied.push("AI Audio Enhance".into());
        }
        if out.plan.run_auto_reframe
            && s.execute("ai.autoReframe", json!({"aspect": "9:16", "splitScreen": out.plan.reframe_split_screen, "wait": true})).is_ok()
        {
            steps_applied.push(if out.plan.reframe_split_screen { "Gaming Split-Screen (9:16)" } else { "AI Auto-Reframe (9:16)" }.into());
        }
        if out.plan.run_bg_removal && s.execute("ai.removeBackground", json!({"mode": "transparentAlpha", "wait": true})).is_ok() {
            steps_applied.push("AI Background Removal".into());
        }
        if out.plan.run_beat_sync && s.execute("ai.addBeatMarkers", json!({"wait": true})).is_ok() {
            steps_applied.push("AI Beat Sync".into());
        }
        if out.plan.run_vod_chapters && s.execute("ai.vodChapters", json!({"wait": true})).is_ok() {
            steps_applied.push("Stream VOD Chapterizer".into());
        }
        if out.plan.run_viral_clips && s.execute("ai.findViralClips", json!({"wait": true})).is_ok() {
            steps_applied.push("AI Viral Clip Finder".into());
        }
        if out.plan.run_smart_triggers && s.execute("ai.applySmartTriggers", json!({"wait": true})).is_ok() {
            steps_applied.push("AI Smart Triggers".into());
        }
        if out.plan.run_subtitles && s.execute("ai.generateSubtitles", json!({"wait": true})).is_ok() {
            steps_applied.push("AI Subtitles".into());
        }
        if out.plan.run_transitions
            && let Some(sid) = seq_id
        {
            let cuts: Vec<(TrackId, Tick)> = s
                .project
                .sequence(sid)
                .map(|q| {
                    q.video_tracks
                        .iter()
                        .filter(|tr| !tr.locked)
                        .flat_map(|tr| tr.items.windows(2).filter(|w| w[0].end() == w[1].start).map(move |w| (tr.id, w[1].start)))
                        .collect()
                })
                .unwrap_or_default();
            let mut added = 0usize;
            for (tid, at) in cuts {
                if s.execute("timeline.addTransition", json!({"track": tid.0, "at": at.0, "effect": "Cross Dissolve"})).is_ok() {
                    added += 1;
                }
            }
            if added > 0 {
                steps_applied.push(format!("Added {added} Cross Dissolve transition(s)"));
            }
        }
        if out.plan.run_color_grade
            && let Some(sid) = seq_id
        {
            let clips: Vec<ClipId> = s
                .project
                .sequence(sid)
                .map(|q| q.video_tracks.iter().filter(|tr| !tr.locked).flat_map(|tr| tr.items.iter().map(|it| it.id)).collect())
                .unwrap_or_default();
            let fx = if out.plan.corkboard_topic.to_ascii_lowercase().contains("black and white")
                || out.plan.corkboard_topic.to_ascii_lowercase().contains("monochrome")
            {
                "black_white"
            } else {
                "lumetri"
            };
            let mut graded = 0usize;
            for cid in clips {
                if s.execute("vfx.apply", json!({"clip": cid.0, "effect": fx})).is_ok() {
                    graded += 1;
                }
            }
            if graded > 0 {
                steps_applied.push(format!("Applied {fx} look to {graded} clip(s)"));
            }
        }
        if out.plan.run_text_overlay
            && let Some(sid) = seq_id
        {
            let headline = extract_clean_headline(&out.plan.corkboard_topic);
            if s.execute("graphics.addText", json!({"seq": sid.0, "text": headline})).is_ok() {
                steps_applied.push("Added Title Overlay".into());
            }
        }
    }

    if steps_applied.is_empty() {
        steps_applied.push("Planned offline (no matching sequence clips found)".into());
    }

    let rep = AllInOneReport {
        seq: seq_id,
        steps_applied,
        generated_layers,
        attachments_imported,
        wikimedia_fetched: out.wikimedia_count,
        used_ollama: out.plan.used_ollama,
    };
    s.all_in_one_report = Some(rep.clone());
    s.toast(format!("Orpheus All-In-One: {}", rep.steps_applied.join(" · ")));
    Ok(rep)
}

/// Apply completed background Orpheus All-In-One jobs on the session thread.
pub fn poll(s: &mut Session) {
    let mut i = 0usize;
    while i < s.all_in_one_jobs.len() {
        let job = s.jobs.iter().find(|j| j.id == s.all_in_one_jobs[i].job);
        let finished = job.is_none_or(|j| j.progress.finished.load(Ordering::Relaxed));
        let cancelled = job.is_some_and(|j| j.progress.cancel.load(Ordering::Relaxed));
        if !finished {
            i += 1;
            continue;
        }
        let pj = s.all_in_one_jobs.remove(i);
        let Some(output) = lock(&pj.output).take().filter(|_| !cancelled) else { continue };
        if pj.epoch != s.document_epoch {
            s.error_toast("ai.allInOne", "Orpheus All-In-One: the project was replaced; nothing was changed");
            continue;
        }
        if let Err(e) = apply_all_in_one_to_session(s, output) {
            s.error_toast("ai.allInOne", format!("Orpheus All-In-One failed: {e}"));
        }
    }
}

fn all_in_one_cmd(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "ai.allInOne";
    let prompt = str_p(p, "prompt").unwrap_or("").trim().to_string();
    let attachment_paths: Vec<String> = p
        .get("attachments")
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(Value::as_str).map(str::to_string).take(MAX_ATTACHMENTS).collect())
        .unwrap_or_default();

    if prompt.is_empty() && attachment_paths.is_empty() {
        return Err(bad(cmd, "Enter a prompt or attach local files first"));
    }

    let fetch_wikimedia = bool_p(p, "fetchWikimedia").unwrap_or(false);
    let ollama_url = str_p(p, "ollamaUrl").unwrap_or("").to_string();
    let dir = media_dir(s, p, "Orpheus AI")?;

    // Extract existing spoken sentence spans from active sequence captions for transcript-driven pacing.
    let transcript_spans: Vec<(f64, f64, String)> = s
        .state
        .active_sequence
        .and_then(|sid| s.project.sequence(sid))
        .and_then(|seq| seq.caption_tracks.first())
        .map(|tr| tr.captions.iter().take(12).map(|c| (c.start.seconds(), (c.start + c.duration).seconds(), c.text.clone())).collect())
        .unwrap_or_default();

    let id = next_job_id(s);
    let prog = Arc::new(Progress::default());
    prog.total.store(100, Ordering::Relaxed);
    *lock(&prog.status) = "Planning Orpheus All-In-One…".into();
    let result: Arc<Mutex<Option<std::result::Result<Report, String>>>> = Arc::default();
    s.jobs.push(Job { id, label: "Orpheus All-In-One".into(), progress: prog.clone(), result: result.clone() });

    let output_slot: Arc<Mutex<Option<PreparedAllInOneOutput>>> = Arc::default();
    s.all_in_one_jobs.push(PendingAllInOneJob { job: id, epoch: s.document_epoch, prompt: prompt.clone(), output: output_slot.clone() });

    let wait = bool_p(p, "wait").unwrap_or(false);
    let run = crate::export_tools::guard_job(prog.clone(), result.clone(), move || {
        let t0 = web_time::Instant::now();
        let outcome = (|| -> std::result::Result<PreparedAllInOneOutput, String> {
            check_cancelled(Some(&prog))?;
            let mut plan = plan_from_prompt(&prompt, attachment_paths.len());
            if !ollama_url.trim().is_empty() {
                *lock(&prog.status) = "Consulting local Ollama planner…".into();
                let _ = try_plan_with_ollama(&ollama_url, &prompt, &mut plan);
            }
            prog.done.store(20, Ordering::Relaxed);

            let mut attachments = Vec::new();
            for path in &attachment_paths {
                check_cancelled(Some(&prog))?;
                if let Ok(meta) = std::fs::metadata(path)
                    && meta.is_file()
                    && meta.len() <= MAX_ATTACHMENT_BYTES
                    && let Ok(bytes) = std::fs::read(path)
                {
                    attachments.push(PreparedAttachment { orig_path: path.clone(), bytes: Arc::from(bytes) });
                }
            }
            if attachments.is_empty() && !attachment_paths.is_empty() && prompt.trim().is_empty() {
                return Err("None of the attached files could be read".to_string());
            }
            prog.done.store(40, Ordering::Relaxed);

            let mut wikimedia_imgs = Vec::new();
            // Prioritize user-attached media: skip Wikimedia stock whenever local attachments are provided.
            if fetch_wikimedia && attachments.is_empty() && (plan.build_scene || plan.build_corkboard || plan.build_tiktok_split) {
                *lock(&prog.status) = "Fetching Wikimedia reference images…".into();
                wikimedia_imgs = fetch_wikimedia_images(&plan.corkboard_topic, Some(&prog));
            }
            let wikimedia_count = wikimedia_imgs.len();
            prog.done.store(65, Ordering::Relaxed);

            let (corkboard_assets, corkboard_elements) = if plan.build_corkboard {
                *lock(&prog.status) = "Generating 2.5D procedural documentary assets…".into();
                prepare_corkboard_assets(&plan.corkboard_topic, &attachments, &wikimedia_imgs, &transcript_spans)?
            } else if plan.build_scene {
                *lock(&prog.status) = "Generating prompt-driven styled scene…".into();
                prepare_styled_scene_assets(plan.scene_style, &plan.corkboard_topic, &attachments, &wikimedia_imgs, &transcript_spans)?
            } else {
                (Vec::new(), Vec::new())
            };
            let (tiktok_broll_assets, tiktok_highlight_bar, tiktok_talking_head) = if plan.build_tiktok_split {
                *lock(&prog.status) = "Generating 50/50 TikTok split B-roll & highlight overlays…".into();
                let (broll, bar, head) = prepare_tiktok_assets(&plan, &attachments, &wikimedia_imgs)?;
                (broll, bar, Some(head))
            } else {
                (Vec::new(), None, None)
            };
            prog.done.store(100, Ordering::Relaxed);

            Ok(PreparedAllInOneOutput {
                plan,
                dir,
                attachments,
                corkboard_assets,
                corkboard_elements,
                tiktok_broll_assets,
                tiktok_highlight_bar,
                tiktok_talking_head,
                transcript_spans,
                wikimedia_count,
            })
        })();

        let rep = match outcome {
            Ok(prep) => {
                let count = prep.corkboard_elements.len() as u64;
                *lock(&output_slot) = Some(prep);
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
        std::thread::Builder::new().name("orpheus-all-in-one".into()).spawn(run).map_err(|e| EngineError::Other(e.to_string()))?;
    }

    if let Some(Err(e)) = s.jobs.iter().find(|j| j.id == id).and_then(|j| lock(&j.result).clone()) {
        return Err(EngineError::Other(e));
    }

    let mut out = json!({ "job": id });
    if wait && let Some(rep) = &s.all_in_one_report {
        out["stepsApplied"] = json!(rep.steps_applied);
        out["generatedLayers"] = json!(rep.generated_layers);
        out["attachmentsImported"] = json!(rep.attachments_imported);
        out["wikimediaFetched"] = json!(rep.wikimedia_fetched);
        out["usedOllama"] = json!(rep.used_ollama);
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
    let report = s.all_in_one_report.as_ref().map(|r| {
        json!({
            "stepsApplied": r.steps_applied,
            "generatedLayers": r.generated_layers,
            "attachmentsImported": r.attachments_imported,
            "wikimediaFetched": r.wikimedia_fetched,
            "usedOllama": r.used_ollama,
        })
    });
    Ok(json!({ "running": running, "progress": progress, "status": status, "error": error, "report": report }))
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "ai.allInOne",
            label: "Orpheus All-In-One (AI)",
            menu: &[],
            shortcut: None,
            params: r#"{"prompt":str?,"attachments":[str]?,"fetchWikimedia":bool?,"ollamaUrl":str?,"dir":str?,"wait":bool?}"#,
            enabled: always,
            run: all_in_one_cmd,
            journal: false,
            named: NamedKeys::NONE,
        },
        CommandSpec {
            id: "ai.allInOneStatus",
            label: "Orpheus All-In-One Status",
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

    #[test]
    fn offline_planner_detects_corkboard_and_chained_ai_tools() {
        let p1 = plan_from_prompt("Create a 2.5D corkboard documentary scene with dynamic shadows", 0);
        assert!(p1.build_corkboard);
        assert!(!p1.used_ollama);

        let p2 = plan_from_prompt("Clean audio, make 9:16 gaming split screen, remove background, add viral clips", 0);
        assert!(p2.run_audio_enhance);
        assert!(p2.run_auto_reframe);
        assert!(p2.reframe_split_screen);
        assert!(p2.run_bg_removal);
        assert!(p2.run_viral_clips);
    }

    #[test]
    fn generic_prompts_select_modern_tech_minimal_or_timeline_without_corkboard() {
        let tech = plan_from_prompt("Create an AI product launch showcase with 4K 60FPS stats", 0);
        assert!(!tech.build_corkboard);
        assert!(tech.build_scene);
        assert_eq!(tech.scene_style, SceneStyle::ModernTech);

        let minimal = plan_from_prompt("Minimalist editorial typography intro for my channel", 0);
        assert!(!minimal.build_corkboard);
        assert!(minimal.build_scene);
        assert_eq!(minimal.scene_style, SceneStyle::MinimalEditorial);

        let timeline = plan_from_prompt("Chronological history timeline of space exploration", 0);
        assert!(!timeline.build_corkboard);
        assert!(timeline.build_scene);
        assert_eq!(timeline.scene_style, SceneStyle::CinematicTimeline);

        let edit_only = plan_from_prompt("Add cross dissolve transitions and cinematic color grading", 0);
        assert!(!edit_only.build_corkboard);
        assert!(!edit_only.build_scene);
        assert!(edit_only.run_transitions);
        assert!(edit_only.run_color_grade);
    }

    #[test]
    fn procedural_corkboard_builds_layered_scene_with_coupled_shadows_and_undoes() {
        let mut s = Session::default();
        s.execute("file.newSequence", json!({"name": "Doc", "video": 1, "audio": 1})).unwrap();
        let tmp_dir = std::env::temp_dir().join(format!("orpheus-allinone-test-{}", std::process::id()));
        let dir_str = tmp_dir.to_string_lossy().into_owned();

        let res = s
            .execute(
                "ai.allInOne",
                json!({
                    "prompt": "Build a 2.5D corkboard documentary scene with dynamic shadows",
                    "fetchWikimedia": false,
                    "dir": dir_str,
                    "wait": true
                }),
            )
            .unwrap();
        assert!(res["generatedLayers"].as_u64().unwrap_or(0) >= 6);

        let seq_id = s.state.active_sequence.unwrap();
        let seq = s.project.sequence(seq_id).unwrap();
        assert!(seq.video_tracks.len() >= 5, "corkboard scene must span parallax background + shadow + card tracks");
        assert!(!seq.video_tracks[0].items.is_empty(), "V1 must hold parallax corkboard backdrop");

        // Dynamic shadow keyframes on V2 must differ from card keyframes on V3.
        let sh_pos = seq.video_tracks[1].items[0].effect("motion").unwrap().param("position").unwrap();
        assert!(sh_pos.keyframes.len() >= 3, "shadow must carry AutoBezier camera-coupled keyframes");

        // Undo removes the entire All-In-One scene cleanly.
        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(s.project.sequence(seq_id).unwrap().video_tracks[0].items.len(), 0);
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn empty_prompt_without_attachments_returns_clean_error() {
        let mut s = Session::default();
        assert!(s.execute("ai.allInOne", json!({"prompt": "   ", "wait": true})).is_err());
    }

    #[test]
    fn tiktok_50_50_split_builds_top_talking_head_bottom_broll_center_pop_subs_and_undoes() {
        let p = plan_from_prompt("Create a 50/50 tiktok split talking head with article highlight", 0);
        assert!(p.build_tiktok_split);
        assert!(p.tiktok_article_highlight);
        assert!(!p.build_corkboard);

        let mut s = Session::default();
        s.execute("file.newSequence", json!({"name": "TikTokSplit", "video": 1, "audio": 1})).unwrap();
        let tmp_dir = std::env::temp_dir().join(format!("orpheus-tiktok-split-test-{}", std::process::id()));
        let dir_str = tmp_dir.to_string_lossy().into_owned();

        let res = s
            .execute(
                "ai.allInOne",
                json!({
                    "prompt": "50/50 tiktok split talking head with article highlight",
                    "fetchWikimedia": false,
                    "dir": dir_str,
                    "wait": true
                }),
            )
            .unwrap();
        assert!(res["generatedLayers"].as_u64().unwrap_or(0) >= 5);

        let seq_id = s.state.active_sequence.unwrap();
        let seq = s.project.sequence(seq_id).unwrap();
        assert_eq!((seq.settings.width, seq.settings.height), (1080, 1920));
        assert!(seq.video_tracks.len() >= 3, "50/50 TikTok split needs V1 B-Roll, V2 Talking-Head, V3 Highlight Bar");
        assert!(!seq.video_tracks[0].items.is_empty(), "V1 must hold bottom 50% B-roll");
        assert!(!seq.video_tracks[1].items.is_empty(), "V2 must hold top 50% talking-head");
        assert!(!seq.video_tracks[2].items.is_empty(), "V3 must hold article highlight bar");

        // V2 Talking-Head must have AutoBezier punch-in scale keyframes and position at y=480.
        let v2_motion = seq.video_tracks[1].items[0].effect("motion").unwrap();
        assert!(v2_motion.param("scale").unwrap().keyframes.len() >= 3);
        assert_eq!(v2_motion.param("position").unwrap().value, ParamValue::Vec2(Vec2::new(540.0, 480.0)));

        // C1 must contain uppercase 1-2 word Center Pop captions at CaptionAnchor::Middle.
        assert!(!seq.caption_tracks.is_empty(), "C1 TikTok Center Pop subtitle track must be inserted");
        let cap_track = &seq.caption_tracks[0];
        assert_eq!(cap_track.style.anchor, filmcraft_project::CaptionAnchor::Middle);
        assert!(!cap_track.captions.is_empty());
        for cap in &cap_track.captions {
            assert!(cap.text.split_whitespace().count() <= 2, "center pop caption must be 1-2 words: {}", cap.text);
            assert_eq!(cap.text, cap.text.to_uppercase(), "center pop caption must be uppercase");
        }

        // Single undo restores the sequence cleanly.
        s.execute("edit.undo", json!({})).unwrap();
        let seq_after = s.project.sequence(seq_id).unwrap();
        assert_eq!(seq_after.video_tracks[0].items.len(), 0);
        assert_eq!(seq_after.caption_tracks.len(), 0);
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn corkboard_overlays_opaque_backdrop_above_existing_talking_head_and_renders_bitmap_text() {
        assert_eq!(extract_clean_headline("Build a 2.5D corkboard documentary scene about Artificial Intelligence"), "ARTIFICIAL INTELLIGENCE");
        let banner = generate_torn_banner(880, 140, "ARTIFICIAL INTELLIGENCE", "EVIDENCE BOARD");
        let dark_ink_pixels = banner.pixels().filter(|px| px[3] == 255 && px[0] < 50 && px[1] < 50 && px[2] < 50).count();
        assert!(dark_ink_pixels > 500, "torn banner must contain crisp dark bitmap headline typography");

        let sticky = generate_sticky_note(340, 340, "ARTIFICIAL INTELLIGENCE", "+340% GROWTH", "AI ADOPTION SURGE");
        let sticky_ink = sticky.pixels().filter(|px| px[3] == 255 && px[0] < 60 && px[1] < 60 && px[2] < 70).count();
        assert!(sticky_ink > 300, "sticky note must contain crisp bitmap note text");

        let mut s = Session::default();
        s.execute("file.newSequence", json!({"name": "TalkingHeadDoc", "video": 1, "audio": 1})).unwrap();
        let seq_id = s.state.active_sequence.unwrap();
        let head_png = encode_png(&generate_talking_head_placeholder(1080, 1920)).unwrap();
        let tmp_dir = std::env::temp_dir().join(format!("orpheus-corkboard-overlay-test-{}", std::process::id()));
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let head_path = tmp_dir.join("host.png").to_string_lossy().into_owned();
        std::fs::write(&head_path, &*head_png).unwrap();
        let host_id = crate::commands::import_bytes(&mut s, &head_path, head_png, None).unwrap();
        s.edit("Place Host on V1", |pr, _| {
            let seq = pr.sequence_mut(seq_id).unwrap();
            seq.settings.width = 1080;
            seq.settings.height = 1920;
            let rate = seq.settings.frame_rate;
            let dur = Tick::from_seconds_f64(12.0);
            let it = pr.make_track_item(host_id, TrackKind::Video, Tick::ZERO, TimeRange::new(Tick::ZERO, dur), rate).unwrap();
            pr.sequence_mut(seq_id).unwrap().video_tracks[0].items.push(it);
            Ok(())
        })
        .unwrap();

        let dir_str = tmp_dir.to_string_lossy().into_owned();
        s.execute(
            "ai.allInOne",
            json!({
                "prompt": "Build a 2.5D corkboard documentary scene about Artificial Intelligence",
                "fetchWikimedia": false,
                "dir": dir_str,
                "wait": true
            }),
        )
        .unwrap();

        let seq = s.project.sequence(seq_id).unwrap();
        // V1 still has the host clip; V2 (index 1) holds the full-frame opaque corkboard backdrop covering the 9:16 canvas.
        assert_eq!(seq.video_tracks[0].items.len(), 1);
        assert_eq!(seq.video_tracks[1].items[0].name, "Corkboard Backdrop (2.5D Parallax)");
        let bg_scale = seq.video_tracks[1].items[0].effect("motion").unwrap().f64_at("scale", Tick::ZERO);
        assert!(bg_scale >= 180.0, "9:16 corkboard backdrop scale ({bg_scale}) must cover full vertical height without exposing talking head");
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn auto_cut_and_polish_plan_and_attachment_priority_skip_wikimedia() {
        let p = plan_from_prompt("Auto-Cut & Polish: remove pauses, clean audio, and generate Shorts subtitles", 0);
        assert!(p.run_remove_pauses);
        assert!(p.run_audio_enhance);
        assert!(p.run_subtitles);
        assert!(!p.build_corkboard);

        let mut s = Session::default();
        s.execute("file.newSequence", json!({"name": "AttachedOnly", "video": 1, "audio": 1})).unwrap();
        let tmp_dir = std::env::temp_dir().join(format!("orpheus-attached-only-test-{}", std::process::id()));
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let attach_path = tmp_dir.join("user_chart.png");
        let img = generate_chart_card(640, 420, "USER REVENUE +42%", "+42%");
        std::fs::write(&attach_path, encode_png(&img).unwrap()).unwrap();

        let res = s
            .execute(
                "ai.allInOne",
                json!({
                    "prompt": "Build a 2.5D corkboard scene about User Revenue +42% in 2026",
                    "attachments": [attach_path.to_string_lossy().into_owned()],
                    "fetchWikimedia": true,
                    "dir": tmp_dir.to_string_lossy().into_owned(),
                    "wait": true
                }),
            )
            .unwrap();
        assert_eq!(res["wikimediaFetched"].as_u64(), Some(0), "local attachments must skip Wikimedia fetching");

        let sticky = generate_sticky_note(340, 340, "USER REVENUE", "+42%", "RECORD GROWTH");
        assert_eq!(sticky.dimensions(), (340, 340));
        let banner = generate_torn_banner(880, 140, "USER REVENUE", "2026 REPORT");
        assert_eq!(banner.dimensions(), (880, 140));
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn missing_corrupt_and_cyrillic_attachments_are_skipped_and_one_undo_reverts_everything() {
        let mut s = Session::default();
        s.execute("file.newSequence", json!({"name": "Attach", "video": 1, "audio": 1})).unwrap();
        let seq_id = s.state.active_sequence.unwrap();
        let tmp_dir = std::env::temp_dir().join(format!("orpheus-attach-edge-{}", std::process::id()));
        std::fs::create_dir_all(&tmp_dir).unwrap();

        let good = tmp_dir.join("Снимка.png");
        std::fs::write(&good, encode_png(&generate_chart_card(320, 200, "ДОКЛАД", "+7%")).unwrap()).unwrap();
        let corrupt = tmp_dir.join("Грешен файл.png");
        std::fs::write(&corrupt, [0xFFu8, 0x00, 0x13, 0x37, 0x42]).unwrap();
        let missing = tmp_dir.join("Липсва.png");

        let items_before = s.project.items.len();
        let res = s
            .execute(
                "ai.allInOne",
                json!({
                    "prompt": "Build a 2.5D corkboard scene",
                    "attachments": [corrupt.to_string_lossy(), missing.to_string_lossy(), good.to_string_lossy()],
                    "dir": tmp_dir.to_string_lossy(),
                    "wait": true
                }),
            )
            .unwrap();
        assert_eq!(res["attachmentsImported"].as_u64(), Some(1), "only the readable Cyrillic-named file imports: {res}");
        assert!(!s.project.sequence(seq_id).unwrap().video_tracks[0].items.is_empty());

        // One Ctrl+Z reverts the whole All-In-One edit group: clips, tracks and imported media.
        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(s.project.sequence(seq_id).unwrap().video_tracks[0].items.len(), 0);
        assert_eq!(s.project.items.len(), items_before, "imported and generated media are removed by the single undo");

        // Only unreadable attachments and no prompt: a clean error, not a generic scene.
        let err = s.execute("ai.allInOne", json!({"attachments": [missing.to_string_lossy()], "dir": tmp_dir.to_string_lossy(), "wait": true}));
        assert!(err.is_err());
        assert_eq!(s.project.items.len(), items_before);
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }
}
