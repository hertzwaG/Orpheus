//! Local AI Subtitles (`ai.subtitleStatus`, `ai.downloadSubtitleModel`, `ai.generateSubtitles`).
//!
//! Invokes a `whisper-cli.exe` (`whisper.cpp`) Vulkan-accelerated sidecar process cleanly as a
//! background job, extracting timeline audio to a temporary 16 kHz mono 16-bit PCM `.wav` file,
//! streaming GGML model downloads on demand into `%APPDATA%\Orpheus\models\`, parsing `-ojf`
//! timestamps into either short Reels/TikTok cues (1-3 words) or standard sentence subtitles, and
//! committing a dedicated caption track on the timeline in a single undoable edit.

use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;
use web_time::Instant;

use filmcraft_export::{Progress, Report};
use filmcraft_geom::Vec2;
use filmcraft_project::{Caption, CaptionAnchor, CaptionFormat, CaptionStyle, CaptionTrack, ClipId, ItemId, ParamValue, Project, Sequence, Track, TrackId};
use filmcraft_render::SourceProvider;
use filmcraft_time::{FrameRate, Tick};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, always, bad, bool_p, clips_p, has_seq, str_p, with_links};
use crate::{EngineError, Job, Result, Session};

#[cfg(windows)]
pub const WHISPER_CLI_NAME: &str = "whisper-cli.exe";
#[cfg(not(windows))]
pub const WHISPER_CLI_NAME: &str = "whisper-cli";

/// Analysis & whisper.cpp input sample rate (16 kHz mono).
const SUBTITLE_SAMPLE_RATE: u32 = 16_000;
/// Audio rendering chunk size (10 seconds @ 16 kHz).
const AUDIO_CHUNK_FRAMES: usize = 160_000;
/// Hard safety bound on extracted audio length (24 hours @ 16 kHz) to support 12-hour+ VODs while bounding corrupt durations.
pub const MAX_AUDIO_SAMPLES: usize = 24 * 3600 * SUBTITLE_SAMPLE_RATE as usize;
/// Streaming Whisper transcription window (20 minutes @ 16 kHz = 19.2M samples = ~38.4 MB WAV on disk).
pub const WHISPER_WINDOW_SAMPLES: usize = 20 * 60 * SUBTITLE_SAMPLE_RATE as usize;
/// Hard bound on parsed whisper JSON size (64 MiB).
const MAX_JSON_BYTES: u64 = 64 << 20;

/// Supported Whisper language presets (Bulgarian default + Auto + major languages).
pub const SUBTITLE_LANGUAGES: &[(&str, &str)] =
    &[("bg", "Bulgarian (bg)"), ("auto", "Auto-detect"), ("en", "English (en)"), ("es", "Spanish (es)"), ("de", "German (de)"), ("ru", "Russian (ru)")];

/// Supported Whisper GGML models downloaded on demand from `ggerganov/whisper.cpp`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WhisperModelSpec {
    pub id: &'static str,
    pub file_name: &'static str,
    pub label: &'static str,
    pub url: &'static str,
    pub approx_bytes: u64,
    pub approx_mb: u64,
}

pub const WHISPER_MODELS: &[WhisperModelSpec] = &[
    WhisperModelSpec {
        id: "large-v3-turbo",
        file_name: "ggml-large-v3-turbo.bin",
        label: "Large v3 Turbo — Most accurate",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
        approx_bytes: 1_624_555_275,
        approx_mb: 1625,
    },
    WhisperModelSpec {
        id: "base",
        file_name: "ggml-base.bin",
        label: "Base — Fast",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
        approx_bytes: 147_951_465,
        approx_mb: 148,
    },
];

pub const DEFAULT_SUBTITLE_MODEL: &str = "large-v3-turbo";
pub const DEFAULT_SUBTITLE_LANGUAGE: &str = "bg";

/// Subtitle formatting preset: punchy 1-3 word Reels/TikTok captions, 1-2 word center-pop, or standard sentences.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SubtitleLengthMode {
    ReelsShort,
    TikTokCenterPop,
    #[default]
    Sentences,
}

impl SubtitleLengthMode {
    pub fn from_str_param(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "reels" | "reelsshort" | "reels_short" | "short" => Some(Self::ReelsShort),
            "tiktok" | "tiktok_center_pop" | "tiktokcenterpop" | "center_pop" | "centerpop" => Some(Self::TikTokCenterPop),
            "sentences" | "sentence" | "standard" => Some(Self::Sentences),
            _ => None,
        }
    }

    pub fn as_id(self) -> &'static str {
        match self {
            Self::ReelsShort => "reels",
            Self::TikTokCenterPop => "tiktok",
            Self::Sentences => "sentences",
        }
    }
}

/// Minimum dedicated GPU VRAM (in MiB) required before recommending Vulkan GPU acceleration.
pub const RECOMMENDED_GPU_MIN_VRAM_MB: u64 = 4096;
/// Minimum dedicated GPU VRAM (8 GB) for optimal Vulkan performance on heavy neural models.
pub const OPTIMAL_GPU_MIN_VRAM_MB: u64 = 8192;
/// Minimum logical CPU cores for moderate performance on CPU-only multi-core workstations.
pub const MODERN_CPU_MIN_CORES: usize = 6;
/// Expected Vulkan backend DLL alongside `whisper-cli.exe` on Windows.
pub const VULKAN_DLL_NAME: &str = "ggml-vulkan.dll";

/// Workload weight category for dynamic hardware capability badging in AI Tools.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AiFeatureWeight {
    /// Pure algorithmic / audio DSP / text heuristics (always optimal).
    Lightweight,
    /// Speech recognition (`whisper-cli.exe`) — optimal with 8+ GB Vulkan GPU, moderate on 4–7 GB VRAM or modern CPU.
    NeuralSpeech,
    /// Heavy visual neural models (Auto-Reframe, Background Removal) — optimal with 8+ GB Vulkan GPU,
    /// moderate on 4–7 GB VRAM or multi-core CPU without integrated GPU bottleneck, slow on < 4 GB / iGPU.
    HeavyVisual,
}

/// Host hardware performance tier shown as a pill badge beside each AI feature.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum HardwareCapabilityTier {
    Optimal,
    Moderate,
    Slow,
}

/// Return logical CPU core count on the current host (at least 1).
pub fn host_cpu_cores() -> usize {
    std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
}

/// Classify expected execution speed (`Optimal`, `Moderate`, or `Slow`) from workload weight,
/// Vulkan DLL readiness, primary GPU adapter metadata, and logical CPU core count.
pub fn feature_capability_tier(weight: AiFeatureWeight, vulkan_dll_found: bool, gpu: Option<&GpuAdapterInfo>, cpu_cores: usize) -> HardwareCapabilityTier {
    if weight == AiFeatureWeight::Lightweight {
        return HardwareCapabilityTier::Optimal;
    }
    if vulkan_dll_found && gpu.is_some_and(|g| g.is_discrete && g.dedicated_vram_mb >= OPTIMAL_GPU_MIN_VRAM_MB) {
        return HardwareCapabilityTier::Optimal;
    }
    let mid_vram = vulkan_dll_found && gpu.is_some_and(|g| g.is_discrete && g.dedicated_vram_mb >= RECOMMENDED_GPU_MIN_VRAM_MB);
    let cpu_fallback_ok = cpu_cores >= MODERN_CPU_MIN_CORES
        && match weight {
            AiFeatureWeight::Lightweight => true,
            AiFeatureWeight::NeuralSpeech => gpu.is_none_or(|_| !vulkan_dll_found),
            AiFeatureWeight::HeavyVisual => gpu.is_none(),
        };
    if mid_vram || cpu_fallback_ok { HardwareCapabilityTier::Moderate } else { HardwareCapabilityTier::Slow }
}

/// Compute backend passed to `whisper-cli` for local AI subtitle transcription.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SubtitleComputeDevice {
    #[default]
    Gpu,
    Cpu,
}

impl SubtitleComputeDevice {
    pub fn from_str_param(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "gpu" | "vulkan" => Some(Self::Gpu),
            "cpu" | "no-gpu" | "nogpu" => Some(Self::Cpu),
            _ => None,
        }
    }

    pub fn as_id(self) -> &'static str {
        match self {
            Self::Gpu => "gpu",
            Self::Cpu => "cpu",
        }
    }
}

/// Primary GPU metadata supplied by the platform layer (DXGI on Windows).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuAdapterInfo {
    pub name: String,
    pub dedicated_vram_mb: u64,
    pub is_discrete: bool,
}

/// Why a particular subtitle compute backend was recommended for the current host.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ComputeRecommendationReason {
    DiscreteGpuSufficientVram,
    VulkanDllMissing,
    LowVram,
    IntegratedGpu,
    NoGpuDetected,
}

/// Hardware acceleration readiness and smart default recommendation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ComputeRecommendation {
    pub recommended: SubtitleComputeDevice,
    pub gpu_available: bool,
    pub reason: ComputeRecommendationReason,
    pub gpu_display_name: Option<String>,
}

static GPU_PROBE: std::sync::OnceLock<fn() -> Option<GpuAdapterInfo>> = std::sync::OnceLock::new();

/// Register the platform-layer primary GPU query callback (`filmcraft-platform` from the app entry point).
pub fn register_gpu_probe(probe: fn() -> Option<GpuAdapterInfo>) {
    let _ = GPU_PROBE.set(probe);
}

/// Query primary hardware GPU metadata using the registered platform probe, if available.
pub fn detect_primary_gpu() -> Option<GpuAdapterInfo> {
    GPU_PROBE.get().and_then(|probe| probe())
}

/// Check whether the Vulkan runtime library exists alongside the resolved `whisper-cli` binary.
pub fn vulkan_dll_alongside(cli_path: &Path) -> bool {
    let Some(dir) = cli_path.parent() else { return false };
    #[cfg(windows)]
    {
        dir.join(VULKAN_DLL_NAME).is_file()
    }
    #[cfg(target_os = "macos")]
    {
        dir.join("libggml-vulkan.dylib").is_file() || dir.join("ggml-metal.dylib").is_file() || dir.join(VULKAN_DLL_NAME).is_file()
    }
    #[cfg(not(any(windows, target_os = "macos")))]
    {
        dir.join("libggml-vulkan.so").is_file() || dir.join(VULKAN_DLL_NAME).is_file()
    }
}

/// Shorten verbose vendor strings (e.g. `"AMD Radeon RX 6700 XT"` -> `"RX 6700 XT"`) for compact buttons.
pub fn compact_gpu_name(raw: &str) -> String {
    let cleaned = raw.replace("(R)", "").replace("(TM)", "").replace("(C)", "").replace("(r)", "").replace("(tm)", "").replace("(c)", "");
    let collapsed: String = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut s = collapsed.as_str();
    if let Some(rest) = s.strip_prefix("NVIDIA GeForce ").or_else(|| s.strip_prefix("NVIDIA ")) {
        s = rest;
    } else if s.starts_with("AMD Radeon RX ") || s.starts_with("AMD Radeon Pro ") {
        if let Some(rest) = s.strip_prefix("AMD Radeon ") {
            s = rest;
        }
    } else if let Some(rest) = s.strip_prefix("AMD ").or_else(|| s.strip_prefix("Intel ")) {
        s = rest;
    }
    s.trim().to_string()
}

/// Evaluate the recommended compute device from Vulkan DLL availability and detected GPU metadata.
pub fn compute_recommendation_for(vulkan_dll_found: bool, gpu: Option<&GpuAdapterInfo>) -> ComputeRecommendation {
    let gpu_display_name = gpu.map(|g| compact_gpu_name(&g.name)).filter(|s| !s.is_empty());
    let capable_gpu = gpu.is_some_and(|g| g.is_discrete && g.dedicated_vram_mb >= RECOMMENDED_GPU_MIN_VRAM_MB);
    if !vulkan_dll_found {
        return ComputeRecommendation {
            recommended: if capable_gpu { SubtitleComputeDevice::Gpu } else { SubtitleComputeDevice::Cpu },
            gpu_available: false,
            reason: ComputeRecommendationReason::VulkanDllMissing,
            gpu_display_name,
        };
    }
    match gpu {
        Some(g) if g.is_discrete && g.dedicated_vram_mb >= RECOMMENDED_GPU_MIN_VRAM_MB => ComputeRecommendation {
            recommended: SubtitleComputeDevice::Gpu,
            gpu_available: true,
            reason: ComputeRecommendationReason::DiscreteGpuSufficientVram,
            gpu_display_name,
        },
        Some(g) if g.is_discrete => ComputeRecommendation {
            recommended: SubtitleComputeDevice::Cpu,
            gpu_available: true,
            reason: ComputeRecommendationReason::LowVram,
            gpu_display_name,
        },
        Some(_) => ComputeRecommendation {
            recommended: SubtitleComputeDevice::Cpu,
            gpu_available: true,
            reason: ComputeRecommendationReason::IntegratedGpu,
            gpu_display_name,
        },
        None => ComputeRecommendation {
            recommended: SubtitleComputeDevice::Cpu,
            gpu_available: false,
            reason: ComputeRecommendationReason::NoGpuDetected,
            gpu_display_name: None,
        },
    }
}

pub fn find_model(id: &str) -> Option<&'static WhisperModelSpec> {
    let norm = id.trim().to_ascii_lowercase();
    WHISPER_MODELS.iter().find(|m| m.id == norm || m.file_name.eq_ignore_ascii_case(&norm))
}

pub fn model_path_in(models_dir: &Path, spec: &WhisperModelSpec) -> PathBuf {
    models_dir.join(spec.file_name)
}

pub fn is_model_installed(models_dir: &Path, spec: &WhisperModelSpec) -> bool {
    let p = model_path_in(models_dir, spec);
    std::fs::metadata(p).is_ok_and(|m| m.is_file() && m.len() > 0)
}

pub fn expected_bin_dir() -> Option<PathBuf> {
    crate::autosave::default_data_dir().map(|d| d.join("bin"))
}

/// Directory where the built-in one-click downloader installs `whisper-cli.exe` and its DLLs
/// (`%APPDATA%\Orpheus\bin` on Windows, `<data_dir>/bin` otherwise).
pub fn install_bin_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    if let Some(appdata) = std::env::var_os("APPDATA").filter(|v| !v.is_empty()).map(PathBuf::from) {
        return Some(appdata.join("Orpheus").join("bin"));
    }
    expected_bin_dir()
}

/// Resolve `whisper-cli.exe` using `exe_dir` first, then `<data_dir>/bin/`, plus `%APPDATA%\Orpheus\bin\` on Windows.
pub fn resolve_whisper_cli_with(exe_dir: Option<&Path>, data_dir: Option<&Path>) -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("ORPHEUS_WHISPER_CLI").filter(|v| !v.is_empty()).map(PathBuf::from)
        && p.is_file()
    {
        return Some(p);
    }
    if let Some(dir) = exe_dir {
        let cand = dir.join(WHISPER_CLI_NAME);
        if cand.is_file() {
            return Some(cand);
        }
    }
    if let Some(dir) = data_dir {
        let cand = dir.join("bin").join(WHISPER_CLI_NAME);
        if cand.is_file() {
            return Some(cand);
        }
    }
    #[cfg(windows)]
    if let Some(appdata) = std::env::var_os("APPDATA").filter(|v| !v.is_empty()).map(PathBuf::from) {
        let cand = appdata.join("Orpheus").join("bin").join(WHISPER_CLI_NAME);
        if cand.is_file() {
            return Some(cand);
        }
    }
    None
}

pub fn resolve_whisper_cli() -> Option<PathBuf> {
    let exe_dir = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf));
    let data_dir = crate::autosave::default_data_dir();
    resolve_whisper_cli_with(exe_dir.as_deref(), data_dir.as_deref())
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SubtitleModelStatus {
    pub id: String,
    pub file_name: String,
    pub label: String,
    pub url: String,
    pub approx_mb: u64,
    pub installed: bool,
    pub path: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SubtitleStatus {
    pub cli_found: bool,
    pub cli_path: Option<String>,
    pub vulkan_dll_found: bool,
    pub cpu_cores: usize,
    pub gpu: Option<GpuAdapterInfo>,
    pub recommendation: ComputeRecommendation,
    pub exe_dir: Option<String>,
    pub bin_dir: Option<String>,
    pub models_dir: Option<String>,
    pub models: Vec<SubtitleModelStatus>,
}

/// Inspect sidecar binary availability, hardware acceleration readiness, and downloaded GGML models.
pub fn status() -> SubtitleStatus {
    let exe_dir = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf));
    let data_dir = crate::autosave::default_data_dir();
    let models_dir = crate::transcript::models_dir();
    let cli = resolve_whisper_cli_with(exe_dir.as_deref(), data_dir.as_deref());
    let vulkan_dll_found = cli.as_deref().is_some_and(vulkan_dll_alongside);
    let cpu_cores = host_cpu_cores();
    let gpu = detect_primary_gpu();
    let recommendation = compute_recommendation_for(vulkan_dll_found, gpu.as_ref());
    let models = WHISPER_MODELS
        .iter()
        .map(|spec| {
            let p = models_dir.as_ref().map(|d| model_path_in(d, spec));
            let installed = models_dir.as_ref().is_some_and(|d| is_model_installed(d, spec));
            SubtitleModelStatus {
                id: spec.id.to_string(),
                file_name: spec.file_name.to_string(),
                label: spec.label.to_string(),
                url: spec.url.to_string(),
                approx_mb: spec.approx_mb,
                installed,
                path: p.map(|x| x.to_string_lossy().into_owned()),
            }
        })
        .collect();
    SubtitleStatus {
        cli_found: cli.is_some(),
        cli_path: cli.map(|p| p.to_string_lossy().into_owned()),
        vulkan_dll_found,
        cpu_cores,
        gpu,
        recommendation,
        exe_dir: exe_dir.map(|p| p.to_string_lossy().into_owned()),
        bin_dir: data_dir.map(|d| d.join("bin").to_string_lossy().into_owned()),
        models_dir: models_dir.map(|d| d.to_string_lossy().into_owned()),
        models,
    }
}

/// One timed word extracted from `whisper-cli -ojf` JSON output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WhisperWord {
    pub text: String,
    pub from_ms: i64,
    pub to_ms: i64,
}

/// One parsed Whisper transcription segment with optional merged word timestamps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WhisperSegment {
    pub from_ms: i64,
    pub to_ms: i64,
    pub text: String,
    pub words: Vec<WhisperWord>,
}

/// One timeline-ready subtitle cue relative to sequence time (`start` includes `range_start`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubtitleCue {
    pub start: Tick,
    pub duration: Tick,
    pub text: String,
}

impl SubtitleCue {
    pub fn end(&self) -> Tick {
        self.start + self.duration
    }
}

#[derive(Deserialize)]
struct RawRoot {
    #[serde(default)]
    transcription: Vec<RawSegment>,
}

#[derive(Deserialize)]
struct RawSegment {
    #[serde(default)]
    timestamps: Option<RawTimestamps>,
    #[serde(default)]
    offsets: Option<RawOffsets>,
    #[serde(default)]
    text: String,
    #[serde(default)]
    tokens: Vec<RawToken>,
}

#[derive(Deserialize)]
struct RawTimestamps {
    #[serde(default)]
    from: String,
    #[serde(default)]
    to: String,
}

#[derive(Deserialize)]
struct RawOffsets {
    from: i64,
    to: i64,
}

#[derive(Deserialize)]
struct RawToken {
    #[serde(default)]
    text: String,
    #[serde(default)]
    timestamps: Option<RawTimestamps>,
    #[serde(default)]
    offsets: Option<RawOffsets>,
}

/// Parse `"HH:MM:SS,mmm"` or `"HH:MM:SS.mmm"` timestamp strings into milliseconds.
fn parse_hms_ms(s: &str) -> Option<i64> {
    let trimmed = s.trim();
    let (hms, ms_part) = trimmed.split_once([',', '.'])?;
    let mut parts = hms.split(':');
    let h: i64 = parts.next()?.parse().ok()?;
    let m: i64 = parts.next()?.parse().ok()?;
    let sec: i64 = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    let ms: i64 = ms_part.chars().take(3).collect::<String>().parse().ok()?;
    h.checked_mul(3600)?.checked_add(m.checked_mul(60)?)?.checked_add(sec)?.checked_mul(1000)?.checked_add(ms)
}

fn bounds_ms(offsets: Option<&RawOffsets>, timestamps: Option<&RawTimestamps>) -> Option<(i64, i64)> {
    if let Some(o) = offsets {
        let a = o.from.max(0);
        let b = o.to.max(a);
        return Some((a, b));
    }
    let ts = timestamps?;
    let a = parse_hms_ms(&ts.from)?.max(0);
    let b = parse_hms_ms(&ts.to)?.max(a);
    Some((a, b))
}

fn is_whisper_meta_token(tok: &str) -> bool {
    let t = tok.trim();
    t.is_empty()
        || (t.starts_with("[_") && t.ends_with(']'))
        || (t.starts_with("<|") && t.ends_with("|>"))
        || matches!(t, "[BLANK_AUDIO]" | "[PAUSE]" | "[SILENCE]")
}

fn merge_tokens_to_words(tokens: &[RawToken], seg_from: i64, seg_to: i64) -> Vec<WhisperWord> {
    let mut words: Vec<WhisperWord> = Vec::new();
    for tok in tokens {
        if is_whisper_meta_token(&tok.text) {
            continue;
        }
        let (t_from, t_to) = bounds_ms(tok.offsets.as_ref(), tok.timestamps.as_ref()).unwrap_or((seg_from, seg_to));
        let starts_with_ws = tok.text.chars().next().is_some_and(char::is_whitespace);
        let clean = tok.text.trim();
        if clean.is_empty() {
            continue;
        }
        if starts_with_ws || words.is_empty() {
            words.push(WhisperWord { text: clean.to_string(), from_ms: t_from, to_ms: t_to.max(t_from) });
        } else if let Some(prev) = words.last_mut() {
            prev.text.push_str(clean);
            prev.to_ms = prev.to_ms.max(t_to);
        }
    }
    words
}

fn proportional_words(text: &str, from_ms: i64, to_ms: i64) -> Vec<WhisperWord> {
    let parts: Vec<&str> = text.split_whitespace().collect();
    if parts.is_empty() {
        return Vec::new();
    }
    let span = (to_ms - from_ms).max(1);
    let total_chars: usize = parts.iter().map(|p| p.chars().count().max(1)).sum::<usize>().max(1);
    let mut cur_chars = 0usize;
    let mut out = Vec::with_capacity(parts.len());
    for part in parts {
        let len = part.chars().count().max(1);
        let w_from = from_ms + (span * cur_chars as i64) / total_chars as i64;
        cur_chars += len;
        let w_to = from_ms + (span * cur_chars as i64) / total_chars as i64;
        out.push(WhisperWord { text: part.to_string(), from_ms: w_from, to_ms: w_to.max(w_from) });
    }
    out
}

/// Parse `whisper-cli -ojf` (or `-oj`) JSON bytes into clean segments with word timestamps.
pub fn parse_whisper_json(bytes: &[u8]) -> std::result::Result<Vec<WhisperSegment>, String> {
    let root: RawRoot = serde_json::from_slice(bytes).map_err(|e| format!("invalid whisper JSON output: {e}"))?;
    let mut segments = Vec::new();
    for raw in root.transcription {
        let text = raw.text.trim().to_string();
        if text.is_empty() || is_whisper_meta_token(&text) {
            continue;
        }
        let (from_ms, to_ms) = bounds_ms(raw.offsets.as_ref(), raw.timestamps.as_ref()).unwrap_or((0, 0));
        let mut words = merge_tokens_to_words(&raw.tokens, from_ms, to_ms);
        if words.is_empty() {
            words = proportional_words(&text, from_ms, to_ms);
        }
        if !words.is_empty() {
            segments.push(WhisperSegment { from_ms, to_ms: to_ms.max(from_ms), text, words });
        }
    }
    Ok(segments)
}

fn ends_clause_or_sentence(word: &str) -> bool {
    word.ends_with(['.', '!', '?', ',', ';', ':', '…'])
}

fn wrap_two_lines(words: &[&WhisperWord], max_line_chars: usize) -> String {
    let mut lines = vec![String::new()];
    for w in words {
        let two_lines = lines.len() >= 2;
        let Some(cur) = lines.last_mut() else { break };
        if cur.is_empty() {
            cur.push_str(&w.text);
        } else if cur.chars().count() + 1 + w.text.chars().count() <= max_line_chars || two_lines {
            cur.push(' ');
            cur.push_str(&w.text);
        } else {
            lines.push(w.text.clone());
        }
    }
    lines.join("\n")
}

/// Group parsed segments/words into `SubtitleCue`s positioned at `range_start + timestamp`,
/// snapped to `rate` frames and guaranteed sorted & non-overlapping.
pub fn build_subtitle_cues(segments: &[WhisperSegment], mode: SubtitleLengthMode, range_start: Tick, rate: FrameRate) -> Vec<SubtitleCue> {
    let raw = match mode {
        SubtitleLengthMode::ReelsShort => build_word_chunks(segments, 3, 24, 300, false),
        SubtitleLengthMode::TikTokCenterPop => build_word_chunks(segments, 2, 14, 220, true),
        SubtitleLengthMode::Sentences => build_sentence_chunks(segments),
    };
    snap_non_overlapping(raw, range_start, rate)
}

/// Build 1–2 word uppercase TikTok Center Pop cues from explicit phrases (used by procedural/demo scenes).
pub fn build_tiktok_center_pop_cues_from_phrases(phrases: &[&str], start_ms: i64, step_ms: i64, range_start: Tick, rate: FrameRate) -> Vec<SubtitleCue> {
    let mut words = Vec::new();
    let mut cursor = start_ms.max(0);
    let step = step_ms.max(180);
    for phrase in phrases {
        for token in phrase.split_whitespace() {
            let trimmed = token.trim();
            if trimmed.is_empty() {
                continue;
            }
            words.push(WhisperWord { from_ms: cursor, to_ms: cursor + step, text: trimmed.to_string() });
            cursor += step;
        }
    }
    if words.is_empty() {
        return Vec::new();
    }
    let seg = WhisperSegment { from_ms: start_ms.max(0), to_ms: cursor, text: phrases.join(" "), words };
    build_subtitle_cues(&[seg], SubtitleLengthMode::TikTokCenterPop, range_start, rate)
}

fn build_word_chunks(segments: &[WhisperSegment], max_words: usize, max_chars: usize, pause_break_ms: i64, uppercase: bool) -> Vec<(i64, i64, String)> {
    let all_words: Vec<&WhisperWord> = segments.iter().flat_map(|s| s.words.iter()).collect();
    if all_words.is_empty() {
        return Vec::new();
    }
    let mut chunks = Vec::new();
    let mut start_idx = 0usize;
    let mut cur_chars = 0usize;

    for i in 0..all_words.len() {
        let w = all_words[i];
        let w_len = w.text.chars().count();
        let count = i - start_idx;
        if count > 0 {
            let prev = all_words[i - 1];
            let gap = w.from_ms - prev.to_ms;
            let should_break = count >= max_words || gap >= pause_break_ms || ends_clause_or_sentence(&prev.text) || (cur_chars + 1 + w_len > max_chars);
            if should_break {
                push_word_slice(&mut chunks, &all_words[start_idx..i], false);
                start_idx = i;
                cur_chars = w_len;
                continue;
            }
            cur_chars += 1 + w_len;
        } else {
            cur_chars = w_len;
        }
    }
    if start_idx < all_words.len() {
        push_word_slice(&mut chunks, &all_words[start_idx..], false);
    }
    if uppercase {
        for (_, _, text) in &mut chunks {
            *text = text.to_uppercase();
        }
    }
    chunks
}

fn build_sentence_chunks(segments: &[WhisperSegment]) -> Vec<(i64, i64, String)> {
    const MAX_CHARS: usize = 84;
    const MAX_LINE_CHARS: usize = 42;
    const MAX_DURATION_MS: i64 = 7000;

    let mut chunks = Vec::new();
    for seg in segments {
        let words: Vec<&WhisperWord> = seg.words.iter().collect();
        if words.is_empty() {
            continue;
        }
        let seg_chars: usize = words.iter().map(|w| w.text.chars().count()).sum::<usize>() + words.len().saturating_sub(1);
        let seg_dur = seg.to_ms - seg.from_ms;
        if seg_chars <= MAX_CHARS && seg_dur <= MAX_DURATION_MS {
            let txt = wrap_two_lines(&words, MAX_LINE_CHARS);
            chunks.push((seg.from_ms, seg.to_ms.max(words.last().map_or(seg.to_ms, |w| w.to_ms)), txt));
            continue;
        }
        let mut start_idx = 0usize;
        let mut cur_chars = 0usize;
        for i in 0..words.len() {
            let w_len = words[i].text.chars().count();
            if i > start_idx {
                let prev = words[i - 1];
                let dur = words[i].to_ms - words[start_idx].from_ms;
                let sentence_end = prev.text.ends_with(['.', '!', '?']) && cur_chars >= 20;
                if cur_chars + 1 + w_len > MAX_CHARS || dur > MAX_DURATION_MS || sentence_end {
                    push_word_slice(&mut chunks, &words[start_idx..i], true);
                    start_idx = i;
                    cur_chars = w_len;
                    continue;
                }
                cur_chars += 1 + w_len;
            } else {
                cur_chars = w_len;
            }
        }
        if start_idx < words.len() {
            push_word_slice(&mut chunks, &words[start_idx..], true);
        }
    }
    chunks
}

fn push_word_slice(out: &mut Vec<(i64, i64, String)>, words: &[&WhisperWord], wrap_lines: bool) {
    let (Some(first), Some(last)) = (words.first(), words.last()) else { return };
    let text = if wrap_lines { wrap_two_lines(words, 42) } else { words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ") };
    out.push((first.from_ms, last.to_ms.max(first.from_ms), text));
}

fn snap_non_overlapping(raw: Vec<(i64, i64, String)>, range_start: Tick, rate: FrameRate) -> Vec<SubtitleCue> {
    let fd = rate.frame_duration().max(Tick(1));
    let mut cues: Vec<SubtitleCue> = Vec::with_capacity(raw.len());
    for (from_ms, to_ms, text) in raw {
        if text.trim().is_empty() {
            continue;
        }
        let s_raw = range_start + Tick::from_units(from_ms.max(0), 1000);
        let e_raw = range_start + Tick::from_units(to_ms.max(from_ms.max(0)), 1000);
        let mut s = rate.snap(s_raw).max(Tick::ZERO);
        if let Some(prev) = cues.last() {
            s = s.max(prev.end());
        }
        let mut e = rate.snap(e_raw);
        if e < e_raw {
            e += fd;
        }
        e = e.max(s + fd);
        cues.push(SubtitleCue { start: s, duration: e - s, text });
    }
    cues
}

pub(crate) struct AudioExtractPlan {
    pub(crate) seq_id: ItemId,
    pub(crate) seq_rate: FrameRate,
    pub(crate) range_start: Tick,
    pub(crate) range_end: Tick,
    pub(crate) tracks: Vec<Track>,
}

pub(crate) fn build_audio_plan(cmd: &str, s: &Session, seq: &Sequence, seq_id: ItemId, p: &Value) -> Result<AudioExtractPlan> {
    let selected = with_links(s, &clips_p(s, p));
    let mut tracks: Vec<Track> = Vec::new();
    let mut min_start: Option<Tick> = None;
    let mut max_end: Option<Tick> = None;

    if !selected.is_empty() {
        for tr in seq.audio_tracks.iter().filter(|t| !t.locked) {
            let items: Vec<_> = tr.items.iter().filter(|it| it.enabled && selected.contains(&it.id)).cloned().collect();
            if items.is_empty() {
                continue;
            }
            for it in &items {
                min_start = Some(min_start.map_or(it.start, |m| m.min(it.start)));
                max_end = Some(max_end.map_or(it.end(), |m| m.max(it.end())));
            }
            let mut copy = tr.clone();
            copy.items = items;
            tracks.push(copy);
        }
    }

    if tracks.is_empty()
        && let Some(tr) = seq.audio_tracks.iter().filter(|t| !t.locked).find(|t| t.items.iter().any(|it| it.enabled))
    {
        for it in tr.items.iter().filter(|it| it.enabled) {
            min_start = Some(min_start.map_or(it.start, |m| m.min(it.start)));
            max_end = Some(max_end.map_or(it.end(), |m| m.max(it.end())));
        }
        tracks.push(tr.clone());
    }

    let (Some(start), Some(end)) = (min_start, max_end) else {
        return Err(bad(cmd, "no enabled audio clips found on unlocked tracks"));
    };
    if end <= start {
        return Err(bad(cmd, "selected audio range is empty"));
    }

    Ok(AudioExtractPlan { seq_id, seq_rate: seq.settings.frame_rate, range_start: start, range_end: end, tracks })
}

pub(crate) fn extract_audio_mono(
    plan: &AudioExtractPlan,
    project: &Project,
    sources: &dyn SourceProvider,
    prog: &Progress,
    progress_pct_cap: f64,
) -> std::result::Result<(Vec<f32>, u32), String> {
    let sr = SUBTITLE_SAMPLE_RATE;
    let s0 = plan.range_start.to_units_floor(i64::from(sr));
    let s1 = plan.range_end.to_units_floor(i64::from(sr));
    let total_samples = ((s1 - s0).max(0) as usize).min(MAX_AUDIO_SAMPLES);
    if total_samples == 0 {
        return Err("audio range has zero samples".into());
    }

    let mut mono = vec![0.0_f32; total_samples];
    let mut offset = 0usize;
    while offset < total_samples {
        if prog.cancel.load(Ordering::Relaxed) {
            return Err("Cancelled".into());
        }
        let n = (total_samples - offset).min(AUDIO_CHUNK_FRAMES);
        let chunk_start = s0 + offset as i64;
        for tr in &plan.tracks {
            let buf = filmcraft_render::audio::track_input(project, tr, chunk_start, n, sr, sources);
            let track_mono = filmcraft_speech::downmix(&buf.channels);
            for (dst, &src) in mono[offset..offset + n].iter_mut().zip(track_mono.iter()) {
                *dst += src;
            }
        }
        offset += n;
        let pct = (offset as f64 / total_samples as f64) * progress_pct_cap;
        prog.done.store(pct.round() as u64, Ordering::Relaxed);
    }
    for s in &mut mono {
        if !s.is_finite() {
            *s = 0.0;
        } else {
            *s = s.clamp(-1.0, 1.0);
        }
    }
    Ok((mono, sr))
}

/// Stream a window of `[window_s0 .. window_s0 + window_samples)` 16 kHz mono samples directly to
/// a 16-bit PCM `.wav` file on disk in 10-second chunks (`AUDIO_CHUNK_FRAMES`), keeping peak RAM
/// under ~2 MB regardless of VOD length.
pub(crate) fn stream_audio_window_to_wav(
    plan: &AudioExtractPlan,
    window: (i64, usize),
    project: &Project,
    sources: &dyn SourceProvider,
    wav_path: &Path,
    prog: &Progress,
    pct: (f64, f64),
) -> std::result::Result<u64, String> {
    let (window_s0, window_samples) = window;
    let (pct_start, pct_span) = pct;
    if window_samples == 0 {
        return Err("audio window has zero samples".into());
    }
    let sr = SUBTITLE_SAMPLE_RATE;
    let data_bytes_u64 = (window_samples as u64).saturating_mul(2);
    let data_bytes = u32::try_from(data_bytes_u64).map_err(|_| "WAV window exceeds 4 GiB RIFF limit".to_string())?;
    let riff_size = 36_u32.saturating_add(data_bytes);

    let file = std::fs::File::create(wav_path).map_err(|e| format!("create temp wav ({}): {e}", wav_path.display()))?;
    let mut writer = BufWriter::with_capacity(256 * 1024, file);

    // Write 44-byte 16-bit mono PCM RIFF/WAVE header.
    let byte_rate = sr.saturating_mul(2);
    let mut hdr = [0u8; 44];
    hdr[0..4].copy_from_slice(b"RIFF");
    hdr[4..8].copy_from_slice(&riff_size.to_le_bytes());
    hdr[8..12].copy_from_slice(b"WAVE");
    hdr[12..16].copy_from_slice(b"fmt ");
    hdr[16..20].copy_from_slice(&16_u32.to_le_bytes());
    hdr[20..22].copy_from_slice(&1_u16.to_le_bytes()); // PCM
    hdr[22..24].copy_from_slice(&1_u16.to_le_bytes()); // 1 channel (mono)
    hdr[24..28].copy_from_slice(&sr.to_le_bytes());
    hdr[28..32].copy_from_slice(&byte_rate.to_le_bytes());
    hdr[32..34].copy_from_slice(&2_u16.to_le_bytes()); // block align
    hdr[34..36].copy_from_slice(&16_u16.to_le_bytes()); // bits per sample
    hdr[36..40].copy_from_slice(b"data");
    hdr[40..44].copy_from_slice(&data_bytes.to_le_bytes());
    writer.write_all(&hdr).map_err(|e| format!("write wav header: {e}"))?;

    let mut scratch = vec![0.0_f32; AUDIO_CHUNK_FRAMES.min(window_samples)];
    let mut pcm_bytes = Vec::with_capacity(scratch.len() * 2);
    let mut offset = 0usize;

    while offset < window_samples {
        if prog.cancel.load(Ordering::Relaxed) {
            return Err("Cancelled".into());
        }
        let n = (window_samples - offset).min(AUDIO_CHUNK_FRAMES);
        let chunk_slice = &mut scratch[..n];
        chunk_slice.fill(0.0);
        let chunk_start = window_s0 + offset as i64;
        for tr in &plan.tracks {
            let buf = filmcraft_render::audio::track_input(project, tr, chunk_start, n, sr, sources);
            let track_mono = filmcraft_speech::downmix(&buf.channels);
            for (dst, &src) in chunk_slice.iter_mut().zip(track_mono.iter()) {
                *dst += src;
            }
        }
        pcm_bytes.clear();
        for &s in chunk_slice.iter() {
            let clean = if s.is_finite() { s.clamp(-1.0, 1.0) } else { 0.0 };
            let q = (clean * 32767.0).round().clamp(-32768.0, 32767.0) as i16;
            pcm_bytes.extend_from_slice(&q.to_le_bytes());
        }
        writer.write_all(&pcm_bytes).map_err(|e| format!("write wav chunk: {e}"))?;
        offset += n;
        let pct = pct_start + (offset as f64 / window_samples as f64) * pct_span;
        prog.done.store((pct.round() as u64).min(95), Ordering::Relaxed);
    }
    writer.flush().map_err(|e| format!("flush temp wav: {e}"))?;
    Ok(44 + data_bytes_u64)
}

#[allow(dead_code)]
fn extract_audio_wav(plan: &AudioExtractPlan, project: &Project, sources: &dyn SourceProvider, prog: &Progress) -> std::result::Result<Vec<u8>, String> {
    let (mono, sr) = extract_audio_mono(plan, project, sources, prog, 30.0)?;
    Ok(filmcraft_media::wav::write_wav16(&mono, 1, sr))
}

struct TempDirGuard(PathBuf);

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn create_job_temp_dir(job_id: u64) -> std::result::Result<TempDirGuard, String> {
    use std::hash::{BuildHasher, Hasher};
    let base = crate::private_temp_dir().map_err(|e| format!("temp folder: {e}"))?;
    let nonce = std::collections::hash_map::RandomState::new().build_hasher().finish();
    let dir = base.join(format!("subtitles-{job_id}-{nonce:08x}"));
    std::fs::create_dir_all(&dir).map_err(|e| format!("create temp dir: {e}"))?;
    Ok(TempDirGuard(dir))
}

#[cfg(not(target_arch = "wasm32"))]
fn read_log_tail(path: &Path) -> String {
    let Ok(bytes) = std::fs::read(path) else { return String::new() };
    let tail_start = bytes.len().saturating_sub(4096);
    String::from_utf8_lossy(&bytes[tail_start..]).lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("").trim().to_string()
}

#[cfg(not(target_arch = "wasm32"))]
fn read_log_head(path: &Path, max_lines: usize) -> String {
    let Ok(bytes) = std::fs::read(path) else { return String::new() };
    let head_end = bytes.len().min(4096);
    String::from_utf8_lossy(&bytes[..head_end]).lines().map(str::trim).filter(|l| !l.is_empty()).take(max_lines).collect::<Vec<_>>().join(" | ")
}

fn parse_srt_timestamp_ms(ts: &str) -> Option<i64> {
    let clean = ts.trim().replace(',', ".");
    let mut parts = clean.split(':');
    let h: i64 = parts.next()?.trim().parse().ok()?;
    let m: i64 = parts.next()?.trim().parse().ok()?;
    let sec_part = parts.next()?.trim();
    let (s_str, ms_str) = sec_part.split_once('.').unwrap_or((sec_part, "0"));
    let s: i64 = s_str.parse().ok()?;
    let ms_digits: String = ms_str.chars().take(3).collect();
    let mut ms: i64 = ms_digits.parse().ok()?;
    for _ in ms_digits.len()..3 {
        ms = ms.saturating_mul(10);
    }
    if h < 0 || m < 0 || s < 0 || ms < 0 {
        return None;
    }
    Some(h.saturating_mul(3_600_000).saturating_add(m.saturating_mul(60_000)).saturating_add(s.saturating_mul(1_000)).saturating_add(ms))
}

/// Parse standard `.srt` subtitle output into `WhisperSegment`s as a fallback when JSON is absent.
pub fn parse_whisper_srt(bytes: &[u8]) -> std::result::Result<Vec<WhisperSegment>, String> {
    let raw = String::from_utf8_lossy(bytes);
    let mut segments = Vec::new();
    for block in raw.replace("\r\n", "\n").split("\n\n") {
        let lines: Vec<&str> = block.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
        let Some(arrow_idx) = lines.iter().position(|l| l.contains("-->")) else { continue };
        let Some((from_str, to_str)) = lines[arrow_idx].split_once("-->") else { continue };
        let Some(from_ms) = parse_srt_timestamp_ms(from_str) else { continue };
        let Some(to_ms) = parse_srt_timestamp_ms(to_str) else { continue };
        let text = lines[arrow_idx + 1..].join(" ").trim().to_string();
        if text.is_empty() || is_whisper_meta_token(&text) {
            continue;
        }
        let words = proportional_words(&text, from_ms, to_ms.max(from_ms));
        if !words.is_empty() {
            segments.push(WhisperSegment { from_ms, to_ms: to_ms.max(from_ms), text, words });
        }
    }
    Ok(segments)
}

/// Parse timestamped `[HH:MM:SS.mmm --> HH:MM:SS.mmm] text` lines from `whisper-cli` stdout/log.
pub fn parse_whisper_stdout(bytes: &[u8]) -> Vec<WhisperSegment> {
    let raw = String::from_utf8_lossy(bytes);
    let mut segments = Vec::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        let Some(rest) = trimmed.strip_prefix('[') else { continue };
        let Some((span, text_raw)) = rest.split_once(']') else { continue };
        let Some((from_str, to_str)) = span.split_once("-->") else { continue };
        let Some(from_ms) = parse_srt_timestamp_ms(from_str) else { continue };
        let Some(to_ms) = parse_srt_timestamp_ms(to_str) else { continue };
        let text = text_raw.trim().to_string();
        if text.is_empty() || is_whisper_meta_token(&text) {
            continue;
        }
        let words = proportional_words(&text, from_ms, to_ms.max(from_ms));
        if !words.is_empty() {
            segments.push(WhisperSegment { from_ms, to_ms: to_ms.max(from_ms), text, words });
        }
    }
    segments
}

pub(crate) struct WhisperRunConfig<'a> {
    pub(crate) model_path: &'a Path,
    pub(crate) language: &'a str,
    pub(crate) mode: SubtitleLengthMode,
    pub(crate) compute: SubtitleComputeDevice,
}

pub(crate) struct ResolvedWhisperConfig {
    pub(crate) language: String,
    pub(crate) mode: SubtitleLengthMode,
    pub(crate) cli_path: PathBuf,
    pub(crate) model_path: PathBuf,
    pub(crate) compute: SubtitleComputeDevice,
}

pub(crate) fn resolve_whisper_run(cmd: &str, p: &Value) -> Result<ResolvedWhisperConfig> {
    let language = validate_lang(str_p(p, "language").unwrap_or(DEFAULT_SUBTITLE_LANGUAGE))?;
    let mode = match str_p(p, "mode") {
        Some(m) => SubtitleLengthMode::from_str_param(m).ok_or_else(|| bad(cmd, format!("unknown subtitle mode `{m}`")))?,
        None => SubtitleLengthMode::Sentences,
    };

    let cli_path = match str_p(p, "cliPath") {
        Some(raw) => {
            let pb = PathBuf::from(raw);
            if pb.is_file() {
                pb
            } else {
                return Err(bad(cmd, format!("whisper-cli executable not found at `{raw}`")));
            }
        }
        None => resolve_whisper_cli().ok_or_else(|| {
            let bin = expected_bin_dir().map_or_else(|| "%APPDATA%\\Orpheus\\bin".into(), |d| d.to_string_lossy().into_owned());
            bad(cmd, format!("{WHISPER_CLI_NAME} not found (place it next to Orpheus or in {bin})"))
        })?,
    };

    let model_id = str_p(p, "model").unwrap_or(DEFAULT_SUBTITLE_MODEL);
    let model_path = match find_model(model_id) {
        Some(spec) => {
            let dir = crate::transcript::models_dir().ok_or_else(|| bad(cmd, "no models directory available"))?;
            let path = model_path_in(&dir, spec);
            if !is_model_installed(&dir, spec) {
                return Err(bad(cmd, format!("Whisper model `{}` is not downloaded yet", spec.file_name)));
            }
            path
        }
        None => return Err(bad(cmd, format!("unknown Whisper model `{model_id}`"))),
    };

    let vulkan_dll_found = vulkan_dll_alongside(&cli_path);
    let rec = compute_recommendation_for(vulkan_dll_found, detect_primary_gpu().as_ref());
    let compute = match str_p(p, "compute") {
        Some(raw) if raw.trim().eq_ignore_ascii_case("auto") => rec.recommended,
        Some(raw) => SubtitleComputeDevice::from_str_param(raw).ok_or_else(|| bad(cmd, format!("unknown compute device `{raw}`")))?,
        None => match crate::ai_models::active_execution_mode() {
            crate::ai_models::AiExecutionMode::Auto => rec.recommended,
            crate::ai_models::AiExecutionMode::GpuDirectMl => {
                if rec.gpu_available {
                    SubtitleComputeDevice::Gpu
                } else {
                    SubtitleComputeDevice::Cpu
                }
            }
            crate::ai_models::AiExecutionMode::Cpu => SubtitleComputeDevice::Cpu,
        },
    };

    Ok(ResolvedWhisperConfig { language, mode, cli_path, model_path, compute })
}

/// Build CLI argument list for `whisper-cli.exe` using only standard supported flags
/// (`-m`, `-f`, `-l`, `-oj`, `-of`, and `-ng` when CPU mode is explicitly selected).
pub fn whisper_cli_args(
    model_path: &Path,
    language: &str,
    wav_path: &Path,
    out_prefix: &Path,
    _mode: SubtitleLengthMode,
    compute: SubtitleComputeDevice,
) -> Vec<std::ffi::OsString> {
    let mut args = vec![
        std::ffi::OsString::from("-m"),
        model_path.as_os_str().to_owned(),
        std::ffi::OsString::from("-f"),
        wav_path.as_os_str().to_owned(),
        std::ffi::OsString::from("-l"),
        std::ffi::OsString::from(language),
        std::ffi::OsString::from("-oj"),
        std::ffi::OsString::from("-of"),
        out_prefix.as_os_str().to_owned(),
    ];
    if matches!(compute, SubtitleComputeDevice::Cpu) {
        args.push(std::ffi::OsString::from("-ng"));
    }
    args
}

fn find_whisper_output_by_ext(dir: &Path, preferred: &[&str], ext: &str) -> Option<PathBuf> {
    for name in preferred {
        let p = dir.join(name);
        if p.is_file() {
            return Some(p);
        }
    }
    let rd = std::fs::read_dir(dir).ok()?;
    for entry in rd.flatten() {
        let p = entry.path();
        if p.is_file() && p.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case(ext)) {
            return Some(p);
        }
    }
    None
}

fn find_whisper_json_output(dir: &Path) -> Option<PathBuf> {
    find_whisper_output_by_ext(dir, &["transcript.json", "transcript.json.json"], "json")
}

fn find_whisper_srt_output(dir: &Path) -> Option<PathBuf> {
    find_whisper_output_by_ext(dir, &["transcript.srt", "transcript.srt.srt"], "srt")
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn run_whisper_cli_on_wav_file(
    cli_path: &Path,
    cfg: &WhisperRunConfig<'_>,
    temp_dir: &Path,
    wav_path: &Path,
    prog: &Progress,
    pct_start: f64,
    pct_span: f64,
) -> std::result::Result<Vec<WhisperSegment>, String> {
    use std::process::{Command, Stdio};

    let wav_meta = std::fs::metadata(wav_path).map_err(|e| format!("stat temp wav: {e}"))?;
    if wav_meta.len() <= 44 {
        return Err("Extracted 16kHz mono WAV file on disk is empty (<= 44 bytes)".into());
    }
    let out_prefix = temp_dir.join("transcript");
    let log_path = temp_dir.join("whisper.log");
    for old in ["transcript.json", "transcript.json.json", "transcript.srt", "transcript.srt.srt", "whisper.log"] {
        let _ = std::fs::remove_file(temp_dir.join(old));
    }

    let log_file = std::fs::File::create(&log_path).map_err(|e| format!("create whisper log: {e}"))?;
    let log_err = log_file.try_clone().map_err(|e| format!("clone log handle: {e}"))?;

    let mut cmd = Command::new(cli_path);
    if let Some(bin_dir) = cli_path.parent().filter(|p| !p.as_os_str().is_empty()) {
        cmd.current_dir(bin_dir);
    }
    cmd.args(whisper_cli_args(cfg.model_path, cfg.language, wav_path, &out_prefix, cfg.mode, cfg.compute))
        .stdin(Stdio::null())
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(log_err));

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    let mut child = cmd.spawn().map_err(|e| format!("failed to start {}: {e}", cli_path.display()))?;
    *prog.status.lock().unwrap_or_else(PoisonError::into_inner) = "Transcribing audio…".into();

    let started = Instant::now();
    let audio_secs = (wav_meta.len().saturating_sub(44) as f64) / (2.0 * f64::from(SUBTITLE_SAMPLE_RATE));
    let max_total_secs = (audio_secs * 5.0).clamp(180.0, 3600.0);
    let stall_limit_secs = 120.0;
    let mut last_log_len = 0u64;
    let mut last_activity = Instant::now();

    let exit_code = loop {
        if prog.cancel.load(Ordering::Relaxed) {
            let _ = child.kill();
            let _ = child.wait();
            return Err("Cancelled".into());
        }
        match child.try_wait() {
            Ok(Some(status)) => {
                let code = status.code();
                if !status.success() {
                    let tail = read_log_tail(&log_path);
                    let code_str = code.map_or_else(|| "unknown".to_string(), |c| c.to_string());
                    let detail = if tail.is_empty() { format!("exit code {code_str}") } else { format!("exit code {code_str}: {tail}") };
                    return Err(format!("whisper-cli failed ({detail})"));
                }
                break code;
            }
            Ok(None) => {
                let elapsed = started.elapsed().as_secs_f64();
                let cur_log_len = std::fs::metadata(&log_path).map_or(last_log_len, |m| m.len());
                if cur_log_len != last_log_len {
                    last_log_len = cur_log_len;
                    last_activity = Instant::now();
                }
                if elapsed > max_total_secs || (elapsed > 45.0 && last_activity.elapsed().as_secs_f64() > stall_limit_secs) {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err("whisper-cli timed out — try switching to CPU mode or the Base model".into());
                }
                let tau = (audio_secs * 0.35).clamp(10.0, 90.0);
                let approx = pct_start + pct_span * (1.0 - (-elapsed / tau).exp());
                prog.done.store((approx.round() as u64).min(95), Ordering::Relaxed);
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(e) => {
                let _ = child.kill();
                return Err(format!("whisper-cli wait failed: {e}"));
            }
        }
    };

    let segments = if let Some(json_path) = find_whisper_json_output(temp_dir) {
        let meta = std::fs::metadata(&json_path).map_err(|e| format!("whisper JSON output unreadable ({}): {e}", json_path.display()))?;
        if meta.len() > MAX_JSON_BYTES {
            return Err("whisper JSON output exceeds 64 MiB limit".into());
        }
        let bytes = std::fs::read(&json_path).map_err(|e| format!("read whisper JSON: {e}"))?;
        parse_whisper_json(&bytes)?
    } else if let Some(srt_path) = find_whisper_srt_output(temp_dir) {
        let bytes = std::fs::read(&srt_path).map_err(|e| format!("read whisper SRT: {e}"))?;
        parse_whisper_srt(&bytes)?
    } else {
        let log_bytes = std::fs::read(&log_path).unwrap_or_default();
        let stdout_segs = parse_whisper_stdout(&log_bytes);
        if !stdout_segs.is_empty() {
            stdout_segs
        } else {
            let head = read_log_head(&log_path, 5);
            let tail = read_log_tail(&log_path);
            let code_str = exit_code.map_or_else(|| "0".to_string(), |c| c.to_string());
            let detail = if !head.is_empty() {
                head
            } else if !tail.is_empty() {
                tail
            } else {
                "no stdout/stderr output".to_string()
            };
            return Err(format!("whisper JSON output missing (exit code {code_str}, dir {}): {detail}", temp_dir.display()));
        }
    };
    Ok(segments)
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn run_whisper_cli(
    cli_path: &Path,
    cfg: &WhisperRunConfig<'_>,
    wav_bytes: &[u8],
    job_id: u64,
    prog: &Progress,
) -> std::result::Result<Vec<WhisperSegment>, String> {
    if wav_bytes.len() <= 44 {
        return Err("Extracted 16kHz mono WAV audio is empty (<= 44 bytes); nothing to transcribe".into());
    }
    let temp_dir = create_job_temp_dir(job_id)?;
    let wav_path = temp_dir.0.join("audio.wav");
    std::fs::write(&wav_path, wav_bytes).map_err(|e| format!("write temp wav: {e}"))?;
    let segs = run_whisper_cli_on_wav_file(cli_path, cfg, &temp_dir.0, &wav_path, prog, 30.0, 65.0)?;
    prog.done.store(100, Ordering::Relaxed);
    Ok(segs)
}

#[cfg(not(target_arch = "wasm32"))]
fn cleanup_window_artifacts(temp_dir: &Path, wav_path: &Path) {
    let _ = std::fs::remove_file(wav_path);
    for artifact in ["transcript.json", "transcript.json.json", "transcript.srt", "transcript.srt.srt", "whisper.log"] {
        let _ = std::fs::remove_file(temp_dir.join(artifact));
    }
}

/// Core chunked transcription loop with per-window fault isolation, drift-free `i128` timestamp
/// offsetting, and guaranteed per-window + directory cleanup on error, cancel, or unwind.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn transcribe_plan_chunked_with<F>(
    plan: &AudioExtractPlan,
    project: &Project,
    sources: &dyn SourceProvider,
    job_id: u64,
    prog: &Progress,
    window_samples_max: usize,
    mut run_window: F,
) -> std::result::Result<Vec<WhisperSegment>, String>
where
    F: FnMut(usize, usize, &Path, &Path, f64, f64) -> std::result::Result<Vec<WhisperSegment>, String>,
{
    let sr = SUBTITLE_SAMPLE_RATE;
    let window_cap = window_samples_max.max(sr as usize);
    let s0 = plan.range_start.to_units_floor(i64::from(sr));
    let s1 = plan.range_end.to_units_floor(i64::from(sr));
    let total_samples = ((s1 - s0).max(0) as usize).min(MAX_AUDIO_SAMPLES);
    if total_samples == 0 {
        return Err("audio range has zero samples".into());
    }

    let temp_dir = create_job_temp_dir(job_id)?;
    let wav_path = temp_dir.0.join("audio.wav");
    let num_windows = total_samples.div_ceil(window_cap).max(1);
    let mut all_segments = Vec::new();
    let mut win_offset_samples = 0usize;
    let mut win_idx = 0usize;
    let mut succeeded_windows = 0usize;
    let mut failed_windows = 0usize;
    let mut first_error: Option<String> = None;

    while win_offset_samples < total_samples {
        if prog.cancel.load(Ordering::Relaxed) {
            cleanup_window_artifacts(&temp_dir.0, &wav_path);
            return Err("Cancelled".into());
        }
        let win_samples = (total_samples - win_offset_samples).min(window_cap);
        let win_s0 = s0.saturating_add(win_offset_samples as i64);
        // Exact i128 rational conversion: 19,200,000 samples / 16,000 Hz = exact 1,200,000 ms/window
        // with zero cumulative floating-point or rounding drift across 36+ windows (12+ hours).
        let win_offset_ms = ((win_offset_samples as i128) * 1000 / i128::from(sr)) as i64;

        let base_pct = (win_idx as f64 / num_windows as f64) * 95.0;
        let win_pct_span = 95.0 / (num_windows as f64);
        if num_windows > 1 {
            *prog.status.lock().unwrap_or_else(PoisonError::into_inner) = format!("Extracting audio window {}/{num_windows}…", win_idx + 1);
        }
        if let Err(err) = stream_audio_window_to_wav(plan, (win_s0, win_samples), project, sources, &wav_path, prog, (base_pct, win_pct_span * 0.30)) {
            cleanup_window_artifacts(&temp_dir.0, &wav_path);
            if prog.cancel.load(Ordering::Relaxed) || err == "Cancelled" {
                return Err("Cancelled".into());
            }
            if num_windows == 1 {
                return Err(err);
            }
            log::warn!("subtitle audio extraction failed on window {}/{num_windows}: {err}", win_idx + 1);
            failed_windows += 1;
            if first_error.is_none() {
                first_error = Some(err);
            }
            win_offset_samples += win_samples;
            win_idx += 1;
            continue;
        }

        if prog.cancel.load(Ordering::Relaxed) {
            cleanup_window_artifacts(&temp_dir.0, &wav_path);
            return Err("Cancelled".into());
        }

        if num_windows > 1 {
            *prog.status.lock().unwrap_or_else(PoisonError::into_inner) = format!("Transcribing window {}/{num_windows}…", win_idx + 1);
        }
        let win_res = run_window(win_idx, num_windows, &temp_dir.0, &wav_path, base_pct + win_pct_span * 0.30, win_pct_span * 0.70);
        cleanup_window_artifacts(&temp_dir.0, &wav_path);

        match win_res {
            Ok(mut win_segs) => {
                succeeded_windows += 1;
                for seg in &mut win_segs {
                    seg.from_ms = seg.from_ms.saturating_add(win_offset_ms);
                    seg.to_ms = seg.to_ms.saturating_add(win_offset_ms);
                    for w in &mut seg.words {
                        w.from_ms = w.from_ms.saturating_add(win_offset_ms);
                        w.to_ms = w.to_ms.saturating_add(win_offset_ms);
                    }
                }
                all_segments.extend(win_segs);
            }
            Err(err) => {
                if prog.cancel.load(Ordering::Relaxed) || err == "Cancelled" {
                    return Err("Cancelled".into());
                }
                if num_windows == 1 {
                    return Err(err);
                }
                // Multi-window VOD resilience: isolate single-window failures (silent/music-only
                // segments, malformed JSON, transient timeout) so hours of completed work survive.
                log::warn!("whisper window {}/{num_windows} failed and was skipped: {err}", win_idx + 1);
                failed_windows += 1;
                if first_error.is_none() {
                    first_error = Some(err);
                }
                *prog.status.lock().unwrap_or_else(PoisonError::into_inner) =
                    format!("Window {}/{num_windows} skipped ({failed_windows} failed) — continuing…", win_idx + 1);
            }
        }
        win_offset_samples += win_samples;
        win_idx += 1;
    }

    if succeeded_windows == 0 {
        return Err(first_error.unwrap_or_else(|| "All transcription windows failed".into()));
    }
    if failed_windows > 0 {
        *prog.status.lock().unwrap_or_else(PoisonError::into_inner) =
            format!("Transcribed {succeeded_windows}/{num_windows} windows ({failed_windows} skipped)");
    }
    prog.done.store(100, Ordering::Relaxed);
    Ok(all_segments)
}

/// Stream-extract and transcribe a sequence's audio range in 20-minute windows (`WHISPER_WINDOW_SAMPLES`),
/// offsetting segment and word timestamps by each window's start time so 12-hour+ VODs transcribe
/// completely with < 60 MB peak RAM.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn transcribe_plan_chunked(
    cli_path: &Path,
    cfg: &WhisperRunConfig<'_>,
    plan: &AudioExtractPlan,
    project: &Project,
    sources: &dyn SourceProvider,
    job_id: u64,
    prog: &Progress,
) -> std::result::Result<Vec<WhisperSegment>, String> {
    transcribe_plan_chunked_with(
        plan,
        project,
        sources,
        job_id,
        prog,
        WHISPER_WINDOW_SAMPLES,
        |_win_idx, _num_windows, temp_dir, wav_path, pct_start, pct_span| {
            run_whisper_cli_on_wav_file(cli_path, cfg, temp_dir, wav_path, prog, pct_start, pct_span)
        },
    )
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn transcribe_plan_chunked(
    _cli_path: &Path,
    _cfg: &WhisperRunConfig<'_>,
    _plan: &AudioExtractPlan,
    _project: &Project,
    _sources: &dyn SourceProvider,
    _job_id: u64,
    _prog: &Progress,
) -> std::result::Result<Vec<WhisperSegment>, String> {
    Err("whisper-cli sidecar execution is unavailable on wasm32".into())
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn run_whisper_cli(
    _cli_path: &Path,
    _cfg: &WhisperRunConfig<'_>,
    _wav_bytes: &[u8],
    _job_id: u64,
    _prog: &Progress,
) -> std::result::Result<Vec<WhisperSegment>, String> {
    Err("whisper-cli sidecar execution is unavailable on wasm32".into())
}

#[cfg(not(target_arch = "wasm32"))]
pub fn download_url_to_path(url: &str, dest: &Path, approx_bytes: u64, prog: &Progress) -> std::result::Result<(), String> {
    let parent = dest.parent().ok_or_else(|| "invalid model path".to_string())?;
    std::fs::create_dir_all(parent).map_err(|e| format!("create models folder: {e}"))?;
    let name = dest.file_name().and_then(|n| n.to_str()).unwrap_or("model.bin");
    let part = parent.join(format!("{name}.part"));

    let tls = ureq::tls::TlsConfig::builder().provider(ureq::tls::TlsProvider::NativeTls).build();
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .tls_config(tls)
        .https_only(!(cfg!(test) && url.starts_with("http://")))
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_recv_response(Some(Duration::from_secs(60)))
        .user_agent(format!("Orpheus/{}", crate::updater::ORPHEUS_VERSION))
        .build()
        .new_agent();

    *prog.status.lock().unwrap_or_else(PoisonError::into_inner) = format!("Connecting for {name}…");
    let resp = agent.get(url).call().map_err(|e| format!("download failed: {e}"))?;
    let content_len = resp.headers().get("Content-Length").and_then(|v| v.to_str().ok()).and_then(|s| s.parse::<u64>().ok()).filter(|&n| n > 0);
    let total = content_len.unwrap_or(approx_bytes.max(1));
    prog.total.store(total, Ordering::Relaxed);

    let file = std::fs::File::create(&part).map_err(|e| format!("create temp model file: {e}"))?;
    let mut writer = BufWriter::with_capacity(128 * 1024, file);
    let mut reader = BufReader::new(resp.into_body().into_reader());
    let mut buf = vec![0u8; 64 * 1024];
    let mut done = 0u64;
    *prog.status.lock().unwrap_or_else(PoisonError::into_inner) = format!("Downloading {name}");

    loop {
        if prog.cancel.load(Ordering::Relaxed) {
            drop(writer);
            let _ = std::fs::remove_file(&part);
            return Err("Cancelled".into());
        }
        let n = match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) => {
                drop(writer);
                let _ = std::fs::remove_file(&part);
                return Err(format!("download read failed: {e}"));
            }
        };
        if let Err(e) = writer.write_all(&buf[..n]) {
            drop(writer);
            let _ = std::fs::remove_file(&part);
            return Err(format!("write model file: {e}"));
        }
        done = done.saturating_add(n as u64);
        prog.done.store(done, Ordering::Relaxed);
        if done > total {
            prog.total.store(done, Ordering::Relaxed);
        }
    }

    writer.flush().map_err(|e| {
        let _ = std::fs::remove_file(&part);
        format!("flush model file: {e}")
    })?;
    drop(writer);
    std::fs::rename(&part, dest).map_err(|e| {
        let _ = std::fs::remove_file(&part);
        format!("finalize model file: {e}")
    })?;
    Ok(())
}

#[cfg(target_arch = "wasm32")]
pub fn download_url_to_path(_url: &str, _dest: &Path, _approx_bytes: u64, _prog: &Progress) -> std::result::Result<(), String> {
    Err("model downloads are unavailable on wasm32".into())
}

/// Completed background subtitle job awaiting main-thread sequence insertion via [`poll`].
pub struct PendingSubtitle {
    pub job: u64,
    pub seq: ItemId,
    pub seq_rate: FrameRate,
    pub range_start: Tick,
    pub language: String,
    pub mode: SubtitleLengthMode,
    pub epoch: u64,
    pub cues: Arc<Mutex<Option<Vec<SubtitleCue>>>>,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

pub fn subtitle_track_style(mode: SubtitleLengthMode) -> CaptionStyle {
    match mode {
        SubtitleLengthMode::ReelsShort => CaptionStyle {
            font: "Inter".into(),
            font_style: "Bold".into(),
            size: 68.0,
            color: [255, 255, 255, 255],
            background: false,
            outline: 5.0,
            outline_color: [0, 0, 0, 255],
            anchor: CaptionAnchor::Bottom,
            margin: 0.14,
            ..Default::default()
        },
        SubtitleLengthMode::TikTokCenterPop => CaptionStyle {
            font: "Inter".into(),
            font_style: "Bold".into(),
            size: 76.0,
            color: [255, 212, 0, 255],
            background: false,
            outline: 7.0,
            outline_color: [0, 0, 0, 255],
            anchor: CaptionAnchor::Middle,
            margin: 0.0,
            ..Default::default()
        },
        SubtitleLengthMode::Sentences => CaptionStyle::default(),
    }
}

/// Detect whether **Game Stream Mode** (vertical split-screen with gameplay on top and FaceCam on
/// the bottom) is active on `seq`, returning the normalized vertical split boundary `split_y`
/// (`0.0..1.0`, top edge of the bottom FaceCam zone, e.g. `0.55` for the 55% / 45% split).
pub fn detect_game_stream_split_y(seq: &Sequence) -> Option<f64> {
    let sh = f64::from(seq.settings.height.max(1));
    for tr in &seq.video_tracks {
        for it in &tr.items {
            if !it.name.ends_with(" (FaceCam)") {
                continue;
            }
            let pos_y = it.effect("motion").map(|m| m.vec2_at("position", Tick::ZERO).y).filter(|y| y.is_finite()).unwrap_or(sh * 0.775);
            if pos_y > sh * 0.5 {
                let cam_cy_norm = (pos_y / sh).clamp(0.55, 0.90);
                let split_y = ((2.0 * cam_cy_norm - 1.0) * 1000.0).round() / 1000.0;
                return Some(split_y.clamp(0.45, 0.75));
            }
        }
    }
    None
}

/// Bottom-anchored caption margin (`fraction of frame height`) that places the subtitle block
/// centered right above the top edge (`split_y`) of the bottom FaceCam in Game Stream Mode.
pub fn game_stream_subtitle_margin(split_y: f64) -> f32 {
    ((1.0 - split_y) as f32).clamp(0.35, 0.45)
}

/// Adjust a [`CaptionStyle`] so subtitles sit centered right above the bottom FaceCam in Game Stream Mode.
pub fn apply_game_stream_subtitle_style(style: &mut CaptionStyle, split_y: f64) {
    if style.anchor != CaptionAnchor::Top {
        style.anchor = CaptionAnchor::Bottom;
        style.margin = style.margin.max(game_stream_subtitle_margin(split_y));
    }
}

/// Scan existing subtitle/caption tracks and graphic text clips on `seq` when Game Stream Mode is
/// active, repositioning any bottom-placed captions or text overlays up to the safe zone centered
/// right above the top edge of the bottom FaceCam. Returns the number of tracks/clips repositioned.
pub fn reposition_subtitles_for_game_stream(seq: &mut Sequence) -> usize {
    let Some(split_y) = detect_game_stream_split_y(seq) else {
        return 0;
    };
    let target_margin = game_stream_subtitle_margin(split_y);
    let mut moved = 0usize;

    for track in &mut seq.caption_tracks {
        if track.style.anchor == CaptionAnchor::Bottom && track.style.margin + 1e-4 < target_margin {
            track.style.margin = target_margin;
            moved += 1;
        }
    }

    let sw = f64::from(seq.settings.width.max(1));
    let sh = f64::from(seq.settings.height.max(1));
    let facecam_top_y = split_y * sh;
    let safe_y = ((split_y - 0.03).clamp(0.45, 0.70) * sh).round();

    for tr in &mut seq.video_tracks {
        if tr.locked {
            continue;
        }
        for it in &mut tr.items {
            if it.name.ends_with(" (FaceCam)") || !it.effects.iter().any(|e| e.effect == "text") {
                continue;
            }
            let mut item_moved = false;
            for ef in &mut it.effects {
                if ef.effect != "text" {
                    continue;
                }
                if let Some(pos_param) = ef.param_mut("position") {
                    let cur = pos_param.value.as_vec2().unwrap_or(Vec2::new(sw * 0.5, sh * 0.5));
                    if cur.y.is_finite() && cur.y > facecam_top_y {
                        let new_x = if cur.x.is_finite() { cur.x } else { sw * 0.5 };
                        pos_param.value = ParamValue::Vec2(Vec2::new(new_x, safe_y));
                        item_moved = true;
                    }
                    for kf in &mut pos_param.keyframes {
                        if let Some(v) = kf.value.as_vec2()
                            && v.y.is_finite()
                            && v.y > facecam_top_y
                        {
                            let new_x = if v.x.is_finite() { v.x } else { sw * 0.5 };
                            kf.value = ParamValue::Vec2(Vec2::new(new_x, safe_y));
                            item_moved = true;
                        }
                    }
                }
            }
            if item_moved {
                moved += 1;
            }
        }
    }

    moved
}

/// Apply generated cues onto a dedicated [`CaptionTrack`] in one undoable sequence edit.
pub fn insert_subtitle_track(s: &mut Session, seq_id: ItemId, language: &str, mode: SubtitleLengthMode, cues: &[SubtitleCue]) -> Result<(TrackId, usize)> {
    if cues.is_empty() {
        s.toast("No speech detected in audio");
        return Ok((TrackId(0), 0));
    }
    let track_name = match mode {
        SubtitleLengthMode::ReelsShort => "AI Subtitles (Reels)",
        SubtitleLengthMode::TikTokCenterPop => "AI Subtitles (TikTok Center Pop)",
        SubtitleLengthMode::Sentences => "AI Subtitles",
    };
    let style = subtitle_track_style(mode);
    let lang = language.to_string();
    let count = cues.len();
    let tid = s.edit("Generate Subtitles", move |pr, st| {
        let tid = TrackId(pr.alloc_id());
        let mut captions = Vec::with_capacity(cues.len());
        for cue in cues {
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
        let q = pr.sequence_mut(seq_id).ok_or(EngineError::NoSequence)?;
        let mut track = CaptionTrack::new(tid, track_name.into(), CaptionFormat::Subtitle);
        track.language = lang;
        track.style = style;
        if let Some(split_y) = detect_game_stream_split_y(q) {
            apply_game_stream_subtitle_style(&mut track.style, split_y);
        }
        track.captions = captions;
        q.caption_tracks.insert(0, track);
        q.check().map_err(EngineError::Other)?;
        st.caption_selection = first_cap.into_iter().collect();
        Ok(tid)
    })?;
    Ok((tid, count))
}

/// Apply finished background subtitle jobs cleanly on the session thread.
pub fn poll(s: &mut Session) {
    let mut i = 0;
    while i < s.subtitle_jobs.len() {
        let job = s.jobs.iter().find(|j| j.id == s.subtitle_jobs[i].job);
        let finished = job.is_none_or(|j| j.progress.finished.load(Ordering::Relaxed));
        let cancelled = job.is_some_and(|j| j.progress.cancel.load(Ordering::Relaxed));
        if !finished {
            i += 1;
            continue;
        }
        let pj = s.subtitle_jobs.remove(i);
        let Some(cues) = lock(&pj.cues).take().filter(|_| !cancelled) else { continue };
        if pj.epoch != s.document_epoch {
            s.error_toast("ai.generateSubtitles", "Generate Subtitles: the project was replaced; nothing was changed");
            continue;
        }
        if let Err(e) = insert_subtitle_track(s, pj.seq, &pj.language, pj.mode, &cues) {
            s.error_toast("ai.generateSubtitles", format!("Generate Subtitles: {e}"));
        }
    }
}

fn validate_lang(raw: &str) -> Result<String> {
    let l = raw.trim();
    if l.is_empty() || l.len() > 16 || !l.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_') {
        return Err(bad("ai.generateSubtitles", format!("invalid language code `{raw}`")));
    }
    Ok(l.to_string())
}

fn status_cmd(_: &mut Session, _: &Value) -> Result<Value> {
    serde_json::to_value(status()).map_err(|e| EngineError::Other(e.to_string()))
}

const WHISPER_LATEST_RELEASE_API_URL: &str = "https://api.github.com/repos/ggml-org/whisper.cpp/releases/latest";
const WHISPER_RELEASES_API_URL: &str = "https://api.github.com/repos/ggml-org/whisper.cpp/releases?per_page=5";
const WHISPER_VULKAN_LATEST_API_URL: &str = "https://api.github.com/repos/jerryshell/whisper.cpp-windows-vulkan-bin/releases/latest";
const WHISPER_VULKAN_FALLBACK_ZIP_URL: &str =
    "https://github.com/jerryshell/whisper.cpp-windows-vulkan-bin/releases/download/v1.0.0/whisper.cpp-windows-vulkan.zip";
const WHISPER_FALLBACK_ZIP_URL: &str = "https://github.com/ggml-org/whisper.cpp/releases/download/b5454/whisper-bin-x64.zip";

/// Parse GitHub Releases API JSON (single release or array of releases) for a Windows x64
/// `whisper.cpp` archive URL, preferring Vulkan-enabled builds if present.
pub fn resolve_whisper_archive_url_from_json(json_str: &str) -> Option<String> {
    resolve_whisper_archive_url_from_json_with_mode(json_str, false)
}

fn resolve_whisper_archive_url_from_json_with_mode(json_str: &str, vulkan_only: bool) -> Option<String> {
    let val = serde_json::from_str::<Value>(json_str).ok()?;
    let releases: Vec<&Value> = match val.as_array() {
        Some(arr) => arr.iter().collect(),
        None => vec![&val],
    };
    let mut fallback: Option<String> = None;
    for rel in releases {
        let Some(assets) = rel.get("assets").and_then(Value::as_array) else { continue };
        for asset in assets {
            let name = asset.get("name").and_then(Value::as_str).unwrap_or("").to_ascii_lowercase();
            let url = asset.get("browser_download_url").and_then(Value::as_str).unwrap_or("");
            if !(url.starts_with("https://") || url.starts_with("http://")) || !name.ends_with(".zip") {
                continue;
            }
            if name.contains("vulkan") && (name.contains("x64") || name.contains("win")) {
                return Some(url.to_string());
            }
            if !vulkan_only && (name == "whisper-bin-x64.zip" || name.contains("bin-x64")) && fallback.is_none() {
                fallback = Some(url.to_string());
            }
        }
    }
    fallback
}

#[cfg(not(target_arch = "wasm32"))]
fn resolve_whisper_archive_url() -> String {
    if let Ok(env_url) = std::env::var("ORPHEUS_WHISPER_BIN_URL")
        && !env_url.trim().is_empty()
    {
        return env_url.trim().to_string();
    }
    let agent = crate::updater::build_http_agent(6, 10);
    // Prefer official ggml-org/whisper.cpp only when it publishes a Vulkan Windows x64 zip;
    // otherwise use the dedicated Windows Vulkan build (includes ggml-vulkan.dll).
    for (api_url, vulkan_only) in
        [(WHISPER_LATEST_RELEASE_API_URL, true), (WHISPER_RELEASES_API_URL, true), (WHISPER_VULKAN_LATEST_API_URL, true), (WHISPER_RELEASES_API_URL, false)]
    {
        if let Ok(resp) = agent.get(api_url).header("Accept", "application/vnd.github+json").call()
            && let Ok(body) = resp.into_body().read_to_string()
            && let Some(url) = resolve_whisper_archive_url_from_json_with_mode(&body, vulkan_only)
        {
            return url;
        }
    }
    if cfg!(windows) { WHISPER_VULKAN_FALLBACK_ZIP_URL.to_string() } else { WHISPER_FALLBACK_ZIP_URL.to_string() }
}

#[cfg(not(target_arch = "wasm32"))]
fn collect_bin_files_recursive(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_bin_files_recursive(&p, out);
        } else if p.is_file()
            && let Some(ext) = p.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase())
            && matches!(ext.as_str(), "exe" | "dll" | "so" | "dylib")
        {
            out.push(p);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn download_and_extract_whisper_engine(bin_dir: &Path, prog: &Progress) -> std::result::Result<u64, String> {
    use std::io::{Read, Write};

    std::fs::create_dir_all(bin_dir).map_err(|e| format!("Failed creating {}: {e}", bin_dir.display()))?;
    *lock(&prog.status) = "Querying whisper.cpp release…".into();
    let url = resolve_whisper_archive_url();

    *lock(&prog.status) = "Downloading Whisper AI engine…".into();
    let agent = crate::updater::build_http_agent(10, 180);
    let resp = agent.get(&url).call().map_err(|e| format!("Engine download failed: {e}"))?;
    let total_len = resp.headers().get("content-length").and_then(|v| v.to_str().ok()).and_then(|s| s.trim().parse::<u64>().ok()).unwrap_or(0);
    if total_len > 0 {
        prog.total.store(total_len, Ordering::Relaxed);
    }

    let zip_part = std::env::temp_dir().join("orpheus-whisper-engine.zip");
    let mut file = std::fs::File::create(&zip_part).map_err(|e| format!("Failed creating temp archive: {e}"))?;
    let mut reader = resp.into_body().into_reader().take(256 * 1024 * 1024);
    let mut buf = [0u8; 64 * 1024];
    let mut downloaded = 0u64;
    loop {
        if prog.cancel.load(Ordering::Relaxed) {
            drop(file);
            let _ = std::fs::remove_file(&zip_part);
            return Err("Cancelled".into());
        }
        let n = reader.read(&mut buf).map_err(|e| format!("Failed reading engine archive: {e}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).map_err(|e| format!("Failed writing engine archive: {e}"))?;
        downloaded = downloaded.saturating_add(n as u64);
        prog.done.store(downloaded, Ordering::Relaxed);
    }
    drop(file);
    if downloaded < 64 * 1024 {
        let _ = std::fs::remove_file(&zip_part);
        return Err("Downloaded engine archive appears empty or truncated".into());
    }

    *lock(&prog.status) = "Extracting whisper-cli.exe and DLLs…".into();
    let extract_tmp = std::env::temp_dir().join("orpheus-whisper-extract");
    let _ = std::fs::remove_dir_all(&extract_tmp);
    std::fs::create_dir_all(&extract_tmp).map_err(|e| format!("Failed creating extract directory: {e}"))?;

    let mut tar_cmd = std::process::Command::new("tar");
    tar_cmd.arg("-xf").arg(&zip_part).arg("-C").arg(&extract_tmp);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        tar_cmd.creation_flags(0x0800_0000);
    }
    let out = tar_cmd.output().map_err(|e| format!("Failed launching archive extractor: {e}"))?;
    let _ = std::fs::remove_file(&zip_part);
    if !out.status.success() {
        let _ = std::fs::remove_dir_all(&extract_tmp);
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(format!("Failed extracting engine archive: {}", err.trim()));
    }

    let mut files = Vec::new();
    collect_bin_files_recursive(&extract_tmp, &mut files);
    for src in files {
        if let Some(name) = src.file_name() {
            let dest = bin_dir.join(name);
            let _ = std::fs::remove_file(&dest);
            if std::fs::rename(&src, &dest).is_err() {
                std::fs::copy(&src, &dest).map_err(|e| format!("Failed installing {}: {e}", name.to_string_lossy()))?;
            }
        }
    }
    let _ = std::fs::remove_dir_all(&extract_tmp);

    let target_cli = bin_dir.join(WHISPER_CLI_NAME);
    if !target_cli.is_file() {
        for alt in if cfg!(windows) { ["main.exe", "whisper.exe"] } else { ["main", "whisper"] } {
            let candidate = bin_dir.join(alt);
            if candidate.is_file() {
                std::fs::rename(&candidate, &target_cli).map_err(|e| format!("Failed renaming {alt}: {e}"))?;
                break;
            }
        }
    }
    if !target_cli.is_file() {
        return Err(format!("{WHISPER_CLI_NAME} was not found inside the downloaded archive"));
    }
    Ok(downloaded)
}

fn download_engine_cmd(s: &mut Session, p: &Value) -> Result<Value> {
    let force = bool_p(p, "force").unwrap_or(false);
    if !force && resolve_whisper_cli().is_some_and(|cli| vulkan_dll_alongside(&cli)) {
        return Ok(json!({ "installed": true }));
    }
    let bin_dir = install_bin_dir().ok_or_else(|| bad("ai.downloadSubtitleEngine", "Cannot determine %APPDATA%\\Orpheus\\bin directory"))?;

    let id = next_job_id(s);
    let prog = Arc::new(Progress::default());
    *lock(&prog.status) = "Preparing Whisper AI engine download…".into();
    let result: Arc<Mutex<Option<std::result::Result<Report, String>>>> = Arc::default();
    s.jobs.push(Job { id, label: "Downloading Whisper AI Engine".into(), progress: prog.clone(), result: result.clone() });

    let wait = bool_p(p, "wait").unwrap_or(false);
    let run = crate::export_tools::guard_job(prog.clone(), result.clone(), move || {
        let t0 = web_time::Instant::now();
        #[cfg(not(target_arch = "wasm32"))]
        let r = download_and_extract_whisper_engine(&bin_dir, &prog).map(|bytes| Report {
            path: bin_dir.join(WHISPER_CLI_NAME).to_string_lossy().into_owned(),
            frames: 0,
            seconds: t0.elapsed().as_secs_f64(),
            bytes,
            render_fps: 0.0,
            ..Default::default()
        });
        #[cfg(target_arch = "wasm32")]
        let r: std::result::Result<Report, String> = {
            let _ = (&bin_dir, &t0);
            Err("Downloading whisper-cli is unavailable on wasm32".into())
        };
        if let Err(ref e) = r {
            *lock(&prog.error) = Some(e.clone());
            *lock(&prog.status) = e.clone();
        } else {
            *lock(&prog.status) = "AI Engine ready".into();
        }
        *lock(&result) = Some(r);
        prog.finished.store(true, Ordering::Relaxed);
    });

    if wait || cfg!(target_arch = "wasm32") {
        run();
    } else {
        std::thread::Builder::new().name("orpheus-whisper-engine-download".into()).spawn(run).map_err(|e| EngineError::Other(e.to_string()))?;
    }
    if let Some(Err(e)) = s.jobs.iter().find(|j| j.id == id).and_then(|j| lock(&j.result).clone()) {
        return Err(EngineError::Other(e));
    }
    Ok(json!({ "job": id, "installed": false }))
}

pub(crate) fn next_job_id(s: &Session) -> u64 {
    s.jobs.iter().map(|j| j.id).max().unwrap_or(0) + 1
}

fn download_model_cmd(s: &mut Session, p: &Value) -> Result<Value> {
    let model_id = str_p(p, "model").unwrap_or(DEFAULT_SUBTITLE_MODEL);
    let spec = find_model(model_id).ok_or_else(|| bad("ai.downloadSubtitleModel", format!("unknown Whisper GGML model `{model_id}`")))?;
    let dir = crate::transcript::models_dir().ok_or_else(|| bad("ai.downloadSubtitleModel", "no models directory available"))?;
    let dest = model_path_in(&dir, spec);
    if is_model_installed(&dir, spec) {
        return Ok(json!({"job": null, "installed": true, "model": spec.id, "path": dest.to_string_lossy()}));
    }

    let id = next_job_id(s);
    let prog = Arc::new(Progress::default());
    prog.total.store(spec.approx_bytes, Ordering::Relaxed);
    *lock(&prog.status) = format!("Downloading {}", spec.file_name);
    let result: Arc<Mutex<Option<std::result::Result<Report, String>>>> = Arc::default();
    s.jobs.push(Job { id, label: format!("Downloading {}", spec.file_name), progress: prog.clone(), result: result.clone() });

    let wait = bool_p(p, "wait").unwrap_or(false);
    let url = spec.url.to_string();
    let approx = spec.approx_bytes;
    let file_name = spec.file_name.to_string();
    let run = crate::export_tools::guard_job(prog.clone(), result.clone(), move || {
        let t0 = web_time::Instant::now();
        let r = download_url_to_path(&url, &dest, approx, &prog).map(|()| Report {
            path: dest.to_string_lossy().into_owned(),
            frames: 0,
            seconds: t0.elapsed().as_secs_f64(),
            bytes: std::fs::metadata(&dest).map_or(0, |m| m.len()),
            render_fps: 0.0,
            ..Default::default()
        });
        if let Err(ref e) = r {
            *lock(&prog.error) = Some(e.clone());
            *lock(&prog.status) = e.clone();
        } else {
            *lock(&prog.status) = format!("Downloaded {file_name}");
        }
        *lock(&result) = Some(r);
        prog.finished.store(true, Ordering::Relaxed);
    });

    if wait || cfg!(target_arch = "wasm32") {
        run();
    } else {
        std::thread::Builder::new().name("orpheus-whisper-download".into()).spawn(run).map_err(|e| EngineError::Other(e.to_string()))?;
    }
    if let Some(Err(e)) = s.jobs.iter().find(|j| j.id == id).and_then(|j| lock(&j.result).clone()) {
        return Err(EngineError::Other(e));
    }
    Ok(json!({"job": id, "installed": false, "model": spec.id}))
}

fn generate_subtitles_cmd(s: &mut Session, p: &Value) -> Result<Value> {
    let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;

    let ResolvedWhisperConfig { language, mode, cli_path, model_path, compute } = resolve_whisper_run("ai.generateSubtitles", p)?;

    let plan = build_audio_plan("ai.generateSubtitles", s, seq, seq_id, p)?;
    let id = next_job_id(s);
    let prog = Arc::new(Progress::default());
    prog.total.store(100, Ordering::Relaxed);
    *lock(&prog.status) = "Preparing audio…".into();
    let result: Arc<Mutex<Option<std::result::Result<Report, String>>>> = Arc::default();
    s.jobs.push(Job { id, label: "Generating AI Subtitles".into(), progress: prog.clone(), result: result.clone() });

    let cues_slot: Arc<Mutex<Option<Vec<SubtitleCue>>>> = Arc::default();
    let pending = PendingSubtitle {
        job: id,
        seq: plan.seq_id,
        seq_rate: plan.seq_rate,
        range_start: plan.range_start,
        language: language.clone(),
        mode,
        epoch: s.document_epoch,
        cues: cues_slot.clone(),
    };
    s.subtitle_jobs.push(pending);

    let project = s.project.clone();
    let provider = s.media.provider(project.clone(), s.services.clone()).own_sources();
    let wait = bool_p(p, "wait").unwrap_or(false);
    let range_start = plan.range_start;
    let seq_rate = plan.seq_rate;
    let lang_run = language.clone();

    let run = crate::export_tools::guard_job(prog.clone(), result.clone(), move || {
        let t0 = web_time::Instant::now();
        let outcome = (|| -> std::result::Result<Vec<SubtitleCue>, String> {
            let cfg = WhisperRunConfig { model_path: &model_path, language: &lang_run, mode, compute };
            let segs = transcribe_plan_chunked(&cli_path, &cfg, &plan, &project, &provider, id, &prog)?;
            Ok(build_subtitle_cues(&segs, mode, range_start, seq_rate))
        })();
        let rep = match outcome {
            Ok(cues) => {
                let n = cues.len();
                *lock(&cues_slot) = Some(cues);
                *lock(&prog.status) = format!("Generated {n} subtitles");
                Ok(Report { frames: n as u64, seconds: t0.elapsed().as_secs_f64(), ..Default::default() })
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
        std::thread::Builder::new().name("orpheus-whisper-cli".into()).spawn(run).map_err(|e| EngineError::Other(e.to_string()))?;
    }

    if let Some(Err(e)) = s.jobs.iter().find(|j| j.id == id).and_then(|j| lock(&j.result).clone()) {
        return Err(EngineError::Other(e));
    }

    let mut out = json!({"job": id, "language": language, "mode": mode.as_id(), "compute": compute.as_id()});
    if wait
        && let Some(q) = s.active_sequence()
        && let Some(tr) = q.caption_tracks.first()
    {
        out["track"] = json!(tr.id.0);
        out["captions"] = json!(tr.captions.len());
    }
    Ok(out)
}

pub(crate) fn commands() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "ai.subtitleStatus",
            label: "AI Subtitles Status",
            menu: &[],
            shortcut: None,
            params: "{}",
            enabled: always,
            run: status_cmd,
            journal: false,
            named: crate::commands::NamedKeys::NONE,
        },
        CommandSpec {
            id: "ai.downloadSubtitleEngine",
            label: "Download AI Engine (whisper-cli)",
            menu: &[],
            shortcut: None,
            params: r#"{"force":bool?,"wait":bool?}"#,
            enabled: always,
            run: download_engine_cmd,
            journal: false,
            named: crate::commands::NamedKeys::NONE,
        },
        CommandSpec {
            id: "ai.downloadSubtitleModel",
            label: "Download Subtitle Model",
            menu: &[],
            shortcut: None,
            params: r#"{"model":"large-v3-turbo|base"?,"wait":bool?}"#,
            enabled: always,
            run: download_model_cmd,
            journal: false,
            named: crate::commands::NamedKeys::NONE,
        },
        CommandSpec {
            id: "ai.generateSubtitles",
            label: "Generate Subtitles",
            menu: &["Sequence"],
            shortcut: None,
            params: r#"{"clips":[id]?,"language":"bg|auto|en|es|de|ru"?,"model":"large-v3-turbo|base"?,"mode":"sentences|reels"?,"compute":"gpu|cpu|auto"?,"wait":bool?}"#,
            enabled: has_seq,
            run: generate_subtitles_cmd,
            journal: true,
            named: crate::commands::NamedKeys::CLIP,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whisper_archive_json_prefers_vulkan_and_falls_back_to_win64_zip() {
        let with_vulkan = r#"[{
            "tag_name": "v1.9.5",
            "assets": [
                { "name": "whisper-bin-x64.zip", "browser_download_url": "https://example.com/whisper-bin-x64.zip" },
                { "name": "whisper-vulkan-bin-x64.zip", "browser_download_url": "https://example.com/whisper-vulkan-bin-x64.zip" }
            ]
        }]"#;
        assert_eq!(resolve_whisper_archive_url_from_json(with_vulkan).as_deref(), Some("https://example.com/whisper-vulkan-bin-x64.zip"));

        let cpu_only = r#"[{
            "tag_name": "b5454",
            "assets": [
                { "name": "whisper-bin-Win32.zip", "browser_download_url": "https://example.com/win32.zip" },
                { "name": "whisper-bin-x64.zip", "browser_download_url": "https://example.com/whisper-bin-x64.zip" }
            ]
        }]"#;
        assert_eq!(resolve_whisper_archive_url_from_json(cpu_only).as_deref(), Some("https://example.com/whisper-bin-x64.zip"));
    }

    #[test]
    fn find_whisper_json_output_locates_json_and_double_json() {
        let dir = std::env::temp_dir().join(format!("orpheus-whisper-json-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(find_whisper_json_output(&dir), None);
        let dbl = dir.join("transcript.json.json");
        std::fs::write(&dbl, b"{}").unwrap();
        assert_eq!(find_whisper_json_output(&dir).as_deref(), Some(dbl.as_path()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    const SAMPLE_OJF_BG: &str = r#"{
        "result": { "language": "bg" },
        "transcription": [
            {
                "timestamps": { "from": "00:00:00,100", "to": "00:00:02,400" },
                "offsets": { "from": 100, "to": 2400 },
                "text": " Здравейте приятели как сте днес?",
                "tokens": [
                    { "text": "[_BEG_]", "offsets": { "from": 0, "to": 0 } },
                    { "text": " Здра", "offsets": { "from": 100, "to": 350 } },
                    { "text": "вейте", "offsets": { "from": 350, "to": 700 } },
                    { "text": " приятели", "offsets": { "from": 720, "to": 1200 } },
                    { "text": ",", "offsets": { "from": 1200, "to": 1250 } },
                    { "text": " как", "offsets": { "from": 1300, "to": 1600 } },
                    { "text": " сте", "offsets": { "from": 1620, "to": 1900 } },
                    { "text": " днес", "offsets": { "from": 1920, "to": 2350 } },
                    { "text": "?", "offsets": { "from": 2350, "to": 2400 } },
                    { "text": "[_EOT_]", "offsets": { "from": 2400, "to": 2400 } }
                ]
            },
            {
                "timestamps": { "from": "00:00:03,000", "to": "00:00:05,200" },
                "offsets": { "from": 3000, "to": 5200 },
                "text": " Това работи офлайн."
            }
        ]
    }"#;

    #[test]
    fn parses_whisper_ojf_and_merges_cyrillic_subword_tokens() {
        let segs = parse_whisper_json(SAMPLE_OJF_BG.as_bytes()).unwrap();
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].words.len(), 5);
        assert_eq!(segs[0].words[0].text, "Здравейте");
        assert_eq!((segs[0].words[0].from_ms, segs[0].words[0].to_ms), (100, 700));
        assert_eq!(segs[0].words[1].text, "приятели,");
        assert_eq!(segs[0].words[4].text, "днес?");
        assert_eq!(segs[1].words.len(), 3);
        assert_eq!(segs[1].words[2].text, "офлайн.");
    }

    #[test]
    fn splits_reels_captions_into_1_to_3_words_and_sentences_by_phrase() {
        let segs = parse_whisper_json(SAMPLE_OJF_BG.as_bytes()).unwrap();
        let rate = FrameRate::FPS_30;

        let reels = build_subtitle_cues(&segs, SubtitleLengthMode::ReelsShort, Tick::ZERO, rate);
        assert!(reels.len() >= 3, "expected compact 1-3 word cues, got {reels:?}");
        for (idx, cue) in reels.iter().enumerate() {
            let wc = cue.text.split_whitespace().count();
            assert!((1..=3).contains(&wc), "cue {idx} `{}` had {wc} words", cue.text);
            if idx > 0 {
                assert!(cue.start >= reels[idx - 1].end(), "cues overlapped");
            }
        }
        assert_eq!(reels[0].text, "Здравейте приятели,");
        assert_eq!(reels[1].text, "как сте днес?");
        assert_eq!(reels[2].text, "Това работи офлайн.");

        let std_cues = build_subtitle_cues(&segs, SubtitleLengthMode::Sentences, Tick::ZERO, rate);
        assert_eq!(std_cues.len(), 2);
        assert!(std_cues[0].text.contains("Здравейте"));
        assert!(std_cues[1].text.contains("Това работи офлайн."));
        assert!(std_cues[1].start >= std_cues[0].end());
    }

    #[test]
    fn resolves_cli_from_exe_or_data_bin_and_inserts_undoable_track() {
        let dir = tempfile::tempdir().unwrap();
        let exe_dir = dir.path().join("app");
        let data_dir = dir.path().join("appdata");
        std::fs::create_dir_all(&exe_dir).unwrap();
        std::fs::create_dir_all(data_dir.join("bin")).unwrap();

        let bin_cli = data_dir.join("bin").join(WHISPER_CLI_NAME);
        std::fs::write(&bin_cli, b"bin").unwrap();
        assert_eq!(resolve_whisper_cli_with(Some(&exe_dir), Some(&data_dir)), Some(bin_cli));

        let exe_cli = exe_dir.join(WHISPER_CLI_NAME);
        std::fs::write(&exe_cli, b"exe").unwrap();
        assert_eq!(resolve_whisper_cli_with(Some(&exe_dir), Some(&data_dir)), Some(exe_cli));

        let mut s = Session::default();
        s.execute("file.newSequence", json!({"name": "BG Test", "width": 1080, "height": 1920, "fps": 30})).unwrap();
        let seq_id = s.state.active_sequence.unwrap();
        let undo_before = s.history.undo.len();

        let segs = parse_whisper_json(SAMPLE_OJF_BG.as_bytes()).unwrap();
        let cues = build_subtitle_cues(&segs, SubtitleLengthMode::ReelsShort, Tick::from_seconds_f64(2.0), FrameRate::FPS_30);
        let (tid, count) = insert_subtitle_track(&mut s, seq_id, "bg", SubtitleLengthMode::ReelsShort, &cues).unwrap();
        assert_eq!(count, cues.len());
        assert_eq!(s.history.undo.len(), undo_before + 1);

        let q = s.active_sequence().unwrap();
        assert_eq!(q.caption_tracks.len(), 1);
        assert_eq!(q.caption_tracks[0].id, tid);
        assert_eq!(q.caption_tracks[0].language, "bg");
        assert_eq!(q.caption_tracks[0].captions.len(), count);
        assert!((q.caption_tracks[0].captions[0].start.seconds() - 2.1).abs() < 0.05);

        s.execute("edit.undo", json!({})).unwrap();
        assert!(s.active_sequence().unwrap().caption_tracks.is_empty());
    }

    #[test]
    fn extracts_16khz_mono_wav_and_handles_hostile_json_inputs() {
        let sr = 48_000u32;
        let samples: Vec<f32> = (0..sr).map(|i| (i as f32 * 0.05).sin() * 0.4).collect();
        let wav = filmcraft_media::wav::write_wav16(&samples, 1, sr);

        let mut s = Session::default();
        let item = crate::commands::import_bytes(&mut s, "speech.wav", wav.into(), None).unwrap();
        s.state.project_selection = vec![item];
        s.execute("file.newSequenceFromClip", json!({"items": [item.0]})).unwrap();

        let seq_id = s.state.active_sequence.unwrap();
        let seq = s.active_sequence().unwrap();
        let plan = build_audio_plan("ai.generateSubtitles", &s, seq, seq_id, &json!({})).unwrap();
        let provider = s.media.provider(s.project.clone(), s.services.clone());
        let prog = Progress::default();
        let out_wav = extract_audio_wav(&plan, &s.project, &provider, &prog).unwrap();

        assert!(out_wav.starts_with(b"RIFF"));
        assert_eq!(&out_wav[8..12], b"WAVE");
        let channels = u16::from_le_bytes([out_wav[22], out_wav[23]]);
        let sample_rate = u32::from_le_bytes([out_wav[24], out_wav[25], out_wav[26], out_wav[27]]);
        let bits = u16::from_le_bytes([out_wav[34], out_wav[35]]);
        assert_eq!((channels, sample_rate, bits), (1, 16_000, 16));

        for bad_json in [
            b"".as_slice(),
            b"{broken",
            br#"{"transcription": "not-an-array"}"#,
            br#"{"transcription": [{"offsets":{"from":-50,"to":-10},"text":"[BLANK_AUDIO]"}]}"#,
            &SAMPLE_OJF_BG.as_bytes()[..SAMPLE_OJF_BG.len() / 2],
        ] {
            let res = std::panic::catch_unwind(|| parse_whisper_json(bad_json));
            assert!(res.is_ok(), "parser panicked on hostile input");
        }

        // poll drops stale jobs whose project epoch no longer matches without mutating the new project
        let cues = build_subtitle_cues(&parse_whisper_json(SAMPLE_OJF_BG.as_bytes()).unwrap(), SubtitleLengthMode::ReelsShort, Tick::ZERO, FrameRate::FPS_30);
        let slot = Arc::new(Mutex::new(Some(cues)));
        s.subtitle_jobs.push(PendingSubtitle {
            job: 9999,
            seq: seq_id,
            seq_rate: FrameRate::FPS_30,
            range_start: Tick::ZERO,
            language: "bg".into(),
            mode: SubtitleLengthMode::ReelsShort,
            epoch: s.document_epoch.wrapping_sub(1),
            cues: slot,
        });
        poll(&mut s);
        assert!(s.subtitle_jobs.is_empty());
        assert!(s.active_sequence().unwrap().caption_tracks.is_empty());

        let st = s.execute("ai.subtitleStatus", json!({})).unwrap();
        assert_eq!(st["models"].as_array().unwrap().len(), 2);
        assert!(st.get("vulkanDllFound").is_some());
        assert!(st.get("recommendation").is_some());
    }

    #[test]
    fn compute_recommendation_and_cli_args() {
        let rx = GpuAdapterInfo { name: "AMD Radeon RX 6700 XT".into(), dedicated_vram_mb: 12_272, is_discrete: true };
        let rec_gpu = compute_recommendation_for(true, Some(&rx));
        assert_eq!(rec_gpu.recommended, SubtitleComputeDevice::Gpu);
        assert!(rec_gpu.gpu_available);
        assert_eq!(rec_gpu.reason, ComputeRecommendationReason::DiscreteGpuSufficientVram);
        assert_eq!(rec_gpu.gpu_display_name.as_deref(), Some("RX 6700 XT"));

        let missing_dll = compute_recommendation_for(false, Some(&rx));
        assert_eq!(missing_dll.recommended, SubtitleComputeDevice::Gpu);
        assert!(!missing_dll.gpu_available);
        assert_eq!(missing_dll.reason, ComputeRecommendationReason::VulkanDllMissing);

        let low_vram = GpuAdapterInfo { name: "NVIDIA GeForce GTX 1050".into(), dedicated_vram_mb: 2048, is_discrete: true };
        let rec_low = compute_recommendation_for(true, Some(&low_vram));
        assert_eq!(rec_low.recommended, SubtitleComputeDevice::Cpu);
        assert!(rec_low.gpu_available);
        assert_eq!(rec_low.reason, ComputeRecommendationReason::LowVram);
        assert_eq!(rec_low.gpu_display_name.as_deref(), Some("GTX 1050"));

        let igpu = GpuAdapterInfo { name: "AMD Radeon(TM) Graphics".into(), dedicated_vram_mb: 512, is_discrete: false };
        let rec_igpu = compute_recommendation_for(true, Some(&igpu));
        assert_eq!(rec_igpu.recommended, SubtitleComputeDevice::Cpu);
        assert_eq!(rec_igpu.reason, ComputeRecommendationReason::IntegratedGpu);
        assert_eq!(rec_igpu.gpu_display_name.as_deref(), Some("Radeon Graphics"));

        let args_gpu = whisper_cli_args(
            Path::new("model.bin"),
            "bg",
            Path::new("audio.wav"),
            Path::new("transcript"),
            SubtitleLengthMode::Sentences,
            SubtitleComputeDevice::Gpu,
        );
        assert!(!args_gpu.iter().any(|a| a == "-ng" || a == "--no-gpu"));
        assert!(args_gpu.windows(2).any(|w| w[0] == "-m" && w[1] == "model.bin"));
        assert!(args_gpu.windows(2).any(|w| w[0] == "-f" && w[1] == "audio.wav"));
        assert!(args_gpu.windows(2).any(|w| w[0] == "-l" && w[1] == "bg"));
        assert!(args_gpu.iter().any(|a| a == "-oj"));
        assert!(args_gpu.windows(2).any(|w| w[0] == "-of" && w[1] == "transcript"));

        let args_cpu = whisper_cli_args(
            Path::new("model.bin"),
            "auto",
            Path::new("audio.wav"),
            Path::new("transcript"),
            SubtitleLengthMode::ReelsShort,
            SubtitleComputeDevice::Cpu,
        );
        assert!(args_cpu.iter().any(|a| a == "-ng"));
        assert!(args_cpu.iter().any(|a| a == "-oj"));
    }

    #[test]
    fn parses_whisper_srt_and_stdout_fallbacks() {
        let srt = b"1\n00:00:00,500 --> 00:00:02,000\nHello world\n\n";
        let srt_segs = parse_whisper_srt(srt).unwrap();
        assert_eq!(srt_segs.len(), 1);
        assert_eq!(srt_segs[0].text, "Hello world");
        assert_eq!(srt_segs[0].from_ms, 500);
        assert_eq!(srt_segs[0].to_ms, 2000);

        let stdout_segs = parse_whisper_stdout(b"[00:00:01.200 --> 00:00:03.400]   Test line\n");
        assert_eq!(stdout_segs.len(), 1);
        assert_eq!(stdout_segs[0].text, "Test line");
        assert_eq!(stdout_segs[0].from_ms, 1200);
        assert_eq!(stdout_segs[0].to_ms, 3400);
    }

    #[test]
    fn dynamic_hardware_capability_tiers() {
        let rx = GpuAdapterInfo { name: "AMD Radeon RX 6700 XT".into(), dedicated_vram_mb: 12_272, is_discrete: true };
        let low_vram = GpuAdapterInfo { name: "NVIDIA GeForce GTX 1050".into(), dedicated_vram_mb: 2048, is_discrete: true };
        let igpu = GpuAdapterInfo { name: "AMD Radeon(TM) Graphics".into(), dedicated_vram_mb: 512, is_discrete: false };
        let mid_gpu = GpuAdapterInfo { name: "NVIDIA GeForce RTX 3060 Laptop".into(), dedicated_vram_mb: 6144, is_discrete: true };

        assert_eq!(feature_capability_tier(AiFeatureWeight::Lightweight, false, None, 2), HardwareCapabilityTier::Optimal);
        assert_eq!(feature_capability_tier(AiFeatureWeight::NeuralSpeech, true, Some(&rx), 12), HardwareCapabilityTier::Optimal);
        assert_eq!(feature_capability_tier(AiFeatureWeight::HeavyVisual, true, Some(&rx), 12), HardwareCapabilityTier::Optimal);
        assert_eq!(feature_capability_tier(AiFeatureWeight::NeuralSpeech, true, Some(&mid_gpu), 8), HardwareCapabilityTier::Moderate);
        assert_eq!(feature_capability_tier(AiFeatureWeight::HeavyVisual, true, Some(&mid_gpu), 8), HardwareCapabilityTier::Moderate);
        assert_eq!(feature_capability_tier(AiFeatureWeight::NeuralSpeech, false, None, 8), HardwareCapabilityTier::Moderate);
        assert_eq!(feature_capability_tier(AiFeatureWeight::HeavyVisual, true, Some(&low_vram), 8), HardwareCapabilityTier::Slow);
        assert_eq!(feature_capability_tier(AiFeatureWeight::HeavyVisual, true, Some(&igpu), 8), HardwareCapabilityTier::Slow);
    }

    #[test]
    fn multi_window_36_chunk_recovery_zero_drift_and_cleanup() {
        // 1. Verify exact 12-hour (36 x 20-minute windows) i128 timestamp offset math has 0 ms drift.
        let sr = i128::from(SUBTITLE_SAMPLE_RATE);
        for win in 0usize..36 {
            let offset_samples = win * WHISPER_WINDOW_SAMPLES;
            let offset_ms = ((offset_samples as i128) * 1000 / sr) as i64;
            assert_eq!(offset_ms, (win as i64) * 1_200_000, "window {win} drifted");
        }
        let end_12h_ms = (((36usize * WHISPER_WINDOW_SAMPLES) as i128) * 1000 / sr) as i64;
        assert_eq!(end_12h_ms, 43_200_000, "12-hour end must be exact 43,200,000 ms");

        // 2. Build a 36-second audio sequence and run 36 consecutive 1-second windows through
        //    `transcribe_plan_chunked_with`, injecting a hard failure on window #14 (index 13).
        let total_secs = 36u32;
        let samples: Vec<f32> = (0..(SUBTITLE_SAMPLE_RATE * total_secs)).map(|i| (i as f32 * 0.02).sin() * 0.3).collect();
        let wav = filmcraft_media::wav::write_wav16(&samples, 1, SUBTITLE_SAMPLE_RATE);

        let mut s = Session::default();
        let item = crate::commands::import_bytes(&mut s, "vod36.wav", wav.into(), None).unwrap();
        s.state.project_selection = vec![item];
        s.execute("file.newSequenceFromClip", json!({"items": [item.0]})).unwrap();
        let seq_id = s.state.active_sequence.unwrap();
        let seq = s.active_sequence().unwrap();
        let plan = build_audio_plan("ai.generateSubtitles", &s, seq, seq_id, &json!({})).unwrap();
        let provider = s.media.provider(s.project.clone(), s.services.clone());

        let prog = Progress::default();
        let mut observed_temp_dir: Option<PathBuf> = None;
        let mut windows_seen = 0usize;
        let segs = transcribe_plan_chunked_with(
            &plan,
            &s.project,
            &provider,
            77_014,
            &prog,
            SUBTITLE_SAMPLE_RATE as usize,
            |win_idx, num_windows, temp_dir, wav_path, _, _| {
                assert_eq!(num_windows, 36);
                assert!(wav_path.is_file(), "window {win_idx} WAV must exist before runner");
                if observed_temp_dir.is_none() {
                    observed_temp_dir = Some(temp_dir.to_path_buf());
                }
                // Previous window's leftover transcript.json must already be deleted before this window runs.
                assert!(!temp_dir.join("transcript.json").exists(), "stale transcript.json leaked into window {win_idx}");
                std::fs::write(temp_dir.join("transcript.json"), b"{}").unwrap();
                windows_seen += 1;

                // Fault injection on window #14 (0-based index 13): simulates timeout / corrupt JSON.
                if win_idx == 13 {
                    return Err("simulated whisper timeout / malformed JSON on window #14".into());
                }
                Ok(vec![WhisperSegment {
                    from_ms: 100,
                    to_ms: 850,
                    text: format!("Window {}", win_idx + 1),
                    words: vec![WhisperWord { from_ms: 100, to_ms: 850, text: format!("W{}", win_idx + 1) }],
                }])
            },
        )
        .expect("35 of 36 windows should succeed when window #14 fails");

        assert_eq!(windows_seen, 36);
        assert_eq!(segs.len(), 35, "35 surviving windows must produce 35 segments");
        assert!(segs.iter().all(|seg| seg.text != "Window 14"), "failed window #14 must be skipped cleanly");
        assert_eq!(segs.first().unwrap().from_ms, 100);
        assert_eq!(segs.last().unwrap().from_ms, 35_100, "window #36 timestamp must have zero drift");
        assert_eq!(segs.last().unwrap().words[0].to_ms, 35_850);
        let dir_path = observed_temp_dir.expect("temp dir recorded");
        assert!(!dir_path.exists(), "TempDirGuard must remove temp dir after completion");

        // 3. Verify cancellation aborts immediately and deletes temp directory.
        let prog_cancel = Progress::default();
        let mut cancel_temp_dir: Option<PathBuf> = None;
        let mut cancel_runs = 0usize;
        let cancel_res =
            transcribe_plan_chunked_with(&plan, &s.project, &provider, 77_015, &prog_cancel, SUBTITLE_SAMPLE_RATE as usize, |win_idx, _, temp_dir, _, _, _| {
                cancel_temp_dir = Some(temp_dir.to_path_buf());
                cancel_runs += 1;
                if win_idx == 2 {
                    prog_cancel.cancel.store(true, Ordering::Relaxed);
                    return Err("Cancelled".into());
                }
                Ok(vec![])
            });
        assert_eq!(cancel_res.unwrap_err(), "Cancelled");
        assert_eq!(cancel_runs, 3, "must stop immediately on cancel instead of processing remaining windows");
        assert!(!cancel_temp_dir.unwrap().exists(), "temp dir must be cleaned up on cancel");

        // 4. Verify failing ALL windows returns an error (never silently claims success).
        let prog_fail = Progress::default();
        let all_fail = transcribe_plan_chunked_with(&plan, &s.project, &provider, 77_016, &prog_fail, SUBTITLE_SAMPLE_RATE as usize, |_, _, _, _, _, _| {
            Err("broken whisper binary".into())
        });
        assert!(all_fail.unwrap_err().contains("broken whisper binary"));
    }
}
