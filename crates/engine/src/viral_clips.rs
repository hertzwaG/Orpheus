//! Local pure-Rust **AI Viral Clip Finder** (`ai.findViralClips`, `ai.jumpToViralClip`,
//! `ai.extractViralClip`, `ai.viralClipsStatus`).
//!
//! Scores candidate short-form segments (**15–30s**, **30–60s**, or **60–90s**) from the active
//! sequence's Whisper subtitle cues (or sequence transcript words) paired with mono PCM audio
//! energy peaks:
//! - **Hook strength** in the first 3–5 seconds (questions `?`, exclamations `!`, digits/stats,
//!   and high-retention Bulgarian & English opening hooks + vocal energy spike);
//! - **Speech pacing & density** (words per second without dead-air pauses);
//! - **Clean sentence boundaries** (snapping window edges to natural sentence/clause cue bounds
//!   so clips never cut mid-word).
//!
//! If no subtitles exist yet on the active sequence and `autoTranscribe` is enabled (the default),
//! the background job seamlessly runs local `whisper-cli.exe` transcription first, inserts the
//! resulting caption track on the main thread via [`crate::subtitles::insert_subtitle_track`], and
//! populates ranked viral clips in one click.

use std::collections::{HashMap, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, PoisonError};

use filmcraft_export::{Progress, Report};
use filmcraft_project::{ItemId, ItemKind, Label, Marker, MarkerId, MarkerKind, Project, Sequence};
use filmcraft_time::{FrameRate, TICKS_PER_SECOND, Tick, TimeDisplay, format_time};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, NamedKeys, always, bad, bool_p, has_seq, str_p, u64_p};
use crate::subtitles::{self, ResolvedWhisperConfig, SubtitleCue, SubtitleLengthMode, WhisperRunConfig, build_audio_plan, build_subtitle_cues, next_job_id};
use crate::{EngineError, Job, Result, Session};

/// Analysis sample rate matching [`crate::subtitles::SUBTITLE_SAMPLE_RATE`].
const VIRAL_SAMPLE_RATE: u32 = 16_000;
/// Hard safety bound on analysed audio length (24 hours @ 16 kHz) to support 12-hour+ VODs.
const MAX_VIRAL_SAMPLES: usize = 24 * 3600 * VIRAL_SAMPLE_RATE as usize;
/// Energy envelope bin step (100 ms per bin).
const ENERGY_HOP_MS: u64 = 100;
/// Streaming decode chunk size (10 seconds @ 16 kHz = 160,000 samples = 100 energy bins).
const VIRAL_CHUNK_SAMPLES: usize = 10 * VIRAL_SAMPLE_RATE as usize;
/// Maximum ranked clips returned per scan.
const MAX_VIRAL_CLIPS: usize = 5;
/// Maximum cached sequence/preset results retained per session.
const MAX_CACHE_ENTRIES: usize = 16;

/// User-selected short-form target clip length range.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ViralDurationPreset {
    #[default]
    Short15To30,
    Medium30To60,
    Long60To90,
}

impl ViralDurationPreset {
    pub fn from_str_param(s: &str) -> Option<Self> {
        let norm = s.trim().to_ascii_lowercase().replace(['–', '—', '_', ' '], "-");
        match norm.as_str() {
            "15-30s" | "15-30" | "short" | "short15to30" => Some(Self::Short15To30),
            "30-60s" | "30-60" | "medium" | "med" | "medium30to60" => Some(Self::Medium30To60),
            "60-90s" | "60-90" | "long" | "long60to90" => Some(Self::Long60To90),
            _ => None,
        }
    }

    pub fn as_id(self) -> &'static str {
        match self {
            Self::Short15To30 => "15-30s",
            Self::Medium30To60 => "30-60s",
            Self::Long60To90 => "60-90s",
        }
    }

    /// `(min_ms, max_ms, ideal_ms)` target clip window.
    pub fn bounds_ms(self) -> (u64, u64, u64) {
        match self {
            Self::Short15To30 => (15_000, 30_000, 22_000),
            Self::Medium30To60 => (30_000, 60_000, 42_000),
            Self::Long60To90 => (60_000, 90_000, 72_000),
        }
    }
}

/// Subtitle cue normalized to exact timeline ticks + milliseconds for viral scoring.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputCue {
    pub start: Tick,
    pub end: Tick,
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
}

/// Compact cue-layout fingerprint `(start_tick, end_tick, text_hash)`.
pub type CueSignature = Vec<(i64, i64, u64)>;

/// One ranked highlight candidate returned by AI Viral Clip Finder.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ViralClip {
    pub rank: usize,
    pub score: u8,
    pub start: Tick,
    pub end: Tick,
    pub start_ms: u64,
    pub end_ms: u64,
    pub duration_ms: u64,
    pub start_timecode: String,
    pub end_timecode: String,
    pub title: String,
    pub cue_count: usize,
}

/// Complete viral analysis output stored cleanly inside `Session`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ViralClipAnalysis {
    pub clips: Vec<ViralClip>,
    pub total_cues: usize,
    pub analyzed_duration_ms: u64,
}

/// Cached viral clip results scoped to one sequence, caption signature, and duration preset.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ViralClipCache {
    pub seq: ItemId,
    pub cue_sig: CueSignature,
    pub preset: ViralDurationPreset,
    pub seq_rate: FrameRate,
    pub drop_frame: bool,
    pub clips: Vec<ViralClip>,
}

/// Generated Whisper subtitles pending main-thread track insertion on completion.
pub struct GeneratedSubtitlesPayload {
    pub language: String,
    pub mode: SubtitleLengthMode,
    pub cues: Vec<SubtitleCue>,
    pub cue_sig: CueSignature,
}

/// In-flight background viral analysis job applied by [`poll`] when completed.
pub struct PendingViralJob {
    pub job: u64,
    pub seq: ItemId,
    pub seq_rate: FrameRate,
    pub drop_frame: bool,
    pub preset: ViralDurationPreset,
    pub epoch: u64,
    pub existing_cue_sig: CueSignature,
    pub generated_subtitles: Arc<Mutex<Option<GeneratedSubtitlesPayload>>>,
    pub analysis: Arc<Mutex<Option<ViralClipAnalysis>>>,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

fn tick_to_ms(t: Tick) -> u64 {
    if t.0 <= 0 {
        return 0;
    }
    ((i128::from(t.0) * 1000) / i128::from(TICKS_PER_SECOND)).clamp(0, i128::from(u64::MAX)) as u64
}

fn clean_text(raw: &str) -> String {
    raw.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn hash_cue_text(s: &str) -> u64 {
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

pub fn cue_signature(cues: &[InputCue]) -> CueSignature {
    cues.iter().map(|c| (c.start.0, c.end.0, hash_cue_text(&c.text))).collect()
}

/// Convert [`SubtitleCue`]s produced by Whisper into [`InputCue`]s for viral scoring.
pub fn input_cues_from_subtitles(cues: &[SubtitleCue]) -> Vec<InputCue> {
    cues.iter()
        .filter_map(|c| {
            let text = clean_text(&c.text);
            let end = c.end();
            if text.is_empty() || end <= c.start {
                return None;
            }
            let start_ms = tick_to_ms(c.start);
            let end_ms = tick_to_ms(end).max(start_ms + 1);
            Some(InputCue { start: c.start, end, start_ms, end_ms, text })
        })
        .collect()
}

/// Extract normalized cues from the active sequence's caption track, or fallback sequence transcript.
pub fn cues_from_sequence(project: &Project, seq: &Sequence) -> Vec<InputCue> {
    let chosen_track = seq
        .caption_tracks
        .iter()
        .find(|tr| tr.enabled && tr.captions.iter().any(|c| !c.text.trim().is_empty()))
        .or_else(|| seq.caption_tracks.iter().find(|tr| tr.captions.iter().any(|c| !c.text.trim().is_empty())));
    if let Some(tr) = chosen_track {
        let mut out: Vec<InputCue> = tr
            .captions
            .iter()
            .filter_map(|c| {
                let text = clean_text(&filmcraft_project::plain_text(&c.text));
                let end = c.end();
                if text.is_empty() || end <= c.start {
                    return None;
                }
                let start_ms = tick_to_ms(c.start);
                let end_ms = tick_to_ms(end).max(start_ms + 1);
                Some(InputCue { start: c.start, end, start_ms, end_ms, text })
            })
            .collect();
        out.sort_by_key(|c| (c.start, c.end));
        if !out.is_empty() {
            return out;
        }
    }

    // Fallback: assemble sentence/phrase cues from sequence transcript words when no caption track exists yet.
    let words = filmcraft_edit::transcript::sequence_words(seq, &project.transcripts);
    if words.is_empty() {
        return Vec::new();
    }
    let mut cues = Vec::new();
    let mut start_idx = 0usize;
    for i in 0..words.len() {
        let gap_after = words.get(i + 1).is_some_and(|next| next.start - words[i].end >= Tick(TICKS_PER_SECOND / 2));
        let sent_end = words[i].text.trim_end().ends_with(['.', '!', '?', '…']);
        let count = i + 1 - start_idx;
        if sent_end || gap_after || count >= 12 || i + 1 == words.len() {
            let chunk = &words[start_idx..=i];
            let text = chunk.iter().map(|w| w.text.trim()).filter(|t| !t.is_empty()).collect::<Vec<_>>().join(" ");
            let start = chunk.first().map_or(Tick::ZERO, |w| w.start);
            let end = chunk.last().map_or(start, |w| w.end);
            if !text.is_empty() && end > start {
                let start_ms = tick_to_ms(start);
                let end_ms = tick_to_ms(end).max(start_ms + 1);
                cues.push(InputCue { start, end, start_ms, end_ms, text });
            }
            start_idx = i + 1;
        }
    }
    cues
}

/// Whether the active sequence already contains subtitle cues or transcribed speech words.
pub fn sequence_has_subtitles(project: &Project, seq: &Sequence) -> bool {
    !cues_from_sequence(project, seq).is_empty()
}

/// Return cached ranked viral clips for the active sequence when its subtitles and preset still match.
pub fn cached_for_session(s: &Session, preset: ViralDurationPreset) -> Option<&ViralClipCache> {
    let seq_id = s.state.active_sequence?;
    let seq = s.active_sequence()?;
    let cues = cues_from_sequence(&s.project, seq);
    if cues.is_empty() {
        return None;
    }
    let sig = cue_signature(&cues);
    s.viral_clips_cache.iter().find(|c| c.seq == seq_id && c.preset == preset && c.cue_sig == sig)
}

/// Look up any cached viral clips for the active sequence matching `preset`.
pub fn latest_cached_clips(s: &Session, preset: ViralDurationPreset) -> Option<&[ViralClip]> {
    cached_for_session(s, preset).map(|c| c.clips.as_slice())
}

/// Coarse 100 ms RMS energy curve derived from mono PCM samples.
struct AudioEnergyProfile {
    bins: Vec<f64>,
    baseline_rms: f64,
    silence_floor: f64,
    origin_ms: u64,
}

impl AudioEnergyProfile {
    fn from_mono(samples: &[f32], sample_rate: u32, origin_ms: u64) -> Option<Self> {
        let sr = sample_rate.clamp(1_000, 384_000);
        let len = samples.len().min(MAX_VIRAL_SAMPLES);
        let hop = ((sr as usize) / (1000 / ENERGY_HOP_MS as usize)).max(1);
        let n_bins = len / hop;
        if n_bins == 0 {
            return None;
        }
        let mut bins = Vec::with_capacity(n_bins);
        let mut active = Vec::new();
        for b in 0..n_bins {
            let s0 = b * hop;
            let mut sum_sq = 0.0_f64;
            for &raw in &samples[s0..s0 + hop] {
                let v = if raw.is_finite() { f64::from(raw.clamp(-1.0, 1.0)) } else { 0.0 };
                sum_sq += v * v;
            }
            let rms = (sum_sq / hop as f64).sqrt();
            bins.push(rms);
            if rms > 1e-4 {
                active.push(rms);
            }
        }
        if active.is_empty() {
            return None;
        }
        active.sort_by(f64::total_cmp);
        let p75_idx = (active.len().saturating_sub(1) * 3) / 4;
        let baseline_rms = active.get(p75_idx).copied().unwrap_or(0.05).max(1e-4);
        let silence_floor = (baseline_rms * 0.14).max(1e-4);
        Some(Self { bins, baseline_rms, silence_floor, origin_ms })
    }

    /// Compute the 100 ms RMS energy envelope in 10-second streaming chunks directly from
    /// `plan.tracks`, avoiding any full-duration `Vec<f32>` allocation on 12-hour VODs.
    fn from_plan_streaming(
        plan: &subtitles::AudioExtractPlan,
        project: &Project,
        sources: &dyn filmcraft_render::SourceProvider,
        prog: &Progress,
        origin_ms: u64,
        progress_pct_cap: f64,
    ) -> std::result::Result<Option<Self>, String> {
        let sr = VIRAL_SAMPLE_RATE;
        let s0 = plan.range_start.to_units_floor(i64::from(sr));
        let s1 = plan.range_end.to_units_floor(i64::from(sr));
        let total_samples = ((s1 - s0).max(0) as usize).min(MAX_VIRAL_SAMPLES);
        let hop = ((sr as usize) / (1000 / ENERGY_HOP_MS as usize)).max(1);
        let n_bins = total_samples / hop;
        if n_bins == 0 {
            return Ok(None);
        }

        let chunk_samples = (VIRAL_CHUNK_SAMPLES / hop).max(1) * hop;
        let mut scratch = vec![0.0_f32; chunk_samples.min(total_samples)];
        let mut bins = Vec::with_capacity(n_bins);
        let mut active = Vec::new();
        let mut offset = 0usize;

        while offset + hop <= total_samples {
            if prog.cancel.load(Ordering::Relaxed) {
                return Err("Cancelled".into());
            }
            let take = ((total_samples - offset) / hop * hop).min(chunk_samples);
            if take == 0 {
                break;
            }
            let chunk_slice = &mut scratch[..take];
            chunk_slice.fill(0.0);
            let chunk_start = s0 + offset as i64;
            for tr in &plan.tracks {
                let buf = filmcraft_render::audio::track_input(project, tr, chunk_start, take, sr, sources);
                let track_mono = filmcraft_speech::downmix(&buf.channels);
                for (dst, &src) in chunk_slice.iter_mut().zip(track_mono.iter()) {
                    *dst += src;
                }
            }
            for bin_samples in chunk_slice.chunks_exact(hop) {
                let mut sum_sq = 0.0_f64;
                for &raw in bin_samples {
                    let v = if raw.is_finite() { f64::from(raw.clamp(-1.0, 1.0)) } else { 0.0 };
                    sum_sq += v * v;
                }
                let rms = (sum_sq / hop as f64).sqrt();
                bins.push(rms);
                if rms > 1e-4 {
                    active.push(rms);
                }
            }
            offset += take;
            let pct = (offset as f64 / total_samples as f64) * progress_pct_cap;
            prog.done.store(pct.round() as u64, Ordering::Relaxed);
        }

        if active.is_empty() {
            return Ok(None);
        }
        active.sort_by(f64::total_cmp);
        let p75_idx = (active.len().saturating_sub(1) * 3) / 4;
        let baseline_rms = active.get(p75_idx).copied().unwrap_or(0.05).max(1e-4);
        let silence_floor = (baseline_rms * 0.14).max(1e-4);
        Ok(Some(Self { bins, baseline_rms, silence_floor, origin_ms }))
    }

    fn bin_slice(&self, start_ms: u64, end_ms: u64) -> &[f64] {
        if end_ms <= self.origin_ms || self.bins.is_empty() {
            return &[];
        }
        let rel_start = start_ms.saturating_sub(self.origin_ms);
        let rel_end = end_ms.saturating_sub(self.origin_ms);
        let i0 = ((rel_start / ENERGY_HOP_MS) as usize).min(self.bins.len());
        let i1 = (rel_end.div_ceil(ENERGY_HOP_MS) as usize).min(self.bins.len());
        if i1 <= i0 { &[] } else { &self.bins[i0..i1] }
    }

    /// Returns `(hook_energy_0_to_1, dynamics_and_bursts_0_to_1, active_ratio_0_to_1)`.
    fn window_metrics(&self, start_ms: u64, end_ms: u64) -> (f64, f64, f64) {
        let hook_end = end_ms.min(start_ms.saturating_add(3_000));
        let hook_bins = self.bin_slice(start_ms, hook_end);
        let hook_energy = if hook_bins.is_empty() {
            0.5
        } else {
            let max_rms = hook_bins.iter().copied().fold(0.0_f64, f64::max);
            let mean_rms = hook_bins.iter().sum::<f64>() / hook_bins.len() as f64;
            let burst_cnt = hook_bins.iter().filter(|&&r| r >= self.baseline_rms * 1.18).count();
            let burst_ratio = burst_cnt as f64 / hook_bins.len() as f64;
            let combined = 0.45 * mean_rms + 0.45 * max_rms + 0.10 * (burst_ratio * self.baseline_rms * 1.4);
            (combined / self.baseline_rms).clamp(0.0, 1.35) / 1.35
        };

        let all_bins = self.bin_slice(start_ms, end_ms);
        if all_bins.is_empty() {
            return (hook_energy, 0.5, 0.75);
        }
        let mean = all_bins.iter().sum::<f64>() / all_bins.len() as f64;
        let peak = all_bins.iter().copied().fold(0.0_f64, f64::max);
        let active_cnt = all_bins.iter().filter(|&&r| r >= self.silence_floor).count();
        let active_ratio = active_cnt as f64 / all_bins.len() as f64;
        let burst_thresh = self.baseline_rms * 1.22;
        let local_bursts = all_bins.windows(3).filter(|w| w[1] >= burst_thresh && w[1] >= w[0] && w[1] >= w[2]).count();
        let burst_score = ((local_bursts as f64) / 3.0).clamp(0.0, 1.0);
        let intensity = (mean / self.baseline_rms).clamp(0.0, 1.2) / 1.2;
        let contrast = if mean > 1e-5 { ((peak / mean) - 1.0).clamp(0.0, 2.5) / 2.5 } else { 0.0 };
        let dynamics = (0.40 * intensity + 0.35 * contrast + 0.25 * burst_score).clamp(0.0, 1.0);
        (hook_energy, dynamics, active_ratio)
    }
}

/// High-retention opening hook words / prefixes in Bulgarian (Cyrillic) & English.
const HOOK_KEYWORDS_BG: &[&str] = &[
    "как ",
    "защо",
    "кой ",
    "какво",
    "кога",
    "никога",
    "винаги",
    "тайна",
    "истина",
    "грешка",
    "грешки",
    "най-",
    "всъщност",
    "внимание",
    "спри",
    "спрете",
    "гледай",
    "виж",
    "слушай",
    "знаеш ли",
    "знаете ли",
    "оказва се",
    "проблем",
    "резултат",
    "трик",
    "съвет",
    "важно",
    "невероятно",
    "шок",
    "топ ",
    "единствен",
    "задължително",
    "процент",
    "хиляди",
    "милион",
    "губят",
    "печелят",
    "но ",
    "обаче",
    "изненада",
    "обрат",
];

const HOOK_KEYWORDS_EN: &[&str] = &[
    "how ",
    "why ",
    "what ",
    "never",
    "always",
    "secret",
    "truth",
    "mistake",
    "mistakes",
    "stop ",
    "wait ",
    "watch ",
    "listen",
    "actually",
    "imagine",
    "worst",
    "best ",
    "most ",
    "top ",
    "reason",
    "warning",
    "nobody",
    "everyone",
    "percent",
    "million",
    "billion",
    "insane",
    "crazy",
    "hack",
    "trick",
    "but ",
    "however",
    "surprise",
    "plot twist",
    "instead",
    "shocking",
];

const FILLER_OPENERS: &[&str] = &["ъъ", "аа", "мм", "хм", "um", "uh", "er", "erm", "hmm"];

struct CueInfo {
    word_count: usize,
    ends_sentence: bool,
    ends_clause: bool,
    starts_sentence: bool,
    starts_clause: bool,
}

fn ends_with_sentence_punct(s: &str) -> bool {
    s.trim_end_matches(['"', '\'', '»', '”', ')']).ends_with(['.', '!', '?', '…'])
}

fn ends_with_clause_punct(s: &str) -> bool {
    ends_with_sentence_punct(s) || s.trim_end_matches(['"', '\'', '»', '”', ')']).ends_with([',', ';', ':', '—', '–'])
}

fn annotate_cues(cues: &[InputCue]) -> Vec<CueInfo> {
    let n = cues.len();
    let mut info = Vec::with_capacity(n);
    let mut ends_sent = vec![false; n];
    let mut ends_cls = vec![false; n];
    for i in 0..n {
        let gap_next = cues.get(i + 1).map_or(u64::MAX, |nxt| nxt.start_ms.saturating_sub(cues[i].end_ms));
        ends_sent[i] = ends_with_sentence_punct(&cues[i].text) || gap_next >= 350 || i + 1 == n;
        ends_cls[i] = ends_sent[i] || ends_with_clause_punct(&cues[i].text) || gap_next >= 200;
    }
    for i in 0..n {
        let word_count = cues[i].text.split_whitespace().count();
        info.push(CueInfo {
            word_count,
            ends_sentence: ends_sent[i],
            ends_clause: ends_cls[i],
            starts_sentence: i == 0 || ends_sent[i - 1],
            starts_clause: i == 0 || ends_cls[i - 1],
        });
    }
    info
}

fn build_hook_title(cues: &[InputCue], start_idx: usize, end_idx: usize) -> String {
    let mut combined = String::new();
    for c in &cues[start_idx..=end_idx] {
        if !combined.is_empty() {
            combined.push(' ');
        }
        combined.push_str(&c.text);
        if ends_with_sentence_punct(&c.text) || combined.split_whitespace().count() >= 10 || combined.chars().count() >= 58 {
            break;
        }
    }
    let words: Vec<&str> = combined.split_whitespace().collect();
    if words.is_empty() {
        return "Highlight clip".into();
    }
    let mut preview = String::new();
    let mut truncated = false;
    for (idx, w) in words.iter().enumerate() {
        if idx >= 10 {
            truncated = true;
            break;
        }
        let next_chars = preview.chars().count() + usize::from(!preview.is_empty()) + w.chars().count();
        if next_chars > 58 && !preview.is_empty() {
            truncated = true;
            break;
        }
        if !preview.is_empty() {
            preview.push(' ');
        }
        preview.push_str(w);
    }
    if truncated {
        preview = preview.trim_end_matches([',', ';', ':', '-', '—', '–']).to_string();
        preview.push('…');
    }
    preview
}

fn score_hook(cues: &[InputCue], start_idx: usize, end_idx: usize, hook_audio: Option<f64>) -> f64 {
    let win_start = cues[start_idx].start_ms;
    let first_3s_horizon = win_start.saturating_add(3_000);
    let mut hook_text = String::new();
    for idx in start_idx..=end_idx {
        if idx > start_idx && cues[idx].start_ms > first_3s_horizon {
            break;
        }
        if !hook_text.is_empty() {
            hook_text.push(' ');
        }
        hook_text.push_str(&cues[idx].text);
    }
    let lower = format!("{} ", hook_text.to_lowercase());
    let mut score = 0.0_f64;

    let q_count = hook_text.matches('?').count();
    let exc_count = hook_text.matches('!').count();
    if q_count > 0 {
        score += 10.0 + ((q_count - 1).min(2) as f64) * 1.5;
    }
    if exc_count > 0 {
        score += 7.0 + ((exc_count - 1).min(2) as f64) * 1.0;
    }
    if hook_text.chars().any(|c| c.is_ascii_digit()) {
        score += 6.0;
    }

    let mut kw_hits = 0usize;
    for kw in HOOK_KEYWORDS_BG.iter().chain(HOOK_KEYWORDS_EN.iter()) {
        if lower.contains(kw) {
            kw_hits += 1;
        }
    }
    score += (kw_hits as f64 * 3.5).min(12.0);

    if let Some(first_word) = lower.split_whitespace().next() {
        let trimmed = first_word.trim_matches(|c: char| !c.is_alphanumeric());
        if FILLER_OPENERS.contains(&trimmed) {
            score -= 6.0;
        }
    }

    let audio_bonus = hook_audio.unwrap_or(0.5) * 6.0;
    (score + audio_bonus).clamp(0.0, 36.0)
}

#[derive(Clone, Debug)]
struct ScoredCandidate {
    start_idx: usize,
    end_idx: usize,
    start_ms: u64,
    end_ms: u64,
    boundary_rank: u8,
    score_quantized: u16,
    score_u8: u8,
}

fn score_window(cues: &[InputCue], info: &[CueInfo], i: usize, j: usize, ideal_ms: u64, audio: Option<&AudioEnergyProfile>) -> ScoredCandidate {
    let start_ms = cues[i].start_ms;
    let end_ms = cues[j].end_ms.max(start_ms + 1);
    let dur_ms = (end_ms - start_ms).max(1);
    let dur_sec = dur_ms as f64 / 1000.0;

    let total_words: usize = info[i..=j].iter().map(|c| c.word_count).sum();
    let wps = total_words as f64 / dur_sec;

    let mut max_gap_ms = 0_u64;
    let mut long_gap_ms = 0_u64;
    for k in i..j {
        let gap = cues[k + 1].start_ms.saturating_sub(cues[k].end_ms);
        max_gap_ms = max_gap_ms.max(gap);
        if gap > 900 {
            long_gap_ms = long_gap_ms.saturating_add(gap - 900);
        }
    }
    let gap_ratio = (long_gap_ms as f64 / dur_ms as f64).clamp(0.0, 1.0);

    let (hook_audio, dynamics, active_audio_ratio) = audio.map_or((None, 0.5, 0.85), |p| {
        let (h, d, a) = p.window_metrics(start_ms, end_ms);
        (Some(h), d, a)
    });

    // 1. Hook strength (0..=36)
    let hook_pts = score_hook(cues, i, j, hook_audio);

    // 2. Speech pacing & density + low dead-air (0..=28)
    let wps_pts = if wps <= 0.1 {
        0.0
    } else if wps < 2.2 {
        (wps / 2.2) * 14.0
    } else if wps <= 4.3 {
        14.0
    } else {
        (14.0 - (wps - 4.3) * 2.5).max(5.0)
    };
    let max_gap_penalty = if max_gap_ms > 2_000 {
        6.0
    } else if max_gap_ms > 1_200 {
        2.5
    } else {
        0.0
    };
    let flow_pts = ((1.0 - gap_ratio) * 8.0 + active_audio_ratio * 6.0 - max_gap_penalty).clamp(0.0, 14.0);
    let pacing_pts = (wps_pts + flow_pts).clamp(0.0, 28.0);

    // 3. Audio energy & vocal dynamics (0..=14)
    let energy_pts = (dynamics * 14.0).clamp(0.0, 14.0);

    // 4. Clean sentence boundaries & target duration fit (0..=22)
    let start_clean = if info[i].starts_sentence {
        7.0
    } else if info[i].starts_clause {
        3.5
    } else {
        0.0
    };
    let end_clean = if info[j].ends_sentence {
        7.0
    } else if info[j].ends_clause {
        3.5
    } else {
        0.0
    };
    let boundary_rank = u8::from(info[i].starts_sentence) + u8::from(info[j].ends_sentence);
    let ideal_f = ideal_ms.max(1_000) as f64;
    let rel_err = ((dur_ms as f64 - ideal_f).abs() / ideal_f).clamp(0.0, 1.0);
    let dur_fit = (1.0 - rel_err * 0.85) * 8.0;
    let bound_pts = (start_clean + end_clean + dur_fit).clamp(0.0, 22.0);

    let total = (hook_pts + pacing_pts + energy_pts + bound_pts).clamp(0.0, 100.0);
    ScoredCandidate {
        start_idx: i,
        end_idx: j,
        start_ms,
        end_ms,
        boundary_rank,
        score_quantized: (total * 100.0).round().clamp(0.0, 10_000.0) as u16,
        score_u8: total.round().clamp(0.0, 100.0) as u8,
    }
}

fn overlap_ratio(a_start: u64, a_end: u64, b_start: u64, b_end: u64) -> f64 {
    let ov_start = a_start.max(b_start);
    let ov_end = a_end.min(b_end);
    if ov_end <= ov_start {
        return 0.0;
    }
    let min_len = (a_end - a_start).min(b_end - b_start).max(1);
    (ov_end - ov_start) as f64 / min_len as f64
}

fn gather_candidates(cues: &[InputCue], info: &[CueInfo], min_ms: u64, max_ms: u64, ideal_ms: u64, audio: Option<&AudioEnergyProfile>) -> Vec<ScoredCandidate> {
    let n = cues.len();
    let mut sentence_bounded = Vec::new();
    let mut clause_bounded = Vec::new();
    let mut any_bounded = Vec::new();
    for i in 0..n {
        let start_ms = cues[i].start_ms;
        for j in i..n {
            let end_ms = cues[j].end_ms;
            let dur = end_ms.saturating_sub(start_ms);
            if dur > max_ms {
                break;
            }
            if dur < min_ms {
                continue;
            }
            if info[i].starts_sentence && info[j].ends_sentence {
                sentence_bounded.push(score_window(cues, info, i, j, ideal_ms, audio));
            } else if info[i].starts_clause && info[j].ends_clause {
                clause_bounded.push(score_window(cues, info, i, j, ideal_ms, audio));
            } else if sentence_bounded.is_empty() && clause_bounded.is_empty() {
                any_bounded.push(score_window(cues, info, i, j, ideal_ms, audio));
            }
        }
    }
    if !sentence_bounded.is_empty() {
        sentence_bounded
    } else if !clause_bounded.is_empty() {
        clause_bounded
    } else {
        any_bounded
    }
}

/// Context needed to format ranked clips for a sequence.
#[derive(Clone, Copy)]
pub struct ScoreContext {
    pub preset: ViralDurationPreset,
    pub audio_origin_ms: u64,
    pub rate: FrameRate,
    pub drop_frame: bool,
}

/// Score sorted subtitle cues paired with optional mono PCM audio peaks into top ranked [`ViralClip`]s.
pub fn score_viral_clips(cues: &[InputCue], mono_pcm: &[f32], sample_rate: u32, ctx: ScoreContext) -> ViralClipAnalysis {
    let audio = AudioEnergyProfile::from_mono(mono_pcm, sample_rate, ctx.audio_origin_ms);
    score_viral_clips_with_profile(cues, audio.as_ref(), ctx)
}

fn score_viral_clips_with_profile(cues: &[InputCue], audio: Option<&AudioEnergyProfile>, ctx: ScoreContext) -> ViralClipAnalysis {
    let valid_cues: Vec<InputCue> = cues.iter().filter(|c| !c.text.trim().is_empty() && c.end > c.start && c.end_ms > c.start_ms).cloned().collect();
    if valid_cues.is_empty() {
        return ViralClipAnalysis { clips: Vec::new(), total_cues: 0, analyzed_duration_ms: 0 };
    }

    let first_ms = valid_cues.first().map_or(0, |c| c.start_ms);
    let last_ms = valid_cues.last().map_or(0, |c| c.end_ms);
    let span_ms = last_ms.saturating_sub(first_ms);
    let info = annotate_cues(&valid_cues);
    let (min_ms, max_ms, ideal_ms) = ctx.preset.bounds_ms();

    let mut candidates = if span_ms >= min_ms {
        let strict = gather_candidates(&valid_cues, &info, min_ms, max_ms, ideal_ms, audio);
        if strict.is_empty() { gather_candidates(&valid_cues, &info, (min_ms * 65) / 100, (max_ms * 125) / 100, ideal_ms, audio) } else { strict }
    } else if span_ms >= 3_000 {
        let eff_min = ((span_ms * 45) / 100).max(3_000).min(span_ms);
        gather_candidates(&valid_cues, &info, eff_min, max_ms.max(span_ms), ideal_ms.min(span_ms), audio)
    } else {
        vec![score_window(&valid_cues, &info, 0, valid_cues.len() - 1, ideal_ms, audio)]
    };

    if candidates.is_empty() {
        candidates.push(score_window(&valid_cues, &info, 0, valid_cues.len() - 1, ideal_ms, audio));
    }

    candidates.sort_by(|a, b| {
        b.score_quantized
            .cmp(&a.score_quantized)
            .then_with(|| b.boundary_rank.cmp(&a.boundary_rank))
            .then_with(|| a.start_ms.cmp(&b.start_ms))
            .then_with(|| a.end_ms.cmp(&b.end_ms))
    });

    // Prefer clean sentence boundaries first during NMS if we have enough distinct candidates;
    // still permit fallback candidates so short sequences never miss valid highlights.
    let mut picked: Vec<ScoredCandidate> = Vec::with_capacity(MAX_VIRAL_CLIPS);
    for cand in &candidates {
        if picked.len() >= MAX_VIRAL_CLIPS {
            break;
        }
        let overlaps =
            picked.iter().any(|p| overlap_ratio(p.start_ms, p.end_ms, cand.start_ms, cand.end_ms) > 0.20 || p.start_ms.abs_diff(cand.start_ms) < 4_000);
        if !overlaps {
            picked.push(cand.clone());
        }
    }

    picked.sort_by(|a, b| b.score_quantized.cmp(&a.score_quantized).then_with(|| a.start_ms.cmp(&b.start_ms)));

    let clips = picked
        .into_iter()
        .enumerate()
        .map(|(idx, c)| {
            let start = valid_cues[c.start_idx].start;
            let end = valid_cues[c.end_idx].end;
            ViralClip {
                rank: idx + 1,
                score: c.score_u8,
                start,
                end,
                start_ms: c.start_ms,
                end_ms: c.end_ms,
                duration_ms: c.end_ms.saturating_sub(c.start_ms),
                start_timecode: format_time(start, ctx.rate, ctx.drop_frame, TimeDisplay::Timecode, 48_000),
                end_timecode: format_time(end, ctx.rate, ctx.drop_frame, TimeDisplay::Timecode, 48_000),
                title: build_hook_title(&valid_cues, c.start_idx, c.end_idx),
                cue_count: c.end_idx - c.start_idx + 1,
            }
        })
        .collect();

    ViralClipAnalysis { clips, total_cues: valid_cues.len(), analyzed_duration_ms: span_ms }
}

fn upsert_cache(s: &mut Session, entry: ViralClipCache) {
    s.viral_clips_cache.retain(|c| !(c.seq == entry.seq && c.preset == entry.preset));
    if s.viral_clips_cache.len() >= MAX_CACHE_ENTRIES {
        s.viral_clips_cache.remove(0);
    }
    s.viral_clips_cache.push(entry);
}

/// Apply completed background viral-clip jobs cleanly on the session thread.
pub fn poll(s: &mut Session) {
    let mut i = 0;
    while i < s.viral_jobs.len() {
        let job = s.jobs.iter().find(|j| j.id == s.viral_jobs[i].job);
        let finished = job.is_none_or(|j| j.progress.finished.load(Ordering::Relaxed));
        let cancelled = job.is_some_and(|j| j.progress.cancel.load(Ordering::Relaxed));
        if !finished {
            i += 1;
            continue;
        }
        let pj = s.viral_jobs.remove(i);
        let analysis = lock(&pj.analysis).take();
        let gen_subs = lock(&pj.generated_subtitles).take();
        if cancelled {
            continue;
        }
        if pj.epoch != s.document_epoch {
            s.error_toast("ai.findViralClips", "AI Viral Clip Finder: the project was replaced; nothing was changed");
            continue;
        }

        let mut final_sig = pj.existing_cue_sig;
        if let Some(subs) = gen_subs
            && !subs.cues.is_empty()
        {
            match subtitles::insert_subtitle_track(s, pj.seq, &subs.language, subs.mode, &subs.cues) {
                Ok(_) => final_sig = subs.cue_sig,
                Err(e) => {
                    s.error_toast("ai.findViralClips", format!("AI Viral Clip Finder: {e}"));
                    continue;
                }
            }
        }

        let Some(analysis) = analysis else { continue };
        let count = analysis.clips.len();
        upsert_cache(
            s,
            ViralClipCache { seq: pj.seq, cue_sig: final_sig, preset: pj.preset, seq_rate: pj.seq_rate, drop_frame: pj.drop_frame, clips: analysis.clips },
        );
        if count == 0 {
            s.toast("No speech detected in audio");
        }
    }
}

fn parse_preset(p: &Value) -> Result<ViralDurationPreset> {
    match str_p(p, "preset") {
        Some(raw) => ViralDurationPreset::from_str_param(raw).ok_or_else(|| bad("ai.findViralClips", format!("unknown clip preset `{raw}`"))),
        None => Ok(ViralDurationPreset::default()),
    }
}

fn find_viral_clips_cmd(s: &mut Session, p: &Value) -> Result<Value> {
    let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let preset = parse_preset(p)?;
    let force = bool_p(p, "force").unwrap_or(false);

    if !force && let Some(cache) = cached_for_session(s, preset) {
        return Ok(json!({
            "cached": true,
            "preset": cache.preset.as_id(),
            "clips": cache.clips,
        }));
    }

    let seq_rate = seq.settings.frame_rate;
    let drop_frame = seq.settings.drop_frame;
    let existing_cues = cues_from_sequence(&s.project, seq);
    let existing_sig = cue_signature(&existing_cues);
    let auto_transcribe = bool_p(p, "autoTranscribe").unwrap_or(true);

    let (audio_plan, whisper_cfg) = if existing_cues.is_empty() {
        if !auto_transcribe {
            return Err(bad("ai.findViralClips", "no subtitles found on the active sequence"));
        }
        let cfg = subtitles::resolve_whisper_run("ai.findViralClips", p)?;
        let plan = build_audio_plan("ai.findViralClips", s, seq, seq_id, p)?;
        (Some(plan), Some(cfg))
    } else {
        (build_audio_plan("ai.findViralClips", s, seq, seq_id, p).ok(), None)
    };

    let id = next_job_id(s);
    let prog = Arc::new(Progress::default());
    prog.total.store(100, Ordering::Relaxed);
    *lock(&prog.status) = if whisper_cfg.is_some() { "Transcribing audio…".into() } else { "Scoring viral highlights…".into() };
    let result: Arc<Mutex<Option<std::result::Result<Report, String>>>> = Arc::default();
    s.jobs.push(Job { id, label: "AI Viral Clip Finder".into(), progress: prog.clone(), result: result.clone() });

    let gen_slot: Arc<Mutex<Option<GeneratedSubtitlesPayload>>> = Arc::default();
    let analysis_slot: Arc<Mutex<Option<ViralClipAnalysis>>> = Arc::default();
    s.viral_jobs.push(PendingViralJob {
        job: id,
        seq: seq_id,
        seq_rate,
        drop_frame,
        preset,
        epoch: s.document_epoch,
        existing_cue_sig: existing_sig,
        generated_subtitles: gen_slot.clone(),
        analysis: analysis_slot.clone(),
    });

    let project = s.project.clone();
    let provider = s.media.provider(project.clone(), s.services.clone()).own_sources();
    let wait = bool_p(p, "wait").unwrap_or(false);
    let will_transcribe = whisper_cfg.is_some();

    let run = crate::export_tools::guard_job(prog.clone(), result.clone(), move || {
        let t0 = web_time::Instant::now();
        let outcome = run_viral_job(ViralJobWorkerInput {
            id,
            existing_cues,
            audio_plan,
            whisper_cfg,
            preset,
            seq_rate,
            drop_frame,
            project,
            provider,
            prog: prog.clone(),
            gen_slot: gen_slot.clone(),
        });
        let rep = match outcome {
            Ok(analysis) => {
                let count = analysis.clips.len() as u64;
                *lock(&analysis_slot) = Some(analysis);
                *lock(&prog.status) = if count > 0 { format!("Found {count} viral clips") } else { "No speech detected in audio".into() };
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
        std::thread::Builder::new().name("orpheus-viral-clips".into()).spawn(run).map_err(|e| EngineError::Other(e.to_string()))?;
    }

    if let Some(Err(e)) = s.jobs.iter().find(|j| j.id == id).and_then(|j| lock(&j.result).clone()) {
        return Err(EngineError::Other(e));
    }

    let mut out = json!({
        "job": id,
        "preset": preset.as_id(),
        "autoTranscribe": will_transcribe,
    });
    if wait && let Some(cache) = cached_for_session(s, preset) {
        out["clips"] = json!(cache.clips);
        out["count"] = json!(cache.clips.len());
    }
    Ok(out)
}

struct ViralJobWorkerInput {
    id: u64,
    existing_cues: Vec<InputCue>,
    audio_plan: Option<subtitles::AudioExtractPlan>,
    whisper_cfg: Option<ResolvedWhisperConfig>,
    preset: ViralDurationPreset,
    seq_rate: FrameRate,
    drop_frame: bool,
    project: Arc<Project>,
    provider: crate::media_pool::PoolProvider,
    prog: Arc<Progress>,
    gen_slot: Arc<Mutex<Option<GeneratedSubtitlesPayload>>>,
}

fn run_viral_job(inp: ViralJobWorkerInput) -> std::result::Result<ViralClipAnalysis, String> {
    let mut energy_profile: Option<AudioEnergyProfile> = None;
    let mut origin_ms = 0_u64;

    if let Some(plan) = inp.audio_plan.as_ref() {
        let cap = if inp.whisper_cfg.is_some() { 20.0 } else { 65.0 };
        origin_ms = tick_to_ms(plan.range_start);
        energy_profile = AudioEnergyProfile::from_plan_streaming(plan, &inp.project, &inp.provider, &inp.prog, origin_ms, cap)?;
    }

    if inp.prog.cancel.load(Ordering::Relaxed) {
        return Err("Cancelled".into());
    }

    let cues = if let Some(cfg) = inp.whisper_cfg {
        let Some(plan) = inp.audio_plan.as_ref() else {
            return Err("no enabled audio clips found on unlocked tracks".into());
        };
        let run_cfg = WhisperRunConfig { model_path: &cfg.model_path, language: &cfg.language, mode: cfg.mode, compute: cfg.compute };
        let segs = subtitles::transcribe_plan_chunked(&cfg.cli_path, &run_cfg, plan, &inp.project, &inp.provider, inp.id, &inp.prog)?;
        let built = build_subtitle_cues(&segs, cfg.mode, plan.range_start, plan.seq_rate);
        let input_cues = input_cues_from_subtitles(&built);
        let sig = cue_signature(&input_cues);
        *lock(&inp.gen_slot) = Some(GeneratedSubtitlesPayload { language: cfg.language, mode: cfg.mode, cues: built, cue_sig: sig });
        input_cues
    } else {
        inp.existing_cues
    };

    if inp.prog.cancel.load(Ordering::Relaxed) {
        return Err("Cancelled".into());
    }

    *lock(&inp.prog.status) = "Scoring viral highlights…".into();
    inp.prog.done.store(85, Ordering::Relaxed);
    let analyzed = score_viral_clips_with_profile(
        &cues,
        energy_profile.as_ref(),
        ScoreContext { preset: inp.preset, audio_origin_ms: origin_ms, rate: inp.seq_rate, drop_frame: inp.drop_frame },
    );
    inp.prog.done.store(100, Ordering::Relaxed);
    Ok(analyzed)
}

struct ClipTarget {
    rank: usize,
    score: u8,
    start: Tick,
    end: Tick,
    title: String,
}

fn resolve_clip_target(s: &Session, p: &Value, cmd: &str) -> Result<ClipTarget> {
    let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let rate = seq.settings.frame_rate;

    if let (Some(st), Some(en)) = (p.get("start").and_then(Value::as_i64), p.get("end").and_then(Value::as_i64)) {
        let start = rate.snap(Tick(st).max(Tick::ZERO));
        let end = rate.snap(Tick(en));
        if end <= start {
            return Err(bad(cmd, "clip end must be after start"));
        }
        let rank = p.get("rank").and_then(Value::as_u64).unwrap_or(1) as usize;
        let score = p.get("score").and_then(Value::as_u64).unwrap_or(0).min(100) as u8;
        let title = str_p(p, "title").unwrap_or("Highlight clip").trim().to_string();
        return Ok(ClipTarget { rank, score, start, end, title });
    }

    let rank = p.get("rank").and_then(Value::as_u64).ok_or_else(|| bad(cmd, "need `rank` or `start`/`end` ticks"))? as usize;
    let preset = match str_p(p, "preset") {
        Some(raw) => ViralDurationPreset::from_str_param(raw),
        None => None,
    };
    let found = s
        .viral_clips_cache
        .iter()
        .rev()
        .filter(|c| c.seq == seq_id && preset.is_none_or(|pr| c.preset == pr))
        .find_map(|c| c.clips.iter().find(|cl| cl.rank == rank))
        .cloned()
        .ok_or_else(|| bad(cmd, format!("no viral clip #{rank} cached for the active sequence")))?;

    let fd = rate.frame_duration().max(Tick(1));
    let start = rate.snap(found.start).max(Tick::ZERO);
    let end = rate.snap(found.end).max(start + fd);
    Ok(ClipTarget { rank: found.rank, score: found.score, start, end, title: found.title })
}

fn jump_to_viral_clip_cmd(s: &mut Session, p: &Value) -> Result<Value> {
    let target = resolve_clip_target(s, p, "ai.jumpToViralClip")?;
    let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let fd = seq.settings.frame_rate.frame_duration().max(Tick(1));
    let marker_name = format!("Viral #{} ({}/100)", target.rank, target.score);
    let marker_comment = target.title.clone();
    let dur = target.end - target.start;
    let tc = format_time(target.start, seq.settings.frame_rate, seq.settings.drop_frame, TimeDisplay::Timecode, 48_000);

    let marker_id = s.edit_sequence("Jump / Mark Viral Clip", |q, ctx, _| {
        q.mark_in = Some(target.start);
        q.mark_out = Some((target.end - fd).max(target.start));
        let existing = q.markers.iter().find(|m| m.start == target.start && m.duration == dur && m.name == marker_name).map(|m| m.id);
        let id = if let Some(eid) = existing {
            eid
        } else {
            let nid = MarkerId(ctx.alloc());
            q.markers.push(Marker {
                url: String::new(),
                author: String::new(),
                custom_color: None,
                id: nid,
                start: target.start,
                duration: dur,
                name: marker_name.clone(),
                comment: marker_comment,
                kind: MarkerKind::Segmentation,
                color: Label::Magenta,
            });
            q.markers.sort_by_key(|m| m.start);
            nid
        };
        Ok(id)
    })?;

    s.set_playhead(target.start);
    s.toast(format!("Marked {marker_name} at {tc}"));
    Ok(json!({
        "rank": target.rank,
        "sequence": seq_id.0,
        "marker": marker_id.0,
        "start": target.start.0,
        "end": target.end.0,
        "timecode": tc,
    }))
}

fn build_trimmed_sequence(pr: &Project, media: &Arc<crate::MediaPool>, src_seq: &Sequence, start: Tick, end: Tick, next: &mut u64) -> Result<Sequence> {
    let snapshot = Arc::new(pr.clone());
    let snap2 = snapshot.clone();
    let pool = Arc::clone(media);
    let durations = move |id: ItemId| crate::media_duration(&snapshot, &pool, id);
    let starts = move |id: ItemId| crate::media_start(&snap2, id);
    let min = src_seq.settings.frame_rate.frame_duration().max(Tick(1));

    let mut q = src_seq.clone();
    let mut ctx = filmcraft_edit::EditCtx { next_id: next, media_duration: &durations, media_start: &starts, min_duration: min };
    let mut links: HashMap<u64, u64> = HashMap::new();

    for tr in q.all_tracks_mut() {
        // Split end first so cutting start afterwards keeps exact segment alignment.
        filmcraft_edit::split_track_at(tr, end, &mut ctx, &mut links);
        filmcraft_edit::split_track_at(tr, start, &mut ctx, &mut links);
        tr.items.retain(|it| it.start >= start && it.end() <= end);
        for it in &mut tr.items {
            it.start -= start;
        }
        tr.transitions.retain(|t| t.start >= start && t.end() <= end);
        for t in &mut tr.transitions {
            t.start -= start;
        }
        filmcraft_edit::remove_orphan_transitions(tr);
        tr.sort();
    }

    for ct in &mut q.caption_tracks {
        let mut kept = Vec::new();
        for mut cap in ct.captions.drain(..) {
            let cs = cap.start.max(start);
            let ce = cap.end().min(end);
            if ce > cs {
                cap.start = cs - start;
                cap.duration = ce - cs;
                kept.push(cap);
            }
        }
        ct.captions = kept;
        ct.sort();
    }

    let clipped_len = end - start;
    q.markers.retain(|m| m.start < end && (m.start + m.duration) >= start);
    for m in &mut q.markers {
        let ms = m.start.max(start);
        let me = (m.start + m.duration).min(end).max(ms);
        m.start = ms - start;
        m.duration = (me - ms).min(clipped_len);
    }
    q.markers.sort_by_key(|m| m.start);
    q.mark_in = None;
    q.mark_out = None;
    q.work_area = None;
    q.split = Default::default();
    q.check().map_err(EngineError::Other)?;
    Ok(q)
}

fn extract_viral_clip_cmd(s: &mut Session, p: &Value) -> Result<Value> {
    let target = resolve_clip_target(s, p, "ai.extractViralClip")?;
    let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let src_item = s.project.item(seq_id).cloned().ok_or(EngineError::NoSequence)?;
    let ItemKind::Sequence(src_seq) = &src_item.kind else {
        return Err(EngineError::NoSequence);
    };
    let seq_len = src_seq.duration();
    let clamped_start = target.start.max(Tick::ZERO);
    let clamped_end = if seq_len > Tick::ZERO { target.end.min(seq_len) } else { target.end };
    if clamped_end <= clamped_start {
        return Err(bad("ai.extractViralClip", "clip range is outside the sequence duration"));
    }

    let new_name = format!("{} · Viral #{}", src_item.name, target.rank);
    let media = s.media.clone();
    let src_clone = (**src_seq).clone();

    let new_id = s.edit("Extract Viral Clip", |pr, st| {
        let mut next = pr.next_id;
        let q = build_trimmed_sequence(pr, &media, &src_clone, clamped_start, clamped_end, &mut next)?;
        pr.next_id = next;
        let bin = pr.root.parent_of(seq_id).filter(|b| b.0 != 0);
        let id = pr.add_item(&new_name, Label::Magenta, ItemKind::Sequence(Arc::new(q)), bin);
        st.active_sequence = Some(id);
        if !st.open_sequences.contains(&id) {
            st.open_sequences.push(id);
        }
        st.selection.clear();
        st.caption_selection.clear();
        st.project_selection = vec![id];
        st.playheads.insert(id, Tick::ZERO);
        Ok(id)
    })?;

    s.events.push(crate::Event::OpenSequence(new_id));
    s.toast(format!("Extracted {new_name}"));
    Ok(json!({
        "rank": target.rank,
        "sequence": new_id.0,
        "name": new_name,
        "start": clamped_start.0,
        "end": clamped_end.0,
        "duration": (clamped_end - clamped_start).0,
    }))
}

fn status_cmd(s: &mut Session, p: &Value) -> Result<Value> {
    let preset = match str_p(p, "preset") {
        Some(raw) => ViralDurationPreset::from_str_param(raw).unwrap_or_default(),
        None => ViralDurationPreset::default(),
    };
    let running = u64_p(p, "job").is_some_and(|jid| s.jobs.iter().any(|j| j.id == jid && !j.progress.finished.load(Ordering::Relaxed)));
    let has_subs = s.active_sequence().is_some_and(|q| sequence_has_subtitles(&s.project, q));
    if let Some(cache) = cached_for_session(s, preset) {
        Ok(json!({
            "cached": true,
            "running": running,
            "hasSubtitles": has_subs,
            "preset": cache.preset.as_id(),
            "clips": cache.clips,
        }))
    } else {
        Ok(json!({
            "cached": false,
            "running": running,
            "hasSubtitles": has_subs,
            "preset": preset.as_id(),
            "clips": [],
        }))
    }
}

pub(crate) fn commands() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "ai.findViralClips",
            label: "Find Viral Clips",
            menu: &["Sequence"],
            shortcut: None,
            params: r#"{"preset":"15-30s|30-60s|60-90s"?,"autoTranscribe":bool?,"force":bool?,"language":"bg|auto|en|es|de|ru"?,"model":"large-v3-turbo|base"?,"mode":"sentences|reels"?,"compute":"gpu|cpu|auto"?,"wait":bool?}"#,
            enabled: has_seq,
            run: find_viral_clips_cmd,
            journal: true,
            named: NamedKeys::NONE,
        },
        CommandSpec {
            id: "ai.jumpToViralClip",
            label: "Jump / Mark Viral Clip",
            menu: &[],
            shortcut: None,
            params: r#"{"rank":int?,"preset":"15-30s|30-60s|60-90s"?,"start":ticks?,"end":ticks?,"score":int?,"title":str?}"#,
            enabled: has_seq,
            run: jump_to_viral_clip_cmd,
            journal: true,
            named: NamedKeys::NONE,
        },
        CommandSpec {
            id: "ai.extractViralClip",
            label: "Extract Viral Clip",
            menu: &[],
            shortcut: None,
            params: r#"{"rank":int?,"preset":"15-30s|30-60s|60-90s"?,"start":ticks?,"end":ticks?,"score":int?,"title":str?}"#,
            enabled: has_seq,
            run: extract_viral_clip_cmd,
            journal: true,
            named: NamedKeys::NONE,
        },
        CommandSpec {
            id: "ai.viralClipsStatus",
            label: "AI Viral Clip Finder Status",
            menu: &[],
            shortcut: None,
            params: r#"{"preset":"15-30s|30-60s|60-90s"?,"job":id?}"#,
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
    use filmcraft_project::{Caption, CaptionFormat, CaptionTrack, ClipId, TrackId};

    fn cue_ms(start_ms: u64, end_ms: u64, text: &str) -> InputCue {
        InputCue { start: Tick::from_units(start_ms as i64, 1000), end: Tick::from_units(end_ms as i64, 1000), start_ms, end_ms, text: text.to_string() }
    }

    #[test]
    fn scores_high_impact_hook_above_flat_or_paused_segments_and_snaps_boundaries() {
        let cues = vec![
            // Segment 1 (0..22s): slow filler segment with 3.5s dead-air pause and no hook
            cue_ms(0, 2_500, "Ъъ, днес просто ще разгледаме някои обикновени бележки."),
            cue_ms(6_000, 12_000, "Понякога има доста паузи между изреченията."),
            cue_ms(16_000, 22_000, "И това продължава бавно до края."),
            // Segment 2 (24..45s): punchy Bulgarian/English question + number hook + steady pace
            cue_ms(24_000, 27_500, "Знаете ли защо 90% от видеата губят зрители?!"),
            cue_ms(27_600, 32_000, "Тайната е в първите 3 секунди и ритъма на монтажа."),
            cue_ms(32_200, 38_000, "Когато премахнете празните паузи, историята оживява!"),
            cue_ms(38_200, 44_000, "Този лесен трик удвоява задържането на вниманието."),
            // Segment 3 (46..67s): solid instructional clip
            cue_ms(46_000, 50_000, "Ето 3 практични съвета как да подобрите звука!"),
            cue_ms(50_200, 56_000, "Първо приближете микрофона и намалете ехото."),
            cue_ms(56_200, 62_000, "Второ изравнете нивата на говора и музиката."),
            cue_ms(62_200, 67_000, "Резултатът е кристален глас във всяко видео."),
            // Segment 4 (69..89s): fourth distinct clip
            cue_ms(69_000, 73_000, "Why do top creators obsess over opening pacing?"),
            cue_ms(73_200, 79_000, "Because viewers decide in seconds whether to watch."),
            cue_ms(79_200, 88_000, "Start directly with value instead of long intros!"),
        ];

        // Synth audio with high energy during 24..44s peak
        let sr = 16_000u32;
        let mut pcm = vec![0.08_f32; sr as usize * 90];
        for s in (24 * sr as usize)..(44 * sr as usize) {
            pcm[s] = if s % 400 < 200 { 0.65 } else { 0.25 };
        }

        let res = score_viral_clips(
            &cues,
            &pcm,
            sr,
            ScoreContext { preset: ViralDurationPreset::Short15To30, audio_origin_ms: 0, rate: FrameRate::FPS_30, drop_frame: false },
        );

        assert!((3..=5).contains(&res.clips.len()), "expected 3..=5 clips, got {:?}", res.clips);
        assert_eq!(res.clips[0].rank, 1);
        assert!(res.clips[0].title.contains("Знаете ли"), "expected viral hook #1 to win, got {:?}", res.clips[0]);
        assert!(res.clips[0].score > 60 && res.clips[0].score <= 100);
        // Must snap cleanly to cue boundaries inside target range
        for clip in &res.clips {
            assert!((15_000..=30_000).contains(&clip.duration_ms), "clip dur {}ms out of 15-30s", clip.duration_ms);
            assert!(cues.iter().any(|c| c.start == clip.start), "start not on cue boundary");
            assert!(cues.iter().any(|c| c.end == clip.end), "end not on cue boundary");
            assert!(!clip.start_timecode.is_empty());
            assert!(!clip.title.is_empty());
        }
    }

    #[test]
    fn handles_short_sequences_reels_cues_and_hostile_inputs_safely() {
        // Reels-style 1-3 word cues where sentence boundaries must prevent mid-sentence cuts
        let reels_cues = vec![
            cue_ms(0, 1_500, "This first part"),
            cue_ms(1_550, 3_500, "finishes an old thought."),
            cue_ms(4_000, 5_500, "Wait — how"),
            cue_ms(5_550, 7_500, "did 5 edits"),
            cue_ms(7_600, 9_500, "change everything?"),
            cue_ms(9_600, 13_000, "Here is why!"),
        ];
        let short_res = score_viral_clips(
            &reels_cues,
            &[],
            16_000,
            ScoreContext { preset: ViralDurationPreset::Medium30To60, audio_origin_ms: 0, rate: FrameRate::FPS_30, drop_frame: false },
        );
        assert!(!short_res.clips.is_empty());
        // Should snap to clean sentence start at 4000ms ("Wait — how") or 0ms, never start mid-sentence at 1550ms
        assert_ne!(short_res.clips[0].start_ms, 1_550);

        // Hostile inputs under catch_unwind: empty, NaN/Inf PCM, reversed cues
        let hostile = std::panic::catch_unwind(|| {
            let bad_cues = vec![cue_ms(5_000, 2_000, "reversed"), cue_ms(1_000, 2_000, "   ")];
            let a = score_viral_clips(
                &bad_cues,
                &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY].repeat(500),
                0,
                ScoreContext { preset: ViralDurationPreset::Long60To90, audio_origin_ms: 0, rate: FrameRate::FPS_30, drop_frame: false },
            );
            assert!(a.clips.is_empty());
            let b = score_viral_clips(
                &[],
                &[],
                16_000,
                ScoreContext { preset: ViralDurationPreset::Short15To30, audio_origin_ms: 0, rate: FrameRate::FPS_30, drop_frame: false },
            );
            assert!(b.clips.is_empty());
        });
        assert!(hostile.is_ok());

        assert_eq!(ViralDurationPreset::from_str_param("30–60s"), Some(ViralDurationPreset::Medium30To60));
        assert_eq!(ViralDurationPreset::from_str_param("60-90s"), Some(ViralDurationPreset::Long60To90));
    }

    #[test]
    fn jump_mark_and_extract_clip_are_undoable_and_preserve_captions_and_audio_sync() {
        let sr = 16_000u32;
        let samples: Vec<f32> = (0..(sr as usize * 36)).map(|i| (i as f32 * 0.04).sin() * 0.35).collect();
        let wav = filmcraft_media::wav::write_wav16(&samples, 1, sr);

        let mut s = Session::default();
        let audio_item = crate::commands::import_bytes(&mut s, "talk.wav", wav.into(), None).unwrap();
        s.state.project_selection = vec![audio_item];
        s.execute("file.newSequenceFromClip", json!({"items": [audio_item.0]})).unwrap();
        let master_seq_id = s.state.active_sequence.unwrap();

        let bars = s.execute("file.newBarsAndTone", json!({"duration": 36.0})).unwrap()["item"].as_u64().unwrap();
        s.edit_sequence("Add Video & AI Captions", |q, ctx, _| {
            let v_id = ClipId(ctx.alloc());
            let mut v = q.audio_tracks[0].items[0].clone();
            v.id = v_id;
            v.item = ItemId(bars);
            q.video_tracks[0].items.push(v);

            let tid = TrackId(ctx.alloc());
            let mut tr = CaptionTrack::new(tid, "AI Subtitles".into(), CaptionFormat::Subtitle);
            let entries = [
                (0, 6_000, "Днес ще покажа нещо ново."),
                (8_000, 14_000, "Първо проверяваме настройките."),
                (14_200, 20_000, "Знаете ли защо 95% от видеата не се гледат?!"),
                (20_100, 26_000, "Тайната е силният въпрос в началото."),
                (26_200, 32_000, "Вижте как енергията скача мигновено!"),
                (32_200, 35_500, "Изпробвайте го още днес."),
            ];
            for (ms0, ms1, txt) in entries {
                let a = Tick::from_units(ms0, 1000);
                let b = Tick::from_units(ms1, 1000);
                tr.captions.push(Caption {
                    id: ClipId(ctx.alloc()),
                    start: a,
                    duration: b - a,
                    text: txt.to_string(),
                    speaker: None,
                    cue_id: None,
                    settings: String::new(),
                });
            }
            q.caption_tracks.push(tr);
            Ok(())
        })
        .unwrap();

        let find_res = s.execute("ai.findViralClips", json!({"preset": "15-30s", "wait": true})).unwrap();
        let clips = find_res["clips"].as_array().unwrap();
        assert!(!clips.is_empty());
        let top_start = clips[0]["start"].as_i64().unwrap();
        let top_end = clips[0]["end"].as_i64().unwrap();
        assert!(top_end > top_start);

        // Action 1: Jump / Mark places playhead, In/Out, and labeled range marker (undoable)
        let jump_res = s.execute("ai.jumpToViralClip", json!({"rank": 1, "preset": "15-30s"})).unwrap();
        let jumped_start = Tick(jump_res["start"].as_i64().unwrap());
        let jumped_end = Tick(jump_res["end"].as_i64().unwrap());
        let fd = s.sequence_rate().frame_duration();
        assert!((jumped_start.0 - top_start).abs() <= fd.0);
        assert_eq!(s.playhead(), jumped_start);
        let q_marked = s.active_sequence().unwrap();
        assert_eq!(q_marked.mark_in, Some(jumped_start));
        assert!(q_marked.mark_out.is_some());
        assert_eq!(q_marked.markers.len(), 1);
        assert!(q_marked.markers[0].name.starts_with("Viral #1"));
        assert_eq!(q_marked.markers[0].start, jumped_start);
        assert_eq!(q_marked.markers[0].duration, jumped_end - jumped_start);

        s.execute("edit.undo", json!({})).unwrap();
        assert!(s.active_sequence().unwrap().markers.is_empty());
        assert_eq!(s.active_sequence().unwrap().mark_in, None);

        // Action 2: Extract Clip creates & opens a trimmed sequence shifted to 0 (undoable)
        let ext_res = s.execute("ai.extractViralClip", json!({"rank": 1, "preset": "15-30s"})).unwrap();
        let extracted_id = ItemId(ext_res["sequence"].as_u64().unwrap());
        assert_ne!(extracted_id, master_seq_id);
        assert_eq!(s.state.active_sequence, Some(extracted_id));

        let q_ext = s.active_sequence().unwrap();
        q_ext.check().unwrap();
        let expected_dur = jumped_end - jumped_start;
        assert!((q_ext.duration().0 - expected_dur.0).abs() <= fd.0);
        assert_eq!(q_ext.video_tracks[0].items.len(), 1);
        assert_eq!(q_ext.audio_tracks[0].items.len(), 1);
        assert_eq!(q_ext.video_tracks[0].items[0].start, Tick::ZERO);
        assert_eq!(q_ext.audio_tracks[0].items[0].start, Tick::ZERO);
        assert!(!q_ext.caption_tracks[0].captions.is_empty());

        // Master sequence remains completely intact
        assert!((s.project.sequence(master_seq_id).unwrap().duration().seconds() - 36.0).abs() < 0.2);

        // Undo removes the extracted sequence cleanly
        s.execute("edit.undo", json!({})).unwrap();
        assert!(s.project.sequence(extracted_id).is_none());
        assert_eq!(s.state.active_sequence, Some(master_seq_id));
    }

    #[test]
    fn missing_subtitles_errors_when_no_auto_transcribe_and_stale_poll_dropped() {
        let mut s = Session::default();
        s.execute("file.newSequence", json!({"name": "No Subs", "width": 1080, "height": 1920, "fps": 30})).unwrap();
        let seq_id = s.state.active_sequence.unwrap();

        let err = s.execute("ai.findViralClips", json!({"autoTranscribe": false})).unwrap_err();
        assert!(err.to_string().contains("no subtitles"));

        // Mismatched document_epoch job must be discarded on poll without mutating cache
        s.viral_jobs.push(PendingViralJob {
            job: 9998,
            seq: seq_id,
            seq_rate: FrameRate::FPS_30,
            drop_frame: false,
            preset: ViralDurationPreset::Short15To30,
            epoch: s.document_epoch.wrapping_sub(1),
            existing_cue_sig: vec![],
            generated_subtitles: Arc::default(),
            analysis: Arc::new(Mutex::new(Some(ViralClipAnalysis { clips: vec![], total_cues: 0, analyzed_duration_ms: 0 }))),
        });
        poll(&mut s);
        assert!(s.viral_jobs.is_empty());
        assert!(s.viral_clips_cache.is_empty());
    }
}
