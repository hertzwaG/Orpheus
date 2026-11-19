//! Pure-Rust **AI Smart Triggers** (`ai.applySmartTriggers`, `ai.smartTriggersStatus`).
//!
//! Scans the active sequence's Whisper subtitle cues (or falls back to sequence transcript
//! words; or automatically runs Whisper transcription first when `autoTranscribe` is enabled) to
//! identify high-impact cue points:
//! - Questions (`?`), exclamations (`!`), numbers/statistics, strong emphasis words in
//!   Bulgarian & English, and topic shifts after pauses.
//!
//! Configurable trigger density (`Subtle` / `Balanced` / `Dynamic`) and three selectable actions
//! committed in one undoable edit:
//! - **Zoom Punch-Ins:** smooth Motion scale keyframes (`100% -> 108% -> 100%` over ~0.35 s) on
//!   the active video clip at emphasis points;
//! - **Synthesized SFX Accents:** clean procedural royalty-free WAV accents (`pop`, `whoosh`,
//!   `ding`) placed on a dedicated `"AI SFX"` audio track;
//! - **Visual Callout Markers:** labeled amber timeline markers (`Label::Mango` / `#F59E0B`) with
//!   the matched trigger keyword.

use std::collections::HashSet;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, PoisonError};

use filmcraft_export::{Progress, Report};
use filmcraft_project::{
    ClipId, HexColor, Interpolation, ItemId, Label, Marker, MarkerId, MarkerKind, ParamValue, Project, Track, TrackId, TrackKind, find_effect,
};
use filmcraft_time::{FrameRate, Tick, TimeRange};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, NamedKeys, always, bad, bool_p, has_seq, str_p, u64_p};
use crate::subtitles::{
    self, ResolvedWhisperConfig, SubtitleCue, SubtitleLengthMode, WhisperRunConfig, build_audio_plan, build_subtitle_cues, extract_audio_mono, next_job_id,
};
use crate::viral_clips::{InputCue, cues_from_sequence, input_cues_from_subtitles, sequence_has_subtitles};
use crate::{EngineError, Job, Result, Session};

/// Zoom punch-in total pulse duration (~0.35 s).
const ZOOM_PULSE_MS: i64 = 350;
/// Zoom punch-in peak offset (~0.14 s after start for crisp snap-in).
const ZOOM_PEAK_MS: i64 = 140;
/// Minimum spacing between two zoom punch-ins on the same video clip.
const MIN_ZOOM_GAP_MS: i64 = 550;
/// Amber hex color for Smart Trigger visual callout markers.
const AMBER_HEX: HexColor = HexColor([0xF5, 0x9E, 0x0B]);

/// Trigger density controlling how selective the cue detector is and minimum spacing between cues.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TriggerDensity {
    Subtle,
    #[default]
    Balanced,
    Dynamic,
}

impl TriggerDensity {
    pub fn from_str_param(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "subtle" | "low" => Some(Self::Subtle),
            "balanced" | "medium" | "normal" => Some(Self::Balanced),
            "dynamic" | "high" => Some(Self::Dynamic),
            _ => None,
        }
    }

    pub fn as_id(self) -> &'static str {
        match self {
            Self::Subtle => "subtle",
            Self::Balanced => "balanced",
            Self::Dynamic => "dynamic",
        }
    }

    /// `(min_gap_ms, min_strength)` thresholds for non-maximum suppression.
    fn thresholds(self) -> (u64, f64) {
        match self {
            Self::Subtle => (5_500, 3.6),
            Self::Balanced => (3_000, 2.7),
            Self::Dynamic => (1_500, 1.8),
        }
    }
}

/// Reason category of a detected high-impact speech trigger.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TriggerKind {
    Question,
    Exclamation,
    Statistic,
    Emphasis,
    TopicShift,
}

impl TriggerKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Question => "question",
            Self::Exclamation => "exclamation",
            Self::Statistic => "statistic",
            Self::Emphasis => "emphasis",
            Self::TopicShift => "topic_shift",
        }
    }
}

/// Procedural sound effect preset aligned to each trigger category.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SfxKind {
    Pop,
    Whoosh,
    Ding,
}

impl SfxKind {
    pub fn file_name(self) -> &'static str {
        match self {
            Self::Pop => "orpheus_sfx_pop.wav",
            Self::Whoosh => "orpheus_sfx_whoosh.wav",
            Self::Ding => "orpheus_sfx_ding.wav",
        }
    }

    fn from_trigger(kind: TriggerKind) -> Self {
        match kind {
            TriggerKind::Question => Self::Ding,
            TriggerKind::Exclamation | TriggerKind::TopicShift => Self::Whoosh,
            TriggerKind::Statistic | TriggerKind::Emphasis => Self::Pop,
        }
    }
}

/// User-configurable options controlling which automated timeline edits are applied.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SmartTriggerOptions {
    pub density: TriggerDensity,
    pub zoom_punch_ins: bool,
    pub sfx_accents: bool,
    pub callout_markers: bool,
}

impl Default for SmartTriggerOptions {
    fn default() -> Self {
        Self { density: TriggerDensity::Balanced, zoom_punch_ins: true, sfx_accents: true, callout_markers: true }
    }
}

impl SmartTriggerOptions {
    pub fn any_action_enabled(self) -> bool {
        self.zoom_punch_ins || self.sfx_accents || self.callout_markers
    }
}

/// One detected emphasis / hook trigger point on the timeline.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DetectedTrigger {
    pub time: Tick,
    pub time_ms: u64,
    pub kind: TriggerKind,
    pub sfx: SfxKind,
    pub keyword: String,
    pub strength: f64,
}

/// Summary of the most recent Smart Triggers application stored on [`Session`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SmartTriggerSummary {
    pub seq: ItemId,
    pub detected: usize,
    pub applied: usize,
    pub zoom_keyframes: usize,
    pub sfx_clips: usize,
    pub callout_markers: usize,
}

/// Generated Whisper subtitles pending insertion before Smart Trigger edits commit.
pub struct GeneratedTriggerSubtitles {
    pub language: String,
    pub mode: SubtitleLengthMode,
    pub cues: Vec<SubtitleCue>,
}

/// In-flight background Smart Triggers job applied cleanly on the session thread by [`poll`].
pub struct PendingSmartTriggerJob {
    pub job: u64,
    pub seq: ItemId,
    pub options: SmartTriggerOptions,
    pub epoch: u64,
    pub generated_subtitles: Arc<Mutex<Option<GeneratedTriggerSubtitles>>>,
    pub triggers: Arc<Mutex<Option<Vec<DetectedTrigger>>>>,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// High-impact emphasis & shift keywords across Bulgarian (Cyrillic) & English.
const EMPHASIS_WORDS_BG: &[&str] = &[
    "тайна",
    "тайната",
    "истина",
    "истината",
    "грешка",
    "грешки",
    "никога",
    "винаги",
    "всъщност",
    "внимание",
    "спри",
    "спрете",
    "задължително",
    "важно",
    "невероятно",
    "шок",
    "проблем",
    "резултат",
    "решение",
    "трик",
    "съвет",
    "единствен",
    "доказано",
    "милион",
    "хиляди",
    "процент",
    "супер",
    "топ",
    "бонус",
    "бързо",
    "лесно",
];

const EMPHASIS_WORDS_EN: &[&str] = &[
    "secret",
    "truth",
    "mistake",
    "mistakes",
    "never",
    "always",
    "actually",
    "important",
    "stop",
    "wait",
    "watch",
    "listen",
    "crazy",
    "insane",
    "amazing",
    "massive",
    "critical",
    "problem",
    "result",
    "proof",
    "proven",
    "hack",
    "trick",
    "tip",
    "warning",
    "nobody",
    "everyone",
    "million",
    "billion",
    "percent",
    "bonus",
    "exactly",
    "immediately",
];

const SHIFT_OPENERS: &[&str] =
    &["сега", "първо", "второ", "трето", "следва", "междувременно", "обаче", "now", "first", "second", "third", "next", "however", "meanwhile"];

fn normalize_token(tok: &str) -> String {
    tok.trim_matches(|c: char| !c.is_alphanumeric() && c != '%' && c != '-' && c != '?').to_lowercase()
}

fn keyword_label(raw: &str) -> String {
    let trimmed = raw.trim_matches(|c: char| c.is_whitespace() || matches!(c, ',' | '.' | ';' | ':' | '"' | '\'' | '»' | '”' | '(' | ')'));
    let mut out = String::new();
    for ch in trimmed.chars().take(28) {
        out.push(ch);
    }
    if out.is_empty() { "Highlight".into() } else { out }
}

fn opening_keyword(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().take(3).collect();
    keyword_label(&words.join(" "))
}

fn interpolated_word_time(cue: &InputCue, word: &str, rate: FrameRate) -> (Tick, u64) {
    let char_pos = cue.text.find(word).map_or(0, |b| cue.text[..b].chars().count());
    let total_chars = cue.text.chars().count().max(1);
    let frac = (char_pos as f64 / total_chars as f64).clamp(0.0, 0.85);
    let span_ticks = (cue.end - cue.start).0.max(0);
    let raw_tick = Tick(cue.start.0.saturating_add((span_ticks as f64 * frac).round() as i64));
    let snapped = rate.snap(raw_tick).max(cue.start);
    let ms = cue.start_ms.saturating_add(((cue.end_ms.saturating_sub(cue.start_ms)) as f64 * frac).round() as u64);
    (snapped, ms)
}

/// Detect high-impact trigger moments across sorted subtitle cues cleanly in pure Rust.
pub fn detect_smart_triggers(cues: &[InputCue], density: TriggerDensity, rate: FrameRate) -> Vec<DetectedTrigger> {
    let valid: Vec<&InputCue> = cues.iter().filter(|c| !c.text.trim().is_empty() && c.end > c.start && c.end_ms > c.start_ms).collect();
    if valid.is_empty() {
        return Vec::new();
    }

    let (min_gap_ms, min_strength) = density.thresholds();
    let mut raw_cands: Vec<DetectedTrigger> = Vec::new();

    for (i, cue) in valid.iter().enumerate() {
        let gap_before = if i > 0 { cue.start_ms.saturating_sub(valid[i - 1].end_ms) } else { 0 };
        let words: Vec<&str> = cue.text.split_whitespace().collect();
        if words.is_empty() {
            continue;
        }

        let mut best_kind: Option<TriggerKind> = None;
        let mut strength = 0.0_f64;
        let mut focus_word: Option<&str> = None;

        // 1. Question (?)
        if cue.text.contains('?') {
            strength += 4.0;
            best_kind = Some(TriggerKind::Question);
            focus_word = words.first().copied();
        }

        // 2. Exclamation (!)
        if cue.text.contains('!') {
            strength += 3.3;
            best_kind.get_or_insert(TriggerKind::Exclamation);
            focus_word.get_or_insert_with(|| words.first().copied().unwrap_or("!"));
        }

        // 3. Numbers / percentages / stats
        if let Some(num_w) = words.iter().copied().find(|w| w.chars().any(|c| c.is_ascii_digit())) {
            strength += 3.6;
            if best_kind.is_none() || best_kind == Some(TriggerKind::Exclamation) {
                best_kind = Some(TriggerKind::Statistic);
                focus_word = Some(num_w);
            }
        }

        // 4. Strong Bulgarian or English emphasis words
        if let Some(emp_w) = words.iter().copied().find(|w| {
            let norm = normalize_token(w);
            let stem = norm.trim_end_matches('?');
            EMPHASIS_WORDS_BG.contains(&stem) || EMPHASIS_WORDS_EN.contains(&stem)
        }) {
            strength += 3.2;
            if best_kind.is_none() {
                best_kind = Some(TriggerKind::Emphasis);
                focus_word = Some(emp_w);
            }
        }

        // 5. Topic shifts after pauses or explicit transition openers
        let opener_norm = normalize_token(words[0]);
        if gap_before >= 850 || (i > 0 && SHIFT_OPENERS.contains(&opener_norm.as_str())) {
            let bonus = if gap_before >= 850 { 3.1 } else { 2.7 };
            strength += bonus;
            if best_kind.is_none() {
                best_kind = Some(TriggerKind::TopicShift);
                focus_word = Some(words[0]);
            }
        }

        if let Some(kind) = best_kind
            && strength >= min_strength
        {
            let (time, time_ms, keyword) = match kind {
                TriggerKind::Statistic | TriggerKind::Emphasis => {
                    if let Some(w) = focus_word {
                        let (t, ms) = interpolated_word_time(cue, w, rate);
                        (t, ms, keyword_label(w))
                    } else {
                        (rate.snap(cue.start).max(Tick::ZERO), cue.start_ms, opening_keyword(&cue.text))
                    }
                }
                _ => (rate.snap(cue.start).max(Tick::ZERO), cue.start_ms, opening_keyword(&cue.text)),
            };
            raw_cands.push(DetectedTrigger { time, time_ms, kind, sfx: SfxKind::from_trigger(kind), keyword, strength });
        }
    }

    // Rank strongest candidates first for non-maximum suppression so peaks win cleanly over nearby weak hits.
    raw_cands.sort_by(|a, b| b.strength.total_cmp(&a.strength).then_with(|| a.time_ms.cmp(&b.time_ms)));
    let mut kept: Vec<DetectedTrigger> = Vec::new();
    for cand in raw_cands {
        if kept.iter().all(|k| k.time_ms.abs_diff(cand.time_ms) >= min_gap_ms) {
            kept.push(cand);
        }
    }
    kept.sort_by_key(|t| (t.time, t.time_ms));
    kept
}

/// Synthesize a crisp, click-free mono 48 kHz 16-bit PCM WAV byte buffer for a trigger SFX accent.
pub fn synthesize_sfx_wav(kind: SfxKind) -> Vec<u8> {
    let mapped = match kind {
        SfxKind::Pop => crate::sfx_pack::CreatorSfx::Pop,
        SfxKind::Whoosh => crate::sfx_pack::CreatorSfx::Whoosh,
        SfxKind::Ding => crate::sfx_pack::CreatorSfx::BellDing,
    };
    crate::sfx_pack::synthesize_wav(mapped)
}

fn ensure_sfx_items(s: &mut Session, p: &Value, triggers: &[DetectedTrigger]) -> Result<Vec<(SfxKind, ItemId, Tick)>> {
    let needed: Vec<SfxKind> = {
        let mut seen = HashSet::new();
        let mut list = Vec::new();
        for t in triggers {
            if seen.insert(t.sfx) {
                list.push(t.sfx);
            }
        }
        list
    };
    if needed.is_empty() {
        return Ok(Vec::new());
    }

    let mut dir: Option<String> = None;
    let mut out = Vec::with_capacity(needed.len());
    for kind in needed {
        let fname = kind.file_name();
        let wav = synthesize_sfx_wav(kind);
        let approx_dur = match kind {
            SfxKind::Pop => Tick::from_units(95, 1000),
            SfxKind::Whoosh => Tick::from_units(260, 1000),
            SfxKind::Ding => Tick::from_units(280, 1000),
        };

        if let Some(existing) = s.project.items.values().find(|it| it.name == fname && !it.as_media().is_none_or(|m| m.offline)).map(|it| it.id) {
            let dur = s.project.item(existing).and_then(|it| it.as_media()).map(|m| m.info.duration).filter(|d| *d > Tick::ZERO).unwrap_or(approx_dur);
            out.push((kind, existing, dur));
            continue;
        }

        let target_dir = match dir.as_deref() {
            Some(d) => d.to_string(),
            None => {
                let d = crate::voiceover::media_dir(s, p, "AI SFX")?;
                if !cfg!(target_arch = "wasm32") {
                    let _ = std::fs::create_dir_all(&d);
                }
                dir = Some(d.clone());
                d
            }
        };
        let path = std::path::Path::new(&target_dir).join(fname).to_string_lossy().into_owned();
        let _ = s.services.write_file(&path, &wav);
        let id = crate::commands::import_bytes(s, &path, Arc::from(wav), None)?;
        let dur = s.project.item(id).and_then(|it| it.as_media()).map(|m| m.info.duration).filter(|d| *d > Tick::ZERO).unwrap_or(approx_dur);
        out.push((kind, id, dur));
    }
    Ok(out)
}

fn upsert_scale_keyframe(param: &mut filmcraft_project::Param, mt: Tick, value: f64) {
    param.put_keyframe(mt, ParamValue::Float(value));
    if let Some(k) = param.keyframes.iter_mut().find(|k| k.time == mt) {
        k.interp = Interpolation::AutoBezier;
    }
}

fn apply_zoom_punch_in(item: &mut filmcraft_project::TrackItem, t_trig: Tick, rate: FrameRate) -> bool {
    let fd = rate.frame_duration().max(Tick(1));
    let clip_end = (item.end() - fd).max(item.start);
    let min_span = Tick::from_units(180, 1000).max(Tick(fd.0.saturating_mul(3)));
    if clip_end - item.start < min_span {
        return false;
    }

    let t_in = (t_trig - fd).clamp(item.start, clip_end);
    let peak_off = Tick::from_units(ZOOM_PEAK_MS, 1000).max(fd);
    let pulse_off = Tick::from_units(ZOOM_PULSE_MS, 1000).max(Tick(fd.0.saturating_mul(2)));
    let t_out = (t_in + pulse_off).min(clip_end);
    let t_peak = (t_in + peak_off).min(t_out - fd);
    if !(t_in < t_peak && t_peak < t_out) {
        return false;
    }

    let mt_in = item.effect_time_at(t_in);
    let mt_peak = item.effect_time_at(t_peak);
    let mt_out = item.effect_time_at(t_out);
    if mt_in == mt_peak || mt_peak == mt_out || mt_in == mt_out {
        return false;
    }

    if item.effect("motion").is_none()
        && let Some(def) = find_effect("motion")
    {
        item.effects.insert(0, def.instance());
    }
    let Some(motion) = item.effect_mut("motion") else {
        return false;
    };

    let uniform = motion.param("uniform_scale").and_then(|p| p.value_at(mt_in).as_bool()).unwrap_or(true);
    let base_scale = motion.f64_at("scale", mt_in).clamp(10.0, 500.0);
    let peak_scale = (base_scale * 1.08).clamp(10.0, 1000.0);
    let Some(scale_param) = motion.param_mut("scale") else {
        return false;
    };
    upsert_scale_keyframe(scale_param, mt_in, base_scale);
    upsert_scale_keyframe(scale_param, mt_peak, peak_scale);
    upsert_scale_keyframe(scale_param, mt_out, base_scale);
    scale_param.keyframes.sort_by_key(|k| k.time);

    if !uniform {
        let base_w = motion.f64_at("scale_width", mt_in).clamp(10.0, 500.0);
        let peak_w = (base_w * 1.08).clamp(10.0, 1000.0);
        if let Some(sw_param) = motion.param_mut("scale_width") {
            upsert_scale_keyframe(sw_param, mt_in, base_w);
            upsert_scale_keyframe(sw_param, mt_peak, peak_w);
            upsert_scale_keyframe(sw_param, mt_out, base_w);
            sw_param.keyframes.sort_by_key(|k| k.time);
        }
    }
    true
}

fn find_target_video_clip<'a>(q: &'a mut filmcraft_project::Sequence, t: Tick, selected: &[ClipId]) -> Option<&'a mut filmcraft_project::TrackItem> {
    let picked = if !selected.is_empty() {
        q.video_tracks
            .iter()
            .rev()
            .filter(|tr| !tr.locked)
            .find_map(|tr| tr.items.iter().find(|it| it.enabled && selected.contains(&it.id) && it.start <= t && t < it.end()).map(|it| it.id))
    } else {
        None
    }
    .or_else(|| {
        q.video_tracks
            .iter()
            .rev()
            .filter(|tr| !tr.locked)
            .find_map(|tr| tr.items.iter().find(|it| it.enabled && it.start <= t && t < it.end()).map(|it| it.id))
    })?;
    q.find_item_mut(picked).map(|(_, it)| it)
}

/// Apply all requested Smart Trigger actions to `seq_id` as one atomic undoable step (`edit_group`).
pub fn apply_triggers_to_session(
    s: &mut Session,
    seq_id: ItemId,
    triggers: &[DetectedTrigger],
    opts: SmartTriggerOptions,
    p: &Value,
) -> Result<SmartTriggerSummary> {
    if !opts.any_action_enabled() {
        return Err(bad("ai.applySmartTriggers", "enable at least one Smart Trigger action"));
    }
    if triggers.is_empty() {
        let sum = SmartTriggerSummary { seq: seq_id, detected: 0, applied: 0, zoom_keyframes: 0, sfx_clips: 0, callout_markers: 0 };
        s.smart_trigger_report = Some(sum.clone());
        return Ok(sum);
    }

    let sfx_map = s.edit_group("Apply AI Smart Triggers", |sess| {
        let sfx_items = if opts.sfx_accents { ensure_sfx_items(sess, p, triggers)? } else { Vec::new() };
        sess.edit("Apply AI Smart Triggers", |pr, st| {
            let selected = st.selection.clone();
            let seq_rate = pr.sequence(seq_id).ok_or(EngineError::NoSequence)?.settings.frame_rate;
            let mut next_id = pr.next_id;

            let sfx_track_id = if opts.sfx_accents && !sfx_items.is_empty() {
                let seq_ro = pr.sequence(seq_id).ok_or(EngineError::NoSequence)?;
                if let Some(tid) = seq_ro.audio_tracks.iter().find(|tr| !tr.locked && tr.name == "AI SFX").map(|tr| tr.id) {
                    Some(tid)
                } else {
                    let tid = TrackId(next_id);
                    next_id += 1;
                    let seq_mut = pr.sequence_mut(seq_id).ok_or(EngineError::NoSequence)?;
                    seq_mut.audio_tracks.push(Track::new(tid, TrackKind::Audio, "AI SFX".into()));
                    Some(tid)
                }
            } else {
                None
            };

            let mut prepared_sfx = Vec::new();
            for trig in triggers {
                if let Some(&(_, item_id, dur)) = sfx_items.iter().find(|(k, _, _)| *k == trig.sfx)
                    && let Some(mut it) = pr.make_track_item(item_id, TrackKind::Audio, trig.time, TimeRange::new(Tick::ZERO, dur), seq_rate)
                {
                    it.id = ClipId(next_id);
                    next_id += 1;
                    it.duration = dur;
                    it.label = Label::Mango;
                    it.name = trig.keyword.clone();
                    prepared_sfx.push((trig.time, it));
                }
            }

            let q = pr.sequence_mut(seq_id).ok_or(EngineError::NoSequence)?;
            let mut zoom_keyframes = 0usize;
            let mut sfx_clips = 0usize;
            let mut callout_markers = 0usize;
            let mut acted_triggers: HashSet<usize> = HashSet::new();
            let mut last_zoom_by_clip: std::collections::HashMap<ClipId, u64> = std::collections::HashMap::new();

            for (idx, trig) in triggers.iter().enumerate() {
                let mut any_for_trig = false;

                if opts.zoom_punch_ins
                    && let Some(item) = find_target_video_clip(q, trig.time, &selected)
                {
                    let can_zoom = last_zoom_by_clip.get(&item.id).is_none_or(|&prev_ms| trig.time_ms.saturating_sub(prev_ms) >= MIN_ZOOM_GAP_MS as u64);
                    if can_zoom && apply_zoom_punch_in(item, trig.time, seq_rate) {
                        last_zoom_by_clip.insert(item.id, trig.time_ms);
                        zoom_keyframes += 3;
                        any_for_trig = true;
                    }
                }

                if opts.callout_markers {
                    let dup = q.markers.iter().any(|m| m.start == trig.time && m.name == trig.keyword);
                    if !dup {
                        let mid = MarkerId(next_id);
                        next_id += 1;
                        q.markers.push(Marker {
                            url: String::new(),
                            author: String::new(),
                            custom_color: Some(AMBER_HEX),
                            id: mid,
                            start: trig.time,
                            duration: Tick::ZERO,
                            name: trig.keyword.clone(),
                            comment: format!("AI Smart Trigger ({})", trig.kind.as_str()),
                            kind: MarkerKind::Comment,
                            color: Label::Mango,
                        });
                        callout_markers += 1;
                        any_for_trig = true;
                    }
                }

                if any_for_trig {
                    acted_triggers.insert(idx);
                }
            }

            if let Some(tid) = sfx_track_id
                && let Some(sfx_tr) = q.audio_tracks.iter_mut().find(|tr| tr.id == tid && !tr.locked)
            {
                for (idx, (t_trig, it)) in prepared_sfx.into_iter().enumerate() {
                    let overlap = sfx_tr.items.iter().any(|ex| ex.start < it.end() && t_trig < ex.end());
                    if !overlap {
                        sfx_tr.items.push(it);
                        sfx_tr.sort();
                        sfx_clips += 1;
                        acted_triggers.insert(idx);
                    }
                }
            }

            q.markers.sort_by_key(|m| m.start);
            q.check().map_err(EngineError::Other)?;
            pr.next_id = next_id;

            Ok(SmartTriggerSummary { seq: seq_id, detected: triggers.len(), applied: acted_triggers.len(), zoom_keyframes, sfx_clips, callout_markers })
        })
    })?;

    s.smart_trigger_report = Some(sfx_map.clone());
    s.toast(format!("Applied {} smart triggers", sfx_map.applied));
    Ok(sfx_map)
}

/// Apply finished background Smart Trigger jobs cleanly on the session thread.
pub fn poll(s: &mut Session) {
    let mut i = 0;
    while i < s.smart_trigger_jobs.len() {
        let job = s.jobs.iter().find(|j| j.id == s.smart_trigger_jobs[i].job);
        let finished = job.is_none_or(|j| j.progress.finished.load(Ordering::Relaxed));
        let cancelled = job.is_some_and(|j| j.progress.cancel.load(Ordering::Relaxed));
        if !finished {
            i += 1;
            continue;
        }
        let pj = s.smart_trigger_jobs.remove(i);
        let gen_subs = lock(&pj.generated_subtitles).take();
        let triggers = lock(&pj.triggers).take();
        if cancelled {
            continue;
        }
        if pj.epoch != s.document_epoch {
            s.error_toast("ai.applySmartTriggers", "AI Smart Triggers: the project was replaced; nothing was changed");
            continue;
        }

        if let Some(subs) = gen_subs
            && !subs.cues.is_empty()
            && let Err(e) = subtitles::insert_subtitle_track(s, pj.seq, &subs.language, subs.mode, &subs.cues)
        {
            s.error_toast("ai.applySmartTriggers", format!("AI Smart Triggers: {e}"));
            continue;
        }

        let Some(trigs) = triggers else { continue };
        if trigs.is_empty() {
            s.toast("No emphasis triggers detected in speech");
            continue;
        }
        if let Err(e) = apply_triggers_to_session(s, pj.seq, &trigs, pj.options, &json!({})) {
            s.error_toast("ai.applySmartTriggers", format!("AI Smart Triggers: {e}"));
        }
    }
}

fn parse_options(p: &Value) -> Result<SmartTriggerOptions> {
    let density = match str_p(p, "density") {
        Some(raw) => TriggerDensity::from_str_param(raw).ok_or_else(|| bad("ai.applySmartTriggers", format!("unknown density `{raw}`")))?,
        None => TriggerDensity::default(),
    };
    let dflt = SmartTriggerOptions::default();
    Ok(SmartTriggerOptions {
        density,
        zoom_punch_ins: bool_p(p, "zoomPunchIns").unwrap_or(dflt.zoom_punch_ins),
        sfx_accents: bool_p(p, "sfxAccents").unwrap_or(dflt.sfx_accents),
        callout_markers: bool_p(p, "calloutMarkers").unwrap_or(dflt.callout_markers),
    })
}

struct TriggerJobInput {
    id: u64,
    existing_cues: Vec<InputCue>,
    audio_plan: Option<subtitles::AudioExtractPlan>,
    whisper_cfg: Option<ResolvedWhisperConfig>,
    density: TriggerDensity,
    seq_rate: FrameRate,
    project: Arc<Project>,
    provider: crate::media_pool::PoolProvider,
    prog: Arc<Progress>,
    gen_slot: Arc<Mutex<Option<GeneratedTriggerSubtitles>>>,
}

fn run_trigger_job(inp: TriggerJobInput) -> std::result::Result<Vec<DetectedTrigger>, String> {
    let cues = if let Some(cfg) = inp.whisper_cfg {
        let Some(plan) = inp.audio_plan.as_ref() else {
            return Err("no enabled audio clips found on unlocked tracks".into());
        };
        let (mono, sr) = extract_audio_mono(plan, &inp.project, &inp.provider, &inp.prog, 25.0)?;
        if inp.prog.cancel.load(Ordering::Relaxed) {
            return Err("Cancelled".into());
        }
        let wav = filmcraft_media::wav::write_wav16(&mono, 1, sr);
        let run_cfg = WhisperRunConfig { model_path: &cfg.model_path, language: &cfg.language, mode: cfg.mode, compute: cfg.compute };
        let segs = subtitles::run_whisper_cli(&cfg.cli_path, &run_cfg, &wav, inp.id, &inp.prog)?;
        let built = build_subtitle_cues(&segs, cfg.mode, plan.range_start, plan.seq_rate);
        let input_cues = input_cues_from_subtitles(&built);
        *lock(&inp.gen_slot) = Some(GeneratedTriggerSubtitles { language: cfg.language, mode: cfg.mode, cues: built });
        input_cues
    } else {
        inp.existing_cues
    };

    if inp.prog.cancel.load(Ordering::Relaxed) {
        return Err("Cancelled".into());
    }
    *lock(&inp.prog.status) = "Detecting smart triggers…".into();
    inp.prog.done.store(85, Ordering::Relaxed);
    let triggers = detect_smart_triggers(&cues, inp.density, inp.seq_rate);
    inp.prog.done.store(100, Ordering::Relaxed);
    Ok(triggers)
}

fn apply_smart_triggers_cmd(s: &mut Session, p: &Value) -> Result<Value> {
    let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let options = parse_options(p)?;
    if !options.any_action_enabled() {
        return Err(bad("ai.applySmartTriggers", "enable at least one Smart Trigger action"));
    }

    let seq_rate = seq.settings.frame_rate;
    let existing_cues = cues_from_sequence(&s.project, seq);
    let auto_transcribe = bool_p(p, "autoTranscribe").unwrap_or(true);

    if !existing_cues.is_empty() && bool_p(p, "wait").unwrap_or(false) {
        let triggers = detect_smart_triggers(&existing_cues, options.density, seq_rate);
        let summary = apply_triggers_to_session(s, seq_id, &triggers, options, p)?;
        return Ok(json!({
            "applied": summary.applied,
            "detected": summary.detected,
            "zoomKeyframes": summary.zoom_keyframes,
            "sfxClips": summary.sfx_clips,
            "calloutMarkers": summary.callout_markers,
        }));
    }

    let (audio_plan, whisper_cfg) = if existing_cues.is_empty() {
        if !auto_transcribe {
            return Err(bad("ai.applySmartTriggers", "no subtitles found on the active sequence"));
        }
        let cfg = subtitles::resolve_whisper_run("ai.applySmartTriggers", p)?;
        let plan = build_audio_plan("ai.applySmartTriggers", s, seq, seq_id, p)?;
        (Some(plan), Some(cfg))
    } else {
        (None, None)
    };

    let id = next_job_id(s);
    let prog = Arc::new(Progress::default());
    prog.total.store(100, Ordering::Relaxed);
    *lock(&prog.status) = if whisper_cfg.is_some() { "Transcribing audio…".into() } else { "Detecting smart triggers…".into() };
    let result: Arc<Mutex<Option<std::result::Result<Report, String>>>> = Arc::default();
    s.jobs.push(Job { id, label: "AI Smart Triggers".into(), progress: prog.clone(), result: result.clone() });

    let gen_slot: Arc<Mutex<Option<GeneratedTriggerSubtitles>>> = Arc::default();
    let trigs_slot: Arc<Mutex<Option<Vec<DetectedTrigger>>>> = Arc::default();
    s.smart_trigger_jobs.push(PendingSmartTriggerJob {
        job: id,
        seq: seq_id,
        options,
        epoch: s.document_epoch,
        generated_subtitles: gen_slot.clone(),
        triggers: trigs_slot.clone(),
    });

    let project = s.project.clone();
    let provider = s.media.provider(project.clone(), s.services.clone());
    let wait = bool_p(p, "wait").unwrap_or(false);
    let will_transcribe = whisper_cfg.is_some();

    let run = crate::export_tools::guard_job(prog.clone(), result.clone(), move || {
        let t0 = web_time::Instant::now();
        let outcome = run_trigger_job(TriggerJobInput {
            id,
            existing_cues,
            audio_plan,
            whisper_cfg,
            density: options.density,
            seq_rate,
            project,
            provider,
            prog: prog.clone(),
            gen_slot: gen_slot.clone(),
        });
        let rep = match outcome {
            Ok(triggers) => {
                let count = triggers.len() as u64;
                *lock(&trigs_slot) = Some(triggers);
                *lock(&prog.status) = if count > 0 { format!("Applied {count} smart triggers") } else { "No emphasis triggers detected in speech".into() };
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
        std::thread::Builder::new().name("orpheus-smart-triggers".into()).spawn(run).map_err(|e| EngineError::Other(e.to_string()))?;
    }

    if let Some(Err(e)) = s.jobs.iter().find(|j| j.id == id).and_then(|j| lock(&j.result).clone()) {
        return Err(EngineError::Other(e));
    }

    let mut out = json!({
        "job": id,
        "density": options.density.as_id(),
        "autoTranscribe": will_transcribe,
    });
    if wait && let Some(rep) = &s.smart_trigger_report {
        out["applied"] = json!(rep.applied);
        out["detected"] = json!(rep.detected);
        out["zoomKeyframes"] = json!(rep.zoom_keyframes);
        out["sfxClips"] = json!(rep.sfx_clips);
        out["calloutMarkers"] = json!(rep.callout_markers);
    }
    Ok(out)
}

fn status_cmd(s: &mut Session, p: &Value) -> Result<Value> {
    let running = u64_p(p, "job").is_some_and(|jid| s.jobs.iter().any(|j| j.id == jid && !j.progress.finished.load(Ordering::Relaxed)));
    let has_subs = s.active_sequence().is_some_and(|q| sequence_has_subtitles(&s.project, q));
    let last = s.smart_trigger_report.as_ref().filter(|r| Some(r.seq) == s.state.active_sequence);
    Ok(json!({
        "running": running,
        "hasSubtitles": has_subs,
        "last": last,
    }))
}

pub(crate) fn commands() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "ai.applySmartTriggers",
            label: "Apply Smart Triggers",
            menu: &["Sequence"],
            shortcut: None,
            params: r#"{"density":"subtle|balanced|dynamic"?,"zoomPunchIns":bool?,"sfxAccents":bool?,"calloutMarkers":bool?,"autoTranscribe":bool?,"language":"bg|auto|en|es|de|ru"?,"model":"large-v3-turbo|base"?,"mode":"sentences|reels"?,"compute":"gpu|cpu|auto"?,"wait":bool?}"#,
            enabled: has_seq,
            run: apply_smart_triggers_cmd,
            journal: true,
            named: NamedKeys::NONE,
        },
        CommandSpec {
            id: "ai.smartTriggersStatus",
            label: "AI Smart Triggers Status",
            menu: &[],
            shortcut: None,
            params: r#"{"job":id?}"#,
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
    use filmcraft_project::{Caption, CaptionFormat, CaptionTrack};

    fn cue_ms(start_ms: u64, end_ms: u64, text: &str) -> InputCue {
        InputCue { start: Tick::from_units(start_ms as i64, 1000), end: Tick::from_units(end_ms as i64, 1000), start_ms, end_ms, text: text.to_string() }
    }

    #[test]
    fn detects_bilingual_emphasis_cues_and_respects_density() {
        let cues = vec![
            cue_ms(0, 2_200, "Знаете ли защо това се случва?"),
            cue_ms(2_400, 4_200, "Дори в 90% от случаите има грешка!"),
            cue_ms(6_000, 8_000, "Тайната е в добрия ритъм."),
            cue_ms(10_500, 12_800, "Сега вижте 5 важни факта."),
            cue_ms(14_800, 17_000, "Never ignore opening pacing!"),
            cue_ms(18_500, 20_500, "Just ordinary filler text without hooks"),
        ];

        let subtle = detect_smart_triggers(&cues, TriggerDensity::Subtle, FrameRate::FPS_30);
        let dynamic = detect_smart_triggers(&cues, TriggerDensity::Dynamic, FrameRate::FPS_30);

        assert!(!subtle.is_empty());
        assert!(dynamic.len() >= subtle.len(), "dynamic={} subtle={}", dynamic.len(), subtle.len());
        assert_eq!(dynamic[0].kind, TriggerKind::Question);
        assert_eq!(dynamic[0].sfx, SfxKind::Ding);
        assert!(dynamic.iter().any(|t| t.kind == TriggerKind::Statistic || t.kind == TriggerKind::Emphasis));

        // Subtle enforces >= 5.5s minimum spacing
        for w in subtle.windows(2) {
            assert!(w[1].time_ms.saturating_sub(w[0].time_ms) >= 5_500);
        }
    }

    #[test]
    fn synthesizes_valid_clean_wav_sfx_and_handles_hostile_inputs() {
        for kind in [SfxKind::Pop, SfxKind::Whoosh, SfxKind::Ding] {
            let wav = synthesize_sfx_wav(kind);
            assert!(wav.starts_with(b"RIFF"));
            assert_eq!(&wav[8..12], b"WAVE");
            assert!(wav.len() > 4_000, "{kind:?} WAV too small: {}", wav.len());
            let mapped = match kind {
                SfxKind::Pop => crate::sfx_pack::CreatorSfx::Pop,
                SfxKind::Whoosh => crate::sfx_pack::CreatorSfx::Whoosh,
                SfxKind::Ding => crate::sfx_pack::CreatorSfx::BellDing,
            };
            let pcm = crate::sfx_pack::synthesize_pcm(mapped);
            assert!(pcm.iter().all(|s| s.is_finite() && (-1.0..=1.0).contains(s)));
        }

        let res = std::panic::catch_unwind(|| {
            let bad = vec![cue_ms(5_000, 2_000, "bad"), cue_ms(100, 500, "   ")];
            assert!(detect_smart_triggers(&bad, TriggerDensity::Dynamic, FrameRate::FPS_30).is_empty());
            assert!(detect_smart_triggers(&[], TriggerDensity::Balanced, FrameRate::FPS_30).is_empty());
        });
        assert!(res.is_ok());
    }

    #[test]
    fn applies_zoom_keyframes_sfx_track_and_amber_markers_in_one_undo_step() {
        let mut s = Session::default();
        s.execute("file.newSequence", json!({"name": "Trigger Seq", "width": 1080, "height": 1920, "fps": 30})).unwrap();
        let bars = s.execute("file.newBarsAndTone", json!({"seconds": 20.0})).unwrap()["item"].as_u64().unwrap();
        let seq_id = s.state.active_sequence.unwrap();

        s.execute("timeline.place", json!({"item": bars, "track": "V1", "frame": 0, "audio": false})).unwrap();

        s.edit_sequence("Setup Captions", |q, ctx, _| {
            let tid = TrackId(ctx.alloc());
            let mut ct = CaptionTrack::new(tid, "AI Subtitles".into(), CaptionFormat::Subtitle);
            for (a_ms, b_ms, txt) in
                [(1_000, 3_200, "Знаете ли защо това работи?"), (5_500, 8_000, "Ето 95% по-добър резултат!"), (11_000, 14_000, "Never skip the secret hook!")]
            {
                let a = Tick::from_units(a_ms, 1000);
                let b = Tick::from_units(b_ms, 1000);
                ct.captions.push(Caption {
                    id: ClipId(ctx.alloc()),
                    start: a,
                    duration: b - a,
                    text: txt.to_string(),
                    speaker: None,
                    cue_id: None,
                    settings: String::new(),
                });
            }
            q.caption_tracks.push(ct);
            Ok(())
        })
        .unwrap();

        let undo_before = s.history.undo.len();
        let res = s
            .execute(
                "ai.applySmartTriggers",
                json!({
                    "density": "balanced",
                    "zoomPunchIns": true,
                    "sfxAccents": true,
                    "calloutMarkers": true,
                    "wait": true
                }),
            )
            .unwrap();

        let applied = res["applied"].as_u64().unwrap();
        assert_eq!(applied, 3, "expected 3 triggers, got {res}");
        assert_eq!(res["zoomKeyframes"].as_u64().unwrap(), 9);
        assert_eq!(res["sfxClips"].as_u64().unwrap(), 3);
        assert_eq!(res["calloutMarkers"].as_u64().unwrap(), 3);
        // Entire multi-part operation collapsed to 1 undoable entry!
        assert_eq!(s.history.undo.len(), undo_before + 1);

        let q = s.active_sequence().unwrap();
        q.check().unwrap();
        let scale_kfs = &q.video_tracks[0].items[0].effect("motion").unwrap().param("scale").unwrap().keyframes;
        assert_eq!(scale_kfs.len(), 9);
        assert!(scale_kfs.iter().any(|k| (k.value.as_f64().unwrap_or(0.0) - 108.0).abs() < 0.01));

        let sfx_tr = q.audio_tracks.iter().find(|t| t.name == "AI SFX").unwrap();
        assert_eq!(sfx_tr.items.len(), 3);
        assert_eq!(q.markers.len(), 3);
        assert_eq!(q.markers[0].color, Label::Mango);
        assert_eq!(q.markers[0].custom_color, Some(AMBER_HEX));

        // Single undo restores pre-trigger project completely!
        s.execute("edit.undo", json!({})).unwrap();
        let q_undo = s.project.sequence(seq_id).unwrap();
        assert!(q_undo.markers.is_empty());
        assert!(q_undo.audio_tracks.iter().all(|t| t.name != "AI SFX"));
        assert!(q_undo.video_tracks[0].items[0].effect("motion").unwrap().param("scale").unwrap().keyframes.is_empty());
    }
}
