//! Local pure-Rust AI Beat Sync (`ai.analyzeBeats`, `ai.addBeatMarkers`, `ai.cutVideoToBeats`,
//! `ai.beatSyncStatus`).
//!
//! Decodes mono PCM from selected audio clips (or the primary unlocked audio track) via
//! [`crate::subtitles::extract_audio_mono`], computes short-window RMS energy, cascaded low-pass
//! bass energy spikes (`~180 Hz`), and STFT spectral flux (`filmcraft_audio_dsp::remix::onset_envelope`)
//! on 10 ms frames, applies O(N) adaptive sliding thresholding with configurable sensitivity and
//! minimum beat spacing, estimates BPM, and applies timeline beat markers or beat-aligned video
//! cuts in a single undoable edit.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, PoisonError};

use filmcraft_export::{Progress, Report};
use filmcraft_project::{ClipId, ItemId, Label, Marker, MarkerId, MarkerKind, TrackId, TrackKind};
use filmcraft_time::{FrameRate, Tick};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, NamedKeys, always, bad, bool_p, has_seq, str_p, u64_p};
use crate::subtitles::{AudioExtractPlan, build_audio_plan, extract_audio_mono, next_job_id};
use crate::{EngineError, Job, Result, Session};

/// Maximum mono samples analysed per job (2 hours at 16 kHz) to bound allocations.
const MAX_BEAT_SAMPLES: usize = 2 * 3600 * 16_000;
/// Spectral-flux chunk size (10 minutes at 16 kHz) so STFT buffers remain bounded.
const FLUX_CHUNK_SECONDS: usize = 600;
/// Minimum RMS energy floor (-80 dBFS) below which audio is treated as silence.
const SILENCE_RMS_FLOOR: f64 = 1e-4;
/// Sliding half-window in 10 ms frames (+/- 400 ms) for adaptive thresholding.
const ADAPTIVE_HALF_WINDOW: usize = 40;
/// Low-pass cutoff frequency (Hz) for isolating kick/sub-bass transients.
const BASS_CUTOFF_HZ: f64 = 180.0;

/// Beat detection sensitivity controlling adaptive threshold tightness and minimum beat spacing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum BeatSensitivity {
    Low,
    #[default]
    Medium,
    High,
}

impl BeatSensitivity {
    pub fn from_str_param(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "low" => Some(Self::Low),
            "medium" | "med" | "normal" => Some(Self::Medium),
            "high" => Some(Self::High),
            _ => None,
        }
    }

    pub fn as_id(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }

    fn params(self) -> (f64, f64, u64) {
        // (std-dev multiplier, global RMS floor fraction, minimum spacing ms)
        match self {
            Self::Low => (1.55, 0.55, 360),
            Self::Medium => (0.95, 0.35, 240),
            Self::High => (0.45, 0.20, 160),
        }
    }
}

/// Beat subdivision interval used when placing timeline markers or cutting video clips.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum BeatInterval {
    #[default]
    EveryBeat,
    Every2Beats,
    Every4Beats,
}

impl BeatInterval {
    pub fn from_str_param(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "1" | "everybeat" | "every_beat" | "beat" => Some(Self::EveryBeat),
            "2" | "every2beats" | "every_2_beats" | "every2" => Some(Self::Every2Beats),
            "4" | "every4beats" | "every_4_beats" | "every4" => Some(Self::Every4Beats),
            _ => None,
        }
    }

    pub fn from_json(p: &Value) -> std::result::Result<Self, String> {
        if let Some(n) = p.get("interval").and_then(Value::as_u64) {
            return match n {
                1 => Ok(Self::EveryBeat),
                2 => Ok(Self::Every2Beats),
                4 => Ok(Self::Every4Beats),
                other => Err(format!("unknown beat interval `{other}` (expected 1, 2, or 4)")),
            };
        }
        if let Some(raw) = str_p(p, "interval") {
            return Self::from_str_param(raw).ok_or_else(|| format!("unknown beat interval `{raw}`"));
        }
        Ok(Self::default())
    }

    pub fn step(self) -> usize {
        match self {
            Self::EveryBeat => 1,
            Self::Every2Beats => 2,
            Self::Every4Beats => 4,
        }
    }

    pub fn as_id(self) -> &'static str {
        match self {
            Self::EveryBeat => "everyBeat",
            Self::Every2Beats => "every2Beats",
            Self::Every4Beats => "every4Beats",
        }
    }
}

/// Action to take immediately after a background beat-analysis job finishes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum BeatPendingAction {
    #[default]
    AnalyzeOnly,
    AddMarkers,
    CutVideo,
}

impl BeatPendingAction {
    pub fn from_str_param(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "analyze" | "analyzeonly" | "none" => Some(Self::AnalyzeOnly),
            "markers" | "addmarkers" | "add_markers" => Some(Self::AddMarkers),
            "cut" | "cutvideo" | "cut_video" | "cutvideotobeats" => Some(Self::CutVideo),
            _ => None,
        }
    }
}

/// Output of pure-Rust beat/onset analysis on a mono PCM buffer.
#[derive(Clone, Debug, PartialEq)]
pub struct BeatAnalysis {
    /// Detected beat timestamps in milliseconds relative to buffer start.
    pub beats_ms: Vec<u64>,
    /// Estimated tempo in beats per minute (`0.0` if fewer than 2 regular beats found).
    pub bpm: f64,
    /// Analysed audio duration in milliseconds.
    pub duration_ms: u64,
}

/// Audio track signature so marker/video edits do not invalidate unchanged audio analysis.
pub type AudioSignature = Vec<(u64, u64, i64, i64, i64)>;

/// Cached beat detection result scoped to one sequence audio state + sensitivity.
#[derive(Clone, Debug, PartialEq)]
pub struct BeatSyncCache {
    pub seq: ItemId,
    pub audio_sig: AudioSignature,
    pub sensitivity: BeatSensitivity,
    pub range_start: Tick,
    pub seq_rate: FrameRate,
    pub beats_ms: Vec<u64>,
    pub bpm: f64,
}

/// In-flight background beat analysis job applied by [`poll`] when completed.
pub struct PendingBeatJob {
    pub job: u64,
    pub seq: ItemId,
    pub audio_sig: AudioSignature,
    pub sensitivity: BeatSensitivity,
    pub interval: BeatInterval,
    pub action: BeatPendingAction,
    pub range_start: Tick,
    pub seq_rate: FrameRate,
    pub epoch: u64,
    pub analysis: Arc<Mutex<Option<BeatAnalysis>>>,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Subdivide detected beats (`EveryBeat` = 1, `Every2Beats` = 2, `Every4Beats` = 4).
pub fn subdivide_beats(beats_ms: &[u64], interval: BeatInterval) -> Vec<u64> {
    let step = interval.step().max(1);
    beats_ms.iter().copied().step_by(step).collect()
}

fn audio_signature(plan: &AudioExtractPlan) -> AudioSignature {
    let mut sig = Vec::new();
    for tr in &plan.tracks {
        for it in &tr.items {
            sig.push((it.id.0, it.item.0, it.start.0, it.duration.0, it.source_in.0));
        }
    }
    sig
}

/// Return cached beat analysis for the active sequence if its audio layout matches and sensitivity agrees.
pub fn cached_for_session(s: &Session, sensitivity: BeatSensitivity) -> Option<&BeatSyncCache> {
    let seq_id = s.state.active_sequence?;
    let seq = s.active_sequence()?;
    let cache = s.beat_sync_cache.as_ref()?;
    if cache.seq != seq_id || cache.sensitivity != sensitivity {
        return None;
    }
    let plan = build_audio_plan("ai.beatSyncStatus", s, seq, seq_id, &json!({})).ok()?;
    (audio_signature(&plan) == cache.audio_sig).then_some(cache)
}

/// Compute STFT spectral flux on 10 ms frames using [`filmcraft_audio_dsp::remix::onset_envelope`],
/// processed in bounded chunks so long sequences keep memory flat.
fn compute_spectral_flux_10ms(samples: &[f32], sr: u32, n_frames: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; n_frames];
    let hop_10ms = (sr as usize / 100).max(1);
    let max_chunk_samples = ((sr as usize) * FLUX_CHUNK_SECONDS).max(hop_10ms * 10);
    let mut start = 0usize;

    while start < samples.len() {
        let end = (start + max_chunk_samples).min(samples.len());
        let slice = &samples[start..end];
        if slice.len() < 32 {
            break;
        }
        let (env, _fft_n, stft_hop) = filmcraft_audio_dsp::remix::onset_envelope(slice, sr);
        if !env.is_empty() && stft_hop > 0 {
            let f_start = start / hop_10ms;
            let f_end = end.div_ceil(hop_10ms).min(n_frames);
            for f_idx in f_start..f_end {
                let sample_pos = (f_idx * hop_10ms).saturating_sub(start);
                let e_pos = sample_pos as f64 / stft_hop as f64;
                let lo = (e_pos.floor() as usize).min(env.len() - 1);
                let hi = (lo + 1).min(env.len() - 1);
                let frac = (e_pos - lo as f64).clamp(0.0, 1.0);
                let val = f64::from(env[lo]) * (1.0 - frac) + f64::from(env[hi]) * frac;
                if val.is_finite() {
                    out[f_idx] = val.max(0.0);
                }
            }
        }
        start = end;
    }
    out
}

fn normalize_by_rms(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let sum_sq: f64 = values.iter().map(|&v| v * v).sum();
    let rms = (sum_sq / values.len() as f64).sqrt();
    if !rms.is_finite() || rms <= 1e-12 {
        return vec![0.0; values.len()];
    }
    values.iter().map(|&v| (v / rms).max(0.0)).collect()
}

fn estimate_bpm(beats_ms: &[u64]) -> f64 {
    if beats_ms.len() < 2 {
        return 0.0;
    }
    let mut ibis: Vec<f64> = beats_ms.windows(2).filter_map(|w| w[1].checked_sub(w[0])).filter(|&d| (250..=2000).contains(&d)).map(|d| d as f64).collect();
    if ibis.is_empty() {
        return 0.0;
    }
    ibis.sort_by(f64::total_cmp);
    let mid = ibis.len() / 2;
    let median_ms = if ibis.len() % 2 == 1 { ibis[mid] } else { (ibis[mid - 1] + ibis[mid]) * 0.5 };
    if median_ms <= 0.0 {
        return 0.0;
    }
    let mut bpm = 60_000.0 / median_ms;
    while bpm < 55.0 {
        bpm *= 2.0;
    }
    while bpm > 195.0 {
        bpm *= 0.5;
    }
    (bpm * 10.0).round() / 10.0
}

/// Detect rhythmic beat timestamps (in milliseconds) and estimate tempo (BPM) from mono PCM
/// samples (`-1.0..=1.0`) using pure Rust RMS energy, low-frequency bass energy spikes, STFT
/// spectral flux, and O(N) adaptive sliding thresholding.
pub fn detect_beats_pcm(mono: &[f32], sample_rate: u32, sensitivity: BeatSensitivity) -> BeatAnalysis {
    let sr = sample_rate.clamp(1_000, 384_000);
    let len = mono.len().min(MAX_BEAT_SAMPLES);
    let duration_ms = ((len as f64 / f64::from(sr)) * 1000.0).round() as u64;
    if len < (sr as usize / 25).max(32) {
        return BeatAnalysis { beats_ms: Vec::new(), bpm: 0.0, duration_ms };
    }

    let clean: Vec<f32> = mono[..len].iter().map(|&s| if s.is_finite() { s.clamp(-1.0, 1.0) } else { 0.0 }).collect();
    let hop = (sr as usize / 100).max(1);
    let n_hops = clean.len() / hop;
    if n_hops < 3 {
        return BeatAnalysis { beats_ms: Vec::new(), bpm: 0.0, duration_ms };
    }

    // Two cascaded 1-pole low-pass stages isolate kick / sub-bass transients (~180 Hz).
    let dt = 1.0 / f64::from(sr);
    let rc = 1.0 / (2.0 * std::f64::consts::PI * BASS_CUTOFF_HZ);
    let alpha = dt / (rc + dt);
    let mut lp1 = 0.0_f64;
    let mut lp2 = 0.0_f64;

    let mut hop_full_sq = vec![0.0_f64; n_hops];
    let mut hop_bass_sq = vec![0.0_f64; n_hops];
    for h in 0..n_hops {
        let s0 = h * hop;
        let mut f_sum = 0.0_f64;
        let mut b_sum = 0.0_f64;
        for &s in &clean[s0..s0 + hop] {
            let x = f64::from(s);
            f_sum += x * x;
            lp1 += alpha * (x - lp1);
            lp2 += alpha * (lp1 - lp2);
            b_sum += lp2 * lp2;
        }
        hop_full_sq[h] = f_sum;
        hop_bass_sq[h] = b_sum;
    }

    let n_frames = n_hops - 1;
    let win_len = (hop * 2) as f64;
    let mut rms = Vec::with_capacity(n_frames);
    let mut bass_rms = Vec::with_capacity(n_frames);
    let mut max_rms = 0.0_f64;

    for t in 0..n_frames {
        let r = ((hop_full_sq[t] + hop_full_sq[t + 1]) / win_len).sqrt();
        let b = ((hop_bass_sq[t] + hop_bass_sq[t + 1]) / win_len).sqrt();
        if r > max_rms {
            max_rms = r;
        }
        rms.push(r);
        bass_rms.push(b);
    }

    if max_rms < SILENCE_RMS_FLOOR {
        return BeatAnalysis { beats_ms: Vec::new(), bpm: 0.0, duration_ms };
    }

    let mut rms_delta = vec![0.0_f64; n_frames];
    let mut bass_delta = vec![0.0_f64; n_frames];
    rms_delta[0] = rms[0];
    bass_delta[0] = bass_rms[0];
    for t in 1..n_frames {
        rms_delta[t] = (rms[t] - rms[t - 1]).max(0.0);
        bass_delta[t] = (bass_rms[t] - bass_rms[t - 1]).max(0.0);
    }

    let flux_raw = compute_spectral_flux_10ms(&clean, sr, n_frames);
    let flux_norm = normalize_by_rms(&flux_raw);
    let bass_norm = normalize_by_rms(&bass_delta);
    let rms_norm = normalize_by_rms(&rms_delta);

    let mut raw_onset = Vec::with_capacity(n_frames);
    for t in 0..n_frames {
        raw_onset.push(0.45 * flux_norm[t] + 0.35 * bass_norm[t] + 0.20 * rms_norm[t]);
    }

    // Light 3-tap smoothing collapses jagged sub-frame spikes onto clean transient centers.
    let mut onset = Vec::with_capacity(n_frames);
    for t in 0..n_frames {
        let prev = raw_onset.get(t.saturating_sub(1)).copied().unwrap_or(0.0);
        let cur = raw_onset[t];
        let next = raw_onset.get(t + 1).copied().unwrap_or(0.0);
        onset.push(0.2 * prev + 0.6 * cur + 0.2 * next);
    }

    let global_onset_rms = {
        let sq: f64 = onset.iter().map(|&v| v * v).sum();
        (sq / n_frames as f64).sqrt()
    };

    // Prefix sums compute adaptive local mean and variance in exact O(N).
    let mut pref = Vec::with_capacity(n_frames + 1);
    let mut pref_sq = Vec::with_capacity(n_frames + 1);
    pref.push(0.0_f64);
    pref_sq.push(0.0_f64);
    for &v in &onset {
        pref.push(pref.last().copied().unwrap_or(0.0) + v);
        pref_sq.push(pref_sq.last().copied().unwrap_or(0.0) + v * v);
    }

    let (k_std, abs_floor_frac, min_spacing_ms) = sensitivity.params();
    let abs_floor = (abs_floor_frac * global_onset_rms).max(1e-6);
    let rms_activity_floor = (0.01 * max_rms).max(SILENCE_RMS_FLOOR);

    let mut beats_ms = Vec::new();
    let mut last_beat_ms: Option<u64> = None;

    for t in 0..n_frames {
        if rms[t] < rms_activity_floor {
            continue;
        }
        let is_peak = match (t > 0, t + 1 < n_frames) {
            (true, true) => onset[t] >= onset[t - 1] && onset[t] > onset[t + 1],
            (false, true) => onset[t] > onset[t + 1],
            (true, false) => onset[t] >= onset[t - 1],
            (false, false) => true,
        };
        if !is_peak {
            continue;
        }

        let lo = t.saturating_sub(ADAPTIVE_HALF_WINDOW);
        let hi = (t + ADAPTIVE_HALF_WINDOW).min(n_frames - 1);
        let count = (hi - lo + 1) as f64;
        let sum = pref[hi + 1] - pref[lo];
        let sum_sq = pref_sq[hi + 1] - pref_sq[lo];
        let mean = sum / count;
        let var = ((sum_sq / count) - mean * mean).max(0.0);
        let std_dev = var.sqrt();
        let threshold = (mean + k_std * std_dev).max(abs_floor);

        if onset[t] <= threshold {
            continue;
        }

        let ms = ((t as f64 * hop as f64 / f64::from(sr)) * 1000.0).round() as u64;
        if let Some(prev) = last_beat_ms
            && ms.saturating_sub(prev) < min_spacing_ms
        {
            continue;
        }
        last_beat_ms = Some(ms);
        beats_ms.push(ms);
    }

    let bpm = estimate_bpm(&beats_ms);
    BeatAnalysis { beats_ms, bpm, duration_ms }
}

fn beat_ticks_for_sequence(beats_ms: &[u64], interval: BeatInterval, range_start: Tick, rate: FrameRate) -> Vec<Tick> {
    let mut out: Vec<Tick> =
        subdivide_beats(beats_ms, interval).into_iter().map(|ms| rate.snap(range_start + Tick::from_units(ms as i64, 1000)).max(Tick::ZERO)).collect();
    out.sort();
    out.dedup();
    out
}

/// Place cyan timeline markers (`"Beat"`) on the active sequence at subdivided beat positions in
/// one undoable edit step.
pub fn apply_beat_markers(s: &mut Session, seq_id: ItemId, beats_ms: &[u64], interval: BeatInterval, range_start: Tick, rate: FrameRate) -> Result<usize> {
    let times = beat_ticks_for_sequence(beats_ms, interval, range_start, rate);
    if times.is_empty() {
        s.toast("No beats detected in audio");
        return Ok(0);
    }
    let added = s.edit("Add Beat Markers", move |pr, _| {
        let mut new_markers = Vec::new();
        let q_existing = pr.sequence(seq_id).ok_or(EngineError::NoSequence)?;
        for &t in &times {
            if !q_existing.markers.iter().any(|m| m.start == t) {
                new_markers.push(t);
            }
        }
        let ids: Vec<u64> = new_markers.iter().map(|_| pr.alloc_id()).collect();
        let q = pr.sequence_mut(seq_id).ok_or(EngineError::NoSequence)?;
        let count = new_markers.len();
        for (t, id) in new_markers.into_iter().zip(ids) {
            q.markers.push(Marker {
                url: String::new(),
                author: String::new(),
                custom_color: None,
                id: MarkerId(id),
                start: t,
                duration: Tick::ZERO,
                name: "Beat".into(),
                comment: String::new(),
                kind: MarkerKind::Comment,
                color: Label::Teal,
            });
        }
        q.markers.sort_by_key(|m| m.start);
        Ok(count)
    })?;
    s.toast(format!("Added {added} beat markers"));
    Ok(added)
}

/// Split selected or active/unlocked video track clips cleanly at subdivided beat timestamps in
/// one undoable edit step (audio tracks stay intact so the music bed is preserved).
pub fn apply_video_cuts(s: &mut Session, seq_id: ItemId, beats_ms: &[u64], interval: BeatInterval, range_start: Tick, rate: FrameRate) -> Result<usize> {
    let times = beat_ticks_for_sequence(beats_ms, interval, range_start, rate);
    if times.is_empty() {
        s.toast("No beats detected in audio");
        return Ok(0);
    }

    let q_ro = s.project.sequence(seq_id).ok_or(EngineError::NoSequence)?;
    let selected_video: Vec<ClipId> = s
        .state
        .selection
        .iter()
        .copied()
        .filter(|cid| q_ro.find_item(*cid).and_then(|(tid, _)| q_ro.track(tid)).is_some_and(|tr| tr.kind == TrackKind::Video && !tr.locked))
        .collect();

    let target_tracks: Vec<TrackId> = if selected_video.is_empty() {
        let targeted: Vec<TrackId> = s
            .targeting()
            .targeted
            .into_iter()
            .filter(|tid| q_ro.track(*tid).is_some_and(|tr| tr.kind == TrackKind::Video && !tr.locked && !tr.items.is_empty()))
            .collect();
        if !targeted.is_empty() { targeted } else { q_ro.video_tracks.iter().filter(|tr| !tr.locked && !tr.items.is_empty()).map(|tr| tr.id).collect() }
    } else {
        Vec::new()
    };

    if selected_video.is_empty() && target_tracks.is_empty() {
        return Err(bad("ai.cutVideoToBeats", "no unlocked video clips found in sequence"));
    }

    let media = s.media.clone();
    let created = s.edit("Auto-Cut Video to Beats", move |pr, _| {
        let snapshot = pr.clone();
        let durations = move |id: ItemId| crate::media_duration(&snapshot, &media, id);
        let min = pr.sequence(seq_id).map(|q| q.settings.frame_rate.frame_duration()).unwrap_or(Tick(1));
        let mut next = pr.next_id;
        let q = pr.sequence_mut(seq_id).ok_or(EngineError::NoSequence)?;
        let mut ctx = filmcraft_edit::EditCtx { next_id: &mut next, media_duration: &durations, media_start: &|_| Tick::ZERO, min_duration: min };
        let mut cuts = 0usize;
        // Latest cut first so the left piece keeps each selected ClipId for earlier cuts.
        for &t in times.iter().rev() {
            let new_ids = if !selected_video.is_empty() {
                filmcraft_edit::razor_items(q, &selected_video, t, &mut ctx)
            } else {
                filmcraft_edit::razor(q, &target_tracks, t, &mut ctx)
            };
            cuts += new_ids.len();
        }
        q.check().map_err(EngineError::Other)?;
        pr.next_id = next;
        Ok(cuts)
    })?;
    s.toast(format!("Applied {created} beat cuts to video"));
    Ok(created)
}

/// Apply finished background beat-detection jobs cleanly on the session thread.
pub fn poll(s: &mut Session) {
    let mut i = 0;
    while i < s.beat_jobs.len() {
        let job = s.jobs.iter().find(|j| j.id == s.beat_jobs[i].job);
        let finished = job.is_none_or(|j| j.progress.finished.load(Ordering::Relaxed));
        let cancelled = job.is_some_and(|j| j.progress.cancel.load(Ordering::Relaxed));
        if !finished {
            i += 1;
            continue;
        }
        let pj = s.beat_jobs.remove(i);
        let Some(analysis) = lock(&pj.analysis).take().filter(|_| !cancelled) else { continue };
        if pj.epoch != s.document_epoch {
            s.error_toast("ai.analyzeBeats", "Beat Sync: the project was replaced; nothing was changed");
            continue;
        }
        s.beat_sync_cache = Some(BeatSyncCache {
            seq: pj.seq,
            audio_sig: pj.audio_sig,
            sensitivity: pj.sensitivity,
            range_start: pj.range_start,
            seq_rate: pj.seq_rate,
            beats_ms: analysis.beats_ms.clone(),
            bpm: analysis.bpm,
        });
        let res = match pj.action {
            BeatPendingAction::AnalyzeOnly => Ok(0),
            BeatPendingAction::AddMarkers => apply_beat_markers(s, pj.seq, &analysis.beats_ms, pj.interval, pj.range_start, pj.seq_rate),
            BeatPendingAction::CutVideo => apply_video_cuts(s, pj.seq, &analysis.beats_ms, pj.interval, pj.range_start, pj.seq_rate),
        };
        if let Err(e) = res {
            s.error_toast("ai.analyzeBeats", format!("Beat Sync: {e}"));
        }
    }
}

fn parse_sensitivity(p: &Value) -> Result<BeatSensitivity> {
    match str_p(p, "sensitivity") {
        Some(raw) => BeatSensitivity::from_str_param(raw).ok_or_else(|| bad("ai.analyzeBeats", format!("unknown sensitivity `{raw}`"))),
        None => Ok(BeatSensitivity::default()),
    }
}

fn run_or_apply(s: &mut Session, p: &Value, default_action: BeatPendingAction, cmd: &str) -> Result<Value> {
    let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let sensitivity = parse_sensitivity(p)?;
    let interval = BeatInterval::from_json(p).map_err(|e| bad(cmd, e))?;
    let action = match str_p(p, "action") {
        Some(raw) => BeatPendingAction::from_str_param(raw).ok_or_else(|| bad(cmd, format!("unknown beat action `{raw}`")))?,
        None => default_action,
    };
    let plan = build_audio_plan(cmd, s, seq, seq_id, p)?;
    let sig = audio_signature(&plan);

    if action != BeatPendingAction::AnalyzeOnly
        && let Some(cache) = s.beat_sync_cache.clone()
        && cache.seq == seq_id
        && cache.audio_sig == sig
        && cache.sensitivity == sensitivity
    {
        let applied = match action {
            BeatPendingAction::AnalyzeOnly => 0,
            BeatPendingAction::AddMarkers => apply_beat_markers(s, seq_id, &cache.beats_ms, interval, cache.range_start, cache.seq_rate)?,
            BeatPendingAction::CutVideo => apply_video_cuts(s, seq_id, &cache.beats_ms, interval, cache.range_start, cache.seq_rate)?,
        };
        return Ok(json!({
            "cached": true,
            "count": cache.beats_ms.len(),
            "bpm": cache.bpm,
            "applied": applied,
            "interval": interval.as_id(),
            "sensitivity": sensitivity.as_id(),
        }));
    }

    let id = next_job_id(s);
    let prog = Arc::new(Progress::default());
    prog.total.store(100, Ordering::Relaxed);
    *lock(&prog.status) = "Analyzing audio beats…".into();
    let result: Arc<Mutex<Option<std::result::Result<Report, String>>>> = Arc::default();
    s.jobs.push(Job { id, label: "AI Beat Sync".into(), progress: prog.clone(), result: result.clone() });

    let analysis_slot: Arc<Mutex<Option<BeatAnalysis>>> = Arc::default();
    let range_start = plan.range_start;
    let seq_rate = plan.seq_rate;
    s.beat_jobs.push(PendingBeatJob {
        job: id,
        seq: seq_id,
        audio_sig: sig,
        sensitivity,
        interval,
        action,
        range_start,
        seq_rate,
        epoch: s.document_epoch,
        analysis: analysis_slot.clone(),
    });

    let project = s.project.clone();
    let provider = s.media.provider(project.clone(), s.services.clone());
    let wait = bool_p(p, "wait").unwrap_or(false);

    let run = crate::export_tools::guard_job(prog.clone(), result.clone(), move || {
        let t0 = web_time::Instant::now();
        let outcome = (|| -> std::result::Result<BeatAnalysis, String> {
            let (mono, sr) = extract_audio_mono(&plan, &project, &provider, &prog, 55.0)?;
            if prog.cancel.load(Ordering::Relaxed) {
                return Err("Cancelled".into());
            }
            *lock(&prog.status) = "Detecting beats…".into();
            prog.done.store(75, Ordering::Relaxed);
            let detected = detect_beats_pcm(&mono, sr, sensitivity);
            prog.done.store(100, Ordering::Relaxed);
            Ok(detected)
        })();
        let rep = match outcome {
            Ok(detected) => {
                let count = detected.beats_ms.len() as u64;
                let bpm = detected.bpm;
                *lock(&analysis_slot) = Some(detected);
                *lock(&prog.status) = format!("Detected {count} beats ({bpm:.0} BPM)");
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
        std::thread::Builder::new().name("orpheus-beat-sync".into()).spawn(run).map_err(|e| EngineError::Other(e.to_string()))?;
    }

    if let Some(Err(e)) = s.jobs.iter().find(|j| j.id == id).and_then(|j| lock(&j.result).clone()) {
        return Err(EngineError::Other(e));
    }

    let mut out = json!({"job": id, "sensitivity": sensitivity.as_id(), "interval": interval.as_id()});
    if wait && let Some(cache) = &s.beat_sync_cache {
        out["count"] = json!(cache.beats_ms.len());
        out["bpm"] = json!(cache.bpm);
        out["beatsMs"] = json!(cache.beats_ms);
    }
    Ok(out)
}

fn analyze_beats_cmd(s: &mut Session, p: &Value) -> Result<Value> {
    run_or_apply(s, p, BeatPendingAction::AnalyzeOnly, "ai.analyzeBeats")
}

fn add_beat_markers_cmd(s: &mut Session, p: &Value) -> Result<Value> {
    run_or_apply(s, p, BeatPendingAction::AddMarkers, "ai.addBeatMarkers")
}

fn cut_video_to_beats_cmd(s: &mut Session, p: &Value) -> Result<Value> {
    run_or_apply(s, p, BeatPendingAction::CutVideo, "ai.cutVideoToBeats")
}

fn status_cmd(s: &mut Session, p: &Value) -> Result<Value> {
    let sens = match str_p(p, "sensitivity") {
        Some(raw) => BeatSensitivity::from_str_param(raw).unwrap_or_default(),
        None => BeatSensitivity::default(),
    };
    let job_running = u64_p(p, "job").is_some_and(|jid| s.jobs.iter().any(|j| j.id == jid && !j.progress.finished.load(Ordering::Relaxed)));
    if let Some(cache) = cached_for_session(s, sens) {
        Ok(json!({
            "cached": true,
            "running": job_running,
            "count": cache.beats_ms.len(),
            "bpm": cache.bpm,
            "beatsMs": cache.beats_ms,
            "sensitivity": cache.sensitivity.as_id(),
        }))
    } else {
        Ok(json!({
            "cached": false,
            "running": job_running,
            "count": 0,
            "bpm": 0.0,
            "beatsMs": [],
            "sensitivity": sens.as_id(),
        }))
    }
}

pub(crate) fn commands() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "ai.analyzeBeats",
            label: "Analyze Beats",
            menu: &[],
            shortcut: None,
            params: r#"{"clips":[id]?,"sensitivity":"low|medium|high"?,"interval":1|2|4?,"action":"analyze|markers|cut"?,"wait":bool?}"#,
            enabled: has_seq,
            run: analyze_beats_cmd,
            journal: true,
            named: NamedKeys::CLIP,
        },
        CommandSpec {
            id: "ai.addBeatMarkers",
            label: "Add Beat Markers",
            menu: &["Markers"],
            shortcut: None,
            params: r#"{"clips":[id]?,"sensitivity":"low|medium|high"?,"interval":1|2|4?,"wait":bool?}"#,
            enabled: has_seq,
            run: add_beat_markers_cmd,
            journal: true,
            named: NamedKeys::CLIP,
        },
        CommandSpec {
            id: "ai.cutVideoToBeats",
            label: "Auto-Cut Video to Beats",
            menu: &["Sequence"],
            shortcut: None,
            params: r#"{"clips":[id]?,"sensitivity":"low|medium|high"?,"interval":1|2|4?,"wait":bool?}"#,
            enabled: has_seq,
            run: cut_video_to_beats_cmd,
            journal: true,
            named: NamedKeys::CLIP,
        },
        CommandSpec {
            id: "ai.beatSyncStatus",
            label: "AI Beat Sync Status",
            menu: &[],
            shortcut: None,
            params: r#"{"sensitivity":"low|medium|high"?,"job":id?}"#,
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
    use std::f32::consts::TAU;

    fn synth_beat_track(sr: u32, bpm: f64, beats: usize, tail_ms: usize, accent_level: f32) -> Vec<f32> {
        let beat_samples = ((60.0 / bpm) * f64::from(sr)).round() as usize;
        let total = beat_samples * beats + (sr as usize * tail_ms) / 1000;
        let mut out = vec![0.0_f32; total];
        for b in 0..beats {
            // Place first beat at +100ms so baseline noise is clear before beat 0.
            let start = (sr as usize / 10) + b * beat_samples;
            let amp = if b % 2 == 0 { 0.85 } else { accent_level };
            let click_len = (sr as usize * 35) / 1000;
            for i in 0..click_len {
                if let Some(slot) = out.get_mut(start + i) {
                    let t = i as f32 / sr as f32;
                    let env = (-t * 65.0).exp();
                    let tone = (t * 95.0 * TAU).sin() + 0.35 * (t * 2200.0 * TAU).sin();
                    *slot = (tone * env * amp).clamp(-1.0, 1.0);
                }
            }
        }
        out
    }

    #[test]
    fn detects_120_bpm_synthetic_beats_and_subdivides_cleanly() {
        let sr = 16_000;
        let pcm = synth_beat_track(sr, 120.0, 8, 500, 0.85);
        let analysis = detect_beats_pcm(&pcm, sr, BeatSensitivity::Medium);

        assert_eq!(analysis.beats_ms.len(), 8, "beats={:?}", analysis.beats_ms);
        assert!((analysis.bpm - 120.0).abs() <= 2.5, "bpm={}", analysis.bpm);
        for (idx, &ms) in analysis.beats_ms.iter().enumerate() {
            let expected = 100 + (idx as u64) * 500;
            assert!(ms.abs_diff(expected) <= 25, "beat {idx}: got {ms}ms, expected ~{expected}ms");
        }

        assert_eq!(subdivide_beats(&analysis.beats_ms, BeatInterval::EveryBeat).len(), 8);
        assert_eq!(subdivide_beats(&analysis.beats_ms, BeatInterval::Every2Beats).len(), 4);
        assert_eq!(subdivide_beats(&analysis.beats_ms, BeatInterval::Every4Beats).len(), 2);
    }

    #[test]
    fn sensitivity_controls_weak_offbeat_detection_and_hostile_inputs_stay_safe() {
        let sr = 16_000;
        // Strong downbeats (0.85) alternating with soft offbeats (0.12)
        let pcm = synth_beat_track(sr, 120.0, 8, 500, 0.12);
        let low = detect_beats_pcm(&pcm, sr, BeatSensitivity::Low);
        let high = detect_beats_pcm(&pcm, sr, BeatSensitivity::High);
        assert!(high.beats_ms.len() >= low.beats_ms.len(), "high={} low={}", high.beats_ms.len(), low.beats_ms.len());
        assert!(low.beats_ms.len() >= 4, "expected downbeats on Low sensitivity, got {:?}", low.beats_ms);

        // Silence and non-finite / tiny hostile inputs must never panic and yield 0 beats.
        assert!(detect_beats_pcm(&[], sr, BeatSensitivity::High).beats_ms.is_empty());
        assert!(detect_beats_pcm(&vec![0.0; 16_000], sr, BeatSensitivity::High).beats_ms.is_empty());
        assert!(detect_beats_pcm(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY].repeat(200), sr, BeatSensitivity::Medium).beats_ms.is_empty());
        assert!(detect_beats_pcm(&[0.5; 10], sr, BeatSensitivity::Medium).beats_ms.is_empty());
    }

    #[test]
    fn add_markers_and_cut_video_to_beats_are_undoable_without_cutting_audio() {
        let sr = 16_000;
        let pcm = synth_beat_track(sr, 120.0, 6, 400, 0.85);
        let wav = filmcraft_media::wav::write_wav16(&pcm, 1, sr);

        let mut s = Session::default();
        let audio_item = crate::commands::import_bytes(&mut s, "beat_track.wav", wav.into(), None).unwrap();
        s.state.project_selection = vec![audio_item];
        s.execute("file.newSequenceFromClip", json!({"items": [audio_item.0]})).unwrap();

        let dur_sec = s.active_sequence().unwrap().duration().seconds().max(3.5);
        let bars = s.execute("file.newBarsAndTone", json!({"duration": dur_sec})).unwrap()["item"].as_u64().unwrap();
        s.edit_sequence("Add Video For Beat Cuts", |q, ctx, _| {
            let v_id = ClipId(ctx.alloc());
            let mut v = q.audio_tracks[0].items[0].clone();
            v.id = v_id;
            v.item = ItemId(bars);
            q.video_tracks[0].items.push(v);
            Ok(())
        })
        .unwrap();

        let m_res = s.execute("ai.addBeatMarkers", json!({"sensitivity": "medium", "interval": 2, "wait": true})).unwrap();
        assert_eq!(m_res["count"].as_u64().unwrap(), 6);
        assert_eq!(s.active_sequence().unwrap().markers.len(), 3);
        assert_eq!(s.active_sequence().unwrap().markers[0].name, "Beat");

        // Cutting video reuses cached beat analysis and splits V1 cleanly while leaving A1 uncut.
        let c_res = s.execute("ai.cutVideoToBeats", json!({"sensitivity": "medium", "interval": 2, "wait": true})).unwrap();
        assert_eq!(c_res["cached"], true);
        let q_after_cut = s.active_sequence().unwrap();
        assert!(q_after_cut.video_tracks[0].items.len() >= 3, "expected split video pieces, got {}", q_after_cut.video_tracks[0].items.len());
        assert_eq!(q_after_cut.audio_tracks[0].items.len(), 1, "audio bed must remain unsplit");

        // Undo restores original single video clip, then undoes markers.
        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(s.active_sequence().unwrap().video_tracks[0].items.len(), 1);
        s.execute("edit.undo", json!({})).unwrap();
        assert!(s.active_sequence().unwrap().markers.is_empty());
    }
}
