//! Offline AI / DSP silence detection and automatic pause removal (`ai.detectSilence`,
//! `ai.removePauses`).
//!
//! Analyses decoded PCM audio frames using pure-Rust RMS amplitude calculation in dBFS across
//! 10 ms windows, identifies contiguous silent segments at or above the minimum pause duration,
//! insets each cut by the requested padding margin so word starts and ends keep natural breathing
//! room, and splits + ripple-deletes those ranges across the sequence in a single undo step.

use filmcraft_edit::transcript as tx;
use filmcraft_project::{ClipId, Sequence, TrackItem};
use filmcraft_render::SourceProvider;
use filmcraft_time::{TICKS_PER_SECOND, Tick, TimeRange};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, NamedKeys, clips_p, f64_p, has_seq, with_links};
use crate::{Result, Session};

/// Analysis sample rate (Hz) used when decoding timeline clips for RMS envelope detection.
const ANALYSIS_SAMPLE_RATE: u32 = 16_000;
/// RMS analysis window duration (10 ms = 1/100 s).
const WINDOW_DIVISOR: usize = 100;
/// Maximum samples analysed per clip (2 hours at 16 kHz) to bound allocations on untrusted input.
const MAX_CLIP_SAMPLES: usize = 2 * 3600 * ANALYSIS_SAMPLE_RATE as usize;

/// Parameters for offline silence detection and pause removal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SilenceOptions {
    /// Silence threshold in dBFS (`-60.0..=-15.0`, default `-38.0`).
    pub threshold_db: f64,
    /// Minimum contiguous silent duration in milliseconds (`150.0..=2000.0`, default `450.0`).
    pub min_pause_ms: f64,
    /// Natural breathing room kept before and after speech in milliseconds (`0.0..=1000.0`, default `80.0`).
    pub padding_ms: f64,
}

impl Default for SilenceOptions {
    fn default() -> Self {
        Self { threshold_db: -38.0, min_pause_ms: 450.0, padding_ms: 80.0 }
    }
}

impl SilenceOptions {
    pub fn sanitized(self) -> Self {
        let threshold_db = if self.threshold_db.is_finite() { self.threshold_db.clamp(-60.0, -15.0) } else { -38.0 };
        let min_pause_ms = if self.min_pause_ms.is_finite() { self.min_pause_ms.clamp(150.0, 2000.0) } else { 450.0 };
        let padding_ms = if self.padding_ms.is_finite() { self.padding_ms.clamp(0.0, 1000.0) } else { 80.0 };
        Self { threshold_db, min_pause_ms, padding_ms }
    }

    pub fn from_json(p: &Value) -> Self {
        let d = Self::default();
        Self {
            threshold_db: f64_p(p, "thresholdDb").unwrap_or(d.threshold_db),
            min_pause_ms: f64_p(p, "minPauseMs").unwrap_or(d.min_pause_ms),
            padding_ms: f64_p(p, "paddingMs").unwrap_or(d.padding_ms),
        }
        .sanitized()
    }
}

pub(crate) fn commands() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "ai.detectSilence",
            label: "Detect Silence",
            menu: &[],
            shortcut: None,
            params: r#"{"clips":[id]?,"thresholdDb":-60..-15=-38,"minPauseMs":150..2000=450,"paddingMs":0..1000=80}"#,
            enabled: has_seq,
            run: detect_cmd,
            journal: false,
            named: NamedKeys::CLIP,
        },
        CommandSpec {
            id: "ai.removePauses",
            label: "Auto-Cut Silence",
            menu: &["Clip"],
            shortcut: None,
            params: r#"{"clips":[id]?,"thresholdDb":-60..-15=-38,"minPauseMs":150..2000=450,"paddingMs":0..1000=80}"#,
            enabled: has_seq,
            run: remove_pauses_cmd,
            journal: true,
            named: NamedKeys::CLIP,
        },
    ]
}

/// Detect silent intervals in an in-memory multi-channel PCM buffer using 10 ms RMS windows in dBFS.
/// Returns clip-local `TimeRange`s after applying `min_pause_ms` and `padding_ms`.
pub fn detect_silence_pcm(channels: &[Vec<f32>], sample_rate: u32, gain_db: f64, opts: SilenceOptions) -> Vec<TimeRange> {
    let opts = opts.sanitized();
    let sr = sample_rate.max(100);
    let total_samples = channels.iter().map(Vec::len).max().unwrap_or(0).min(MAX_CLIP_SAMPLES);
    if total_samples == 0 || channels.is_empty() {
        return Vec::new();
    }

    let window_samples = ((sr as usize) / WINDOW_DIVISOR).max(1);
    let gain_lin = if gain_db.is_finite() { 10.0_f64.powf(gain_db.clamp(-96.0, 48.0) / 20.0) } else { 1.0 };
    let thresh_lin = 10.0_f64.powf(opts.threshold_db / 20.0);
    let min_pause_tick = Tick::from_seconds_f64(opts.min_pause_ms / 1000.0);
    let padding_tick = Tick::from_seconds_f64(opts.padding_ms / 1000.0);
    let total_dur_tick = Tick::from_units(total_samples as i64, i64::from(sr));

    let num_ch = channels.len().max(1);
    let mut out = Vec::new();
    let mut silent_start: Option<usize> = None;
    let mut pos = 0usize;

    while pos < total_samples {
        let end = (pos + window_samples).min(total_samples);
        let frames = (end - pos).max(1);
        let mut sum_sq = 0.0_f64;
        for ch in channels {
            if let Some(slice) = ch.get(pos..end) {
                for &sample in slice {
                    let v = if sample.is_finite() { f64::from(sample) } else { 0.0 };
                    sum_sq += v * v;
                }
            }
        }
        let rms = (sum_sq / ((frames * num_ch) as f64)).sqrt() * gain_lin;
        let is_silent = rms < thresh_lin;

        if is_silent {
            if silent_start.is_none() {
                silent_start = Some(pos);
            }
        } else if let Some(s0) = silent_start.take() {
            push_silent_interval(&mut out, s0, pos, sr, total_samples, total_dur_tick, min_pause_tick, padding_tick);
        }
        pos = end;
    }

    if let Some(s0) = silent_start.take() {
        push_silent_interval(&mut out, s0, total_samples, sr, total_samples, total_dur_tick, min_pause_tick, padding_tick);
    }

    out
}

#[allow(clippy::too_many_arguments)]
fn push_silent_interval(
    out: &mut Vec<TimeRange>,
    start_sample: usize,
    end_sample: usize,
    sr: u32,
    total_samples: usize,
    total_dur_tick: Tick,
    min_pause_tick: Tick,
    padding_tick: Tick,
) {
    let raw_start = Tick::from_units(start_sample as i64, i64::from(sr));
    let raw_end = Tick::from_units(end_sample as i64, i64::from(sr));
    if raw_end - raw_start < min_pause_tick {
        return;
    }
    let cut_start = if start_sample > 0 { raw_start + padding_tick } else { Tick::ZERO };
    let cut_end = if end_sample < total_samples { raw_end - padding_tick } else { total_dur_tick };
    if cut_end > cut_start {
        out.push(TimeRange::from_bounds(cut_start, cut_end));
    }
}

/// Resolve the audio track items to analyse: selected clips (expanded with linked partners), or the
/// first unlocked audio track with enabled clips when nothing with audio is selected.
fn target_audio_clips<'a>(s: &Session, seq: &'a Sequence, p: &Value) -> Vec<&'a TrackItem> {
    let explicit = clips_p(s, p);
    let expanded = with_links(s, &explicit);
    if !expanded.is_empty() {
        let selected: Vec<&'a TrackItem> =
            seq.audio_tracks.iter().filter(|t| !t.locked).flat_map(|t| t.items.iter()).filter(|it| it.enabled && expanded.contains(&it.id)).collect();
        if !selected.is_empty() {
            return selected;
        }
    }
    seq.audio_tracks
        .iter()
        .filter(|t| !t.locked)
        .find_map(|t| {
            let items: Vec<&'a TrackItem> = t.items.iter().filter(|it| it.enabled).collect();
            (!items.is_empty()).then_some(items)
        })
        .unwrap_or_default()
}

/// Detect frame-snapped sequence `TimeRange`s of silent pauses across the target audio clips.
pub fn detect_sequence_silence(s: &Session, p: &Value, opts: SilenceOptions) -> Vec<TimeRange> {
    let Some(seq) = s.active_sequence() else { return Vec::new() };
    let opts = opts.sanitized();
    let targets = target_audio_clips(s, seq, p);
    if targets.is_empty() {
        return Vec::new();
    }

    let provider = s.media.provider(s.project.clone(), s.services.clone());
    let rate = seq.settings.frame_rate;
    let fd = rate.frame_duration();
    let sr = ANALYSIS_SAMPLE_RATE;
    let mut ranges = Vec::new();

    for it in targets {
        if !it.enabled || it.reverse || it.frame_hold.is_some() || !it.speed.is_finite() || it.speed <= 0.0 {
            continue;
        }
        let Some(src) = provider.source(it.item) else { continue };
        if !src.info().has_audio() {
            continue;
        }
        let start_sample = it.source_in.to_units_floor(i64::from(sr));
        let source_ticks = ((it.duration.0 as f64) * it.speed).round() as i64;
        let len_samples = Tick(source_ticks.max(0)).to_units_floor(i64::from(sr)).clamp(0, MAX_CLIP_SAMPLES as i64) as usize;
        if len_samples == 0 {
            continue;
        }
        let Ok(buf) = src.audio_stream(it.audio_stream, start_sample, len_samples, sr) else { continue };
        let local_ranges = detect_silence_pcm(&buf.channels, sr, it.gain_db, opts);
        for local in local_ranges {
            let tl_s = it.start + Tick(((local.start.0 as f64) / it.speed).round() as i64);
            let tl_e = it.start + Tick(((local.end().0 as f64) / it.speed).round() as i64);
            let clamped_s = tl_s.clamp(it.start, it.end());
            let clamped_e = tl_e.clamp(it.start, it.end());
            let mut s_snap = rate.snap(clamped_s);
            if s_snap < clamped_s {
                s_snap += fd;
            }
            let e_snap = rate.snap(clamped_e);
            if e_snap > s_snap {
                ranges.push(TimeRange::from_bounds(s_snap, e_snap));
            }
        }
    }

    tx::merge_ranges(ranges)
}

fn detect_cmd(s: &mut Session, p: &Value) -> Result<Value> {
    let opts = SilenceOptions::from_json(p);
    let ranges = detect_sequence_silence(s, p, opts);
    let total_ticks: i64 = ranges.iter().map(|r| r.duration.0).sum();
    Ok(json!({
        "pauses": ranges.len(),
        "ticks": total_ticks,
        "seconds": total_ticks as f64 / TICKS_PER_SECOND as f64,
        "ranges": ranges.iter().map(|r| json!({"start": r.start.0, "end": r.end().0, "duration": r.duration.0})).collect::<Vec<_>>(),
    }))
}

fn remove_pauses_cmd(s: &mut Session, p: &Value) -> Result<Value> {
    let opts = SilenceOptions::from_json(p);
    let ranges = detect_sequence_silence(s, p, opts);
    let count = ranges.len();
    if count == 0 {
        return Ok(json!({"removed": 0, "ticks": 0, "seconds": 0.0}));
    }
    let removed_ticks = s.edit_sequence("Remove Pauses", |q, ctx, st| {
        let total = tx::ripple_delete_ranges(q, ranges, ctx);
        let surviving: Vec<ClipId> = st.selection.iter().copied().filter(|id| q.find_item(*id).is_some()).collect();
        st.selection = surviving;
        Ok(total)
    })?;
    Ok(json!({
        "removed": count,
        "ticks": removed_ticks.0,
        "seconds": removed_ticks.0 as f64 / TICKS_PER_SECOND as f64,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    fn synth_tone_silence_tone(sr: u32) -> Vec<f32> {
        let mut samples = Vec::with_capacity(sr as usize * 3);
        // 0.0s .. 1.0s: 440 Hz sine wave at -6 dBFS (amplitude 0.5)
        for i in 0..sr {
            let phase = (i as f32 / sr as f32) * 440.0 * TAU;
            samples.push(phase.sin() * 0.5);
        }
        // 1.0s .. 2.0s: complete silence (1000 ms)
        samples.extend(std::iter::repeat_n(0.0_f32, sr as usize));
        // 2.0s .. 3.0s: 440 Hz sine wave at -6 dBFS
        for i in 0..sr {
            let phase = (i as f32 / sr as f32) * 440.0 * TAU;
            samples.push(phase.sin() * 0.5);
        }
        samples
    }

    #[test]
    fn detects_silent_interval_and_applies_padding() {
        let sr = 16_000;
        let pcm = vec![synth_tone_silence_tone(sr)];
        let opts = SilenceOptions { threshold_db: -38.0, min_pause_ms: 450.0, padding_ms: 80.0 };
        let ranges = detect_silence_pcm(&pcm, sr, 0.0, opts);
        assert_eq!(ranges.len(), 1);
        let r = ranges[0];
        // Raw silence is [1.00s, 2.00s]; with 80 ms padding on each side, cut is [1.08s, 1.92s].
        assert!((r.start.seconds() - 1.08).abs() < 0.015, "start={}", r.start.seconds());
        assert!((r.end().seconds() - 1.92).abs() < 0.015, "end={}", r.end().seconds());
    }

    #[test]
    fn skips_pauses_shorter_than_minimum_duration() {
        let sr = 16_000;
        let pcm = vec![synth_tone_silence_tone(sr)];
        let opts = SilenceOptions { threshold_db: -38.0, min_pause_ms: 1200.0, padding_ms: 80.0 };
        let ranges = detect_silence_pcm(&pcm, sr, 0.0, opts);
        assert!(ranges.is_empty());
    }

    #[test]
    fn remove_pauses_splits_and_ripples_linked_video_and_audio_with_undo() {
        let sr = 16_000u32;
        let samples = synth_tone_silence_tone(sr);
        let wav = filmcraft_media::wav::write_wav16(&samples, 1, sr);

        let mut s = Session::default();
        let item = crate::commands::import_bytes(&mut s, "speech_pause.wav", wav.into(), None).unwrap();
        s.state.project_selection = vec![item];
        s.execute("file.newSequenceFromClip", json!({"items": [item.0]})).unwrap();

        // Also place a video generator on V1 over the same [0..3s] span and link it with the audio clip
        let bars = s.execute("file.newBarsAndTone", json!({"duration": 3.0})).unwrap()["item"].as_u64().unwrap();
        let seq_id = s.state.active_sequence.unwrap();
        let dur_before = s.active_sequence().unwrap().duration();
        let a_clip = s.active_sequence().unwrap().audio_tracks[0].items[0].id;
        s.edit_sequence("Add Linked Video", |q, ctx, _| {
            let v_id = ClipId(ctx.alloc());
            let mut v_item = q.audio_tracks[0].items[0].clone();
            v_item.id = v_id;
            v_item.item = filmcraft_project::ItemId(bars);
            v_item.link = Some(a_clip.0);
            q.video_tracks[0].items.push(v_item);
            q.audio_tracks[0].items[0].link = Some(a_clip.0);
            Ok(())
        })
        .unwrap();
        let _ = seq_id;

        let preview = s.execute("ai.detectSilence", json!({"thresholdDb": -38.0, "minPauseMs": 450.0, "paddingMs": 80.0})).unwrap();
        assert_eq!(preview["pauses"], 1);
        assert!(preview["seconds"].as_f64().unwrap() > 0.75);

        let res = s.execute("ai.removePauses", json!({"thresholdDb": -38.0, "minPauseMs": 450.0, "paddingMs": 80.0})).unwrap();
        assert_eq!(res["removed"], 1);

        let q = s.active_sequence().unwrap();
        // Both V1 and A1 were split into 2 contiguous pieces and rippled left in sync
        assert_eq!(q.video_tracks[0].items.len(), 2);
        assert_eq!(q.audio_tracks[0].items.len(), 2);
        assert_eq!(q.video_tracks[0].items[0].range(), q.audio_tracks[0].items[0].range());
        assert_eq!(q.video_tracks[0].items[1].range(), q.audio_tracks[0].items[1].range());
        assert_eq!(q.video_tracks[0].items[1].start, q.video_tracks[0].items[0].end());
        assert!(q.duration() < dur_before);

        // Undo restores the original unsplit 3-second timeline
        s.execute("edit.undo", json!({})).unwrap();
        let q_undo = s.active_sequence().unwrap();
        assert_eq!(q_undo.video_tracks[0].items.len(), 1);
        assert_eq!(q_undo.audio_tracks[0].items.len(), 1);
        assert_eq!(q_undo.duration(), dur_before);
    }
}
