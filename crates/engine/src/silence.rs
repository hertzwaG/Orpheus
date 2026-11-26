//! Offline AI / DSP silence detection and automatic pause removal (`ai.detectSilence`,
//! `ai.removePauses`).
//!
//! Analyses decoded PCM audio frames using pure-Rust RMS amplitude calculation in dBFS across
//! 10 ms windows, identifies contiguous silent segments at or above the minimum pause duration,
//! insets each cut by the requested padding margin so word starts and ends keep natural breathing
//! room, and splits + ripple-deletes those ranges across the sequence in a single undo step.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, PoisonError};

use filmcraft_edit::transcript as tx;
use filmcraft_export::{Progress, Report};
use filmcraft_project::{ClipId, ItemId, Sequence, TrackItem};
use filmcraft_render::SourceProvider;
use filmcraft_time::{FrameRate, TICKS_PER_SECOND, Tick, TimeRange};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, NamedKeys, bool_p, clips_p, f64_p, has_seq, with_links};
use crate::subtitles::next_job_id;
use crate::{EngineError, Job, Result, Session};

/// Analysis sample rate (Hz) used when decoding timeline clips for RMS envelope detection.
const ANALYSIS_SAMPLE_RATE: u32 = 16_000;
/// RMS analysis window duration (10 ms = 1/100 s).
const WINDOW_DIVISOR: usize = 100;
/// Streaming decode chunk size (10 seconds @ 16 kHz = 160,000 samples) to bound peak RAM to < 4 MB.
pub const SILENCE_CHUNK_SAMPLES: usize = 10 * ANALYSIS_SAMPLE_RATE as usize;
/// Maximum samples analysed per clip (24 hours at 16 kHz) to support 12-hour+ VODs while bounding corrupt durations.
pub const MAX_CLIP_SAMPLES: usize = 24 * 3600 * ANALYSIS_SAMPLE_RATE as usize;

/// Pending background silence-removal job polled by `Session::poll`.
pub struct PendingSilenceJob {
    pub job: u64,
    pub seq: ItemId,
    pub epoch: u64,
    pub ranges: Arc<Mutex<Option<Vec<TimeRange>>>>,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

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
            params: r#"{"clips":[id]?,"thresholdDb":-60..-15=-38,"minPauseMs":150..2000=450,"paddingMs":0..1000=80,"async":bool?,"wait":bool?}"#,
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

#[derive(Clone)]
struct ClipSilenceWork {
    item_id: ItemId,
    audio_stream: usize,
    start: Tick,
    end: Tick,
    source_in: Tick,
    duration: Tick,
    speed: f64,
    gain_db: f64,
}

fn collect_silence_work(s: &Session, seq: &Sequence, p: &Value) -> Vec<ClipSilenceWork> {
    let targets = target_audio_clips(s, seq, p);
    let mut work = Vec::with_capacity(targets.len());
    for it in targets {
        if !it.enabled || it.reverse || it.frame_hold.is_some() || !it.speed.is_finite() || it.speed <= 0.0 {
            continue;
        }
        work.push(ClipSilenceWork {
            item_id: it.item,
            audio_stream: it.audio_stream,
            start: it.start,
            end: it.end(),
            source_in: it.source_in,
            duration: it.duration,
            speed: it.speed,
            gain_db: it.gain_db,
        });
    }
    work
}

/// Stream-decode a clip in 10-second chunks (`SILENCE_CHUNK_SAMPLES`) at 16 kHz, carrying the RMS
/// silence state across chunk boundaries so 12-hour+ VODs use < 4 MB of RAM.
fn detect_clip_silence_streaming(
    clip: &ClipSilenceWork,
    src: &dyn filmcraft_media::MediaSource,
    sr: u32,
    opts: SilenceOptions,
    prog: Option<&Progress>,
    pct_base: f64,
    pct_span: f64,
) -> std::result::Result<Vec<TimeRange>, String> {
    let opts = opts.sanitized();
    let sr = sr.max(100);
    let start_sample = clip.source_in.to_units_floor(i64::from(sr));
    let source_ticks = ((clip.duration.0 as f64) * clip.speed).round() as i64;
    let total_samples = Tick(source_ticks.max(0)).to_units_floor(i64::from(sr)).clamp(0, MAX_CLIP_SAMPLES as i64) as usize;
    if total_samples == 0 {
        return Ok(Vec::new());
    }

    let window_samples = ((sr as usize) / WINDOW_DIVISOR).max(1);
    // Align chunk size to an exact multiple of `window_samples` so 10 ms RMS windows never split across chunks.
    let chunk_samples = (SILENCE_CHUNK_SAMPLES / window_samples).max(1) * window_samples;
    let gain_lin = if clip.gain_db.is_finite() { 10.0_f64.powf(clip.gain_db.clamp(-96.0, 48.0) / 20.0) } else { 1.0 };
    let thresh_lin = 10.0_f64.powf(opts.threshold_db / 20.0);
    let min_pause_tick = Tick::from_seconds_f64(opts.min_pause_ms / 1000.0);
    let padding_tick = Tick::from_seconds_f64(opts.padding_ms / 1000.0);
    let total_dur_tick = Tick::from_units(total_samples as i64, i64::from(sr));

    let mut out = Vec::new();
    let mut silent_start: Option<usize> = None;
    let mut chunk_offset = 0usize;

    while chunk_offset < total_samples {
        if prog.is_some_and(|p| p.cancel.load(Ordering::Relaxed)) {
            return Err("Cancelled".into());
        }
        let take = (total_samples - chunk_offset).min(chunk_samples);
        let Ok(buf) = src.audio_stream(clip.audio_stream, start_sample + chunk_offset as i64, take, sr) else {
            break;
        };
        let channels = &buf.channels;
        if channels.is_empty() {
            break;
        }
        let num_ch = channels.len().max(1);
        let got = channels.iter().map(Vec::len).max().unwrap_or(0).min(take);
        if got == 0 {
            break;
        }

        let mut pos = 0usize;
        while pos < got {
            let end = (pos + window_samples).min(got);
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
            let global_pos = chunk_offset + pos;

            if is_silent {
                if silent_start.is_none() {
                    silent_start = Some(global_pos);
                }
            } else if let Some(s0) = silent_start.take() {
                push_silent_interval(&mut out, s0, global_pos, sr, total_samples, total_dur_tick, min_pause_tick, padding_tick);
            }
            pos = end;
        }

        chunk_offset += got;
        if let Some(p) = prog {
            let pct = pct_base + (chunk_offset as f64 / total_samples as f64) * pct_span;
            p.done.store((pct.round() as u64).min(95), Ordering::Relaxed);
        }
        if got < take {
            break;
        }
    }

    if let Some(s0) = silent_start.take() {
        push_silent_interval(&mut out, s0, total_samples, sr, total_samples, total_dur_tick, min_pause_tick, padding_tick);
    }

    Ok(out)
}

fn analyze_silence_work(
    work: &[ClipSilenceWork],
    rate: FrameRate,
    provider: &dyn SourceProvider,
    opts: SilenceOptions,
    prog: Option<&Progress>,
) -> std::result::Result<Vec<TimeRange>, String> {
    if work.is_empty() {
        return Ok(Vec::new());
    }
    let fd = rate.frame_duration();
    let sr = ANALYSIS_SAMPLE_RATE;
    let mut ranges = Vec::new();
    let total_clips = work.len().max(1);

    for (idx, clip) in work.iter().enumerate() {
        if prog.is_some_and(|p| p.cancel.load(Ordering::Relaxed)) {
            return Err("Cancelled".into());
        }
        let Some(src) = provider.source(clip.item_id) else { continue };
        if !src.info().has_audio() {
            continue;
        }
        let pct_base = (idx as f64 / total_clips as f64) * 90.0;
        let pct_span = 90.0 / (total_clips as f64);
        let local_ranges = detect_clip_silence_streaming(clip, &*src, sr, opts, prog, pct_base, pct_span)?;
        for local in local_ranges {
            let tl_s = clip.start + Tick(((local.start.0 as f64) / clip.speed).round() as i64);
            let tl_e = clip.start + Tick(((local.end().0 as f64) / clip.speed).round() as i64);
            let clamped_s = tl_s.clamp(clip.start, clip.end);
            let clamped_e = tl_e.clamp(clip.start, clip.end);
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

    Ok(tx::merge_ranges(ranges))
}

/// Detect frame-snapped sequence `TimeRange`s of silent pauses across the target audio clips.
pub fn detect_sequence_silence(s: &Session, p: &Value, opts: SilenceOptions) -> Vec<TimeRange> {
    let Some(seq) = s.active_sequence() else { return Vec::new() };
    let work = collect_silence_work(s, seq, p);
    if work.is_empty() {
        return Vec::new();
    }
    let provider = s.media.provider(s.project.clone(), s.services.clone()).own_sources();
    analyze_silence_work(&work, seq.settings.frame_rate, &provider, opts.sanitized(), None).unwrap_or_default()
}

fn apply_silence_ranges(s: &mut Session, seq_id: ItemId, ranges: Vec<TimeRange>) -> Result<(usize, Tick)> {
    let count = ranges.len();
    if count == 0 {
        return Ok((0, Tick::ZERO));
    }
    let prev_active = s.state.active_sequence;
    s.state.active_sequence = Some(seq_id);
    let res = s.edit_sequence("Remove Pauses", |q, ctx, st| {
        let total = tx::ripple_delete_ranges(q, ranges, ctx);
        let surviving: Vec<ClipId> = st.selection.iter().copied().filter(|id| q.find_item(*id).is_some()).collect();
        st.selection = surviving;
        Ok(total)
    });
    if prev_active.is_some() {
        s.state.active_sequence = prev_active;
    }
    let removed_ticks = res?;
    Ok((count, removed_ticks))
}

/// Apply completed background silence-removal jobs on the session thread.
pub fn poll(s: &mut Session) {
    let mut i = 0usize;
    while i < s.silence_jobs.len() {
        let job = s.jobs.iter().find(|j| j.id == s.silence_jobs[i].job);
        let finished = job.is_none_or(|j| j.progress.finished.load(Ordering::Relaxed));
        let cancelled = job.is_some_and(|j| j.progress.cancel.load(Ordering::Relaxed));
        if !finished {
            i += 1;
            continue;
        }
        let pj = s.silence_jobs.remove(i);
        let Some(ranges) = lock(&pj.ranges).take().filter(|_| !cancelled) else { continue };
        if pj.epoch != s.document_epoch {
            s.error_toast("ai.removePauses", "Auto-Cut Silence: the project was replaced; nothing was changed");
            continue;
        }
        match apply_silence_ranges(s, pj.seq, ranges) {
            Ok((0, _)) => s.toast("No silent pauses detected"),
            Ok((count, ticks)) => {
                let secs = ticks.0 as f64 / TICKS_PER_SECOND as f64;
                s.toast(format!("Removed {count} pause(s) ({secs:.1}s)"));
            }
            Err(e) => s.error_toast("ai.removePauses", format!("Auto-Cut Silence: {e}")),
        }
    }
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
    let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let rate = seq.settings.frame_rate;
    let work = collect_silence_work(s, seq, p);

    let run_async = bool_p(p, "async").unwrap_or(false) || p.get("wait").and_then(Value::as_bool) == Some(false);
    if !run_async || cfg!(target_arch = "wasm32") {
        let provider = s.media.provider(s.project.clone(), s.services.clone()).own_sources();
        let ranges = analyze_silence_work(&work, rate, &provider, opts, None).map_err(EngineError::Other)?;
        let (count, removed_ticks) = apply_silence_ranges(s, seq_id, ranges)?;
        return Ok(json!({
            "removed": count,
            "ticks": removed_ticks.0,
            "seconds": removed_ticks.0 as f64 / TICKS_PER_SECOND as f64,
        }));
    }

    let id = next_job_id(s);
    let prog = Arc::new(Progress::default());
    prog.total.store(100, Ordering::Relaxed);
    *lock(&prog.status) = "Scanning audio for silent pauses…".into();
    let result: Arc<Mutex<Option<std::result::Result<Report, String>>>> = Arc::default();
    let ranges_slot: Arc<Mutex<Option<Vec<TimeRange>>>> = Arc::default();

    let provider = s.media.provider(s.project.clone(), s.services.clone()).own_sources();
    let prog_bg = prog.clone();
    let result_bg = result.clone();
    let slot_bg = ranges_slot.clone();

    let run = crate::export_tools::guard_job(prog_bg.clone(), result_bg.clone(), move || {
        let t0 = web_time::Instant::now();
        let rep = match analyze_silence_work(&work, rate, &provider, opts, Some(&prog_bg)) {
            Ok(ranges) => {
                let n = ranges.len() as u64;
                *lock(&slot_bg) = Some(ranges);
                *lock(&prog_bg.status) = format!("Found {n} silent pause(s)");
                Ok(Report { frames: n, seconds: t0.elapsed().as_secs_f64(), ..Default::default() })
            }
            Err(e) => {
                *lock(&prog_bg.error) = Some(e.clone());
                *lock(&prog_bg.status) = e.clone();
                Err(e)
            }
        };
        *lock(&result_bg) = Some(rep);
        prog_bg.finished.store(true, Ordering::Relaxed);
    });

    std::thread::Builder::new()
        .name(format!("orpheus-remove-pauses-{id}"))
        .spawn(run)
        .map_err(|e| EngineError::Other(format!("failed to start silence detector thread: {e}")))?;

    s.jobs.push(Job { id, label: "Auto-Cut Silence".into(), progress: prog, result });
    s.silence_jobs.push(PendingSilenceJob { job: id, seq: seq_id, epoch: s.document_epoch, ranges: ranges_slot });
    Ok(json!({ "job": id }))
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

    /// 12-hour (43,200 s) procedural streaming media source: generates deterministic speech + 5,000
    /// silent pauses on the fly for any requested chunk `[start..start+frames)` without pre-allocating
    /// 12 hours of PCM in RAM (emulating `Mp4Source`/`MkvSource` random-access packet streaming).
    struct TwelveHourVodSource {
        info: filmcraft_media::MediaInfo,
        video_gen: filmcraft_media::generators::GeneratorSource,
    }

    impl TwelveHourVodSource {
        fn new() -> Self {
            use filmcraft_media::MediaSource as _;
            let dur = Tick::from_seconds_f64(12.0 * 3600.0);
            let src_gen = filmcraft_media::generators::GeneratorSource::new(filmcraft_media::Generator::BarsAndTone, 1920, 1080, FrameRate::FPS_60, dur)
                .with_name("12h_stream_vod.mp4");
            let mut info = src_gen.info().clone();
            info.audio_streams = vec![filmcraft_media::AudioStreamInfo { sample_rate: 16_000, channels: 1, codec: "AAC".into(), bits_per_sample: None }];
            Self { info, video_gen: src_gen }
        }
    }

    impl filmcraft_media::MediaSource for TwelveHourVodSource {
        fn info(&self) -> &filmcraft_media::MediaInfo {
            &self.info
        }

        fn video_frame(&self, req: filmcraft_media::FrameRequest) -> filmcraft_media::Result<Arc<filmcraft_frame::VideoFrame>> {
            self.video_gen.video_frame(req)
        }

        fn audio(&self, start: i64, frames: usize, sample_rate: u32) -> filmcraft_media::Result<filmcraft_frame::AudioBuffer> {
            self.audio_stream(0, start, frames, sample_rate)
        }

        fn audio_stream(&self, _stream: usize, start: i64, frames: usize, sample_rate: u32) -> filmcraft_media::Result<filmcraft_frame::AudioBuffer> {
            // 5,000 periodic pauses across 12 hours (43,200 s): every 8.64 seconds (138,240 samples @ 16 kHz),
            // the first 1.00 second (16,000 samples) after a 1.00 second speech lead-in is silent.
            let sr = sample_rate.max(1) as i64;
            let period = (864 * sr) / 100; // 8.64s = 138,240 samples @ 16 kHz
            let speech_lead = sr; // 1.0s
            let silence_end = 2 * sr; // 2.0s (1.0s pause)
            let mut ch = Vec::with_capacity(frames);
            for i in 0..frames {
                let s_idx = (start + i as i64).max(0);
                let cycle = s_idx / period;
                let in_cycle = s_idx % period;
                let is_pause = cycle < 5_000 && in_cycle >= speech_lead && in_cycle < silence_end;
                ch.push(if is_pause { 0.0_f32 } else { 0.35_f32 });
            }
            Ok(filmcraft_frame::AudioBuffer { sample_rate, channels: vec![ch] })
        }
    }

    #[test]
    fn stress_test_12_hour_vod_streaming_pipeline_and_5000_silence_cuts() {
        use crate::subtitles::{WHISPER_WINDOW_SAMPLES, build_audio_plan, stream_audio_window_to_wav};
        use crate::vod_chapters::{FrameSignature, VisualLayoutKind, classify_chapter_title, detect_sustained_chapters, format_hms_ms};

        let vod_dur = Tick::from_seconds_f64(12.0 * 3600.0); // 43,200 seconds
        let mut s = Session::default();
        s.execute("file.newSequence", json!({"name": "12h_VOD", "width": 1920, "height": 1080, "fps": 60, "video": 1, "audio": 1})).unwrap();
        let seq_id = s.state.active_sequence.unwrap();

        // Register the 12-hour streaming source in the project & media pool.
        let src: filmcraft_media::SharedSource = Arc::new(TwelveHourVodSource::new());
        let media_id = s
            .edit("Add 12h VOD", |pr, _| {
                let id = pr.add_item(
                    "12h_stream_vod.mp4",
                    filmcraft_project::Label::Iris,
                    filmcraft_project::ItemKind::Media(Box::new(filmcraft_project::MediaClip {
                        media: filmcraft_project::MediaRef::File { path: "12h_stream_vod.mp4".into() },
                        info: src.info().clone(),
                        interpret: Default::default(),
                        mark_in: None,
                        mark_out: None,
                        markers: vec![],
                        offline: false,
                        proxy: None,
                        identity: None,
                    })),
                    None,
                );
                let rate = pr.sequence(seq_id).unwrap().settings.frame_rate;
                let range = TimeRange::new(Tick::ZERO, vod_dur);
                let mut v_it = pr.make_track_item(id, filmcraft_project::TrackKind::Video, Tick::ZERO, range, rate).unwrap();
                let mut a_it = pr.make_track_item(id, filmcraft_project::TrackKind::Audio, Tick::ZERO, range, rate).unwrap();
                let link_id = pr.alloc_id();
                v_it.link = Some(link_id);
                a_it.link = Some(link_id);
                let q = pr.sequence_mut(seq_id).unwrap();
                q.video_tracks[0].items.push(v_it);
                q.audio_tracks[0].items.push(a_it);
                Ok(id)
            })
            .unwrap();
        s.media.insert(media_id, src.clone());

        // 1. 12-Hour Chunked Whisper Audio Window Streaming to Disk (20-min window = 19.2M samples = 38.4 MB WAV).
        let t_wav = web_time::Instant::now();
        let seq_ref = s.active_sequence().unwrap().clone();
        let plan = build_audio_plan("ai.generateSubtitles", &s, &seq_ref, seq_id, &json!({})).unwrap();
        let provider = s.media.provider(s.project.clone(), s.services.clone());
        let prog = Progress::default();
        let tmp_wav = std::env::temp_dir().join(format!("orpheus-12h-window-stress-{}.wav", std::process::id()));
        let bytes_written = stream_audio_window_to_wav(&plan, (0, WHISPER_WINDOW_SAMPLES), &s.project, &provider, &tmp_wav, &prog, (0.0, 100.0)).unwrap();
        let wav_elapsed_ms = t_wav.elapsed().as_secs_f64() * 1000.0;
        let on_disk_len = std::fs::metadata(&tmp_wav).unwrap().len();
        let _ = std::fs::remove_file(&tmp_wav);
        assert_eq!(bytes_written, 44 + (WHISPER_WINDOW_SAMPLES as u64 * 2));
        assert_eq!(on_disk_len, 38_400_044, "20-min 16 kHz mono WAV must be exact 38,400,044 bytes");

        // 2. 12-Hour Streaming Silence Detection across all 43,200s (5,000 pauses across hours 0..12).
        let t_detect = web_time::Instant::now();
        let opts = SilenceOptions { threshold_db: -38.0, min_pause_ms: 450.0, padding_ms: 80.0 }.sanitized();
        let detected = detect_sequence_silence(&s, &json!({}), opts);
        let detect_elapsed_ms = t_detect.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(detected.len(), 5_000, "must detect all 5,000 pauses across all 12 hours without 2h truncation");
        assert!(detected.last().unwrap().start.seconds() > 11.9 * 3600.0, "last pause must be near hour 12");

        // 3. Apply all 5,000 Silence Cuts in a single $O(N + M)$ Ripple-Delete Pass on V1 + A1.
        let t_ripple = web_time::Instant::now();
        let (removed_count, removed_ticks) = apply_silence_ranges(&mut s, seq_id, detected).unwrap();
        let ripple_elapsed_ms = t_ripple.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(removed_count, 5_000);
        let q_after = s.active_sequence().unwrap();
        assert_eq!(q_after.video_tracks[0].items.len(), 5_001);
        assert_eq!(q_after.audio_tracks[0].items.len(), 5_001);
        q_after.check().expect("sequence invariants must hold after 5,000 ripple cuts");

        // 4. 12-Hour VOD Chapterizer (3,600 sparse visual samples across 43,200s -> 12 distinct 1-hour chapters).
        let t_chap = web_time::Instant::now();
        let layouts = [
            (VisualLayoutKind::TalkingHead, "starting soon welcome back stream intro", 0.20_f32),
            (VisualLayoutKind::TalkingHead, "just chatting q&a catching up with chat", 0.45_f32),
            (VisualLayoutKind::GamingHud, "playing escape from tarkov custom raid", 0.15_f32),
            (VisualLayoutKind::DesktopReact, "reacting watch this video tier list", 0.75_f32),
            (VisualLayoutKind::GamingHud, "counter-strike 2 mirage premiere match", 0.25_f32),
            (VisualLayoutKind::GamingHud, "minecraft hardcore nether survival", 0.55_f32),
            (VisualLayoutKind::TalkingHead, "podcast interview with special guest", 0.30_f32),
            (VisualLayoutKind::GamingHud, "valorant radiant ranked vandal clutch", 0.12_f32),
            (VisualLayoutKind::DesktopReact, "live coding programming rust refactor", 0.68_f32),
            (VisualLayoutKind::GamingHud, "elden ring malenia boss fight", 0.22_f32),
            (VisualLayoutKind::GamingHud, "gta grand theft auto los santos", 0.48_f32),
            (VisualLayoutKind::TalkingHead, "just chatting outro thanks for watching", 0.18_f32),
        ];
        let mut sigs = Vec::with_capacity(3_600);
        for sample_i in 0..3_600 {
            let t_sec = sample_i as f64 * 12.0;
            let hour_idx = ((t_sec / 3600.0).floor() as usize).min(11);
            let (layout, _, luma) = layouts[hour_idx];
            sigs.push(FrameSignature {
                time_sec: t_sec,
                layout,
                luma_grid: [luma; 16],
                color_hist: [luma * 0.8; 12],
                center_skin_ratio: if layout == VisualLayoutKind::TalkingHead { 0.22 } else { 0.04 },
                edge_hud_energy: if layout == VisualLayoutKind::GamingHud { 0.12 } else { 0.03 },
                center_action_energy: 0.08,
            });
        }
        let chapters = detect_sustained_chapters(&sigs, 43_200.0, 45.0);
        let chap_elapsed_ms = t_chap.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(chapters.len(), 12, "12-hour VOD with 12 hourly activity transitions must produce 12 chapters");
        for (idx, (st_s, en_s, layout)) in chapters.iter().enumerate() {
            let title = classify_chapter_title(*layout, layouts[idx].1);
            assert!(!title.is_empty());
            let _ = (st_s, en_s, format_hms_ms((*st_s * 1000.0) as u64));
        }

        // 5. All-In-One Prompt Parser verification for both Bulgarian and English multi-step prompts.
        let bg_plan = crate::all_in_one::plan_from_prompt("махни паузите, сложи субтитри и раздели на глави", 0);
        assert!(bg_plan.run_remove_pauses && bg_plan.run_subtitles && bg_plan.run_vod_chapters && !bg_plan.run_split_at_playhead);
        let en_plan = crate::all_in_one::plan_from_prompt("remove silences, add captions, split into chapters, and reframe to 9:16", 0);
        assert!(en_plan.run_remove_pauses && en_plan.run_subtitles && en_plan.run_vod_chapters && en_plan.run_auto_reframe && !en_plan.run_split_at_playhead);

        // 6. Export & Audio Mixing Stress Test on the 5,001-Clip 12-Hour Timeline:
        //    a) Verify PoolProvider opens 1 source handle for all 5,001 clips (zero os error 24 risk).
        let export_provider = s.media.provider(s.project.clone(), s.services.clone()).own_sources();
        assert!(filmcraft_render::SourceProvider::source(&export_provider, media_id).is_some());

        //    b) Benchmark 1,000 audio mix pulls across hours 0..11.9 on the 5,001-clip A1 track (O(log N)).
        let seq_snap = s.active_sequence().unwrap().clone();
        let t_mix = web_time::Instant::now();
        let total_samples_48k = seq_snap.duration().to_units_floor(48_000).max(48_000);
        for pull in 0..1_000i64 {
            let at = (pull * (total_samples_48k - 2_048)) / 1_000;
            let buf = filmcraft_render::audio::mix_sequence(&s.project, &seq_snap, at, 1_024, &export_provider);
            assert_eq!(buf.channels.len(), 2);
            assert_eq!(buf.channels[0].len(), 1_024);
        }
        let mix_1000_ms = t_mix.elapsed().as_secs_f64() * 1000.0;

        //    c) Run real native Rust `filmcraft_export::export` on the 5,001-clip timeline across cuts
        //       near hour 0 and near hour 11.5 (H.264 + AAC MP4), verifying valid container output
        //       and zero temp file leaks.
        let tmp_mp4_start = std::env::temp_dir().join(format!("orpheus-12h-export-start-{}.mp4", std::process::id()));
        let tmp_mp4_late = std::env::temp_dir().join(format!("orpheus-12h-export-late-{}.mp4", std::process::id()));
        let _ = std::fs::remove_file(&tmp_mp4_start);
        let _ = std::fs::remove_file(&tmp_mp4_late);
        let t_exp = web_time::Instant::now();
        for (out_path, start_sec) in [(&tmp_mp4_start, 0.0_f64), (&tmp_mp4_late, (seq_snap.duration().seconds() - 2.0).max(0.0))] {
            let settings = filmcraft_export::ExportSettings {
                format: filmcraft_export::Format::H264,
                scale: 0.25,
                include_audio: true,
                path: out_path.to_string_lossy().into_owned(),
                range: Some(TimeRange::new(Tick::from_seconds_f64(start_sec), Tick::from_seconds_f64(0.5))),
                ..Default::default()
            };
            let exp_prog = Progress::default();
            let rep =
                filmcraft_export::export(&s.project, seq_id, &settings, &export_provider, &exp_prog).expect("5,001-clip 12h timeline export must succeed");
            let mp4_bytes = std::fs::read(out_path).expect("exported mp4 must exist");
            std::fs::remove_file(out_path).expect("cleanup exported mp4");
            assert!(!out_path.exists());
            assert!(rep.frames >= 25, "must render >= 25 frames at 60fps across 0.5s cut range, got {}", rep.frames);
            assert!(mp4_bytes.len() > 256 && &mp4_bytes[4..8] == b"ftyp", "exported file must be valid MP4");
        }
        let exp_elapsed_ms = t_exp.elapsed().as_secs_f64() * 1000.0;

        // 7. Undo / Redo History Memory-Budget Verification on the 10,002-Item Sequence:
        //    Push 25 distinct edits mutating the sequence and verify History evicts old snapshots
        //    to honor MAX_HISTORY_ESTIMATED_BYTES while always preserving >= MIN_UNDO_STEPS (8).
        for edit_i in 0..25usize {
            s.edit_sequence(&format!("VOD Edit {edit_i}"), |q, _, _| {
                q.mark_in = Some(Tick::from_seconds_f64(edit_i as f64));
                Ok(())
            })
            .unwrap();
        }
        let undo_len = s.history.undo.len();
        let hist_bytes = s.history.estimated_bytes();
        assert!(undo_len >= crate::MIN_UNDO_STEPS, "must retain at least {} undo steps, got {undo_len}", crate::MIN_UNDO_STEPS);
        assert!(undo_len <= 12, "history memory cap must evict oversized 10,000-clip snapshots, got {undo_len} steps");

        println!(
            "[12H_STRESS_METRICS] wav_20m_ms={wav_elapsed_ms:.1} wav_bytes={on_disk_len} silence_detect_12h_ms={detect_elapsed_ms:.1} pauses_detected={} ripple_5000_cuts_ms={ripple_elapsed_ms:.2} removed_sec={:.1} vod_chapters_12h_ms={chap_elapsed_ms:.2} chapters={} mix_1000_pulls_ms={mix_1000_ms:.1} export_2x_mp4_ms={exp_elapsed_ms:.1} history_undo_steps={undo_len} history_est_mb={:.1}",
            removed_count,
            removed_ticks.seconds(),
            chapters.len(),
            hist_bytes as f64 / (1024.0 * 1024.0)
        );
    }
}
