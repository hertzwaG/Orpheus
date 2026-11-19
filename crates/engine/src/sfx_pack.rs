//! 100% pure-Rust procedural Creator SFX Pack synthesizer & timeline inserter (`ai.insertCreatorSfx`).
//!
//! Generates clean, deterministic, royalty-free 48 kHz mono 16-bit PCM WAV sound effects on demand
//! with zero external audio assets or runtime FFmpeg dependencies:
//! - **Pop** (`pop`): crisp pitch-drop bubble pop
//! - **Whoosh** (`whoosh`): smooth bandpass-swept air transition
//! - **Deep Bass Hit** (`bass_hit`): cinematic 95 Hz -> 38 Hz sub impact
//! - **Bell Ding** (`bell_ding`): bright harmonic chime
//! - **Camera Shutter** (`shutter`): dual transient mechanical click
//! - **Sub Riser** (`sub_riser`): rising 45 Hz -> 220 Hz tension swell

use std::path::Path;

use filmcraft_project::{ItemId, Track, TrackId, TrackKind};
use filmcraft_time::{Tick, TimeRange};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::commands::{CommandSpec, NamedKeys, always, bad, str_p};
use crate::voiceover::media_dir;
use crate::{EngineError, Result, Session};

pub const SFX_SAMPLE_RATE: u32 = 48_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CreatorSfx {
    Pop,
    Whoosh,
    BassHit,
    BellDing,
    Shutter,
    SubRiser,
}

impl CreatorSfx {
    pub const ALL: [Self; 6] = [Self::Pop, Self::Whoosh, Self::BassHit, Self::BellDing, Self::Shutter, Self::SubRiser];

    pub fn from_str_param(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "pop" => Some(Self::Pop),
            "whoosh" => Some(Self::Whoosh),
            "bass_hit" | "basshit" | "bass" | "hit" => Some(Self::BassHit),
            "bell_ding" | "bellding" | "bell" | "ding" => Some(Self::BellDing),
            "shutter" | "camera_shutter" | "camerashutter" | "click" => Some(Self::Shutter),
            "sub_riser" | "subriser" | "riser" => Some(Self::SubRiser),
            _ => None,
        }
    }

    pub fn as_id(self) -> &'static str {
        match self {
            Self::Pop => "pop",
            Self::Whoosh => "whoosh",
            Self::BassHit => "bass_hit",
            Self::BellDing => "bell_ding",
            Self::Shutter => "shutter",
            Self::SubRiser => "sub_riser",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Pop => "Pop",
            Self::Whoosh => "Whoosh",
            Self::BassHit => "Deep Bass Hit",
            Self::BellDing => "Bell Ding",
            Self::Shutter => "Camera Shutter",
            Self::SubRiser => "Sub Riser",
        }
    }

    pub fn file_name(self) -> &'static str {
        match self {
            Self::Pop => "orpheus_sfx_pop.wav",
            Self::Whoosh => "orpheus_sfx_whoosh.wav",
            Self::BassHit => "orpheus_sfx_bass_hit.wav",
            Self::BellDing => "orpheus_sfx_bell_ding.wav",
            Self::Shutter => "orpheus_sfx_shutter.wav",
            Self::SubRiser => "orpheus_sfx_sub_riser.wav",
        }
    }
}

/// Synthesize a deterministic 48 kHz 16-bit PCM WAV byte buffer for `kind`.
pub fn synthesize_wav(kind: CreatorSfx) -> Vec<u8> {
    let samples = synthesize_pcm(kind);
    filmcraft_media::wav::write_wav16(&samples, 1, SFX_SAMPLE_RATE)
}

pub fn synthesize_pcm(kind: CreatorSfx) -> Vec<f32> {
    let sr = f64::from(SFX_SAMPLE_RATE);
    let tau = std::f64::consts::TAU;
    match kind {
        CreatorSfx::Pop => {
            let n = ((sr * 0.095).round() as usize).max(16);
            let mut phase = 0.0_f64;
            (0..n)
                .map(|i| {
                    let t = i as f64 / sr;
                    let u = i as f64 / n as f64;
                    let freq = 270.0 - 185.0 * u.powf(0.55);
                    phase += (tau * freq) / sr;
                    let attack = (t / 0.003).clamp(0.0, 1.0);
                    let decay = (-t * 40.0).exp();
                    let fade_out = ((1.0 - u) / 0.12).clamp(0.0, 1.0);
                    ((phase.sin() + 0.25 * (phase * 2.0).sin()) * attack * decay * fade_out * 0.62) as f32
                })
                .collect()
        }
        CreatorSfx::BellDing => {
            let n = ((sr * 0.28).round() as usize).max(16);
            (0..n)
                .map(|i| {
                    let t = i as f64 / sr;
                    let u = i as f64 / n as f64;
                    let attack = (t / 0.005).clamp(0.0, 1.0);
                    let decay = (-t * 11.5).exp();
                    let fade_out = ((1.0 - u) / 0.15).clamp(0.0, 1.0);
                    let f0 = (t * 880.0 * tau).sin();
                    let f1 = 0.35 * (t * 1_760.0 * tau).sin();
                    let f2 = 0.15 * (t * 2_640.0 * tau).sin();
                    ((f0 + f1 + f2) * attack * decay * fade_out * 0.42) as f32
                })
                .collect()
        }
        CreatorSfx::Whoosh => synthesize_noise_sweep(0.26, 280.0, 1_450.0, 0.72),
        CreatorSfx::BassHit => {
            let n = ((sr * 0.45).round() as usize).max(16);
            let mut phase = 0.0_f64;
            (0..n)
                .map(|i| {
                    let t = i as f64 / sr;
                    let u = i as f64 / n as f64;
                    let freq = 96.0 - 58.0 * u.powf(0.42);
                    phase += (tau * freq) / sr;
                    let attack = (t / 0.004).clamp(0.0, 1.0);
                    let decay = (-t * 7.2).exp();
                    let punch = (-t * 55.0).exp() * 0.35;
                    let body = phase.sin() + 0.2 * (phase * 2.0).sin();
                    ((body * decay + punch * (phase * 3.0).sin()) * attack * 0.78).clamp(-0.95, 0.95) as f32
                })
                .collect()
        }
        CreatorSfx::Shutter => {
            let n = ((sr * 0.14).round() as usize).max(16);
            let mut rng: u64 = 0xCAFE_1234_ABCD_9876;
            (0..n)
                .map(|i| {
                    let t = i as f64 / sr;
                    rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                    let white = ((rng >> 33) as f64 / (f64::from(u32::MAX) * 0.5)) - 1.0;
                    // Dual crisp mechanical click pulses at t=0.002s and t=0.048s.
                    let c1 = (-((t - 0.004).abs()) * 420.0).exp();
                    let c2 = (-((t - 0.048).abs()) * 340.0).exp() * 0.75;
                    let tone = (t * 2_400.0 * tau).sin() * 0.45;
                    ((white * 0.65 + tone) * (c1 + c2) * 0.75).clamp(-0.92, 0.92) as f32
                })
                .collect()
        }
        CreatorSfx::SubRiser => {
            let n = ((sr * 0.85).round() as usize).max(16);
            let mut phase = 0.0_f64;
            let mut rng: u64 = 0x91E1_0DA5_C47B_2113;
            let mut lp = 0.0_f64;
            (0..n)
                .map(|i| {
                    let u = i as f64 / n as f64;
                    let freq = 45.0 + 175.0 * u.powf(1.8);
                    phase += (tau * freq) / sr;
                    rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                    let white = ((rng >> 33) as f64 / (f64::from(u32::MAX) * 0.5)) - 1.0;
                    let cutoff = 180.0 + 1_600.0 * u.powi(2);
                    let rc = 1.0 / (tau * cutoff);
                    let dt = 1.0 / sr;
                    lp += (dt / (rc + dt)) * (white - lp);
                    let swell = u.powf(1.35);
                    let tail = ((1.0 - u) / 0.05).clamp(0.0, 1.0);
                    ((phase.sin() * 0.6 + lp * 0.45) * swell * tail * 0.75).clamp(-0.92, 0.92) as f32
                })
                .collect()
        }
    }
}

fn synthesize_noise_sweep(dur_sec: f64, f_min: f64, f_max: f64, gain: f64) -> Vec<f32> {
    let sr = f64::from(SFX_SAMPLE_RATE);
    let tau = std::f64::consts::TAU;
    let n = ((sr * dur_sec).round() as usize).max(16);
    let mut rng: u64 = 0x5EED_CAFE_BABE_1337;
    let mut lp = 0.0_f64;
    let mut hp_prev = 0.0_f64;
    let mut hp_out = 0.0_f64;
    (0..n)
        .map(|i| {
            let u = i as f64 / n as f64;
            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            let white = ((rng >> 33) as f64 / (f64::from(u32::MAX) * 0.5)) - 1.0;
            let sweep = f_min + (f_max - f_min) * (std::f64::consts::PI * u).sin().powi(2);
            let rc = 1.0 / (tau * sweep);
            let dt = 1.0 / sr;
            let alpha = dt / (rc + dt);
            lp += alpha * (white - lp);
            hp_out = 0.965 * (hp_out + lp - hp_prev);
            hp_prev = lp;
            let env = (std::f64::consts::PI * u).sin().powf(1.4);
            (hp_out * env * gain).clamp(-0.9, 0.9) as f32
        })
        .collect()
}

/// Ensure a procedural SFX WAV exists on disk + imported into the project, then return its [`ItemId`].
pub fn ensure_sfx_item(s: &mut Session, kind: CreatorSfx, dir_override: Option<&str>) -> Result<ItemId> {
    let dir_val = dir_override.map(|d| json!({ "dir": d })).unwrap_or_else(|| json!({}));
    let dir = media_dir(s, &dir_val, "sfx")?;
    if !cfg!(target_arch = "wasm32") {
        std::fs::create_dir_all(&dir).map_err(|e| EngineError::Other(format!("{dir}: {e}")))?;
    }
    let path = Path::new(&dir).join(kind.file_name()).to_string_lossy().into_owned();
    if let Some(existing) = s.project.items.iter().find(|(_, it)| it.name == kind.file_name()).map(|(&id, _)| id) {
        return Ok(existing);
    }
    let wav = synthesize_wav(kind);
    s.services.write_file(&path, &wav).map_err(|e| EngineError::Other(format!("{path}: {e}")))?;
    crate::commands::import_bytes(s, &path, wav.into(), None)
}

/// Insert `kind` at `at` (defaults to current playhead) onto a `"Creator SFX"` audio track in one undo step.
pub fn insert_sfx_at(s: &mut Session, kind: CreatorSfx, at: Option<Tick>, dir_override: Option<&str>) -> Result<ItemId> {
    let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let playhead = s.state.playheads.get(&seq_id).copied().unwrap_or(Tick::ZERO);
    let pos = at.unwrap_or(playhead).max(Tick::ZERO);
    let group_label = format!("Insert SFX: {}", kind.label());
    let item_id = s.edit_group(&group_label, |s| {
        let imported_id = ensure_sfx_item(s, kind, dir_override)?;
        let rate = s.project.sequence(seq_id).map(|q| q.settings.frame_rate).unwrap_or(filmcraft_time::FrameRate::FPS_24);
        let dur = Tick::from_units(synthesize_pcm(kind).len() as i64, i64::from(SFX_SAMPLE_RATE)).max(rate.frame_duration().max(Tick(1)));

        s.edit("Insert Creator SFX", |pr, _| {
            let need_track = !pr.sequence(seq_id).ok_or(EngineError::NoSequence)?.audio_tracks.iter().any(|t| t.name == "Creator SFX" && !t.locked);
            let new_tid = need_track.then(|| TrackId(pr.alloc_id()));
            let mut it = pr
                .make_track_item(imported_id, TrackKind::Audio, rate.snap(pos), TimeRange::new(Tick::ZERO, dur), rate)
                .ok_or_else(|| EngineError::Other("failed to create SFX clip".into()))?;
            it.name = format!("SFX: {}", kind.label());
            it.duration = dur;

            let seq = pr.sequence_mut(seq_id).ok_or(EngineError::NoSequence)?;
            if let Some(tid) = new_tid {
                seq.audio_tracks.push(Track::new(tid, TrackKind::Audio, "Creator SFX".into()));
            }
            let target_idx = seq
                .audio_tracks
                .iter()
                .position(|t| t.name == "Creator SFX" && !t.locked)
                .or_else(|| seq.audio_tracks.iter().position(|t| !t.locked))
                .ok_or_else(|| EngineError::Other("no unlocked audio track available".into()))?;
            let tr = &mut seq.audio_tracks[target_idx];
            if tr.items.iter().any(|x| x.start < it.end() && it.start < x.end()) {
                let end_max = tr.items.iter().filter(|x| x.start < it.end() && it.start < x.end()).map(|x| x.end()).max().unwrap_or(it.start);
                it.start = rate.snap(end_max);
            }
            tr.items.push(it);
            seq.check().map_err(EngineError::Other)?;
            Ok(imported_id)
        })
    })?;
    s.toast(format!("Inserted {}", kind.label()));
    Ok(item_id)
}

pub(crate) fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: "ai.insertCreatorSfx",
        label: "Insert Creator SFX at Playhead",
        menu: &[],
        shortcut: None,
        params: r#"{"sfx":str,"dir":str?}"#,
        enabled: always,
        run: |s, p| {
            let raw = str_p(p, "sfx").ok_or_else(|| bad("ai.insertCreatorSfx", "missing sfx parameter"))?;
            let kind = CreatorSfx::from_str_param(raw).ok_or_else(|| bad("ai.insertCreatorSfx", format!("unknown creator sfx `{raw}`")))?;
            let dir = str_p(p, "dir");
            let item_id = insert_sfx_at(s, kind, None, dir)?;
            Ok(json!({ "sfx": kind.as_id(), "item": item_id }))
        },
        journal: false,
        named: NamedKeys::NONE,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_six_creator_sfx_synthesize_valid_non_empty_wavs() {
        for kind in CreatorSfx::ALL {
            let wav = synthesize_wav(kind);
            assert!(wav.len() > 44, "{} WAV must have PCM payload", kind.label());
            assert_eq!(&wav[0..4], b"RIFF");
            assert_eq!(&wav[8..12], b"WAVE");
        }
    }

    #[test]
    fn insert_creator_sfx_places_audio_clip_and_undoes_cleanly() {
        let mut s = Session::default();
        s.execute("file.newSequence", json!({"name": "SfxTest", "video": 1, "audio": 1})).unwrap();
        let tmp_dir = std::env::temp_dir().join(format!("orpheus-sfx-test-{}", std::process::id()));
        let dir_str = tmp_dir.to_string_lossy().into_owned();

        s.execute("ai.insertCreatorSfx", json!({"sfx": "bass_hit", "dir": dir_str})).unwrap();
        let seq_id = s.state.active_sequence.unwrap();
        let seq = s.project.sequence(seq_id).unwrap();
        let sfx_track = seq.audio_tracks.iter().find(|t| t.name == "Creator SFX").unwrap();
        assert_eq!(sfx_track.items.len(), 1);

        s.execute("edit.undo", json!({})).unwrap();
        let seq_after = s.project.sequence(seq_id).unwrap();
        assert!(seq_after.audio_tracks.iter().all(|t| t.items.is_empty()));
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }
}
