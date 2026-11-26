//! Orpheus All-In-One Timeline & Video Editing Command Executor
//! (`ai.allInOne`, `ai.allInOneStatus`).
//!
//! Parses natural-language editing instructions in both English and Bulgarian (with optional
//! local Ollama planning and deterministic offline fallback) and chains real, undoable timeline
//! operations on the active sequence:
//! - Subtitles & transcription (`ai.generateSubtitles`)
//! - Native text / title overlays (`graphics.addText`)
//! - Cuts, splits, silence removal, and clip speed (`timeline.razor`, `ai.removePauses`, `clip.speedDuration`)
//! - Transitions, color grading, B&W, blur, and zoom (`timeline.addTransition`, `effects.apply`, `effects.setParam`)
//! - Audio mute, gain/volume, normalization, and AI audio enhance (`effects.apply`, `clip.audioGain`, `ai.enhanceAudio`)
//! - AI reframe, Game Stream split-screen, background removal, viral clips, smart triggers, and VOD chapters

use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use filmcraft_export::{Progress, Report};
use filmcraft_project::{ClipId, ItemId};
use filmcraft_time::Tick;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, NamedKeys, always, bad, bool_p, str_p, u64_p};
use crate::subtitles::next_job_id;
use crate::{EngineError, Job, Result, Session};

const MAX_ATTACHMENTS: usize = 16;
const MAX_ATTACHMENT_BYTES: u64 = 64 * 1024 * 1024;

/// Structured multi-step timeline execution plan derived from a natural-language prompt.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AllInOnePlan {
    pub prompt_summary: String,
    pub run_subtitles: bool,
    pub subtitle_language: Option<String>,
    pub run_text_overlay: bool,
    pub text_overlay_content: Option<String>,
    pub run_remove_pauses: bool,
    pub run_split_at_playhead: bool,
    pub speed_percent: Option<f64>,
    pub transition_effect: Option<String>,
    pub apply_bw: bool,
    pub apply_blur: bool,
    pub apply_color_grade: bool,
    pub zoom_scale: Option<f64>,
    pub audio_mute: bool,
    pub audio_normalize: bool,
    pub audio_gain_db: Option<f64>,
    pub run_audio_enhance: bool,
    pub run_auto_reframe: bool,
    pub reframe_split_screen: bool,
    pub run_bg_removal: bool,
    pub run_viral_clips: bool,
    pub run_smart_triggers: bool,
    pub run_vod_chapters: bool,
    pub used_ollama: bool,
}

/// Result summary stored on `Session` after an All-In-One job completes.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AllInOneReport {
    pub seq: Option<ItemId>,
    pub steps_applied: Vec<String>,
    pub generated_layers: usize,
    pub attachments_imported: usize,
    pub wikimedia_fetched: usize,
    pub used_ollama: bool,
}

#[derive(Clone)]
struct PreparedAttachment {
    orig_path: String,
    bytes: Arc<[u8]>,
}

struct PreparedAllInOneOutput {
    plan: AllInOnePlan,
    attachments: Vec<PreparedAttachment>,
}

/// Sequential execution stage for an async multi-step Orpheus All-In-One prompt pipeline.
///
/// Guarantees that timeline-mutating background jobs (`PendingSilenceJob` -> `PendingAudioEnhanceJob`
/// -> `PendingAutoReframeJob` -> `PendingBgRemovalJob` -> `PendingSubtitle` -> `PendingVodChapterJob`
/// -> `PendingViralJob` -> `PendingSmartTriggerJob`) execute and commit to the sequence one at a
/// time in strict dependency order, preventing timestamp drift on 12-hour VODs while keeping the
/// UI thread 100% non-blocking.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum PipelineStage {
    #[default]
    Planning,
    AttachmentsAndPreCuts,
    RemovePauses,
    AudioOps,
    AutoReframe,
    BgRemoval,
    Subtitles,
    VodChapters,
    ViralClips,
    SmartTriggers,
    FinalVisualsAndText,
    Done,
}

pub struct PendingAllInOneJob {
    pub(crate) job: u64,
    pub(crate) epoch: u64,
    pub(crate) prompt: String,
    plan_ready: Arc<AtomicBool>,
    output: Arc<Mutex<Option<PreparedAllInOneOutput>>>,
    plan: Option<AllInOnePlan>,
    attachments: Vec<PreparedAttachment>,
    seq_id: Option<ItemId>,
    stage: PipelineStage,
    waiting_on_job: Option<u64>,
    waiting_step_success: Option<String>,
    waiting_step_fail_prefix: Option<String>,
    steps_applied: Vec<String>,
    attachments_imported: usize,
    started_at: web_time::Instant,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| haystack.contains(n))
}

fn extract_quoted_text(prompt: &str) -> Option<String> {
    const PAIRS: [(char, char); 4] = [('"', '"'), ('\'', '\''), ('„', '“'), ('«', '»')];
    for (open, close) in PAIRS {
        if let Some(start) = prompt.find(open) {
            let rest = &prompt[start + open.len_utf8()..];
            if let Some(end) = rest.find(close) {
                let candidate = rest[..end].trim();
                if !candidate.is_empty() {
                    return Some(candidate.chars().take(120).collect());
                }
            }
        }
    }
    None
}

fn extract_overlay_text(prompt: &str) -> String {
    if let Some(quoted) = extract_quoted_text(prompt) {
        return quoted;
    }
    let lower = prompt.to_lowercase();
    for kw in ["заглавие", "надпис", "текст", "title", "headline", "text", "intro", "outro"] {
        if let Some(byte_pos) = lower.find(kw) {
            // Map byte offset safely via char count fallback if lengths differ, or slice if valid char boundary.
            let after_kw = if prompt.is_char_boundary(byte_pos + kw.len()) { &prompt[byte_pos + kw.len()..] } else { "" };
            let trimmed = after_kw.trim_start_matches(|c: char| c == ':' || c == '-' || c.is_whitespace());
            let mut cut = trimmed;
            for sep in [" и ", " and ", ",", ".", ";"] {
                if let Some(idx) = cut.to_lowercase().find(sep)
                    && cut.is_char_boundary(idx)
                {
                    cut = &cut[..idx];
                }
            }
            let cleaned = cut.trim();
            if !cleaned.is_empty() && cleaned.chars().count() <= 80 {
                return cleaned.to_string();
            }
        }
    }
    "Title".to_string()
}

fn detect_subtitle_language(lower: &str) -> Option<String> {
    if contains_any(lower, &["български", "bulgarian"]) {
        Some("bg".to_string())
    } else if contains_any(lower, &["английски", "english"]) {
        Some("en".to_string())
    } else if contains_any(lower, &["немски", "german"]) {
        Some("de".to_string())
    } else if contains_any(lower, &["френски", "french"]) {
        Some("fr".to_string())
    } else if contains_any(lower, &["испански", "spanish"]) {
        Some("es".to_string())
    } else {
        None
    }
}

fn parse_speed_percent(lower: &str) -> Option<f64> {
    for token in lower.split_whitespace() {
        let clean = token.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '.' && c != '%');
        if let Some(pct_str) = clean.strip_suffix('%')
            && let Ok(pct) = pct_str.parse::<f64>()
            && pct.is_finite()
            && (10.0..=1000.0).contains(&pct)
        {
            return Some(pct);
        }
        if let Some(mult_str) = clean.strip_suffix('x')
            && let Ok(mult) = mult_str.parse::<f64>()
            && mult.is_finite()
            && (0.1..=10.0).contains(&mult)
        {
            return Some(mult * 100.0);
        }
    }
    if contains_any(lower, &["ускори", "по-бързо", "faster", "speed up"]) {
        Some(150.0)
    } else if contains_any(lower, &["забави", "по-бавно", "slower", "slow down", "slow motion"]) {
        Some(50.0)
    } else if contains_any(lower, &["скорост", "speed"]) {
        Some(150.0)
    } else {
        None
    }
}

/// Deterministic offline English + Bulgarian natural-language prompt planner.
pub fn plan_from_prompt(prompt: &str, _attachment_count: usize) -> AllInOnePlan {
    let trimmed = prompt.trim();
    let lower = trimmed.to_lowercase();

    let run_subtitles = contains_any(&lower, &["субтитри", "captions", "subtitles", "subtitle", "transcribe", "транскрибирай"])
        || (lower.contains("надписи") && !contains_any(&lower, &["заглавие", "текст", "title"]));
    let subtitle_language = if run_subtitles { detect_subtitle_language(&lower) } else { None };

    let run_text_overlay = contains_any(&lower, &["текст", "заглавие", "интро", "аутро", "title", "text", "intro", "outro", "lower third", "headline"])
        || (lower.contains("надпис") && !lower.contains("надписи"));
    let text_overlay_content = if run_text_overlay { Some(extract_overlay_text(trimmed)) } else { None };

    let run_remove_pauses =
        contains_any(&lower, &["silence", "silences", "pause", "pauses", "dead air", "паузи", "паузите", "тишина", "тишината", "празни места"]);
    let wants_vertical =
        contains_any(&lower, &["9:16", "vertical", "tiktok", "reels", "shorts", "шортс", "reframe", "вертикално", "вертикален", "тикток", "рийлс"]);
    let mentions_facecam = contains_any(&lower, &["фейскам", "facecam", "face cam", "webcam", "камера", "game stream", "стрийм"]);
    let explicit_split_screen = contains_any(&lower, &["split screen", "split-screen", "gaming split", "game stream"]);
    let reframe_split_screen = explicit_split_screen || (wants_vertical && mentions_facecam);
    let run_auto_reframe = reframe_split_screen || wants_vertical;
    let run_bg_removal = contains_any(&lower, &["remove background", "transparent", "green screen", "премахни фон"]);
    let run_viral_clips = contains_any(&lower, &["viral", "highlight", "вайръл", "акценти", "най-добри моменти"]);
    let run_smart_triggers = contains_any(&lower, &["smart trigger", "auto b-roll", "смарт тригери", "тригери"]);
    let run_vod_chapters = contains_any(
        &lower,
        &[
            "vod",
            "chapter",
            "chapters",
            "глави",
            "глава",
            "секции",
            "секция",
            "раздели по теми",
            "раздели на глави",
            "split into chapters",
            "split by topic",
            "split into sections",
        ],
    );
    let explicit_playhead_cut = contains_any(&lower, &["playhead", "плейхед", "razor"]);
    let run_split_at_playhead = explicit_playhead_cut
        || (contains_any(&lower, &["изрежи", "срежи", "раздели", "отрежи", "cut", "trim", "split"])
            && !run_remove_pauses
            && !run_vod_chapters
            && !explicit_split_screen);
    let speed_percent = parse_speed_percent(&lower);

    let transition_effect = if contains_any(&lower, &["затъмняване", "dip to black", "fade to black", "fade out", "fade in"]) {
        Some("Dip to Black".to_string())
    } else if contains_any(&lower, &["преход", "преходи", "появяване", "transition", "fade", "dissolve"]) {
        Some("Cross Dissolve".to_string())
    } else {
        None
    };

    let apply_bw = contains_any(&lower, &["черно-бял", "черно бял", "чернобял", "black and white", "black & white", "b&w", "monochrome", "grayscale"]);
    let apply_blur = contains_any(&lower, &["замъгли", "замъгляване", "блъър", "blur"]);
    let apply_color_grade =
        !apply_bw && contains_any(&lower, &["цвят", "цветове", "контраст", "наситеност", "color", "grade", "lumetri", "contrast", "cinematic"]);
    let zoom_scale = if contains_any(&lower, &["приближи", "увеличение", "зуум", "zoom in", "zoom"]) { Some(125.0) } else { None };

    let audio_mute = contains_any(&lower, &["заглуши", "без звук", "mute"]);
    let audio_normalize = contains_any(&lower, &["нормализирай", "normalize"]);
    let run_audio_enhance = contains_any(&lower, &["изчисти звук", "изчисти аудио", "clean audio", "enhance audio", "denoise", "podcast"]);
    let audio_gain_db = if contains_any(&lower, &["усили", "по-силно", "boost", "louder", "volume up"]) {
        Some(6.0)
    } else if contains_any(&lower, &["намали звук", "по-тихо", "quieter", "volume down", "lower volume"])
        || (lower.contains("намали") && !lower.contains("скорост"))
    {
        Some(-6.0)
    } else {
        None
    };

    AllInOnePlan {
        prompt_summary: trimmed.chars().take(120).collect(),
        run_subtitles,
        subtitle_language,
        run_text_overlay,
        text_overlay_content,
        run_remove_pauses,
        run_split_at_playhead,
        speed_percent,
        transition_effect,
        apply_bw,
        apply_blur,
        apply_color_grade,
        zoom_scale,
        audio_mute,
        audio_normalize,
        audio_gain_db,
        run_audio_enhance,
        run_auto_reframe,
        reframe_split_screen,
        run_bg_removal,
        run_viral_clips,
        run_smart_triggers,
        run_vod_chapters,
        used_ollama: false,
    }
}

fn try_plan_with_ollama(endpoint: &str, prompt: &str, base: &mut AllInOnePlan) -> bool {
    let url = format!("{}/api/generate", endpoint.trim_end_matches('/'));
    let sys = "You are a video editing command planner. Return ONLY compact JSON with boolean keys: run_subtitles, run_text_overlay, run_remove_pauses, run_split_at_playhead, apply_bw, apply_blur, apply_color_grade, audio_mute, audio_normalize, run_audio_enhance, run_auto_reframe, reframe_split_screen, run_bg_removal, run_viral_clips, run_smart_triggers, run_vod_chapters, plus optional string text_overlay_content.";
    let body = json!({
        "model": "llama3.2",
        "prompt": format!("{sys}\nUser request: {prompt}"),
        "stream": false,
        "format": "json"
    });
    let agent = ureq::Agent::new_with_config(ureq::config::Config::builder().timeout_global(Some(Duration::from_secs(6))).build());
    let Ok(resp) = agent.post(&url).header("Content-Type", "application/json").send(body.to_string().as_bytes()) else {
        return false;
    };
    let mut text = String::new();
    if resp.into_body().into_reader().take(64 * 1024).read_to_string(&mut text).is_err() {
        return false;
    }
    let Ok(outer) = serde_json::from_str::<Value>(&text) else {
        return false;
    };
    let raw_inner = outer.get("response").and_then(Value::as_str).unwrap_or("");
    let Ok(parsed) = serde_json::from_str::<Value>(raw_inner) else {
        return false;
    };
    if parsed.get("run_subtitles").and_then(Value::as_bool) == Some(true) {
        base.run_subtitles = true;
    }
    if parsed.get("run_text_overlay").and_then(Value::as_bool) == Some(true) {
        base.run_text_overlay = true;
    }
    if let Some(txt) = parsed.get("text_overlay_content").and_then(Value::as_str)
        && !txt.trim().is_empty()
    {
        base.text_overlay_content = Some(txt.trim().to_string());
    }
    if parsed.get("run_remove_pauses").and_then(Value::as_bool) == Some(true) {
        base.run_remove_pauses = true;
    }
    if parsed.get("run_split_at_playhead").and_then(Value::as_bool) == Some(true) {
        base.run_split_at_playhead = true;
    }
    if parsed.get("apply_bw").and_then(Value::as_bool) == Some(true) {
        base.apply_bw = true;
    }
    if parsed.get("apply_blur").and_then(Value::as_bool) == Some(true) {
        base.apply_blur = true;
    }
    if parsed.get("apply_color_grade").and_then(Value::as_bool) == Some(true) {
        base.apply_color_grade = true;
    }
    if parsed.get("audio_mute").and_then(Value::as_bool) == Some(true) {
        base.audio_mute = true;
    }
    if parsed.get("audio_normalize").and_then(Value::as_bool) == Some(true) {
        base.audio_normalize = true;
    }
    if parsed.get("run_audio_enhance").and_then(Value::as_bool) == Some(true) {
        base.run_audio_enhance = true;
    }
    if parsed.get("run_auto_reframe").and_then(Value::as_bool) == Some(true) {
        base.run_auto_reframe = true;
    }
    if parsed.get("reframe_split_screen").and_then(Value::as_bool) == Some(true) {
        base.reframe_split_screen = true;
        base.run_auto_reframe = true;
    }
    if parsed.get("run_bg_removal").and_then(Value::as_bool) == Some(true) {
        base.run_bg_removal = true;
    }
    if parsed.get("run_viral_clips").and_then(Value::as_bool) == Some(true) {
        base.run_viral_clips = true;
    }
    if parsed.get("run_smart_triggers").and_then(Value::as_bool) == Some(true) {
        base.run_smart_triggers = true;
    }
    if parsed.get("run_vod_chapters").and_then(Value::as_bool) == Some(true) {
        base.run_vod_chapters = true;
    }
    base.used_ollama = true;
    true
}

fn check_cancelled(prog: Option<&Progress>) -> std::result::Result<(), String> {
    if prog.is_some_and(|p| p.cancel.load(Ordering::Relaxed)) {
        return Err("cancelled".to_string());
    }
    Ok(())
}

fn target_video_clips(s: &Session, sid: ItemId) -> Vec<ClipId> {
    let Some(seq) = s.project.sequence(sid) else {
        return Vec::new();
    };
    let selected: Vec<ClipId> =
        s.state.selection.iter().copied().filter(|cid| seq.video_tracks.iter().any(|tr| !tr.locked && tr.items.iter().any(|it| it.id == *cid))).collect();
    if !selected.is_empty() {
        return selected;
    }
    seq.video_tracks.iter().filter(|tr| !tr.locked).flat_map(|tr| tr.items.iter().map(|it| it.id)).collect()
}

fn target_audio_clips(s: &Session, sid: ItemId) -> Vec<ClipId> {
    let Some(seq) = s.project.sequence(sid) else {
        return Vec::new();
    };
    let selected: Vec<ClipId> =
        s.state.selection.iter().copied().filter(|cid| seq.audio_tracks.iter().any(|tr| !tr.locked && tr.items.iter().any(|it| it.id == *cid))).collect();
    if !selected.is_empty() {
        return selected;
    }
    seq.audio_tracks.iter().filter(|tr| !tr.locked).flat_map(|tr| tr.items.iter().map(|it| it.id)).collect()
}

fn target_all_clips(s: &Session, sid: ItemId) -> Vec<ClipId> {
    if !s.state.selection.is_empty() {
        return s.state.selection.clone();
    }
    let Some(seq) = s.project.sequence(sid) else {
        return Vec::new();
    };
    seq.video_tracks
        .iter()
        .filter(|tr| !tr.locked)
        .flat_map(|tr| tr.items.iter().map(|it| it.id))
        .chain(seq.audio_tracks.iter().filter(|tr| !tr.locked).flat_map(|tr| tr.items.iter().map(|it| it.id)))
        .collect()
}

fn apply_timing_and_audio_steps(s: &mut Session, sid: ItemId, plan: &AllInOnePlan, steps_applied: &mut Vec<String>) {
    // 2. Cuts, splits, and speed.
    if plan.run_split_at_playhead {
        let t = s.playhead();
        match s.execute("timeline.razor", json!({"time": t.0})) {
            Ok(v) => {
                let cuts = v.get("cuts").and_then(Value::as_u64).unwrap_or(0);
                steps_applied.push(if cuts > 0 { format!("Split {cuts} clip(s) at playhead") } else { "No clip under playhead to split".into() });
            }
            Err(e) => steps_applied.push(format!("Split failed: {e}")),
        }
    }

    if let Some(speed) = plan.speed_percent {
        let clips: Vec<u64> = target_all_clips(s, sid).into_iter().map(|c| c.0).collect();
        if clips.is_empty() {
            steps_applied.push("Speed change skipped (no unlocked clips on timeline)".into());
        } else if s.execute("clip.speedDuration", json!({"clips": clips, "speed": speed})).is_ok() {
            steps_applied.push(format!("Set clip speed to {speed:.0}%"));
        }
    }

    if plan.run_remove_pauses {
        match s.execute("ai.removePauses", json!({"wait": true})) {
            Ok(_) => steps_applied.push("Removed silences".into()),
            Err(e) => steps_applied.push(format!("Remove silences skipped: {e}")),
        }
    }

    // 3. Audio operations.
    if plan.run_audio_enhance && s.execute("ai.enhanceAudio", json!({"wait": true})).is_ok() {
        steps_applied.push("AI Audio Enhance".into());
    }

    if plan.audio_mute {
        let aclips = target_audio_clips(s, sid);
        let mut muted = 0usize;
        for cid in aclips {
            if s.execute("effects.apply", json!({"clips": [cid.0], "effect": "Mute"})).is_ok()
                && s.execute("effects.setParam", json!({"clip": cid.0, "effect": "mute", "param": "mute", "value": true})).is_ok()
            {
                muted += 1;
            }
        }
        if muted > 0 {
            steps_applied.push(format!("Muted {muted} audio clip(s)"));
        } else {
            steps_applied.push("Mute skipped (no unlocked audio clips)".into());
        }
    }

    if plan.audio_normalize {
        let aclips: Vec<u64> = target_audio_clips(s, sid).into_iter().map(|c| c.0).collect();
        if !aclips.is_empty() && s.execute("clip.audioGain", json!({"clips": aclips, "mode": "normalizeAll", "db": -3.0})).is_ok() {
            steps_applied.push("Normalized audio peaks to -3 dB".into());
        }
    }

    if let Some(db) = plan.audio_gain_db {
        let aclips: Vec<u64> = target_audio_clips(s, sid).into_iter().map(|c| c.0).collect();
        if !aclips.is_empty() && s.execute("clip.audioGain", json!({"clips": aclips, "mode": "adjust", "db": db})).is_ok() {
            steps_applied.push(format!("Adjusted audio gain by {db:+.1} dB"));
        }
    }
}

fn apply_ai_and_subtitle_steps(s: &mut Session, plan: &AllInOnePlan, steps_applied: &mut Vec<String>) {
    // 4. AI framing & visual timeline tools.
    // Ordering invariant: `ai.autoReframe` MUST execute BEFORE `ai.generateSubtitles` (step 5)
    // and text overlays (step 8) so any Game Stream ` (FaceCam)` split clip already exists on
    // the sequence when `generate_subtitles` runs `detect_game_stream_split_y` (`split_y = 0.55`).
    if plan.run_auto_reframe {
        let mut params = json!({
            "aspect": "9:16",
            "resizeSequence": true,
            "splitScreen": plan.reframe_split_screen,
            "wait": true
        });
        if plan.reframe_split_screen {
            params["facecamPlacement"] = json!("bottom");
            params["facecamRegion"] = json!("auto");
        }
        if s.execute("ai.autoReframe", params).is_ok() {
            steps_applied.push(if plan.reframe_split_screen { "Gaming Split-Screen (9:16)" } else { "AI Auto-Reframe (9:16)" }.into());
        }
    }
    if plan.run_bg_removal && s.execute("ai.removeBackground", json!({"mode": "transparentAlpha", "wait": true})).is_ok() {
        steps_applied.push("AI Background Removal".into());
    }

    // 5. Subtitles (always after `ai.removePauses` and `ai.autoReframe` so timestamps match the
    // post-cut timeline and Game Stream captions anchor above FaceCam, and BEFORE `ai.vodChapters` /
    // `ai.findViralClips` / `ai.applySmartTriggers` so downstream tools reuse the generated captions).
    if plan.run_subtitles {
        let mut params = json!({"wait": true});
        if let Some(lang) = &plan.subtitle_language {
            params["language"] = json!(lang);
        }
        match s.execute("ai.generateSubtitles", params) {
            Ok(_) => steps_applied.push(match &plan.subtitle_language {
                Some(lang) => format!("Generated AI Subtitles ({lang})"),
                None => "Generated AI Subtitles".into(),
            }),
            Err(e) => steps_applied.push(format!("AI Subtitles failed: {e}")),
        }
    }

    if plan.run_vod_chapters && s.execute("ai.vodChapters", json!({"wait": true})).is_ok() {
        steps_applied.push("Stream VOD Chapterizer".into());
    }
    if plan.run_viral_clips && s.execute("ai.findViralClips", json!({"wait": true})).is_ok() {
        steps_applied.push("AI Viral Clip Finder".into());
    }
    if plan.run_smart_triggers && s.execute("ai.applySmartTriggers", json!({"wait": true})).is_ok() {
        steps_applied.push("AI Smart Triggers".into());
    }
}

fn apply_visual_and_text_steps(s: &mut Session, sid: ItemId, plan: &AllInOnePlan, steps_applied: &mut Vec<String>) {
    // 6. Transitions.
    if let Some(effect_name) = &plan.transition_effect {
        let cuts: Vec<(filmcraft_project::TrackId, Tick)> = s
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
            if s.execute("timeline.addTransition", json!({"track": tid.0, "at": at.0, "effect": effect_name})).is_ok() {
                added += 1;
            }
        }
        if added > 0 {
            steps_applied.push(format!("Added {added} {effect_name} transition(s)"));
        } else {
            steps_applied.push(format!("No adjacent clip edit points found for {effect_name}"));
        }
    }

    // 7. Visual effects, color grading, and zoom.
    let vclips = target_video_clips(s, sid);
    if plan.apply_bw {
        let applied = vclips.iter().filter(|cid| s.execute("effects.apply", json!({"clips": [cid.0], "effect": "black_white"})).is_ok()).count();
        if applied > 0 {
            steps_applied.push(format!("Applied Black & White to {applied} clip(s)"));
        }
    }
    if plan.apply_blur {
        let applied = vclips.iter().filter(|cid| s.execute("effects.apply", json!({"clips": [cid.0], "effect": "gaussian_blur"})).is_ok()).count();
        if applied > 0 {
            steps_applied.push(format!("Applied Gaussian Blur to {applied} clip(s)"));
        }
    }
    if plan.apply_color_grade {
        let applied = vclips.iter().filter(|cid| s.execute("effects.apply", json!({"clips": [cid.0], "effect": "lumetri"})).is_ok()).count();
        if applied > 0 {
            steps_applied.push(format!("Applied Lumetri Color to {applied} clip(s)"));
        }
    }
    if let Some(scale) = plan.zoom_scale {
        let zoomed = vclips
            .iter()
            .filter(|cid| s.execute("effects.setParam", json!({"clip": cid.0, "effect": "motion", "param": "scale", "value": scale})).is_ok())
            .count();
        if zoomed > 0 {
            steps_applied.push(format!("Set zoom scale to {scale:.0}% on {zoomed} clip(s)"));
        }
    }

    // 8. Native text / title overlay on the timeline (after autoReframe so Game Stream split is respected).
    if plan.run_text_overlay {
        let text = plan.text_overlay_content.clone().unwrap_or_else(|| "Title".to_string());
        let mut params = json!({"text": text, "newClip": true});
        if let Some(seq) = s.project.sequence(sid)
            && let Some(split_y) = crate::subtitles::detect_game_stream_split_y(seq)
        {
            let cx = f64::from(seq.settings.width.max(1)) * 0.5;
            let cy = (f64::from(seq.settings.height.max(1)) * (split_y - 0.03).clamp(0.45, 0.70)).round();
            params["position"] = json!([cx, cy]);
        }
        if s.execute("graphics.newText", params).is_ok() {
            steps_applied.push(format!("Added Text Overlay: \"{text}\""));
        }
    }
}

fn apply_all_in_one_to_session(s: &mut Session, out: PreparedAllInOneOutput) -> Result<AllInOneReport> {
    let seq_id = s.state.active_sequence.or_else(|| s.project.sequences().next().map(|q| q.id));
    let mut steps_applied = Vec::new();
    let mut attachments_imported = 0usize;

    let report_res: Result<()> = s.edit_group("Orpheus All-In-One", |s| {
        // 1. Import user-attached local media files into the Project bin.
        let mut attachments_skipped = 0usize;
        for att in &out.attachments {
            match crate::commands::import_bytes(s, &att.orig_path, att.bytes.clone(), None) {
                Ok(_) => attachments_imported += 1,
                Err(_) => attachments_skipped += 1,
            }
        }
        if attachments_imported > 0 {
            steps_applied.push(format!("Imported {attachments_imported} attachment(s)"));
        }
        if attachments_skipped > 0 {
            steps_applied.push(format!("Skipped {attachments_skipped} unreadable attachment(s)"));
        }

        let Some(sid) = seq_id else {
            steps_applied.push("No active sequence open — open or create a sequence first".into());
            return Ok(());
        };

        apply_timing_and_audio_steps(s, sid, &out.plan, &mut steps_applied);
        apply_ai_and_subtitle_steps(s, &out.plan, &mut steps_applied);
        apply_visual_and_text_steps(s, sid, &out.plan, &mut steps_applied);
        Ok(())
    });
    report_res?;

    if steps_applied.is_empty() {
        steps_applied.push("No matching timeline editing actions recognized in prompt".into());
    }

    let rep = AllInOneReport { seq: seq_id, steps_applied, generated_layers: 0, attachments_imported, wikimedia_fetched: 0, used_ollama: out.plan.used_ollama };
    s.all_in_one_report = Some(rep.clone());
    s.toast(format!("Orpheus All-In-One: {}", rep.steps_applied.join(" · ")));
    Ok(rep)
}

fn poll_child_queues(s: &mut Session) {
    crate::silence::poll(s);
    crate::audio_enhance::poll(s);
    crate::auto_reframe::poll(s);
    crate::bg_removal::poll(s);
    crate::subtitles::poll(s);
    crate::vod_chapters::poll(s);
    crate::viral_clips::poll(s);
    crate::smart_triggers::poll(s);
}

fn is_child_job_active(s: &Session, child_id: u64) -> bool {
    let job_running = s.jobs.iter().find(|j| j.id == child_id).is_some_and(|j| !j.progress.finished.load(Ordering::Relaxed));
    if job_running {
        return true;
    }
    s.silence_jobs.iter().any(|j| j.job == child_id)
        || s.audio_enhance_jobs.iter().any(|j| j.job == child_id)
        || s.auto_reframe_jobs.iter().any(|j| j.job == child_id)
        || s.bg_removal_jobs.iter().any(|j| j.job == child_id)
        || s.subtitle_jobs.iter().any(|j| j.job == child_id)
        || s.vod_chapters_jobs.iter().any(|j| j.job == child_id)
        || s.viral_jobs.iter().any(|j| j.job == child_id)
        || s.smart_trigger_jobs.iter().any(|j| j.job == child_id)
}

fn child_job_error(s: &Session, child_id: u64) -> Option<String> {
    let job = s.jobs.iter().find(|j| j.id == child_id)?;
    if let Some(Err(e)) = lock(&job.result).clone() {
        return Some(e);
    }
    lock(&job.progress.error).clone()
}

fn apply_audio_instant_ops(s: &mut Session, sid: ItemId, plan: &AllInOnePlan, steps_applied: &mut Vec<String>) {
    if plan.audio_mute {
        let aclips = target_audio_clips(s, sid);
        let mut muted = 0usize;
        for cid in aclips {
            if s.execute("effects.apply", json!({"clips": [cid.0], "effect": "Mute"})).is_ok()
                && s.execute("effects.setParam", json!({"clip": cid.0, "effect": "mute", "param": "mute", "value": true})).is_ok()
            {
                muted += 1;
            }
        }
        if muted > 0 {
            steps_applied.push(format!("Muted {muted} audio clip(s)"));
        } else {
            steps_applied.push("Mute skipped (no unlocked audio clips)".into());
        }
    }
    if plan.audio_normalize {
        let aclips: Vec<u64> = target_audio_clips(s, sid).into_iter().map(|c| c.0).collect();
        if !aclips.is_empty() && s.execute("clip.audioGain", json!({"clips": aclips, "mode": "normalizeAll", "db": -3.0})).is_ok() {
            steps_applied.push("Normalized audio peaks to -3 dB".into());
        }
    }
    if let Some(db) = plan.audio_gain_db {
        let aclips: Vec<u64> = target_audio_clips(s, sid).into_iter().map(|c| c.0).collect();
        if !aclips.is_empty() && s.execute("clip.audioGain", json!({"clips": aclips, "mode": "adjust", "db": db})).is_ok() {
            steps_applied.push(format!("Adjusted audio gain by {db:+.1} dB"));
        }
    }
}

fn apply_attachments_and_precuts(s: &mut Session, pj: &mut PendingAllInOneJob, plan: &AllInOnePlan) -> bool {
    let mut attachments_skipped = 0usize;
    for att in &pj.attachments {
        match crate::commands::import_bytes(s, &att.orig_path, att.bytes.clone(), None) {
            Ok(_) => pj.attachments_imported += 1,
            Err(_) => attachments_skipped += 1,
        }
    }
    if pj.attachments_imported > 0 {
        pj.steps_applied.push(format!("Imported {} attachment(s)", pj.attachments_imported));
    }
    if attachments_skipped > 0 {
        pj.steps_applied.push(format!("Skipped {attachments_skipped} unreadable attachment(s)"));
    }
    let seq_id = s.state.active_sequence.or_else(|| s.project.sequences().next().map(|q| q.id));
    pj.seq_id = seq_id;
    let Some(sid) = seq_id else {
        pj.steps_applied.push("No active sequence open — open or create a sequence first".into());
        return false;
    };
    if plan.run_split_at_playhead {
        let t = s.playhead();
        match s.execute("timeline.razor", json!({"time": t.0})) {
            Ok(v) => {
                let cuts = v.get("cuts").and_then(Value::as_u64).unwrap_or(0);
                pj.steps_applied.push(if cuts > 0 { format!("Split {cuts} clip(s) at playhead") } else { "No clip under playhead to split".into() });
            }
            Err(e) => pj.steps_applied.push(format!("Split failed: {e}")),
        }
    }
    if let Some(speed) = plan.speed_percent {
        let clips: Vec<u64> = target_all_clips(s, sid).into_iter().map(|c| c.0).collect();
        if clips.is_empty() {
            pj.steps_applied.push("Speed change skipped (no unlocked clips on timeline)".into());
        } else if s.execute("clip.speedDuration", json!({"clips": clips, "speed": speed})).is_ok() {
            pj.steps_applied.push(format!("Set clip speed to {speed:.0}%"));
        }
    }
    true
}

fn advance_pipeline_step(s: &mut Session, pj: &mut PendingAllInOneJob) {
    let Some(plan) = pj.plan.clone() else { return };

    loop {
        match pj.stage {
            PipelineStage::Planning => {
                pj.stage = PipelineStage::AttachmentsAndPreCuts;
            }
            PipelineStage::AttachmentsAndPreCuts => {
                if !apply_attachments_and_precuts(s, pj, &plan) {
                    pj.stage = PipelineStage::Done;
                    continue;
                }
                pj.stage = PipelineStage::RemovePauses;
            }
            PipelineStage::RemovePauses => {
                pj.stage = PipelineStage::AudioOps;
                if plan.run_remove_pauses {
                    match s.execute("ai.removePauses", json!({"async": true})) {
                        Ok(v) => {
                            if let Some(child_id) = v.get("job").and_then(Value::as_u64) {
                                pj.waiting_on_job = Some(child_id);
                                pj.waiting_step_success = Some("Removed silences".into());
                                pj.waiting_step_fail_prefix = Some("Remove silences skipped".into());
                                return;
                            }
                            pj.steps_applied.push("Removed silences".into());
                        }
                        Err(e) => pj.steps_applied.push(format!("Remove silences skipped: {e}")),
                    }
                }
            }
            PipelineStage::AudioOps => {
                pj.stage = PipelineStage::AutoReframe;
                if let Some(sid) = pj.seq_id {
                    apply_audio_instant_ops(s, sid, &plan, &mut pj.steps_applied);
                }
                if plan.run_audio_enhance
                    && let Ok(v) = s.execute("ai.enhanceAudio", json!({"wait": false}))
                {
                    if let Some(child_id) = v.get("job").and_then(Value::as_u64) {
                        pj.waiting_on_job = Some(child_id);
                        pj.waiting_step_success = Some("AI Audio Enhance".into());
                        pj.waiting_step_fail_prefix = None;
                        return;
                    }
                    pj.steps_applied.push("AI Audio Enhance".into());
                }
            }
            PipelineStage::AutoReframe => {
                pj.stage = PipelineStage::BgRemoval;
                if plan.run_auto_reframe {
                    let mut params = json!({
                        "aspect": "9:16",
                        "resizeSequence": true,
                        "splitScreen": plan.reframe_split_screen,
                        "wait": false
                    });
                    if plan.reframe_split_screen {
                        params["facecamPlacement"] = json!("bottom");
                        params["facecamRegion"] = json!("auto");
                    }
                    let label = if plan.reframe_split_screen { "Gaming Split-Screen (9:16)" } else { "AI Auto-Reframe (9:16)" };
                    if let Ok(v) = s.execute("ai.autoReframe", params) {
                        if let Some(child_id) = v.get("job").and_then(Value::as_u64) {
                            pj.waiting_on_job = Some(child_id);
                            pj.waiting_step_success = Some(label.into());
                            pj.waiting_step_fail_prefix = None;
                            return;
                        }
                        pj.steps_applied.push(label.into());
                    }
                }
            }
            PipelineStage::BgRemoval => {
                pj.stage = PipelineStage::Subtitles;
                if plan.run_bg_removal
                    && let Ok(v) = s.execute("ai.removeBackground", json!({"mode": "transparentAlpha", "wait": false}))
                {
                    if let Some(child_id) = v.get("job").and_then(Value::as_u64) {
                        pj.waiting_on_job = Some(child_id);
                        pj.waiting_step_success = Some("AI Background Removal".into());
                        pj.waiting_step_fail_prefix = None;
                        return;
                    }
                    pj.steps_applied.push("AI Background Removal".into());
                }
            }
            PipelineStage::Subtitles => {
                pj.stage = PipelineStage::VodChapters;
                if plan.run_subtitles {
                    let mut params = json!({"wait": false});
                    if let Some(lang) = &plan.subtitle_language {
                        params["language"] = json!(lang);
                    }
                    let ok_label = match &plan.subtitle_language {
                        Some(lang) => format!("Generated AI Subtitles ({lang})"),
                        None => "Generated AI Subtitles".into(),
                    };
                    match s.execute("ai.generateSubtitles", params) {
                        Ok(v) => {
                            if let Some(child_id) = v.get("job").and_then(Value::as_u64) {
                                pj.waiting_on_job = Some(child_id);
                                pj.waiting_step_success = Some(ok_label);
                                pj.waiting_step_fail_prefix = Some("AI Subtitles failed".into());
                                return;
                            }
                            pj.steps_applied.push(ok_label);
                        }
                        Err(e) => pj.steps_applied.push(format!("AI Subtitles failed: {e}")),
                    }
                }
            }
            PipelineStage::VodChapters => {
                pj.stage = PipelineStage::ViralClips;
                if plan.run_vod_chapters
                    && let Ok(v) = s.execute("ai.vodChapters", json!({"wait": false}))
                {
                    if let Some(child_id) = v.get("job").and_then(Value::as_u64) {
                        pj.waiting_on_job = Some(child_id);
                        pj.waiting_step_success = Some("Stream VOD Chapterizer".into());
                        pj.waiting_step_fail_prefix = None;
                        return;
                    }
                    pj.steps_applied.push("Stream VOD Chapterizer".into());
                }
            }
            PipelineStage::ViralClips => {
                pj.stage = PipelineStage::SmartTriggers;
                if plan.run_viral_clips
                    && let Ok(v) = s.execute("ai.findViralClips", json!({"wait": false}))
                {
                    if let Some(child_id) = v.get("job").and_then(Value::as_u64) {
                        pj.waiting_on_job = Some(child_id);
                        pj.waiting_step_success = Some("AI Viral Clip Finder".into());
                        pj.waiting_step_fail_prefix = None;
                        return;
                    }
                    pj.steps_applied.push("AI Viral Clip Finder".into());
                }
            }
            PipelineStage::SmartTriggers => {
                pj.stage = PipelineStage::FinalVisualsAndText;
                if plan.run_smart_triggers
                    && let Ok(v) = s.execute("ai.applySmartTriggers", json!({"wait": false}))
                {
                    if let Some(child_id) = v.get("job").and_then(Value::as_u64) {
                        pj.waiting_on_job = Some(child_id);
                        pj.waiting_step_success = Some("AI Smart Triggers".into());
                        pj.waiting_step_fail_prefix = None;
                        return;
                    }
                    pj.steps_applied.push("AI Smart Triggers".into());
                }
            }
            PipelineStage::FinalVisualsAndText => {
                if let Some(sid) = pj.seq_id {
                    apply_visual_and_text_steps(s, sid, &plan, &mut pj.steps_applied);
                }
                pj.stage = PipelineStage::Done;
            }
            PipelineStage::Done => break,
        }
    }
}

/// Advance active multi-step Orpheus All-In-One pipelines on the session thread.
///
/// Ensures child background jobs (`PendingSilenceJob`, `PendingSubtitle`, `PendingVodChapterJob`,
/// etc.) finish and commit their timeline edits one stage at a time before the next stage begins.
pub fn poll(s: &mut Session) {
    if s.all_in_one_jobs.is_empty() {
        return;
    }
    poll_child_queues(s);

    let mut i = 0usize;
    while i < s.all_in_one_jobs.len() {
        let master_id = s.all_in_one_jobs[i].job;
        let master_cancelled = s.jobs.iter().find(|j| j.id == master_id).is_some_and(|j| j.progress.cancel.load(Ordering::Relaxed));
        let master_failed_early = s
            .jobs
            .iter()
            .find(|j| j.id == master_id)
            .is_some_and(|j| j.progress.finished.load(Ordering::Relaxed) && lock(&j.result).as_ref().is_some_and(std::result::Result::is_err));

        if master_failed_early {
            s.all_in_one_jobs.remove(i);
            continue;
        }

        if master_cancelled {
            if let Some(child_id) = s.all_in_one_jobs[i].waiting_on_job
                && let Some(cj) = s.jobs.iter().find(|j| j.id == child_id)
            {
                cj.progress.cancel.store(true, Ordering::Relaxed);
            }
            if let Some(mj) = s.jobs.iter().find(|j| j.id == master_id) {
                *lock(&mj.progress.status) = "Cancelled".into();
                *lock(&mj.progress.error) = Some("Cancelled".into());
                *lock(&mj.result) = Some(Err("Cancelled".into()));
                mj.progress.finished.store(true, Ordering::Relaxed);
            }
            s.all_in_one_jobs.remove(i);
            continue;
        }

        if s.all_in_one_jobs[i].epoch != s.document_epoch {
            if let Some(child_id) = s.all_in_one_jobs[i].waiting_on_job
                && let Some(cj) = s.jobs.iter().find(|j| j.id == child_id)
            {
                cj.progress.cancel.store(true, Ordering::Relaxed);
            }
            if let Some(mj) = s.jobs.iter().find(|j| j.id == master_id) {
                mj.progress.finished.store(true, Ordering::Relaxed);
            }
            s.all_in_one_jobs.remove(i);
            s.error_toast("ai.allInOne", "Orpheus All-In-One: the project was replaced; nothing was changed");
            continue;
        }

        // Wait for the background planning worker to finish building `PreparedAllInOneOutput`.
        if s.all_in_one_jobs[i].plan.is_none() {
            if !s.all_in_one_jobs[i].plan_ready.load(Ordering::Acquire) {
                i += 1;
                continue;
            }
            let Some(prep) = lock(&s.all_in_one_jobs[i].output).take() else {
                s.all_in_one_jobs.remove(i);
                continue;
            };
            s.all_in_one_jobs[i].plan = Some(prep.plan);
            s.all_in_one_jobs[i].attachments = prep.attachments;
            s.all_in_one_jobs[i].stage = PipelineStage::AttachmentsAndPreCuts;
        }

        // If waiting on a child background job, check whether it has finished and committed.
        if let Some(child_id) = s.all_in_one_jobs[i].waiting_on_job {
            if is_child_job_active(s, child_id) {
                let child_info = s.jobs.iter().find(|j| j.id == child_id).map(|j| (lock(&j.progress.status).clone(), j.progress.fraction()));
                if let Some((child_status, child_frac)) = child_info
                    && let Some(mj) = s.jobs.iter().find(|j| j.id == master_id)
                {
                    if !child_status.is_empty() {
                        *lock(&mj.progress.status) = child_status;
                    }
                    let done_pct = (15.0 + child_frac.clamp(0.0, 1.0) * 80.0).round() as u64;
                    mj.progress.done.store(done_pct.min(95), Ordering::Relaxed);
                }
                i += 1;
                continue;
            }
            let err = child_job_error(s, child_id);
            let pj = &mut s.all_in_one_jobs[i];
            pj.waiting_on_job = None;
            let ok_label = pj.waiting_step_success.take();
            let fail_prefix = pj.waiting_step_fail_prefix.take();
            match (err, ok_label, fail_prefix) {
                (None, Some(lbl), _) => pj.steps_applied.push(lbl),
                (Some(e), _, Some(prefix)) => pj.steps_applied.push(format!("{prefix}: {e}")),
                _ => {}
            }
        }

        // Temporarily take `pj` out of `s.all_in_one_jobs` so `advance_pipeline_step` can mutate `s`.
        let mut pj = s.all_in_one_jobs.remove(i);
        advance_pipeline_step(s, &mut pj);

        if pj.waiting_on_job.is_some() && pj.stage != PipelineStage::Done {
            s.all_in_one_jobs.insert(i, pj);
            i += 1;
            continue;
        }

        // Pipeline completed all stages!
        if pj.steps_applied.is_empty() {
            pj.steps_applied.push("No matching timeline editing actions recognized in prompt".into());
        }
        let used_ollama = pj.plan.as_ref().is_some_and(|pl| pl.used_ollama);
        let rep = AllInOneReport {
            seq: pj.seq_id,
            steps_applied: pj.steps_applied,
            generated_layers: 0,
            attachments_imported: pj.attachments_imported,
            wikimedia_fetched: 0,
            used_ollama,
        };
        s.all_in_one_report = Some(rep.clone());
        s.toast(format!("Orpheus All-In-One: {}", rep.steps_applied.join(" · ")));

        if let Some(mj) = s.jobs.iter().find(|j| j.id == master_id) {
            *lock(&mj.progress.status) = rep.steps_applied.join(" · ");
            mj.progress.done.store(100, Ordering::Relaxed);
            *lock(&mj.result) =
                Some(Ok(Report { frames: rep.steps_applied.len() as u64, seconds: pj.started_at.elapsed().as_secs_f64(), ..Default::default() }));
            mj.progress.finished.store(true, Ordering::Relaxed);
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

    let ollama_url = str_p(p, "ollamaUrl").unwrap_or("").to_string();

    let id = next_job_id(s);
    let prog = Arc::new(Progress::default());
    prog.total.store(100, Ordering::Relaxed);
    *lock(&prog.status) = "Planning Orpheus All-In-One…".into();
    let result: Arc<Mutex<Option<std::result::Result<Report, String>>>> = Arc::default();
    s.jobs.push(Job { id, label: "Orpheus All-In-One".into(), progress: prog.clone(), result: result.clone() });

    let output_slot: Arc<Mutex<Option<PreparedAllInOneOutput>>> = Arc::default();
    let plan_ready = Arc::new(AtomicBool::new(false));
    s.all_in_one_jobs.push(PendingAllInOneJob {
        job: id,
        epoch: s.document_epoch,
        prompt: prompt.clone(),
        plan_ready: plan_ready.clone(),
        output: output_slot.clone(),
        plan: None,
        attachments: Vec::new(),
        seq_id: None,
        stage: PipelineStage::Planning,
        waiting_on_job: None,
        waiting_step_success: None,
        waiting_step_fail_prefix: None,
        steps_applied: Vec::new(),
        attachments_imported: 0,
        started_at: web_time::Instant::now(),
    });

    let wait = bool_p(p, "wait").unwrap_or(false);
    let plan_ready_bg = plan_ready.clone();
    let prog_bg = prog.clone();
    let result_bg = result.clone();
    let run = move || {
        let outcome = (|| -> std::result::Result<PreparedAllInOneOutput, String> {
            check_cancelled(Some(&prog_bg))?;
            let mut plan = plan_from_prompt(&prompt, attachment_paths.len());
            if !ollama_url.trim().is_empty() {
                *lock(&prog_bg.status) = "Consulting local Ollama planner…".into();
                let _ = try_plan_with_ollama(&ollama_url, &prompt, &mut plan);
            }
            prog_bg.done.store(15, Ordering::Relaxed);

            let mut attachments = Vec::new();
            for path in &attachment_paths {
                check_cancelled(Some(&prog_bg))?;
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
            Ok(PreparedAllInOneOutput { plan, attachments })
        })();

        match outcome {
            Ok(prep) => {
                *lock(&output_slot) = Some(prep);
                plan_ready_bg.store(true, Ordering::Release);
            }
            Err(e) => {
                *lock(&prog_bg.error) = Some(e.clone());
                *lock(&prog_bg.status) = e.clone();
                *lock(&result_bg) = Some(Err(e));
                prog_bg.finished.store(true, Ordering::Relaxed);
            }
        }
    };

    if wait || cfg!(target_arch = "wasm32") {
        run();
        if let Some(idx) = s.all_in_one_jobs.iter().position(|j| j.job == id) {
            let pj = s.all_in_one_jobs.remove(idx);
            if let Some(prep) = lock(&pj.output).take() {
                let t0 = pj.started_at;
                match apply_all_in_one_to_session(s, prep) {
                    Ok(rep) => {
                        *lock(&prog.status) = rep.steps_applied.join(" · ");
                        prog.done.store(100, Ordering::Relaxed);
                        *lock(&result) = Some(Ok(Report { frames: rep.steps_applied.len() as u64, seconds: t0.elapsed().as_secs_f64(), ..Default::default() }));
                        prog.finished.store(true, Ordering::Relaxed);
                    }
                    Err(e) => {
                        let msg = e.to_string();
                        *lock(&prog.error) = Some(msg.clone());
                        *lock(&prog.status) = msg.clone();
                        *lock(&result) = Some(Err(msg));
                        prog.finished.store(true, Ordering::Relaxed);
                    }
                }
            }
        }
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
            let error = lock(&j.progress.error).clone();
            json!({
                "job": j.id,
                "running": running,
                "progress": (done as f64) / (total as f64),
                "status": status,
                "error": error,
            })
        })
    });
    let report = s.all_in_one_report.as_ref().map(|r| {
        json!({
            "stepsApplied": r.steps_applied,
            "generatedLayers": r.generated_layers,
            "attachmentsImported": r.attachments_imported,
            "wikimediaFetched": r.wikimedia_fetched,
            "usedOllama": r.used_ollama,
        })
    });
    let pending_prompts: Vec<String> = s.all_in_one_jobs.iter().map(|j| j.prompt.clone()).collect();
    Ok(json!({
        "jobState": job_state,
        "report": report,
        "pendingPrompts": pending_prompts,
    }))
}

pub(crate) fn commands() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "ai.allInOne",
            label: "Orpheus All-In-One…",
            menu: &[],
            shortcut: None,
            params: r#"{"prompt":str?,"attachments":[str]?,"ollamaUrl":str?,"wait":bool?}"#,
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

    fn make_session_with_demo() -> Session {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        s
    }

    #[test]
    fn planner_recognizes_english_and_bulgarian_timeline_ops() {
        let bg = plan_from_prompt("Добави субтитри на български, заглавие \"Новини\", преход и черно-бял ефект", 0);
        assert!(bg.run_subtitles);
        assert_eq!(bg.subtitle_language.as_deref(), Some("bg"));
        assert!(bg.run_text_overlay);
        assert_eq!(bg.text_overlay_content.as_deref(), Some("Новини"));
        assert_eq!(bg.transition_effect.as_deref(), Some("Cross Dissolve"));
        assert!(bg.apply_bw);

        let en = plan_from_prompt("Cut at playhead, speed up 200%, zoom in, normalize audio and add blur", 0);
        assert!(en.run_split_at_playhead);
        assert_eq!(en.speed_percent, Some(200.0));
        assert_eq!(en.zoom_scale, Some(125.0));
        assert!(en.audio_normalize);
        assert!(en.apply_blur);
    }

    #[test]
    fn executes_chained_timeline_edits_without_corkboard() {
        let mut s = make_session_with_demo();
        let res = s
            .execute(
                "ai.allInOne",
                json!({
                    "prompt": "Добави заглавие \"Епизод 1\", преход, черно-бял ефект и приближи",
                    "wait": true
                }),
            )
            .unwrap();
        let steps = res["stepsApplied"].as_array().unwrap();
        assert!(!steps.is_empty());
        assert_eq!(res["generatedLayers"].as_u64(), Some(0));

        let sid = s.state.active_sequence.unwrap();
        let seq = s.project.sequence(sid).unwrap();
        // Text overlay added on a video track, B&W + zoom applied on V1 clips.
        assert!(seq.video_tracks.iter().any(|tr| tr.items.iter().any(|it| it.name.contains("Епизод 1"))));
        assert!(seq.video_tracks[0].items[0].effects.iter().any(|e| e.effect == "black_white"));
        assert_eq!(seq.video_tracks[0].items[0].effect("motion").and_then(|e| e.param("scale")).and_then(|p| p.value.as_f64()), Some(125.0));
    }

    #[test]
    fn split_and_speed_execute_on_active_sequence() {
        let mut s = make_session_with_demo();
        let sid = s.state.active_sequence.unwrap();
        let before_count = s.project.sequence(sid).unwrap().video_tracks[0].items.len();
        let first_item = &s.project.sequence(sid).unwrap().video_tracks[0].items[0];
        let first_mid = first_item.start + Tick(first_item.duration.0 / 2);
        s.state.playheads.insert(sid, first_mid);
        let res = s.execute("ai.allInOne", json!({"prompt": "Split at playhead and set speed 200%", "wait": true})).unwrap();
        let steps = res["stepsApplied"].as_array().unwrap();
        assert!(steps.iter().any(|v| v.as_str().is_some_and(|st| st.contains("Split"))));
        assert!(steps.iter().any(|v| v.as_str().is_some_and(|st| st.contains("200%"))));
        let after_count = s.project.sequence(sid).unwrap().video_tracks[0].items.len();
        assert!(after_count > before_count);
    }

    #[test]
    fn planner_recognizes_shorts_facecam_bottom_and_subtitles_in_order() {
        let prompt = "направи формата на видеото за шортс като фейскама отиде долу и добави субтитри";
        let plan = plan_from_prompt(prompt, 0);
        assert!(plan.run_auto_reframe, "must plan ai.autoReframe for Bulgarian Shorts prompt");
        assert!(plan.reframe_split_screen, "must enable Game Stream / FaceCam split mode");
        assert!(plan.run_subtitles, "must plan ai.generateSubtitles");

        // English equivalent should plan the same Game Stream + Subtitles pipeline.
        let en = plan_from_prompt("Make a 9:16 shorts video with facecam at the bottom and add subtitles", 0);
        assert!(en.run_auto_reframe);
        assert!(en.reframe_split_screen);
        assert!(en.run_subtitles);

        // Execute on demo sequence and verify `ai.autoReframe` (Game Stream Mode) runs BEFORE text/subtitle overlays.
        let mut s = make_session_with_demo();
        let res = s
            .execute(
                "ai.allInOne",
                json!({
                    "prompt": "направи формата на видеото за шортс като фейскама отиде долу и добави заглавие \"Шортс\"",
                    "wait": true
                }),
            )
            .unwrap();
        let steps: Vec<&str> = res["stepsApplied"].as_array().unwrap().iter().filter_map(Value::as_str).collect();
        let reframe_idx = steps.iter().position(|st| st.contains("Gaming Split-Screen (9:16)"));
        let txt_idx = steps.iter().position(|st| st.contains("Added Text Overlay"));
        assert!(reframe_idx.is_some(), "expected Gaming Split-Screen step in {steps:?}");
        assert!(txt_idx.is_some(), "expected Added Text Overlay step in {steps:?}");
        assert!(reframe_idx.unwrap() < txt_idx.unwrap(), "ai.autoReframe must execute before text/subtitle overlay: {steps:?}");

        // Sequence canvas must have been resized to 9:16 (1080x1920) by autoReframe.
        let sid = s.state.active_sequence.unwrap();
        let seq = s.project.sequence(sid).unwrap();
        assert_eq!((seq.settings.width, seq.settings.height), (1080, 1920));

        // Beat Sync command is completely removed.
        assert!(crate::find_command("ai.addBeatMarkers").is_none());
    }

    #[test]
    fn async_multi_step_pipeline_sequences_silence_subtitles_and_chapters_without_drift() {
        use filmcraft_project::MarkerKind;
        use std::f32::consts::TAU;

        // Verify prompt planner parses both English and Bulgarian multi-step streamer prompts
        // WITHOUT falsely triggering `run_split_at_playhead` on "split into chapters".
        let en_plan = plan_from_prompt("cut silences, add subtitles, split into chapters, and reframe to 9:16", 0);
        assert!(en_plan.run_remove_pauses);
        assert!(en_plan.run_subtitles);
        assert!(en_plan.run_vod_chapters);
        assert!(en_plan.run_auto_reframe);
        assert!(!en_plan.run_split_at_playhead, "'split into chapters' must not trigger a playhead razor cut");

        let bg_plan = plan_from_prompt("изрежи паузите, добави субтитри, раздели на глави и направи вертикално 9:16", 0);
        assert!(bg_plan.run_remove_pauses);
        assert!(bg_plan.run_subtitles);
        assert!(bg_plan.run_vod_chapters);
        assert!(bg_plan.run_auto_reframe);
        assert!(!bg_plan.run_split_at_playhead, "'раздели на глави' must not trigger a playhead razor cut");

        // Build a 3-second sequence with a 1-second silent pause in the middle [1.0s .. 2.0s].
        let sr = 16_000u32;
        let mut samples = Vec::with_capacity(sr as usize * 3);
        for i in 0..sr {
            samples.push(((i as f32 / sr as f32) * 440.0 * TAU).sin() * 0.5);
        }
        samples.extend(std::iter::repeat_n(0.0_f32, sr as usize));
        for i in 0..sr {
            samples.push(((i as f32 / sr as f32) * 440.0 * TAU).sin() * 0.5);
        }
        let wav = filmcraft_media::wav::write_wav16(&samples, 1, sr);

        let mut s = Session::default();
        let audio_item = crate::commands::import_bytes(&mut s, "vod_pause.wav", wav.into(), None).unwrap();
        s.state.project_selection = vec![audio_item];
        s.execute("file.newSequenceFromClip", json!({"items": [audio_item.0]})).unwrap();
        let bars = s.execute("file.newBarsAndTone", json!({"duration": 3.0})).unwrap()["item"].as_u64().unwrap();
        let seq_id = s.state.active_sequence.unwrap();
        let a_clip = s.active_sequence().unwrap().audio_tracks[0].items[0].id;
        s.edit_sequence("Link Video", |q, ctx, _| {
            let v_id = ClipId(ctx.alloc());
            let mut v_item = q.audio_tracks[0].items[0].clone();
            v_item.id = v_id;
            v_item.item = ItemId(bars);
            v_item.link = Some(a_clip.0);
            q.video_tracks[0].items.push(v_item);
            q.audio_tracks[0].items[0].link = Some(a_clip.0);
            Ok(())
        })
        .unwrap();

        let dur_before = s.project.sequence(seq_id).unwrap().duration();
        assert!((dur_before.seconds() - 3.0).abs() < 0.05);

        // Dispatch `ai.allInOne` in async UI mode (`wait: false`) and drive the state machine via `poll(&mut s)`.
        let out = s
            .execute(
                "ai.allInOne",
                json!({
                    "prompt": "cut silences, split into chapters, and reframe to 9:16",
                    "wait": false
                }),
            )
            .unwrap();
        let master_job_id = out["job"].as_u64().unwrap();

        let deadline = web_time::Instant::now() + Duration::from_secs(10);
        while s.all_in_one_jobs.iter().any(|j| j.job == master_job_id) {
            assert!(web_time::Instant::now() < deadline, "async multi-step pipeline timed out");
            // When `silence_jobs` is active, downstream `vod_chapters_jobs` and `auto_reframe_jobs` must NOT have started yet!
            if !s.silence_jobs.is_empty() {
                assert!(s.auto_reframe_jobs.is_empty(), "autoReframe must wait for silence removal to finish");
                assert!(s.vod_chapters_jobs.is_empty(), "vodChapters must wait for silence removal to finish");
            }
            // When `auto_reframe_jobs` is active, `vod_chapters_jobs` must NOT have started yet!
            if !s.auto_reframe_jobs.is_empty() {
                assert!(s.vod_chapters_jobs.is_empty(), "vodChapters must wait for autoReframe to finish");
            }
            poll(&mut s);
            std::thread::sleep(Duration::from_millis(2));
        }

        let rep = s.all_in_one_report.as_ref().expect("AllInOneReport must be populated");
        let steps = &rep.steps_applied;
        let silence_pos = steps.iter().position(|st| st.contains("Removed silences")).expect("Removed silences step");
        let reframe_pos = steps.iter().position(|st| st.contains("Auto-Reframe (9:16)")).expect("Auto-Reframe step");
        let chap_pos = steps.iter().position(|st| st.contains("Stream VOD Chapterizer")).expect("VOD Chapterizer step");

        // Verify strict execution order: RemovePauses -> AutoReframe -> VodChapters
        assert!(silence_pos < reframe_pos, "RemovePauses must finish before AutoReframe: {steps:?}");
        assert!(reframe_pos < chap_pos, "AutoReframe must finish before VodChapters: {steps:?}");

        // Verify timeline was rippled (~2.16s after removing ~0.84s pause), reframed to 9:16 (1080x1920),
        // and chapter markers end at the post-ripple duration (NOT the old 3.0s duration!).
        let seq_after = s.project.sequence(seq_id).unwrap();
        let dur_after = seq_after.duration();
        assert!(dur_after < dur_before, "sequence must be shortened by silence removal");
        assert_eq!((seq_after.settings.width, seq_after.settings.height), (1080, 1920));
        let chapter_markers: Vec<_> = seq_after.markers.iter().filter(|m| m.kind == MarkerKind::Chapter).collect();
        assert!(!chapter_markers.is_empty(), "VOD chapterizer must attach chapter markers");
        for m in chapter_markers {
            assert!(
                (m.start + m.duration) <= dur_after + Tick::from_seconds_f64(0.05),
                "chapter marker end ({:.2}s) must match post-ripple duration ({:.2}s) with zero drift",
                (m.start + m.duration).seconds(),
                dur_after.seconds()
            );
        }
    }
}
