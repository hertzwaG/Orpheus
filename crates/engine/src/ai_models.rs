//! Unified Local ONNX Model Manager & Pure-Rust ONNX Protobuf Tensor Runtime.
//!
//! Manages on-demand local `.onnx` downloads into `<data_dir>/models/` (reusing
//! [`crate::subtitles::download_url_to_path`] with TLS, redirects, and atomic `.part` -> `.onnx`
//! rename) and provides a zero-dependency, `unsafe`-free pure-Rust ONNX protobuf parser +
//! lightweight tensor inference engine for:
//! - **Background Removal** (`u2netp.onnx` / `rmbg-1.4.onnx`)
//! - **Speech Denoising** (`rnnoise.onnx` / `deepfilternet3.onnx`)
//! - **Face Detection** (`ultraface-rfb-320.onnx` / `blazeface.onnx`)

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use filmcraft_export::{Progress, Report};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, NamedKeys, always, bad, str_p};
use crate::subtitles::{GpuAdapterInfo, RECOMMENDED_GPU_MIN_VRAM_MB, detect_primary_gpu, host_cpu_cores, next_job_id};
use crate::{Job, Result, Session};

/// User-selectable execution mode for ONNX Runtime (`ort`) and Whisper AI tools.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AiExecutionMode {
    #[default]
    Auto,
    GpuDirectMl,
    Cpu,
}

impl AiExecutionMode {
    pub const ALL: [Self; 3] = [Self::Auto, Self::GpuDirectMl, Self::Cpu];

    pub fn from_str_param(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().replace(['-', '_', ' ', '(', ')'], "").as_str() {
            "auto" | "autorecommended" | "recommended" | "default" => Some(Self::Auto),
            "gpu" | "directml" | "gpudirectml" | "dml" | "vulkan" => Some(Self::GpuDirectMl),
            "cpu" => Some(Self::Cpu),
            _ => None,
        }
    }

    pub fn as_id(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::GpuDirectMl => "gpu",
            Self::Cpu => "cpu",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "Auto (Recommended)",
            Self::GpuDirectMl => "GPU (DirectML)",
            Self::Cpu => "CPU",
        }
    }

    fn to_u8(self) -> u8 {
        match self {
            Self::Auto => 0,
            Self::GpuDirectMl => 1,
            Self::Cpu => 2,
        }
    }

    fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::GpuDirectMl,
            2 => Self::Cpu,
            _ => Self::Auto,
        }
    }
}

/// Concrete ONNX Runtime execution provider resolved from [`AiExecutionMode`] and detected hardware.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ResolvedExecutionProvider {
    DirectML,
    Cpu,
}

impl ResolvedExecutionProvider {
    pub fn as_id(self) -> &'static str {
        match self {
            Self::DirectML => "DirectMLExecutionProvider",
            Self::Cpu => "CPUExecutionProvider",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::DirectML => "GPU (DirectML)",
            Self::Cpu => "CPU",
        }
    }
}

/// Machine compatibility indicator shown as a Green / Yellow / Red readiness pill.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum MachineReadinessLevel {
    /// Optimal GPU acceleration ready (discrete GPU with sufficient VRAM and system RAM).
    Green,
    /// Compatible via CPU or tight VRAM (integrated GPU, 2–4 GB VRAM, or multi-core CPU).
    Yellow,
    /// Insufficient RAM/VRAM or missing runtime for the requested configuration.
    Red,
}

impl MachineReadinessLevel {
    pub fn as_id(self) -> &'static str {
        match self {
            Self::Green => "green",
            Self::Yellow => "yellow",
            Self::Red => "red",
        }
    }
}

/// Complete hardware readiness & recommendation summary for Settings ▸ AI Models.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AiSpecsAssessment {
    pub mode: AiExecutionMode,
    pub recommended_mode: AiExecutionMode,
    pub resolved_provider: ResolvedExecutionProvider,
    pub readiness: MachineReadinessLevel,
    pub readiness_label: String,
    pub gpu_name: Option<String>,
    pub dedicated_vram_mb: u64,
    pub is_discrete_gpu: bool,
    pub system_ram_mb: u64,
    pub cpu_cores: usize,
}

static ACTIVE_EXECUTION_MODE: AtomicU8 = AtomicU8::new(0);
static EXEC_MODE_LOADED: OnceLock<()> = OnceLock::new();
static SYSTEM_RAM_PROBE: OnceLock<fn() -> u64> = OnceLock::new();

fn execution_mode_file() -> Option<PathBuf> {
    crate::autosave::default_data_dir().map(|d| d.join("ai_execution_mode.txt"))
}

/// Read the active user-selected [`AiExecutionMode`] (`Auto`, `GpuDirectMl`, or `Cpu`).
pub fn active_execution_mode() -> AiExecutionMode {
    EXEC_MODE_LOADED.get_or_init(|| {
        if let Some(path) = execution_mode_file()
            && let Ok(raw) = std::fs::read_to_string(path)
            && let Some(parsed) = AiExecutionMode::from_str_param(&raw)
        {
            ACTIVE_EXECUTION_MODE.store(parsed.to_u8(), Ordering::Relaxed);
        }
    });
    AiExecutionMode::from_u8(ACTIVE_EXECUTION_MODE.load(Ordering::Relaxed))
}

/// Update and persist the active [`AiExecutionMode`].
pub fn set_active_execution_mode(mode: AiExecutionMode) {
    let _ = EXEC_MODE_LOADED.set(());
    ACTIVE_EXECUTION_MODE.store(mode.to_u8(), Ordering::Relaxed);
    if let Some(path) = execution_mode_file() {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(path, mode.as_id());
    }
}

/// Register the platform-layer system RAM query callback (`filmcraft-platform` from the app entry point).
pub fn register_system_ram_probe(probe: fn() -> u64) {
    let _ = SYSTEM_RAM_PROBE.set(probe);
}

/// Return detected physical system RAM in MiB (defaults to 8192 MiB when no platform probe is registered).
pub fn detect_system_ram_mb() -> u64 {
    SYSTEM_RAM_PROBE.get().map_or(8192, |probe| probe().max(1024))
}

/// Resolve whether `mode` should execute on `DirectMLExecutionProvider` or `CPUExecutionProvider`.
pub fn resolve_execution_provider(mode: AiExecutionMode, gpu: Option<&GpuAdapterInfo>) -> ResolvedExecutionProvider {
    match mode {
        AiExecutionMode::Cpu => ResolvedExecutionProvider::Cpu,
        AiExecutionMode::GpuDirectMl => ResolvedExecutionProvider::DirectML,
        AiExecutionMode::Auto => {
            if gpu.is_some_and(|g| g.is_discrete && g.dedicated_vram_mb >= 2048) {
                ResolvedExecutionProvider::DirectML
            } else {
                ResolvedExecutionProvider::Cpu
            }
        }
    }
}

/// Evaluate overall machine readiness (`Green` / `Yellow` / `Red`), optimal execution mode, and
/// resolved ONNX Runtime execution provider from GPU VRAM, system RAM, and CPU thread count.
pub fn assess_machine_specs(mode: AiExecutionMode, gpu: Option<&GpuAdapterInfo>, system_ram_mb: u64, cpu_cores: usize) -> AiSpecsAssessment {
    let vram_mb = gpu.map_or(0, |g| g.dedicated_vram_mb);
    let is_discrete = gpu.is_some_and(|g| g.is_discrete);
    let gpu_name = gpu.map(|g| crate::subtitles::compact_gpu_name(&g.name)).filter(|s| !s.is_empty());

    let recommended_mode = if is_discrete && vram_mb >= RECOMMENDED_GPU_MIN_VRAM_MB { AiExecutionMode::GpuDirectMl } else { AiExecutionMode::Cpu };
    let resolved_provider = resolve_execution_provider(mode, gpu);

    let (readiness, readiness_label) = if system_ram_mb < 4096 {
        (MachineReadinessLevel::Red, format!("Insufficient RAM ({:.1} GB < 4 GB minimum)", system_ram_mb as f64 / 1024.0))
    } else if mode == AiExecutionMode::GpuDirectMl && (gpu.is_none() || (!is_discrete && vram_mb < 1024)) {
        (MachineReadinessLevel::Red, "DirectML GPU requested but no capable GPU VRAM detected".to_string())
    } else if resolved_provider == ResolvedExecutionProvider::DirectML && is_discrete && vram_mb >= RECOMMENDED_GPU_MIN_VRAM_MB && system_ram_mb >= 8192 {
        (MachineReadinessLevel::Green, "Optimal GPU Acceleration Ready (DirectML)".to_string())
    } else if system_ram_mb >= 4096 && (cpu_cores >= 4 || vram_mb >= 2048) {
        (MachineReadinessLevel::Yellow, "Compatible (CPU / Moderate VRAM)".to_string())
    } else {
        (MachineReadinessLevel::Red, "Limited Hardware (Low RAM / CPU Cores)".to_string())
    };

    AiSpecsAssessment {
        mode,
        recommended_mode,
        resolved_provider,
        readiness,
        readiness_label,
        gpu_name,
        dedicated_vram_mb: vram_mb,
        is_discrete_gpu: is_discrete,
        system_ram_mb,
        cpu_cores,
    }
}

/// Evaluate `(MachineReadinessLevel, recommended_for_specs)` for an ONNX model kind on the host specs.
pub fn model_readiness_for_specs(kind: AiModelKind, gpu: Option<&GpuAdapterInfo>, system_ram_mb: u64) -> (MachineReadinessLevel, bool) {
    let vram_mb = gpu.map_or(0, |g| g.dedicated_vram_mb);
    let is_discrete = gpu.is_some_and(|g| g.is_discrete);
    if system_ram_mb < 3072 {
        return (MachineReadinessLevel::Red, false);
    }
    match kind {
        AiModelKind::BgMatting => {
            if is_discrete && vram_mb >= RECOMMENDED_GPU_MIN_VRAM_MB && system_ram_mb >= 8192 {
                (MachineReadinessLevel::Green, true)
            } else if system_ram_mb >= 4096 {
                (MachineReadinessLevel::Yellow, true)
            } else {
                (MachineReadinessLevel::Red, false)
            }
        }
        AiModelKind::SpeechDenoise | AiModelKind::FaceDetect => {
            if (is_discrete && vram_mb >= 2048 && system_ram_mb >= 6144) || system_ram_mb >= 8192 {
                (MachineReadinessLevel::Green, true)
            } else if system_ram_mb >= 4096 {
                (MachineReadinessLevel::Yellow, true)
            } else {
                (MachineReadinessLevel::Red, false)
            }
        }
    }
}

/// Evaluate `(MachineReadinessLevel, recommended_for_specs)` for a Whisper speech model tier (`large-v3-turbo` vs `base`).
pub fn whisper_tier_readiness_for_specs(model_id: &str, gpu: Option<&GpuAdapterInfo>, system_ram_mb: u64) -> (MachineReadinessLevel, bool) {
    let vram_mb = gpu.map_or(0, |g| g.dedicated_vram_mb);
    let is_discrete = gpu.is_some_and(|g| g.is_discrete);
    let high_spec = is_discrete && vram_mb >= 6144 && system_ram_mb >= 12288;

    if model_id.eq_ignore_ascii_case("large-v3-turbo") {
        if high_spec {
            (MachineReadinessLevel::Green, true)
        } else if (is_discrete && vram_mb >= 4096 && system_ram_mb >= 8192) || system_ram_mb >= 12288 {
            (MachineReadinessLevel::Yellow, false)
        } else {
            (MachineReadinessLevel::Red, false)
        }
    } else if system_ram_mb < 3072 {
        (MachineReadinessLevel::Red, false)
    } else if (is_discrete && vram_mb >= 2048) || system_ram_mb >= 8192 {
        (MachineReadinessLevel::Green, !high_spec)
    } else {
        (MachineReadinessLevel::Yellow, !high_spec)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AiModelKind {
    BgMatting,
    SpeechDenoise,
    FaceDetect,
}

impl AiModelKind {
    pub const ALL: [Self; 3] = [Self::BgMatting, Self::SpeechDenoise, Self::FaceDetect];

    pub fn from_str_param(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "bg_matting" | "bg" | "matting" | "u2netp" | "rmbg" => Some(Self::BgMatting),
            "speech_denoise" | "denoise" | "audio" | "rnnoise" | "deepfilternet" => Some(Self::SpeechDenoise),
            "face_detect" | "face" | "ultraface" | "blazeface" => Some(Self::FaceDetect),
            _ => None,
        }
    }

    pub fn as_id(self) -> &'static str {
        match self {
            Self::BgMatting => "bg_matting",
            Self::SpeechDenoise => "speech_denoise",
            Self::FaceDetect => "face_detect",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::BgMatting => "Background Removal (u2netp / RMBG-1.4)",
            Self::SpeechDenoise => "Speech Denoising (RNNoise / DeepFilterNet)",
            Self::FaceDetect => "Face Detection (UltraFace RFB-320)",
        }
    }

    pub fn primary_file_name(self) -> &'static str {
        match self {
            Self::BgMatting => "u2netp.onnx",
            Self::SpeechDenoise => "rnnoise.onnx",
            Self::FaceDetect => "ultraface-rfb-320.onnx",
        }
    }

    pub fn candidate_file_names(self) -> &'static [&'static str] {
        match self {
            Self::BgMatting => &["rmbg-1.4.onnx", "u2netp.onnx", "modnet.onnx", "rmbg.onnx", "u2net.onnx"],
            Self::SpeechDenoise => &["deepfilternet3.onnx", "rnnoise.onnx", "denoiser.onnx"],
            Self::FaceDetect => &["ultraface-rfb-320.onnx", "version-RFB-320.onnx", "yolov8n-face.onnx", "blazeface.onnx"],
        }
    }

    fn matches_file_stem(self, lower_name: &str) -> bool {
        if !lower_name.ends_with(".onnx") {
            return false;
        }
        match self {
            Self::BgMatting => lower_name.contains("rmbg") || lower_name.contains("modnet") || lower_name.contains("u2net"),
            Self::SpeechDenoise => lower_name.contains("deepfilter") || lower_name.contains("rnnoise") || lower_name.contains("denoise"),
            Self::FaceDetect => {
                lower_name.contains("ultraface")
                    || lower_name.contains("rfb-320")
                    || lower_name.contains("yolov8")
                    || lower_name.contains("blazeface")
                    || lower_name.contains("face")
            }
        }
    }

    pub fn approx_bytes(self) -> u64 {
        match self {
            Self::BgMatting => 4_600_000,
            Self::SpeechDenoise => 1_800_000,
            Self::FaceDetect => 1_250_000,
        }
    }

    pub fn download_mirrors(self) -> &'static [&'static str] {
        match self {
            Self::BgMatting => {
                &["https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx", "https://huggingface.co/TryOn/u2netp/resolve/main/u2netp.onnx"]
            }
            Self::SpeechDenoise => &[
                "https://huggingface.co/Xenova/rnnoise-onnx/resolve/main/rnnoise.onnx",
                "https://github.com/breizhn/DTLN/raw/master/pretrained_model/model_1.onnx",
            ],
            Self::FaceDetect => &[
                "https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/raw/master/models/onnx/version-RFB-320.onnx",
                "https://huggingface.co/spaces/onnx/Ultra-Light-Fast-Generic-Face-Detector/resolve/main/version-RFB-320.onnx",
            ],
        }
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

fn model_search_dirs(override_dir: Option<&Path>) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(d) = override_dir {
        dirs.push(d.to_path_buf());
    }
    if let Some(d) = crate::transcript::models_dir() {
        dirs.push(d);
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(parent) = exe.parent()
    {
        dirs.push(parent.join("models"));
    }
    dirs
}

/// Locate a cached local `.onnx` file for `kind` in `<data_dir>/models/` or `<exe_dir>/models/`.
pub fn detect_model(kind: AiModelKind, override_dir: Option<&Path>) -> Option<PathBuf> {
    for dir in model_search_dirs(override_dir) {
        for name in kind.candidate_file_names() {
            let p = dir.join(name);
            if std::fs::metadata(&p).is_ok_and(|m| m.is_file() && m.len() > 1024) {
                return Some(p);
            }
        }
        if let Ok(entries) = std::fs::read_dir(&dir) {
            let mut matched: Vec<PathBuf> = entries
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    p.file_name().and_then(|n| n.to_str()).is_some_and(|n| kind.matches_file_stem(&n.to_ascii_lowercase()))
                        && std::fs::metadata(p).is_ok_and(|m| m.is_file() && m.len() > 1024)
                })
                .collect();
            matched.sort();
            if let Some(first) = matched.into_iter().next() {
                return Some(first);
            }
        }
    }
    None
}

pub fn all_models_installed() -> bool {
    AiModelKind::ALL.iter().all(|&k| detect_model(k, None).is_some())
}

// ---------------------------------------------------------------------------------------------
// Pure-Rust ONNX Protobuf (`onnx.ModelProto`) Parser & Lightweight Tensor Runtime
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct OnnxTensor {
    pub dims: Vec<usize>,
    pub data: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct OnnxNode {
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub strides: Vec<usize>,
    pub pads: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct OnnxGraph {
    pub name: String,
    pub ir_version: i64,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub initializers: HashMap<String, OnnxTensor>,
    pub nodes: Vec<OnnxNode>,
}

struct ProtoReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> ProtoReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    fn read_varint(&mut self) -> Option<u64> {
        let mut out = 0u64;
        let mut shift = 0u32;
        for _ in 0..10 {
            let b = *self.buf.get(self.pos)?;
            self.pos += 1;
            out |= u64::from(b & 0x7F) << shift;
            if b & 0x80 == 0 {
                return Some(out);
            }
            shift += 7;
        }
        None
    }

    fn read_bytes(&mut self) -> Option<&'a [u8]> {
        let len = usize::try_from(self.read_varint()?).ok()?;
        let end = self.pos.checked_add(len)?;
        let slice = self.buf.get(self.pos..end)?;
        self.pos = end;
        Some(slice)
    }

    fn skip_field(&mut self, wire_type: u64) -> Option<()> {
        match wire_type {
            0 => {
                self.read_varint()?;
                Some(())
            }
            1 => {
                self.pos = self.pos.checked_add(8)?;
                if self.pos <= self.buf.len() { Some(()) } else { None }
            }
            2 => {
                self.read_bytes()?;
                Some(())
            }
            5 => {
                self.pos = self.pos.checked_add(4)?;
                if self.pos <= self.buf.len() { Some(()) } else { None }
            }
            _ => None,
        }
    }
}

fn parse_tensor_proto(bytes: &[u8]) -> Option<(String, OnnxTensor)> {
    let mut r = ProtoReader::new(bytes);
    let mut name = String::new();
    let mut dims = Vec::new();
    let mut data_type = 0u64;
    let mut float_data = Vec::new();
    let mut raw_data: &[u8] = &[];

    while r.remaining() > 0 {
        let tag = r.read_varint()?;
        let field = tag >> 3;
        let wire = tag & 7;
        match (field, wire) {
            (1, 0) => {
                if let Some(d) = r.read_varint().and_then(|v| usize::try_from(v).ok()) {
                    dims.push(d);
                }
            }
            (1, 2) => {
                let packed = r.read_bytes()?;
                let mut pr = ProtoReader::new(packed);
                while pr.remaining() > 0 {
                    if let Some(d) = pr.read_varint().and_then(|v| usize::try_from(v).ok()) {
                        dims.push(d);
                    }
                }
            }
            (2, 0) => data_type = r.read_varint()?,
            (4, 5) => {
                let end = r.pos.checked_add(4)?;
                let b = r.buf.get(r.pos..end)?;
                r.pos = end;
                float_data.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
            }
            (4, 2) => {
                let packed = r.read_bytes()?;
                for &ch in packed.as_chunks::<4>().0 {
                    float_data.push(f32::from_le_bytes(ch));
                }
            }
            (8, 2) => name = String::from_utf8_lossy(r.read_bytes()?).into_owned(),
            (9, 2) => raw_data = r.read_bytes()?,
            _ => r.skip_field(wire)?,
        }
    }

    if float_data.is_empty() && !raw_data.is_empty() && (data_type == 1 || data_type == 0) {
        // Limit decoded tensor weight allocation to 128 MB worth of f32s.
        let count = (raw_data.len() / 4).min(32 * 1024 * 1024);
        float_data.reserve(count);
        for &ch in raw_data.as_chunks::<4>().0.iter().take(count) {
            float_data.push(f32::from_le_bytes(ch));
        }
    }
    if name.is_empty() {
        return None;
    }
    Some((name, OnnxTensor { dims, data: float_data }))
}

fn parse_node_proto(bytes: &[u8]) -> Option<OnnxNode> {
    let mut r = ProtoReader::new(bytes);
    let mut op_type = String::new();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut strides = Vec::new();
    let mut pads = Vec::new();

    while r.remaining() > 0 {
        let tag = r.read_varint()?;
        let field = tag >> 3;
        let wire = tag & 7;
        match (field, wire) {
            (1, 2) => inputs.push(String::from_utf8_lossy(r.read_bytes()?).into_owned()),
            (2, 2) => outputs.push(String::from_utf8_lossy(r.read_bytes()?).into_owned()),
            (4, 2) => op_type = String::from_utf8_lossy(r.read_bytes()?).into_owned(),
            (5, 2) => {
                // AttributeProto: parse name + ints (field 8) for strides/pads.
                let attr_bytes = r.read_bytes()?;
                let mut ar = ProtoReader::new(attr_bytes);
                let mut attr_name = String::new();
                let mut ints = Vec::new();
                while ar.remaining() > 0 {
                    let atag = ar.read_varint()?;
                    match (atag >> 3, atag & 7) {
                        (1, 2) => attr_name = String::from_utf8_lossy(ar.read_bytes()?).into_owned(),
                        (8, 0) => {
                            if let Some(v) = ar.read_varint().and_then(|x| usize::try_from(x).ok()) {
                                ints.push(v);
                            }
                        }
                        (8, 2) => {
                            let packed = ar.read_bytes()?;
                            let mut pr = ProtoReader::new(packed);
                            while pr.remaining() > 0 {
                                if let Some(v) = pr.read_varint().and_then(|x| usize::try_from(x).ok()) {
                                    ints.push(v);
                                }
                            }
                        }
                        (_, w) => ar.skip_field(w)?,
                    }
                }
                if attr_name == "strides" {
                    strides = ints;
                } else if attr_name == "pads" {
                    pads = ints;
                }
            }
            _ => r.skip_field(wire)?,
        }
    }
    Some(OnnxNode { op_type, inputs, outputs, strides, pads })
}

fn parse_value_info_name(bytes: &[u8]) -> Option<String> {
    let mut r = ProtoReader::new(bytes);
    while r.remaining() > 0 {
        let tag = r.read_varint()?;
        match (tag >> 3, tag & 7) {
            (1, 2) => return Some(String::from_utf8_lossy(r.read_bytes()?).into_owned()),
            (_, w) => r.skip_field(w)?,
        }
    }
    None
}

fn parse_graph_proto(bytes: &[u8], ir_version: i64) -> Option<OnnxGraph> {
    let mut r = ProtoReader::new(bytes);
    let mut name = String::new();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut initializers = HashMap::new();
    let mut nodes = Vec::new();

    while r.remaining() > 0 {
        let tag = r.read_varint()?;
        let field = tag >> 3;
        let wire = tag & 7;
        match (field, wire) {
            (1, 2) => {
                if let Some(n) = parse_node_proto(r.read_bytes()?) {
                    nodes.push(n);
                }
            }
            (2, 2) => name = String::from_utf8_lossy(r.read_bytes()?).into_owned(),
            (5, 2) => {
                if let Some((k, t)) = parse_tensor_proto(r.read_bytes()?) {
                    initializers.insert(k, t);
                }
            }
            (11, 2) => {
                if let Some(n) = parse_value_info_name(r.read_bytes()?) {
                    inputs.push(n);
                }
            }
            (12, 2) => {
                if let Some(n) = parse_value_info_name(r.read_bytes()?) {
                    outputs.push(n);
                }
            }
            _ => r.skip_field(wire)?,
        }
    }
    Some(OnnxGraph { name, ir_version, inputs, outputs, initializers, nodes })
}

/// Parse an ONNX `.onnx` binary (`ModelProto`) into [`OnnxGraph`], returning `None` if invalid.
pub fn parse_onnx_bytes(bytes: &[u8]) -> Option<OnnxGraph> {
    let mut r = ProtoReader::new(bytes);
    let mut ir_version = 0i64;
    let mut graph_bytes: Option<&[u8]> = None;

    while r.remaining() > 0 {
        let tag = r.read_varint()?;
        match (tag >> 3, tag & 7) {
            (1, 0) => ir_version = i64::try_from(r.read_varint()?).unwrap_or(0),
            (7, 2) => graph_bytes = Some(r.read_bytes()?),
            (_, w) => r.skip_field(w)?,
        }
    }
    parse_graph_proto(graph_bytes?, ir_version)
}

fn cached_graph(path: &Path) -> Option<Arc<OnnxGraph>> {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, Arc<OnnxGraph>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(g) = lock(cache).get(path).cloned() {
        return Some(g);
    }
    let meta = std::fs::metadata(path).ok()?;
    if !meta.is_file() || meta.len() > 256 * 1024 * 1024 {
        return None;
    }
    let bytes = std::fs::read(path).ok()?;
    let parsed = Arc::new(parse_onnx_bytes(&bytes)?);
    lock(cache).insert(path.to_path_buf(), parsed.clone());
    Some(parsed)
}

fn run_conv2d(inp: &OnnxTensor, weight: &OnnxTensor, bias: Option<&OnnxTensor>, stride: usize, pad: usize) -> Option<OnnxTensor> {
    if inp.dims.len() != 4 || weight.dims.len() != 4 {
        return None;
    }
    let (c_in, h_in, w_in) = (inp.dims[1], inp.dims[2], inp.dims[3]);
    let (c_out, w_c_in, kh, kw) = (weight.dims[0], weight.dims[1], weight.dims[2], weight.dims[3]);
    if w_c_in != c_in || kh == 0 || kw == 0 || stride == 0 {
        return None;
    }
    // a hostile model can carry any pad/stride: no unchecked arithmetic on them
    let padded_h = pad.checked_mul(2).and_then(|p| h_in.checked_add(p))?;
    let padded_w = pad.checked_mul(2).and_then(|p| w_in.checked_add(p))?;
    let h_out = (padded_h.checked_sub(kh)? / stride).checked_add(1)?;
    let w_out = (padded_w.checked_sub(kw)? / stride).checked_add(1)?;
    let total = c_out.checked_mul(h_out)?.checked_mul(w_out)?;
    if total > 16 * 1024 * 1024 {
        return None;
    }
    let mut out = vec![0.0_f32; total];
    for oc in 0..c_out {
        let b_val = bias.and_then(|b| b.data.get(oc).copied()).unwrap_or(0.0);
        for oy in 0..h_out {
            for ox in 0..w_out {
                let mut acc = b_val;
                for ic in 0..c_in {
                    for ky in 0..kh {
                        let iy = (oy * stride + ky) as isize - pad as isize;
                        if iy < 0 || iy >= h_in as isize {
                            continue;
                        }
                        for kx in 0..kw {
                            let ix = (ox * stride + kx) as isize - pad as isize;
                            if ix < 0 || ix >= w_in as isize {
                                continue;
                            }
                            let iv = inp.data.get((ic * h_in + iy as usize) * w_in + ix as usize).copied().unwrap_or(0.0);
                            let wv = weight.data.get(((oc * c_in + ic) * kh + ky) * kw + kx).copied().unwrap_or(0.0);
                            acc += iv * wv;
                        }
                    }
                }
                out[(oc * h_out + oy) * w_out + ox] = acc;
            }
        }
    }
    Some(OnnxTensor { dims: vec![1, c_out, h_out, w_out], data: out })
}

fn eval_simple_onnx_feedforward(graph: &OnnxGraph, input: OnnxTensor, max_nodes: usize) -> Option<OnnxTensor> {
    let mut env: HashMap<String, OnnxTensor> = graph.initializers.clone();
    let input_name = graph.inputs.first()?.clone();
    env.insert(input_name, input);

    for node in graph.nodes.iter().take(max_nodes) {
        let out_name = node.outputs.first()?.clone();
        let next = match node.op_type.as_str() {
            "Conv" => {
                let x = env.get(node.inputs.first()?)?;
                let w = env.get(node.inputs.get(1)?)?;
                let b = node.inputs.get(2).and_then(|n| env.get(n));
                let stride = node.strides.first().copied().unwrap_or(1).max(1);
                let pad = node.pads.first().copied().unwrap_or(0);
                run_conv2d(x, w, b, stride, pad)?
            }
            "Relu" => {
                let x = env.get(node.inputs.first()?)?;
                OnnxTensor { dims: x.dims.clone(), data: x.data.iter().map(|v| v.max(0.0)).collect() }
            }
            "Sigmoid" => {
                let x = env.get(node.inputs.first()?)?;
                OnnxTensor { dims: x.dims.clone(), data: x.data.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect() }
            }
            _ => return None,
        };
        env.insert(out_name, next);
    }
    let final_name = graph.outputs.first()?;
    env.remove(final_name)
}

/// Input RGB normalization scheme for ONNX vision models.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RgbNormMode {
    /// ImageNet mean/std (`u2netp.onnx`).
    ImageNet,
    /// `rgb / 255.0 - 0.5` (`[-0.5, 0.5]`, `rmbg-1.4.onnx`).
    ZeroCenterHalf,
    /// `(rgb / 255.0 - 0.5) / 0.5` (`[-1.0, 1.0]`, `modnet.onnx`, `ultraface`).
    MinusOneToOne,
    /// `rgb / 255.0` (`[0.0, 1.0]`, `yolov8`).
    UnitZeroToOne,
}

/// Bilinearly resize an RGBA8 frame (`src_w × src_h`) into a normalized planar RGB NCHW tensor
/// `[1, 3, dst_h, dst_w]` using `norm_mode`.
pub fn preprocess_rgba_to_nchw_mode(rgba: &[u8], src_w: usize, src_h: usize, dst_w: usize, dst_h: usize, norm_mode: RgbNormMode) -> Option<Vec<f32>> {
    let needed = src_w.checked_mul(src_h)?.checked_mul(4)?;
    let plane_len = dst_w.checked_mul(dst_h)?;
    let total_len = plane_len.checked_mul(3)?;
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 || rgba.len() < needed || total_len > 32 * 1024 * 1024 {
        return None;
    }
    let mut nchw = vec![0.0_f32; total_len];
    let scale_x = src_w as f32 / dst_w as f32;
    let scale_y = src_h as f32 / dst_h as f32;

    for dy in 0..dst_h {
        let fy = ((dy as f32 + 0.5) * scale_y - 0.5).clamp(0.0, (src_h - 1) as f32);
        let y0 = (fy.floor() as usize).min(src_h - 1);
        let y1 = (y0 + 1).min(src_h - 1);
        let wy = fy - y0 as f32;

        for dx in 0..dst_w {
            let fx = ((dx as f32 + 0.5) * scale_x - 0.5).clamp(0.0, (src_w - 1) as f32);
            let x0 = (fx.floor() as usize).min(src_w - 1);
            let x1 = (x0 + 1).min(src_w - 1);
            let wx = fx - x0 as f32;

            let i00 = (y0 * src_w + x0) * 4;
            let i10 = (y0 * src_w + x1) * 4;
            let i01 = (y1 * src_w + x0) * 4;
            let i11 = (y1 * src_w + x1) * 4;
            let dst_idx = dy * dst_w + dx;

            for c in 0..3 {
                let p00 = f32::from(rgba[i00 + c]);
                let p10 = f32::from(rgba[i10 + c]);
                let p01 = f32::from(rgba[i01 + c]);
                let p11 = f32::from(rgba[i11 + c]);
                let top = p00 + (p10 - p00) * wx;
                let bot = p01 + (p11 - p01) * wx;
                let val = top + (bot - top) * wy;
                let norm = match norm_mode {
                    RgbNormMode::ImageNet => {
                        const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
                        const STD: [f32; 3] = [0.229, 0.224, 0.225];
                        (val / 255.0 - MEAN[c]) / STD[c]
                    }
                    RgbNormMode::ZeroCenterHalf => (val / 255.0) - 0.5,
                    RgbNormMode::MinusOneToOne => (val / 127.5) - 1.0,
                    RgbNormMode::UnitZeroToOne => val / 255.0,
                };
                nchw[c * plane_len + dst_idx] = norm;
            }
        }
    }
    Some(nchw)
}

/// Bilinearly resize an RGBA8 frame (`src_w × src_h`) into a normalized planar RGB NCHW tensor
/// `[1, 3, dst_h, dst_w]`. When `imagenet_norm` is true, applies ImageNet mean/std; otherwise
/// scales to `[-1.0, 1.0]`.
pub fn preprocess_rgba_to_nchw(rgba: &[u8], src_w: usize, src_h: usize, dst_w: usize, dst_h: usize, imagenet_norm: bool) -> Option<Vec<f32>> {
    let mode = if imagenet_norm { RgbNormMode::ImageNet } else { RgbNormMode::MinusOneToOne };
    preprocess_rgba_to_nchw_mode(rgba, src_w, src_h, dst_w, dst_h, mode)
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Normalize a raw `[1, 1, matte_h, matte_w]` model output tensor into `[0.0, 1.0]` probabilities
/// (applying sigmoid if raw logits fall outside `[0, 1]`, plus a crisp `smoothstep(0.35, 0.65, p)`
/// contrast curve to reject background bleed) and bilinearly resize back to `dst_w × dst_h`.
pub fn postprocess_matte_bilinear(raw_matte: &[f32], matte_w: usize, matte_h: usize, dst_w: usize, dst_h: usize) -> Option<Vec<f32>> {
    let src_len = matte_w.checked_mul(matte_h)?;
    let dst_len = dst_w.checked_mul(dst_h)?;
    if matte_w == 0 || matte_h == 0 || dst_w == 0 || dst_h == 0 || raw_matte.len() < src_len || dst_len > 32 * 1024 * 1024 {
        return None;
    }
    let slice = &raw_matte[..src_len];
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    for &v in slice {
        if v.is_finite() {
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
    }
    let is_logits = min_v < -0.05 || max_v > 1.05;
    let mut probs = Vec::with_capacity(src_len);
    for &v in slice {
        let p = if !v.is_finite() {
            0.0
        } else if is_logits {
            1.0 / (1.0 + (-v.clamp(-20.0, 20.0)).exp())
        } else {
            v.clamp(0.0, 1.0)
        };
        probs.push(smoothstep(0.35, 0.65, p));
    }

    let mut out = vec![0.0_f32; dst_len];
    let scale_x = matte_w as f32 / dst_w as f32;
    let scale_y = matte_h as f32 / dst_h as f32;

    for dy in 0..dst_h {
        let fy = ((dy as f32 + 0.5) * scale_y - 0.5).clamp(0.0, (matte_h - 1) as f32);
        let y0 = (fy.floor() as usize).min(matte_h - 1);
        let y1 = (y0 + 1).min(matte_h - 1);
        let wy = fy - y0 as f32;

        for dx in 0..dst_w {
            let fx = ((dx as f32 + 0.5) * scale_x - 0.5).clamp(0.0, (matte_w - 1) as f32);
            let x0 = (fx.floor() as usize).min(matte_w - 1);
            let x1 = (x0 + 1).min(matte_w - 1);
            let wx = fx - x0 as f32;

            let p00 = probs[y0 * matte_w + x0];
            let p10 = probs[y0 * matte_w + x1];
            let p01 = probs[y1 * matte_w + x0];
            let p11 = probs[y1 * matte_w + x1];
            let top = p00 + (p10 - p00) * wx;
            let bot = p01 + (p11 - p01) * wx;
            out[dy * dst_w + dx] = (top + (bot - top) * wy).clamp(0.0, 1.0);
        }
    }
    Some(out)
}

#[cfg(not(target_arch = "wasm32"))]
struct CachedOrtMattingSession {
    path: PathBuf,
    modified: Option<std::time::SystemTime>,
    provider: ResolvedExecutionProvider,
    session: ort::session::Session,
    input_h: usize,
    input_w: usize,
    norm_mode: RgbNormMode,
}

#[cfg(not(target_arch = "wasm32"))]
fn ort_environment() -> Option<&'static ort::environment::Environment> {
    static ORT_ENV: OnceLock<Option<ort::environment::Environment>> = OnceLock::new();
    ORT_ENV.get_or_init(|| ort::init().with_name("orpheus").with_telemetry(false).build().ok()).as_ref()
}

#[cfg(not(target_arch = "wasm32"))]
fn resolve_model_input_dims(session: &ort::session::Session, model_path: &Path) -> (usize, usize, RgbNormMode) {
    let fname = model_path.file_name().and_then(|s| s.to_str()).unwrap_or("").to_ascii_lowercase();
    let norm_mode = if fname.contains("rmbg") {
        RgbNormMode::ZeroCenterHalf
    } else if fname.contains("modnet") {
        RgbNormMode::MinusOneToOne
    } else {
        RgbNormMode::ImageNet
    };
    let default_side = if fname.contains("rmbg") || fname.contains("modnet") { 512 } else { 320 };
    if let Some(inp) = session.inputs().first()
        && let ort::value::ValueType::Tensor { shape, .. } = inp.dtype()
        && shape.len() >= 4
    {
        let h = usize::try_from(shape[2]).ok().filter(|&d| (8..=2048).contains(&d)).unwrap_or(default_side);
        let w = usize::try_from(shape[3]).ok().filter(|&d| (8..=2048).contains(&d)).unwrap_or(default_side);
        return (h, w, norm_mode);
    }
    (default_side, default_side, norm_mode)
}

#[cfg(not(target_arch = "wasm32"))]
fn build_ort_matting_session(model_path: &Path, provider: ResolvedExecutionProvider) -> Option<CachedOrtMattingSession> {
    let meta = std::fs::metadata(model_path).ok()?;
    if !meta.is_file() || meta.len() < 64 {
        return None;
    }
    let modified = meta.modified().ok();
    let env = ort_environment()?;
    let builder = ort::session::Session::builder(env).ok()?;
    let mut builder = match provider {
        ResolvedExecutionProvider::DirectML => builder
            .with_memory_pattern(false)
            .unwrap_or_else(|e| e.recover())
            .with_parallel_execution(false)
            .unwrap_or_else(|e| e.recover())
            .with_execution_providers([ort::ep::DirectML::default().build(), ort::ep::CPU::default().build()])
            .unwrap_or_else(|e| e.recover()),
        ResolvedExecutionProvider::Cpu => builder.with_execution_providers([ort::ep::CPU::default().build()]).unwrap_or_else(|e| e.recover()),
    };
    let session = builder.commit_from_file(model_path).ok()?;
    let (input_h, input_w, norm_mode) = resolve_model_input_dims(&session, model_path);
    Some(CachedOrtMattingSession { path: model_path.to_path_buf(), modified, provider, session, input_h, input_w, norm_mode })
}

#[cfg(not(target_arch = "wasm32"))]
fn infer_bg_matte_ort(model_path: &Path, rgba: &[u8], width: usize, height: usize) -> Option<Vec<f32>> {
    static ORT_BG_SESSION: OnceLock<Mutex<Option<CachedOrtMattingSession>>> = OnceLock::new();
    let slot = ORT_BG_SESSION.get_or_init(|| Mutex::new(None));
    let mode = active_execution_mode();
    let gpu = detect_primary_gpu();
    let provider = resolve_execution_provider(mode, gpu.as_ref());
    let modified = std::fs::metadata(model_path).ok().and_then(|m| m.modified().ok());

    let mut guard = lock(slot);
    let needs_reload = guard.as_ref().is_none_or(|c| c.path != model_path || c.modified != modified || c.provider != provider);
    if needs_reload {
        *guard = build_ort_matting_session(model_path, provider);
    }
    let cached = guard.as_mut()?;
    let (th, tw, norm_mode) = (cached.input_h, cached.input_w, cached.norm_mode);
    let nchw = preprocess_rgba_to_nchw_mode(rgba, width, height, tw, th, norm_mode)?;
    let input_tensor = ort::value::Tensor::<f32>::from_array(([1usize, 3, th, tw], nchw.into_boxed_slice())).ok()?;
    let outputs = cached.session.run(ort::inputs![input_tensor]).ok()?;
    if outputs.len() == 0 {
        return None;
    }
    let (out_shape, out_data) = outputs[0].try_extract_tensor::<f32>().ok()?;
    let (oh, ow) = match out_shape.len() {
        4 => (usize::try_from(out_shape[2]).ok()?, usize::try_from(out_shape[3]).ok()?),
        3 => (usize::try_from(out_shape[1]).ok()?, usize::try_from(out_shape[2]).ok()?),
        2 => (usize::try_from(out_shape[0]).ok()?, usize::try_from(out_shape[1]).ok()?),
        _ => (th, tw),
    };
    postprocess_matte_bilinear(out_data, ow, oh, width, height)
}

/// Run local ONNX background matting when a cached `.onnx` model is present and valid.
/// Uses the cached `ort::session::Session` (with DirectML / CPU execution provider selected by
/// [`active_execution_mode`]), falling back to the lightweight pure-Rust evaluator for synthetic
/// micro-graphs.
pub fn infer_bg_matte_onnx(model_path: &Path, rgba: &[u8], width: usize, height: usize) -> Option<Vec<f32>> {
    let needed = width.checked_mul(height).and_then(|p| p.checked_mul(4))?;
    if width < 4 || height < 4 || rgba.len() < needed {
        return None;
    }
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(matte) = infer_bg_matte_ort(model_path, rgba, width, height) {
        return Some(matte);
    }
    let graph = cached_graph(model_path)?;
    if graph.nodes.is_empty() {
        return None;
    }
    let (tw, th) = (64usize, 64usize);
    let nchw = preprocess_rgba_to_nchw(rgba, width, height, tw, th, true)?;
    let out = eval_simple_onnx_feedforward(&graph, OnnxTensor { dims: vec![1, 3, th, tw], data: nchw }, 128)?;
    let (oh, ow) = (*out.dims.get(2)?, *out.dims.get(3)?);
    postprocess_matte_bilinear(&out.data, ow, oh, width, height)
}

/// Run local ONNX neural speech denoising when a cached `.onnx` model is present and valid.
pub fn infer_speech_denoise_onnx(model_path: &Path, channels: &mut [Vec<f32>]) -> bool {
    let Some(graph) = cached_graph(model_path) else {
        return false;
    };
    if graph.nodes.is_empty() || channels.is_empty() {
        return false;
    }
    // Check if the loaded ONNX graph is a feed-forward Conv/Relu/Sigmoid denoiser we can evaluate directly.
    let first_len = channels[0].len();
    if first_len == 0 {
        return false;
    }
    let probe = OnnxTensor { dims: vec![1, 1, 1, first_len.min(480)], data: channels[0][..first_len.min(480)].to_vec() };
    if let Some(out) = eval_simple_onnx_feedforward(&graph, probe, 64)
        && out.data.len() == first_len.min(480)
    {
        for ch in channels.iter_mut() {
            for chunk in ch.chunks_mut(480) {
                let len = chunk.len();
                if let Some(res) = eval_simple_onnx_feedforward(&graph, OnnxTensor { dims: vec![1, 1, 1, len], data: chunk.to_vec() }, 64)
                    && res.data.len() == len
                {
                    chunk.copy_from_slice(&res.data);
                }
            }
        }
        return true;
    }
    false
}

/// Normalized face bounding box `[0.0..=1.0]` returned by ONNX face detection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OnnxFaceBox {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
    pub cx: f64,
    pub cy: f64,
    pub conf: f64,
}

#[cfg(not(target_arch = "wasm32"))]
struct CachedOrtFaceSession {
    path: PathBuf,
    modified: Option<std::time::SystemTime>,
    provider: ResolvedExecutionProvider,
    session: ort::session::Session,
    input_h: usize,
    input_w: usize,
    norm_mode: RgbNormMode,
}

#[cfg(not(target_arch = "wasm32"))]
fn build_ort_face_session(model_path: &Path, provider: ResolvedExecutionProvider) -> Option<CachedOrtFaceSession> {
    let meta = std::fs::metadata(model_path).ok()?;
    if !meta.is_file() || meta.len() < 64 {
        return None;
    }
    let modified = meta.modified().ok();
    let env = ort_environment()?;
    let builder = ort::session::Session::builder(env).ok()?;
    let mut builder = match provider {
        ResolvedExecutionProvider::DirectML => builder
            .with_memory_pattern(false)
            .unwrap_or_else(|e| e.recover())
            .with_parallel_execution(false)
            .unwrap_or_else(|e| e.recover())
            .with_execution_providers([ort::ep::DirectML::default().build(), ort::ep::CPU::default().build()])
            .unwrap_or_else(|e| e.recover()),
        ResolvedExecutionProvider::Cpu => builder.with_execution_providers([ort::ep::CPU::default().build()]).unwrap_or_else(|e| e.recover()),
    };
    let session = builder.commit_from_file(model_path).ok()?;
    let fname = model_path.file_name().and_then(|s| s.to_str()).unwrap_or("").to_ascii_lowercase();
    let is_yolo = fname.contains("yolo");
    let (def_h, def_w, norm_mode) = if is_yolo { (640usize, 640usize, RgbNormMode::UnitZeroToOne) } else { (240usize, 320usize, RgbNormMode::MinusOneToOne) };
    let (input_h, input_w) = if let Some(inp) = session.inputs().first()
        && let ort::value::ValueType::Tensor { shape, .. } = inp.dtype()
        && shape.len() >= 4
    {
        let h = usize::try_from(shape[2]).ok().filter(|&d| (32..=1280).contains(&d)).unwrap_or(def_h);
        let w = usize::try_from(shape[3]).ok().filter(|&d| (32..=1280).contains(&d)).unwrap_or(def_w);
        (h, w)
    } else {
        (def_h, def_w)
    };
    Some(CachedOrtFaceSession { path: model_path.to_path_buf(), modified, provider, session, input_h, input_w, norm_mode })
}

fn best_ultraface_box(scores: &[f32], boxes: &[f32], n_anchors: usize) -> Option<OnnxFaceBox> {
    if scores.len() < n_anchors * 2 || boxes.len() < n_anchors * 4 {
        return None;
    }
    let mut best: Option<OnnxFaceBox> = None;
    for i in 0..n_anchors {
        let conf = f64::from(scores[i * 2 + 1]);
        if conf <= 0.45 {
            continue;
        }
        let x0 = f64::from(boxes[i * 4]).clamp(0.0, 1.0);
        let y0 = f64::from(boxes[i * 4 + 1]).clamp(0.0, 1.0);
        let x1 = f64::from(boxes[i * 4 + 2]).clamp(0.0, 1.0);
        let y1 = f64::from(boxes[i * 4 + 3]).clamp(0.0, 1.0);
        if x1 <= x0 + 0.01 || y1 <= y0 + 0.01 {
            continue;
        }
        if best.is_none_or(|b| conf > b.conf) {
            best = Some(OnnxFaceBox { x0, y0, x1, y1, cx: (x0 + x1) * 0.5, cy: (y0 + y1) * 0.5, conf: conf.min(1.0) });
        }
    }
    best
}

fn best_yolo_face_box(data: &[f32], channels: usize, n_anchors: usize, in_w: usize, in_h: usize) -> Option<OnnxFaceBox> {
    if channels < 5 || n_anchors == 0 || data.len() < channels * n_anchors {
        return None;
    }
    let mut best: Option<OnnxFaceBox> = None;
    for i in 0..n_anchors {
        let conf = f64::from(data[4 * n_anchors + i]);
        if conf <= 0.40 || conf > 1.05 {
            continue;
        }
        let cx = f64::from(data[i]) / in_w.max(1) as f64;
        let cy = f64::from(data[n_anchors + i]) / in_h.max(1) as f64;
        let bw = f64::from(data[2 * n_anchors + i]) / in_w.max(1) as f64;
        let bh = f64::from(data[3 * n_anchors + i]) / in_h.max(1) as f64;
        let x0 = (cx - bw * 0.5).clamp(0.0, 1.0);
        let y0 = (cy - bh * 0.5).clamp(0.0, 1.0);
        let x1 = (cx + bw * 0.5).clamp(0.0, 1.0);
        let y1 = (cy + bh * 0.5).clamp(0.0, 1.0);
        if x1 <= x0 + 0.01 || y1 <= y0 + 0.01 {
            continue;
        }
        if best.is_none_or(|b| conf > b.conf) {
            best = Some(OnnxFaceBox { x0, y0, x1, y1, cx: cx.clamp(0.05, 0.95), cy: cy.clamp(0.05, 0.95), conf: conf.min(1.0) });
        }
    }
    best
}

#[cfg(not(target_arch = "wasm32"))]
fn infer_face_box_ort(model_path: &Path, rgba: &[u8], width: usize, height: usize) -> Option<OnnxFaceBox> {
    static ORT_FACE_SESSION: OnceLock<Mutex<Option<CachedOrtFaceSession>>> = OnceLock::new();
    let slot = ORT_FACE_SESSION.get_or_init(|| Mutex::new(None));
    let mode = active_execution_mode();
    let gpu = detect_primary_gpu();
    let provider = resolve_execution_provider(mode, gpu.as_ref());
    let modified = std::fs::metadata(model_path).ok().and_then(|m| m.modified().ok());

    let mut guard = lock(slot);
    let needs_reload = guard.as_ref().is_none_or(|c| c.path != model_path || c.modified != modified || c.provider != provider);
    if needs_reload {
        *guard = build_ort_face_session(model_path, provider);
    }
    let cached = guard.as_mut()?;
    let (th, tw, norm_mode) = (cached.input_h, cached.input_w, cached.norm_mode);
    let nchw = preprocess_rgba_to_nchw_mode(rgba, width, height, tw, th, norm_mode)?;
    let input_tensor = ort::value::Tensor::<f32>::from_array(([1usize, 3, th, tw], nchw.into_boxed_slice())).ok()?;
    let outputs = cached.session.run(ort::inputs![input_tensor]).ok()?;

    if outputs.len() >= 2 {
        let (s0, d0) = outputs[0].try_extract_tensor::<f32>().ok()?;
        let (s1, d1) = outputs[1].try_extract_tensor::<f32>().ok()?;
        let last0 = s0.last().copied().unwrap_or(0);
        let last1 = s1.last().copied().unwrap_or(0);
        if last0 == 2 && last1 == 4 {
            let n = usize::try_from(*s0.get(1)?).ok()?;
            return best_ultraface_box(d0, d1, n);
        } else if last0 == 4 && last1 == 2 {
            let n = usize::try_from(*s1.get(1)?).ok()?;
            return best_ultraface_box(d1, d0, n);
        }
    }
    if outputs.len() >= 1 {
        let (s, d) = outputs[0].try_extract_tensor::<f32>().ok()?;
        if s.len() == 3 {
            let c = usize::try_from(s[1]).ok()?;
            let n = usize::try_from(s[2]).ok()?;
            return best_yolo_face_box(d, c, n, tw, th);
        }
    }
    None
}

/// Run local ONNX face detection when a cached `.onnx` model is present and valid, returning [`OnnxFaceBox`].
pub fn infer_face_box_onnx(model_path: &Path, rgba: &[u8], width: usize, height: usize) -> Option<OnnxFaceBox> {
    let needed = width.checked_mul(height).and_then(|p| p.checked_mul(4))?;
    if width < 8 || height < 8 || rgba.len() < needed {
        return None;
    }
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(b) = infer_face_box_ort(model_path, rgba, width, height) {
        return Some(b);
    }
    let graph = cached_graph(model_path)?;
    if graph.nodes.is_empty() {
        return None;
    }
    let (tw, th) = (64usize, 48usize);
    let nchw = preprocess_rgba_to_nchw_mode(rgba, width, height, tw, th, RgbNormMode::MinusOneToOne)?;
    let out = eval_simple_onnx_feedforward(&graph, OnnxTensor { dims: vec![1, 3, th, tw], data: nchw }, 128)?;
    let (oh, ow) = (*out.dims.get(2)?, *out.dims.get(3)?);
    if oh == 0 || ow == 0 || out.data.len() < oh * ow {
        return None;
    }
    let (mut best_x, mut best_y, mut best_v) = (0.5_f64, 0.4_f64, 0.0_f64);
    for y in 0..oh {
        for x in 0..ow {
            let v = f64::from(out.data[y * ow + x]);
            if v > best_v {
                best_v = v;
                best_x = (x as f64 + 0.5) / ow as f64;
                best_y = (y as f64 + 0.5) / oh as f64;
            }
        }
    }
    if best_v > 0.35 {
        Some(OnnxFaceBox {
            x0: (best_x - 0.12).clamp(0.0, 1.0),
            y0: (best_y - 0.16).clamp(0.0, 1.0),
            x1: (best_x + 0.12).clamp(0.0, 1.0),
            y1: (best_y + 0.16).clamp(0.0, 1.0),
            cx: best_x,
            cy: best_y,
            conf: best_v.min(1.0),
        })
    } else {
        None
    }
}

/// Run local ONNX face detection when a cached `.onnx` model is present and valid, returning `(cx_norm, cy_norm, conf)`.
pub fn infer_face_bbox_onnx(model_path: &Path, rgba: &[u8], width: usize, height: usize) -> Option<(f64, f64, f64)> {
    infer_face_box_onnx(model_path, rgba, width, height).map(|b| (b.cx, b.cy, b.conf))
}

// ---------------------------------------------------------------------------------------------
// Commands: `ai.modelStatus`, `ai.setExecutionMode`, `ai.downloadModel`, `ai.downloadAllAiModels`
// ---------------------------------------------------------------------------------------------

fn model_status_cmd(_s: &mut Session, _p: &Value) -> Result<Value> {
    let mode = active_execution_mode();
    let gpu = detect_primary_gpu();
    let system_ram_mb = detect_system_ram_mb();
    let cpu_cores = host_cpu_cores();
    let specs = assess_machine_specs(mode, gpu.as_ref(), system_ram_mb, cpu_cores);

    let items: Vec<Value> = AiModelKind::ALL
        .iter()
        .map(|&kind| {
            let path = detect_model(kind, None);
            let size_bytes = path.as_ref().and_then(|p| std::fs::metadata(p).ok()).map(|m| m.len()).unwrap_or(0);
            let (readiness, recommended_for_specs) = model_readiness_for_specs(kind, gpu.as_ref(), system_ram_mb);
            json!({
                "id": kind.as_id(),
                "label": kind.label(),
                "installed": path.is_some(),
                "path": path.map(|p| p.to_string_lossy().into_owned()),
                "sizeBytes": size_bytes,
                "approxBytes": kind.approx_bytes(),
                "readiness": readiness.as_id(),
                "recommendedForSpecs": recommended_for_specs,
            })
        })
        .collect();
    Ok(json!({
        "allInstalled": all_models_installed(),
        "executionMode": specs.mode.as_id(),
        "recommendedExecutionMode": specs.recommended_mode.as_id(),
        "resolvedProvider": specs.resolved_provider.as_id(),
        "readiness": specs.readiness.as_id(),
        "readinessLabel": specs.readiness_label,
        "gpuName": specs.gpu_name,
        "dedicatedVramMb": specs.dedicated_vram_mb,
        "systemRamMb": specs.system_ram_mb,
        "cpuCores": specs.cpu_cores,
        "models": items,
    }))
}

fn set_execution_mode_cmd(s: &mut Session, p: &Value) -> Result<Value> {
    let raw = str_p(p, "mode").ok_or_else(|| bad("ai.setExecutionMode", "missing `mode` parameter (`auto`, `gpu`, or `cpu`)"))?;
    let mode = AiExecutionMode::from_str_param(raw).ok_or_else(|| bad("ai.setExecutionMode", format!("unknown execution mode `{raw}`")))?;
    set_active_execution_mode(mode);
    let gpu = detect_primary_gpu();
    let provider = resolve_execution_provider(mode, gpu.as_ref());
    s.toast(format!("AI Execution Mode: {} ({})", mode.label(), provider.as_id()));
    Ok(json!({
        "executionMode": mode.as_id(),
        "resolvedProvider": provider.as_id(),
    }))
}

fn spawn_models_download_job(s: &mut Session, targets: Vec<AiModelKind>, label: &str) -> Result<Value> {
    let target_dir = crate::transcript::models_dir().unwrap_or_else(|| std::env::temp_dir().join("orpheus-models"));
    let job_id = next_job_id(s);
    let progress = Arc::new(Progress::default());
    let approx_total: u64 = targets.iter().map(|k| k.approx_bytes()).sum::<u64>().max(1);
    progress.total.store(approx_total, Ordering::Relaxed);
    *lock(&progress.status) = format!("Downloading {} AI model(s)…", targets.len());
    let result_slot: Arc<Mutex<Option<std::result::Result<Report, String>>>> = Arc::default();
    s.jobs.push(Job { id: job_id, label: label.to_string(), progress: progress.clone(), result: result_slot.clone() });

    let prog = progress.clone();
    std::thread::spawn(move || {
        let t0 = web_time::Instant::now();
        let outcome = (|| -> std::result::Result<u64, String> {
            let mut total_downloaded = 0u64;
            for (idx, kind) in targets.iter().enumerate() {
                if prog.cancel.load(Ordering::Relaxed) {
                    return Err("Cancelled".into());
                }
                let dest = target_dir.join(kind.primary_file_name());
                if std::fs::metadata(&dest).is_ok_and(|m| m.is_file() && m.len() > 1024) {
                    total_downloaded += std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
                    continue;
                }
                *lock(&prog.status) = format!("Downloading {} ({}/{})…", kind.primary_file_name(), idx + 1, targets.len());
                let sub_prog = Progress::default();
                let mut last_err = String::from("Download failed");
                let mut ok = false;
                for &url in kind.download_mirrors() {
                    if prog.cancel.load(Ordering::Relaxed) {
                        return Err("Cancelled".into());
                    }
                    match crate::subtitles::download_url_to_path(url, &dest, kind.approx_bytes(), &sub_prog) {
                        Ok(()) => {
                            let bytes = std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
                            if bytes > 1024 {
                                total_downloaded += bytes;
                                ok = true;
                                break;
                            }
                            let _ = std::fs::remove_file(&dest);
                            last_err = format!("{} was truncated", kind.primary_file_name());
                        }
                        Err(e) => {
                            last_err = e;
                        }
                    }
                }
                if !ok {
                    return Err(last_err);
                }
                prog.done.store(total_downloaded.min(approx_total), Ordering::Relaxed);
            }
            prog.done.store(approx_total, Ordering::Relaxed);
            Ok(total_downloaded)
        })();
        match outcome {
            Ok(bytes) => {
                *lock(&result_slot) = Some(Ok(Report { frames: targets.len() as u64, bytes, seconds: t0.elapsed().as_secs_f64(), ..Default::default() }));
            }
            Err(e) => {
                *lock(&prog.error) = Some(e.clone());
                *lock(&result_slot) = Some(Err(e));
            }
        }
        prog.finished.store(true, Ordering::Relaxed);
    });

    Ok(json!({ "job": job_id }))
}

fn download_model_cmd(s: &mut Session, p: &Value) -> Result<Value> {
    let raw = str_p(p, "model").ok_or_else(|| bad("ai.downloadModel", "missing model parameter"))?;
    let kind = AiModelKind::from_str_param(raw).ok_or_else(|| bad("ai.downloadModel", format!("unknown AI model `{raw}`")))?;
    if let Some(existing) = detect_model(kind, None) {
        s.toast(format!("{} is already installed", kind.primary_file_name()));
        return Ok(json!({ "installed": true, "path": existing.to_string_lossy() }));
    }
    spawn_models_download_job(s, vec![kind], "Download AI Model")
}

fn download_all_models_cmd(s: &mut Session, _p: &Value) -> Result<Value> {
    let missing: Vec<AiModelKind> = AiModelKind::ALL.into_iter().filter(|&k| detect_model(k, None).is_none()).collect();
    if missing.is_empty() {
        s.toast("All AI ONNX models are ready offline");
        return Ok(json!({ "installed": true }));
    }
    spawn_models_download_job(s, missing, "Download All AI Models")
}

pub(crate) fn commands() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "ai.modelStatus",
            label: "AI ONNX Model Status",
            menu: &[],
            shortcut: None,
            params: "{}",
            enabled: always,
            run: model_status_cmd,
            journal: false,
            named: NamedKeys::NONE,
        },
        CommandSpec {
            id: "ai.setExecutionMode",
            label: "Set AI Execution Mode (Auto / DirectML GPU / CPU)",
            menu: &[],
            shortcut: None,
            params: r#"{"mode":"auto|gpu|cpu"}"#,
            enabled: always,
            run: set_execution_mode_cmd,
            journal: false,
            named: NamedKeys::NONE,
        },
        CommandSpec {
            id: "ai.downloadModel",
            label: "Download AI ONNX Model",
            menu: &[],
            shortcut: None,
            params: r#"{"model":"bg_matting|speech_denoise|face_detect"}"#,
            enabled: always,
            run: download_model_cmd,
            journal: false,
            named: NamedKeys::NONE,
        },
        CommandSpec {
            id: "ai.downloadAllAiModels",
            label: "Download All AI ONNX Models",
            menu: &[],
            shortcut: None,
            params: "{}",
            enabled: always,
            run: download_all_models_cmd,
            journal: false,
            named: NamedKeys::NONE,
        },
    ]
}

#[cfg(test)]
mod tests {
    #[test]
    fn hostile_conv_pad_stride_and_kernel_return_none_instead_of_overflowing() {
        let inp = OnnxTensor { dims: vec![1, 1, 4, 4], data: vec![0.5; 16] };
        let w = OnnxTensor { dims: vec![1, 1, 3, 3], data: vec![0.1; 9] };
        assert!(run_conv2d(&inp, &w, None, 1, 1).is_some(), "a sane conv still runs");
        for pad in [usize::MAX, usize::MAX / 2, usize::MAX / 2 + 1, 1 << 40] {
            assert!(run_conv2d(&inp, &w, None, 1, pad).is_none(), "pad {pad}");
        }
        assert!(run_conv2d(&inp, &w, None, 0, 0).is_none(), "zero stride");
        assert!(run_conv2d(&inp, &w, None, usize::MAX, 0).is_some(), "a huge stride just yields a single output");
        let huge_kernel = OnnxTensor { dims: vec![1, 1, usize::MAX, usize::MAX], data: Vec::new() };
        assert!(run_conv2d(&inp, &huge_kernel, None, 1, 0).is_none());
    }

    use super::*;

    fn encode_varint(mut v: u64, out: &mut Vec<u8>) {
        loop {
            let b = (v & 0x7F) as u8;
            v >>= 7;
            if v == 0 {
                out.push(b);
                break;
            } else {
                out.push(b | 0x80);
            }
        }
    }

    fn push_len_field(field: u64, data: &[u8], out: &mut Vec<u8>) {
        encode_varint((field << 3) | 2, out);
        encode_varint(data.len() as u64, out);
        out.extend_from_slice(data);
    }

    #[test]
    fn pure_rust_onnx_parser_decodes_valid_conv_sigmoid_graph_and_runs_inference() {
        // Build a minimal valid ONNX ModelProto: x -> Conv(3->1, 1x1) -> Sigmoid -> y
        let mut weight_proto = Vec::new();
        for &d in &[1u64, 3, 1, 1] {
            encode_varint(1 << 3, &mut weight_proto);
            encode_varint(d, &mut weight_proto);
        }
        push_len_field(8, b"w", &mut weight_proto);
        let mut w_raw = Vec::new();
        for f in [1.0_f32, 1.0, 1.0] {
            w_raw.extend_from_slice(&f.to_le_bytes());
        }
        push_len_field(9, &w_raw, &mut weight_proto);

        let mut conv_node = Vec::new();
        push_len_field(1, b"x", &mut conv_node);
        push_len_field(1, b"w", &mut conv_node);
        push_len_field(2, b"h", &mut conv_node);
        push_len_field(4, b"Conv", &mut conv_node);

        let mut sig_node = Vec::new();
        push_len_field(1, b"h", &mut sig_node);
        push_len_field(2, b"y", &mut sig_node);
        push_len_field(4, b"Sigmoid", &mut sig_node);

        let mut vi_in = Vec::new();
        push_len_field(1, b"x", &mut vi_in);
        let mut vi_out = Vec::new();
        push_len_field(1, b"y", &mut vi_out);

        let mut graph = Vec::new();
        push_len_field(1, &conv_node, &mut graph);
        push_len_field(1, &sig_node, &mut graph);
        push_len_field(2, b"test_onnx", &mut graph);
        push_len_field(5, &weight_proto, &mut graph);
        push_len_field(11, &vi_in, &mut graph);
        push_len_field(12, &vi_out, &mut graph);

        let mut model = Vec::new();
        encode_varint(1 << 3, &mut model);
        encode_varint(8, &mut model);
        push_len_field(7, &graph, &mut model);

        let parsed = parse_onnx_bytes(&model).expect("valid ONNX bytes must parse");
        assert_eq!(parsed.name, "test_onnx");
        assert_eq!(parsed.nodes.len(), 2);

        let inp = OnnxTensor { dims: vec![1, 3, 2, 2], data: vec![0.5; 12] };
        let out = eval_simple_onnx_feedforward(&parsed, inp, 16).expect("feedforward eval must succeed");
        assert_eq!(out.dims, vec![1, 1, 2, 2]);
        assert!(out.data.iter().all(|&v| v > 0.7 && v < 0.9), "sigmoid(1.5) ~= 0.817");
    }

    #[test]
    fn hostile_corrupt_onnx_bytes_return_none_without_panicking() {
        assert!(parse_onnx_bytes(&[]).is_none());
        assert!(parse_onnx_bytes(&[0xFF; 64]).is_none());
    }

    #[test]
    fn preprocess_and_postprocess_matte_dimensions_and_bilinear_resize() {
        let (src_w, src_h) = (64usize, 36usize);
        let mut rgba = vec![0u8; src_w * src_h * 4];
        for y in 0..src_h {
            for x in 0..src_w {
                let i = (y * src_w + x) * 4;
                rgba[i] = 200;
                rgba[i + 1] = 120;
                rgba[i + 2] = 60;
                rgba[i + 3] = 255;
            }
        }
        let nchw = preprocess_rgba_to_nchw(&rgba, src_w, src_h, 32, 32, true).expect("preprocess should succeed");
        assert_eq!(nchw.len(), 3 * 32 * 32, "expected [1, 3, 32, 32] NCHW tensor");
        assert!(nchw.iter().all(|v| v.is_finite()));

        // Raw logits (-4.0 background, +4.0 center foreground) -> sigmoid + bilinear resize to 64x36.
        let mut raw_logits = vec![-4.0_f32; 16 * 16];
        for y in 4..12 {
            for x in 4..12 {
                raw_logits[y * 16 + x] = 4.0;
            }
        }
        let matte = postprocess_matte_bilinear(&raw_logits, 16, 16, src_w, src_h).expect("postprocess should succeed");
        assert_eq!(matte.len(), src_w * src_h);
        let center = matte[(src_h / 2) * src_w + (src_w / 2)];
        let corner = matte[0];
        assert!(center > 0.95, "center foreground logit (+4.0) must sigmoid to > 0.95, got {center}");
        assert!(corner < 0.05, "corner background logit (-4.0) must sigmoid to < 0.05, got {corner}");
    }

    #[test]
    fn machine_specs_green_yellow_red_and_recommended_tiers() {
        let rtx = GpuAdapterInfo { name: "NVIDIA GeForce RTX 3060".into(), dedicated_vram_mb: 12_288, is_discrete: true };
        let igpu = GpuAdapterInfo { name: "Intel UHD Graphics".into(), dedicated_vram_mb: 512, is_discrete: false };

        let green = assess_machine_specs(AiExecutionMode::Auto, Some(&rtx), 16_384, 12);
        assert_eq!(green.readiness, MachineReadinessLevel::Green);
        assert_eq!(green.recommended_mode, AiExecutionMode::GpuDirectMl);
        assert_eq!(green.resolved_provider, ResolvedExecutionProvider::DirectML);

        let yellow = assess_machine_specs(AiExecutionMode::Auto, Some(&igpu), 8_192, 8);
        assert_eq!(yellow.readiness, MachineReadinessLevel::Yellow);
        assert_eq!(yellow.recommended_mode, AiExecutionMode::Cpu);
        assert_eq!(yellow.resolved_provider, ResolvedExecutionProvider::Cpu);

        let red_low_ram = assess_machine_specs(AiExecutionMode::Auto, None, 2_048, 2);
        assert_eq!(red_low_ram.readiness, MachineReadinessLevel::Red);

        let red_forced_gpu = assess_machine_specs(AiExecutionMode::GpuDirectMl, Some(&igpu), 8_192, 8);
        assert_eq!(red_forced_gpu.readiness, MachineReadinessLevel::Red);

        // Whisper model tier recommendations: 12 GB VRAM + 16 GB RAM recommends Large v3 Turbo; iGPU + 8 GB RAM recommends Base.
        let (large_rtx_lvl, large_rtx_rec) = whisper_tier_readiness_for_specs("large-v3-turbo", Some(&rtx), 16_384);
        assert_eq!((large_rtx_lvl, large_rtx_rec), (MachineReadinessLevel::Green, true));
        let (base_igpu_lvl, base_igpu_rec) = whisper_tier_readiness_for_specs("base", Some(&igpu), 8_192);
        assert_eq!((base_igpu_lvl, base_igpu_rec), (MachineReadinessLevel::Green, true));
    }
}
