//! Primary GPU adapter query (dedicated VRAM and device name) for compute recommendations.
//!
//! Uses DXGI on Windows (`CreateDXGIFactory1` → `EnumAdapters1` → `GetDesc1`) so 64-bit
//! `DedicatedVideoMemory` is read directly without spawning a subprocess or hitting WMI's 4 GB
//! 32-bit `AdapterRAM` cap. Gracefully returns `None` on non-Windows targets.

/// Detected graphics adapter metadata returned by [`primary_gpu`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DetectedGpu {
    pub name: String,
    pub dedicated_vram_mb: u64,
    pub is_discrete: bool,
}

#[cfg(target_os = "windows")]
#[allow(unsafe_code)]
mod imp {
    use super::DetectedGpu;
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE, IDXGIFactory1};

    const MICROSOFT_VENDOR_ID: u32 = 0x1414;

    fn looks_integrated(name: &str, vram_mb: u64) -> bool {
        let lower = name.to_ascii_lowercase();
        let explicit_discrete = [" rtx", " gtx", " rx ", "arc ", "quadro", "radeon pro", "firepro", "titan"].iter().any(|k| lower.contains(k));
        if explicit_discrete {
            return false;
        }
        vram_mb < 1024
            || lower.contains("uhd graphics")
            || lower.contains("hd graphics")
            || lower.contains("iris")
            || lower == "amd radeon(tm) graphics"
            || lower == "amd radeon graphics"
    }

    pub(super) fn primary_gpu() -> Option<DetectedGpu> {
        let mut best: Option<DetectedGpu> = None;
        // SAFETY: `CreateDXGIFactory1` returns a newly created COM interface pointer and `EnumAdapters1` /
        // `GetDesc1` populate plain POD structs (`DXGI_ADAPTER_DESC1`) for each valid adapter index.
        unsafe {
            let Ok(factory): Result<IDXGIFactory1, _> = CreateDXGIFactory1() else {
                return None;
            };
            for i in 0..32u32 {
                let Ok(adapter) = factory.EnumAdapters1(i) else { break };
                let Ok(desc) = adapter.GetDesc1() else { continue };
                let is_software = (desc.Flags & (DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32)) != 0 || desc.VendorId == MICROSOFT_VENDOR_ID;
                if is_software {
                    continue;
                }
                let end = desc.Description.iter().position(|&c| c == 0).unwrap_or(desc.Description.len());
                let name = String::from_utf16_lossy(&desc.Description[..end]).trim().to_string();
                if name.is_empty() {
                    continue;
                }
                let vram_mb = (desc.DedicatedVideoMemory as u64) / (1024 * 1024);
                let is_discrete = !looks_integrated(&name, vram_mb);
                let cand = DetectedGpu { name, dedicated_vram_mb: vram_mb, is_discrete };
                let replace = best.as_ref().is_none_or(|cur| (cand.is_discrete, cand.dedicated_vram_mb) > (cur.is_discrete, cur.dedicated_vram_mb));
                if replace {
                    best = Some(cand);
                }
            }
        }
        best
    }

    pub(super) fn system_ram_mb() -> u64 {
        let mut max_shared_bytes = 0u64;
        // SAFETY: `CreateDXGIFactory1` and `EnumAdapters1`/`GetDesc1` read plain adapter descriptors;
        // under WDDM/DXGI, `SharedSystemMemory` is set by the OS kernel to 50% of physical system RAM.
        unsafe {
            if let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory1>() {
                for i in 0..32u32 {
                    let Ok(adapter) = factory.EnumAdapters1(i) else { break };
                    let Ok(desc) = adapter.GetDesc1() else { continue };
                    max_shared_bytes = max_shared_bytes.max(desc.SharedSystemMemory as u64);
                }
            }
        }
        let est_mb = max_shared_bytes.saturating_mul(2) / (1024 * 1024);
        if est_mb >= 1024 { est_mb } else { 8192 }
    }
}

/// Detect the primary hardware GPU adapter on the system, preferring discrete GPUs with larger
/// dedicated VRAM. Returns `None` on non-Windows platforms or when no hardware adapter is found.
pub fn primary_gpu() -> Option<DetectedGpu> {
    #[cfg(target_os = "windows")]
    {
        imp::primary_gpu()
    }
    #[cfg(not(target_os = "windows"))]
    {
        None
    }
}

/// Return estimated physical system RAM in MiB (at least 1024 MiB).
pub fn system_ram_mb() -> u64 {
    #[cfg(target_os = "windows")]
    {
        imp::system_ram_mb()
    }
    #[cfg(not(target_os = "windows"))]
    {
        8192
    }
}

/// Return the host's logical CPU thread count (at least 1).
pub fn logical_cpu_cores() -> usize {
    std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_core_count_is_positive_and_gpu_probe_is_safe() {
        assert!(logical_cpu_cores() >= 1);
        assert!(system_ram_mb() >= 1024);
        if let Some(gpu) = primary_gpu() {
            assert!(!gpu.name.is_empty());
        }
    }
}
