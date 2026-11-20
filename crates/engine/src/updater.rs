//! Non-intrusive GitHub Releases update checker & Windows self-updater.
//!
//! Queries a configurable GitHub Releases API endpoint on a background thread with a strict
//! 4-second timeout, failing silently on offline/network errors without blocking or crashing the UI.
//! On Windows, **Install Update & Restart** downloads the new `.exe`, renames the running
//! `orpheus.exe` to `orpheus.exe.bak`, places the new binary, relaunches, and cleans up `.bak` on
//! the next startup.

use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const DEFAULT_RELEASES_API_URL: &str = "https://api.github.com/repos/hertzwaG/Orpheus/releases/latest";
pub const DEFAULT_UPDATE_URL: &str = DEFAULT_RELEASES_API_URL;
pub const DEFAULT_RELEASES_PAGE_URL: &str = "https://github.com/hertzwaG/Orpheus/releases";
pub const DEFAULT_LATEST_RELEASE_PAGE_URL: &str = "https://github.com/hertzwaG/Orpheus/releases/latest";
pub const ORPHEUS_VERSION: &str = "1.0.2";

const MAX_RELEASE_JSON_BYTES: u64 = 512 * 1024;
const MAX_INSTALLER_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Default)]
pub struct UpdateDownloadProgress {
    pub downloaded: AtomicU64,
    pub total: AtomicU64,
}

impl UpdateDownloadProgress {
    pub fn fraction(&self) -> Option<f32> {
        let total = self.total.load(Ordering::Relaxed);
        if total == 0 {
            return None;
        }
        let dl = self.downloaded.load(Ordering::Relaxed);
        Some((dl as f32 / total as f32).clamp(0.0, 1.0))
    }
}

/// Resolve the update endpoint URL: `ORPHEUS_UPDATE_URL` env override -> `configured` -> default.
pub fn resolve_update_endpoint(configured: &str) -> String {
    if let Ok(env_url) = std::env::var("ORPHEUS_UPDATE_URL")
        && !env_url.trim().is_empty()
    {
        return env_url.trim().to_string();
    }
    let trimmed = configured.trim();
    if trimmed.is_empty() { DEFAULT_RELEASES_API_URL.to_string() } else { trimmed.to_string() }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReleaseInfo {
    pub tag_name: String,
    pub version: String,
    pub html_url: String,
    pub download_url: Option<String>,
    pub notes: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UpdateCheckOutcome {
    UpToDate(String),
    Available(ReleaseInfo),
    OfflineOrError(String),
}

/// Parse a semantic version string (`v1.2.3`, `1.2.3-rc1`, etc.) into `(major, minor, patch)`.
pub fn parse_semver(raw: &str) -> Option<(u64, u64, u64)> {
    let trimmed = raw.trim().trim_start_matches(['v', 'V']);
    let core = trimmed.split(['-', '+']).next()?;
    let mut parts = core.split('.');
    let major = parts.next()?.parse::<u64>().ok()?;
    let minor = parts.next().unwrap_or("0").parse::<u64>().ok()?;
    let patch = parts.next().unwrap_or("0").parse::<u64>().ok()?;
    Some((major, minor, patch))
}

/// Returns `true` when `latest` represents a strictly newer semantic version than `current`.
pub fn is_newer_version(current: &str, latest: &str) -> bool {
    match (parse_semver(current), parse_semver(latest)) {
        (Some(cur), Some(lat)) => lat > cur,
        _ => false,
    }
}

/// Parse a GitHub Releases `/releases/latest` JSON payload into [`ReleaseInfo`].
pub fn parse_github_release_json(json_str: &str) -> Option<ReleaseInfo> {
    let val = serde_json::from_str::<Value>(json_str).ok()?;
    let tag_name = val.get("tag_name").and_then(Value::as_str)?.trim().to_string();
    let (maj, min, pat) = parse_semver(&tag_name)?;
    let version = format!("{maj}.{min}.{pat}");
    let html_url = val
        .get("html_url")
        .and_then(Value::as_str)
        .filter(|s| s.starts_with("https://") || s.starts_with("http://"))
        .unwrap_or(DEFAULT_RELEASES_PAGE_URL)
        .to_string();
    let notes = val.get("body").and_then(Value::as_str).unwrap_or("").trim().chars().take(2000).collect();

    let mut setup_url: Option<String> = None;
    let mut fallback_exe_url: Option<String> = None;
    if let Some(assets) = val.get("assets").and_then(Value::as_array) {
        for asset in assets {
            let name = asset.get("name").and_then(Value::as_str).unwrap_or("").to_ascii_lowercase();
            let url = asset.get("browser_download_url").and_then(Value::as_str).unwrap_or("");
            if !(url.starts_with("https://") || url.starts_with("http://")) {
                continue;
            }
            if name.ends_with("setup.exe") {
                setup_url = Some(url.to_string());
                break;
            }
            if name.ends_with(".exe") && fallback_exe_url.is_none() {
                fallback_exe_url = Some(url.to_string());
            }
        }
    }
    let download_url = setup_url.or(fallback_exe_url);

    Some(ReleaseInfo { tag_name, version, html_url, download_url, notes })
}

fn build_http_agent(connect_secs: u64, recv_secs: u64) -> ureq::Agent {
    let tls = ureq::tls::TlsConfig::builder().provider(ureq::tls::TlsProvider::NativeTls).build();
    ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(connect_secs)))
        .timeout_recv_response(Some(Duration::from_secs(recv_secs)))
        .max_redirects(5)
        .tls_config(tls)
        .user_agent(format!("Orpheus/{ORPHEUS_VERSION}"))
        .build()
        .new_agent()
}

/// Query `endpoint` synchronously with a strict timeout (safe for background threads).
pub fn check_for_updates_sync(endpoint: &str, current_version: &str) -> UpdateCheckOutcome {
    let trimmed = endpoint.trim();
    if !(trimmed.starts_with("https://") || trimmed.starts_with("http://")) {
        return UpdateCheckOutcome::OfflineOrError("Invalid update endpoint URL".into());
    }
    let agent = build_http_agent(6, 8);

    let resp = match agent.get(trimmed).header("Accept", "application/vnd.github+json").call() {
        Ok(r) => r,
        Err(ureq::Error::StatusCode(404)) => {
            return UpdateCheckOutcome::OfflineOrError("No published releases found on GitHub yet (HTTP 404).".into());
        }
        Err(ureq::Error::StatusCode(code)) => {
            return UpdateCheckOutcome::OfflineOrError(format!("Update server returned HTTP {code}."));
        }
        Err(e) => return UpdateCheckOutcome::OfflineOrError(format!("Update check failed: {e}")),
    };
    let mut body = String::new();
    if let Err(e) = resp.into_body().into_reader().take(MAX_RELEASE_JSON_BYTES).read_to_string(&mut body) {
        return UpdateCheckOutcome::OfflineOrError(format!("Failed reading release info: {e}"));
    }
    let Some(info) = parse_github_release_json(&body) else {
        return UpdateCheckOutcome::OfflineOrError("Invalid release metadata".into());
    };
    if is_newer_version(current_version, &info.version) {
        UpdateCheckOutcome::Available(info)
    } else {
        UpdateCheckOutcome::UpToDate(current_version.to_string())
    }
}

/// Spawn a non-blocking background update check; never panics or stalls the UI.
pub fn spawn_update_check(endpoint: String, current_version: String) -> mpsc::Receiver<UpdateCheckOutcome> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let outcome = check_for_updates_sync(&endpoint, &current_version);
            let _ = tx.send(outcome);
        }));
    });
    rx
}

/// Remove any leftover `<exe>.bak` created by a previous Windows self-update.
pub fn cleanup_stale_backup_exe() {
    if let Ok(exe) = std::env::current_exe() {
        let mut bak = exe.into_os_string();
        bak.push(".bak");
        let bak_path = PathBuf::from(bak);
        if bak_path.exists() {
            let _ = std::fs::remove_file(&bak_path);
        }
    }
}

/// Download the Windows installer (`*Setup.exe`) into `%TEMP%\Orpheus-Update-Setup.exe` with
/// progress feedback, launch it silently (`/SILENT /CLOSEAPPLICATIONS /RESTARTAPPLICATIONS`),
/// and return its path so the caller can exit cleanly.
pub fn install_windows_update_and_relaunch_with_progress(
    download_url: &str,
    progress: Option<&UpdateDownloadProgress>,
) -> std::result::Result<PathBuf, String> {
    if !cfg!(windows) {
        return Err("Windows installer self-update is supported on Windows only; please use Open Download Page.".into());
    }
    let trimmed = download_url.trim();
    if !(trimmed.starts_with("https://") || trimmed.starts_with("http://")) {
        return Err("Invalid download URL".into());
    }
    let installer_path = std::env::temp_dir().join("Orpheus-Update-Setup.exe");

    let agent = build_http_agent(8, 120);

    let resp = agent.get(trimmed).call().map_err(|e| format!("Download failed: {e}"))?;
    if let Some(p) = progress {
        p.downloaded.store(0, Ordering::Relaxed);
        let len = resp.headers().get("content-length").and_then(|v| v.to_str().ok()).and_then(|s| s.trim().parse::<u64>().ok()).unwrap_or(0);
        p.total.store(len, Ordering::Relaxed);
    }

    let mut file = std::fs::File::create(&installer_path).map_err(|e| format!("Failed creating temp installer: {e}"))?;
    let mut reader = resp.into_body().into_reader().take(MAX_INSTALLER_BYTES);
    let mut buf = [0u8; 64 * 1024];
    let mut total_written = 0u64;
    loop {
        let n = reader.read(&mut buf).map_err(|e| format!("Failed downloading update installer: {e}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).map_err(|e| format!("Failed writing update installer: {e}"))?;
        total_written = total_written.saturating_add(n as u64);
        if let Some(p) = progress {
            p.downloaded.store(total_written, Ordering::Relaxed);
        }
    }
    drop(file);
    if total_written < 1024 {
        let _ = std::fs::remove_file(&installer_path);
        return Err("Downloaded update installer appears empty or truncated".into());
    }

    std::process::Command::new(&installer_path)
        .args(["/SILENT", "/CLOSEAPPLICATIONS", "/RESTARTAPPLICATIONS"])
        .spawn()
        .map_err(|e| format!("Installer downloaded, but launch failed: {e}"))?;
    Ok(installer_path)
}

/// Download and silently launch the Windows update installer without progress tracking.
pub fn install_windows_update_and_relaunch(download_url: &str) -> std::result::Result<PathBuf, String> {
    install_windows_update_and_relaunch_with_progress(download_url, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semver_comparison_handles_v_prefixes_and_prereleases() {
        assert_eq!(parse_semver("v1.2.3"), Some((1, 2, 3)));
        assert_eq!(parse_semver("1.0.0-rc1"), Some((1, 0, 0)));
        assert!(is_newer_version("1.0.0", "v1.0.1"));
        assert!(is_newer_version("1.0.0", "1.1.0"));
        assert!(!is_newer_version("1.0.0", "1.0.0"));
        assert!(!is_newer_version("1.2.0", "1.1.9"));
    }

    #[test]
    fn github_release_json_extracts_windows_exe_asset_and_notes() {
        let payload = r#"{
            "tag_name": "v1.2.0",
            "html_url": "https://github.com/hertzwaG/Orpheus/releases/tag/v1.2.0",
            "body": "Faster Vulkan Whisper + TikTok 50/50 split.",
            "assets": [
                { "name": "orpheus-linux.AppImage", "browser_download_url": "https://example.com/orpheus.AppImage" },
                { "name": "orpheus-v1.2.0-win64.exe", "browser_download_url": "https://example.com/orpheus.exe" }
            ]
        }"#;
        let info = parse_github_release_json(payload).unwrap();
        assert_eq!(info.version, "1.2.0");
        assert_eq!(info.download_url.as_deref(), Some("https://example.com/orpheus.exe"));
        assert!(info.notes.contains("TikTok 50/50"));

        let setup_payload = r#"{
            "tag_name": "v1.3.0",
            "html_url": "https://github.com/hertzwaG/Orpheus/releases/tag/v1.3.0",
            "body": "Inno Setup installer release.",
            "assets": [
                { "name": "orpheus-v1.3.0-portable.exe", "browser_download_url": "https://example.com/portable.exe" },
                { "name": "Orpheus-v1.3.0-Setup.exe", "browser_download_url": "https://example.com/Orpheus-v1.3.0-Setup.exe" }
            ]
        }"#;
        let setup_info = parse_github_release_json(setup_payload).unwrap();
        assert_eq!(setup_info.version, "1.3.0");
        assert_eq!(setup_info.download_url.as_deref(), Some("https://example.com/Orpheus-v1.3.0-Setup.exe"));
    }

    #[test]
    fn offline_or_bad_endpoint_returns_clean_error_without_panicking() {
        let out = check_for_updates_sync("not-a-url", "1.0.1");
        assert!(matches!(out, UpdateCheckOutcome::OfflineOrError(_)));
    }

    #[test]
    fn github_release_json_v101_matches_current_version_and_setup_exe() {
        let payload = r#"{
            "tag_name": "v1.0.1",
            "html_url": "https://github.com/hertzwaG/Orpheus/releases/tag/v1.0.1",
            "body": "Orpheus v1.0.1 release.",
            "assets": [
                { "name": "Orpheus-v1.0.1-Setup.exe", "browser_download_url": "https://github.com/hertzwaG/Orpheus/releases/download/v1.0.1/Orpheus-v1.0.1-Setup.exe" }
            ]
        }"#;
        let info = parse_github_release_json(payload).unwrap();
        assert_eq!(parse_semver(&info.tag_name), Some((1, 0, 1)));
        assert_eq!(info.version, "1.0.1");
        assert!(!is_newer_version("1.0.1", &info.version));
        assert_eq!(info.download_url.as_deref(), Some("https://github.com/hertzwaG/Orpheus/releases/download/v1.0.1/Orpheus-v1.0.1-Setup.exe"));
    }
}
