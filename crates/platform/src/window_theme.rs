//! Native window title-bar theme synchronisation.
//!
//! Keeps the Windows caption bar, border and title text matched to Orpheus's
//! light/dark theme in both active and inactive window states so the title bar
//! never flips to light-on-light or white when losing focus.

/// Synchronise a Win32 window's title-bar appearance with Orpheus's theme.
///
/// `hwnd` is the raw `HWND` value (`isize` from `Win32WindowHandle::hwnd`).
/// Safe no-op on non-Windows platforms or when `hwnd == 0`.
#[cfg(target_os = "windows")]
#[allow(unsafe_code)]
pub fn sync_titlebar(hwnd: isize, dark: bool, focused: bool) {
    use windows::Win32::Foundation::{COLORREF, HWND};
    use windows::Win32::Graphics::Dwm::{
        DWMWA_BORDER_COLOR, DWMWA_CAPTION_COLOR, DWMWA_COLOR_DEFAULT, DWMWA_TEXT_COLOR, DWMWA_USE_IMMERSIVE_DARK_MODE, DwmSetWindowAttribute,
    };

    if hwnd == 0 {
        return;
    }
    let hwnd = HWND(hwnd as *mut core::ffi::c_void);
    let use_dark: i32 = i32::from(dark);
    // 0x00BBGGRR matching `Tokens::for_kind(ThemeKind::Dark)` (`#181A20`, `#292D37`, `#DCE0E8` / `#9AA1B0`).
    let (caption, border, text) = if dark {
        let cap = if focused { COLORREF(0x0020_1A18) } else { COLORREF(0x0022_1C19) };
        let brd = COLORREF(0x0037_2D29);
        let txt = if focused { COLORREF(0x00E8_E0DC) } else { COLORREF(0x00B0_A19A) };
        (cap, brd, txt)
    } else {
        (COLORREF(DWMWA_COLOR_DEFAULT), COLORREF(DWMWA_COLOR_DEFAULT), COLORREF(DWMWA_COLOR_DEFAULT))
    };

    // SAFETY: `hwnd` is a valid top-level window handle for the duration of the call, and each
    // attribute pointer points to a stack-allocated value of the exact size passed to DwmSetWindowAttribute.
    unsafe {
        let _ = DwmSetWindowAttribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, (&raw const use_dark).cast(), size_of::<i32>() as u32);
        let _ = DwmSetWindowAttribute(hwnd, DWMWA_CAPTION_COLOR, (&raw const caption).cast(), size_of::<COLORREF>() as u32);
        let _ = DwmSetWindowAttribute(hwnd, DWMWA_BORDER_COLOR, (&raw const border).cast(), size_of::<COLORREF>() as u32);
        let _ = DwmSetWindowAttribute(hwnd, DWMWA_TEXT_COLOR, (&raw const text).cast(), size_of::<COLORREF>() as u32);
    }
}

#[cfg(not(target_os = "windows"))]
pub fn sync_titlebar(_hwnd: isize, _dark: bool, _focused: bool) {}
