# Orpheus

**Orpheus** (formerly working title **FrameForge**) is a fork/modification of **FilmCraft** (an open-source, pure-Rust non-linear video editor by the ArtCraft team, licensed under `MIT OR Apache-2.0`).

> **Note:** The visible product name is centralized in `filmcraft_ui_egui::APP_NAME` (`crates/ui-egui/src/lib.rs`).

## Running Orpheus

```powershell
# Desktop application (builds and launches target\release\orpheus.exe)
cargo run --release -p filmcraft

# Open a specific project file
cargo run --release -p filmcraft -- path\to\project.fcproj

# Launch with the JSON-lines control server
cargo run --release -p filmcraft -- --control 9876
```

- **Windows executable:** `target\release\orpheus.exe`
- **Quality checks before committing:** `cargo xtask ci` (or `cargo fmt --check`, `cargo clippy`, `cargo xtask layers`, `cargo xtask assets`)

## Compatibility & Internal Identifiers

To preserve Cargo workspace tooling and compatibility with existing `.fcproj` projects, the following apply:
- Workspace crate names remain `filmcraft-*` and package name `-p filmcraft` (`default-run = "orpheus"`).
- Project files (`.fcproj`) and preset/template JSON files accept legacy `filmcraft.*` / `FilmCraft` headers while writing `orpheus.*` / `Orpheus`.
- Default data directory uses `Orpheus` (`%APPDATA%\Orpheus` on Windows, `~/Library/Application Support/Orpheus` on macOS, `$XDG_DATA_HOME/orpheus` / `~/.local/share/orpheus` on Linux).
- Environment variables accept `ORPHEUS_*` first (`ORPHEUS_CONTROL_PORT`, `ORPHEUS_CONTROL_TOKEN`, `ORPHEUS_DATA_DIR`, etc.) with `FILMCRAFT_*` fallbacks.

## Credits & Licensing

Orpheus is built on [FilmCraft](https://github.com/storytold/filmcraft) and retains all upstream contributor credits (`contributors/`), open-source licenses (`LICENSE-MIT`, `LICENSE-APACHE`, `NOTICE`), and asset attributions (`ATTRIBUTION.md`). Upstream ArtCraft trademarked wordmarks in `docs/brand/` are not used in the Orpheus UI per `AGENTS.md` §1.8 and `docs/brand/LICENSE-brand.txt`.
