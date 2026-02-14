# AGENTS.md
Guidance for coding agents working in this repository.

## Scope
- Applies to the whole repo (`/Users/ripopov/work/rust/open-codex`).
- Project type: Rust binary crate with `eframe` + `wgpu` + WGSL compute shader targeting native desktop and `wasm32` web.
- Primary sources: `src/main.rs` and `src/pathtracer.wgsl`.

## Cursor / Copilot Rule Sources
- `.cursor/rules/`: not present.
- `.cursorrules`: not present.
- `.github/copilot-instructions.md`: not present.
- There are no repository-local Cursor/Copilot instruction files today.

## Repository Layout
- `src/main.rs`: app bootstrap, scene building, camera controls, GPU orchestration.
- `src/pathtracer.wgsl`: path tracer + fast raytracer shader logic.
- `web/index.html`: browser host page and wasm bootstrapping logic.
- `web/pkg/`: generated wasm-bindgen artifacts (ignored by git).
- `snapshot.png`: runtime output image (ignored by git).
- `target/`: build artifacts (ignored by git).

## Toolchain and Runtime Expectations
- Rust edition: 2021.
- Build tool: Cargo.
- Rendering backend: WGPU.
- Browser target: `wasm32-unknown-unknown`.
- Browser packaging: `wasm-bindgen` CLI output to `web/pkg/`.
- Runtime smoke tests require a working graphics environment.
- Browser smoke tests require a WebGPU-capable browser (Chrome with hardware acceleration enabled).

## Build / Run / Lint / Test Commands

### Build and Run
- Type-check quickly: `cargo check`
- Debug build: `cargo build`
- Debug run: `cargo run`
- Release build: `cargo build --release`
- Release run: `cargo run --release`

### Browser (WASM)
- Install target (one-time): `rustup target add wasm32-unknown-unknown`
- Type-check wasm target: `cargo check --target wasm32-unknown-unknown`
- Build wasm release: `cargo build --target wasm32-unknown-unknown --release`
- Generate web bindings:
  - `wasm-bindgen --target web --out-dir web/pkg target/wasm32-unknown-unknown/release/cornell_egui_pathtracer.wasm`
- Serve web root locally: `python3 -m http.server 8080 --directory web`
- Open in Chrome: `open -a "Google Chrome" "http://127.0.0.1:8080"`

### Formatting
- Format all Rust code: `cargo fmt`
- Verify formatting only: `cargo fmt -- --check`

### Linting
- Standard lint pass: `cargo clippy --all-targets --all-features`
- Strict mode (may fail on current warnings):
  - `cargo clippy --all-targets --all-features -- -D warnings`
- Current known warnings in strict mode include `too_many_arguments` and `manual_div_ceil`.
- Do not hide warnings with broad `allow` attributes unless explicitly requested.

### Tests
- Run all tests: `cargo test`
- List all tests/benches: `cargo test -- --list`
- Current status: no tests are defined yet (`0 tests`).

### Running a Single Test (important)
- By name substring: `cargo test <test_name_substring>`
- Exact unit test name: `cargo test <test_name> -- --exact`
- Exact test with logs: `cargo test <test_name> -- --exact --nocapture`
- Single integration test file: `cargo test --test <file_stem>`
- Single test in integration file: `cargo test --test <file_stem> <test_name_substring>`

### Quick Validation Matrix
- Code only (no shader logic): `cargo fmt -- --check && cargo check`
- CPU logic + tests: `cargo fmt -- --check && cargo test`
- Renderer or WGSL edits: `cargo check && cargo run`
- Browser-target edits: `cargo check --target wasm32-unknown-unknown`
- Browser runtime edits: `cargo build --target wasm32-unknown-unknown --release && wasm-bindgen --target web --out-dir web/pkg target/wasm32-unknown-unknown/release/cornell_egui_pathtracer.wasm`
- Pre-PR sanity: `cargo fmt -- --check && cargo clippy --all-targets --all-features`

## Recommended Agent Workflow
- Read existing code paths before editing; match local conventions first.
- Keep diffs focused; avoid unrelated refactors.
- Run `cargo fmt` after Rust edits.
- Run `cargo check` after any non-trivial change.
- For web-target changes, run `cargo check --target wasm32-unknown-unknown`.
- For browser startup/runtime fixes, regenerate `web/pkg` and validate via local server in Chrome.
- If shader/render flow changed, run `cargo run` for a visual smoke test.
- When browser changes are not reflected, use a hard refresh or cache-busted URL.
- Do not commit generated `web/pkg` artifacts.
- Do not revert unrelated user changes in the working tree.

## Rust Style Guidelines

### Formatting and Structure
- Trust `rustfmt`; do not preserve custom spacing against formatter output.
- Prefer short, composable helper functions over monolithic functions.
- Use early returns for guard clauses and error exits.
- Keep comments sparse; only explain non-obvious intent or invariants.

### Imports
- Keep imports explicit (no glob imports).
- Follow the existing file style: external crates first, `std` imports after.
- Remove unused imports and dead helper code when touched.

### Naming Conventions
- Types/enums/traits: `UpperCamelCase`.
- Functions/methods/variables: `snake_case`.
- Constants: `SCREAMING_SNAKE_CASE`.
- Prefer semantic names tied to rendering behavior (`sample_index`, `render_mode`).

### Types and Numeric Discipline
- Be explicit at boundaries (`u32`, `usize`, `f32`) and conversions.
- Prefer `f32` for renderer math unless precision needs force otherwise.
- Use `saturating_*`, `wrapping_*`, or checked math when overflow behavior matters.
- Clamp and epsilon-check float math in camera/intersection logic.
- Be careful with time arithmetic on wasm; prefer checked operations when subtracting durations.

### Native vs Web Runtime
- Use `std::time::Instant` on native targets and `web_time::Instant` on `wasm32`.
- Avoid assumptions about filesystem/background threads on wasm paths.
- Keep snapshot writing disabled on browser target unless explicit web storage support is added.

### GPU Boundary Rules
- Keep Rust/WGSL constants in sync (primitive/material/render-mode IDs).
- Preserve binary layout for GPU structs (`#[repr(C)]`, `Pod`, `Zeroable`).
- Update both sides when changing uniforms or storage buffers.
- Keep texture/buffer usage flags minimal but sufficient for operations performed.

### Error Handling
- Current project style favors `Result<T, String>` in setup/runtime boundaries.
- Include actionable context in error strings.
- Avoid `unwrap()`/`expect()` in runtime code paths.
- Best-effort background tasks may swallow errors only when non-fatal.

### Concurrency and State
- Use channels for snapshot handoff and atomics for stop signaling.
- Keep render mode transitions explicit and deterministic.
- Reset/clear accumulation whenever camera or mode invalidates prior samples.
- Prefer cheap per-frame state checks over expensive rebuilds.

### Performance Expectations
- Avoid per-frame allocations in hot paths.
- Reuse GPU resources; avoid rebuilding pipelines/buffers per frame.
- Keep input responsiveness high by using the fast rendering path while moving.
- Let path accumulation resume only after idle timeout.

## WGSL Style Guidelines
- Use small utility functions for repeated math (`safe_normalize`, `reflect`, etc.).
- Keep branches readable; avoid deeply nested expressions where possible.
- Maintain two clear modes:
  - Fast mode: immediate, non-accumulating preview.
  - Path mode: stochastic accumulation over frames.
- Keep color mapping centralized in `linear_to_display`.
- Keep fog/lighting math stable and bounded (`clamp`, controlled falloff).

## Testing Guidance for New Changes
- Add unit tests for pure CPU utilities when practical.
- For render features, validate with:
  - `cargo check`
  - `cargo clippy --all-targets --all-features`
  - `cargo run` visual verification
- For browser features, validate with:
  - `cargo check --target wasm32-unknown-unknown`
  - wasm release build + `wasm-bindgen`
  - local web server + Chrome smoke run
- If adding tests, include single-test invocation examples in PR notes.

## Commit and Change Hygiene
- Match current commit style: concise, imperative, behavior-oriented titles.
- Explain visible rendering changes in commit body when behavior shifts.
- Avoid mixing formatting-only and functional changes unless required.

## Agent Completion Checklist
- Formatting done (`cargo fmt`).
- Build check done (`cargo check`).
- For web edits: wasm build check done (`cargo check --target wasm32-unknown-unknown`).
- Lint status reported (pass or known warnings/failures).
- Runtime smoke test done for shader/render-loop edits.
- Browser smoke test done for wasm/web edits.
- Any limitations and follow-up work documented clearly.
