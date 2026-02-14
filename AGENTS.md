# AGENTS.md
Guidance for coding agents working in this repository.

## Scope
- Applies to the whole repo (`/Users/ripopov/work/rust/open-codex`).
- Project type: Rust binary crate with `eframe` + `wgpu` + WGSL compute shader.
- Primary sources: `src/main.rs` and `src/pathtracer.wgsl`.

## Cursor / Copilot Rule Sources
- `.cursor/rules/`: not present.
- `.cursorrules`: not present.
- `.github/copilot-instructions.md`: not present.
- There are no repository-local Cursor/Copilot instruction files today.

## Repository Layout
- `src/main.rs`: app bootstrap, scene building, camera controls, GPU orchestration.
- `src/pathtracer.wgsl`: path tracer + fast raytracer shader logic.
- `snapshot.png`: runtime output image (ignored by git).
- `target/`: build artifacts (ignored by git).

## Toolchain and Runtime Expectations
- Rust edition: 2021.
- Build tool: Cargo.
- Rendering backend: WGPU.
- Runtime smoke tests require a working graphics environment.

## Build / Run / Lint / Test Commands

### Build and Run
- Type-check quickly: `cargo check`
- Debug build: `cargo build`
- Debug run: `cargo run`
- Release build: `cargo build --release`
- Release run: `cargo run --release`

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
- Pre-PR sanity: `cargo fmt -- --check && cargo clippy --all-targets --all-features`

## Recommended Agent Workflow
- Read existing code paths before editing; match local conventions first.
- Keep diffs focused; avoid unrelated refactors.
- Run `cargo fmt` after Rust edits.
- Run `cargo check` after any non-trivial change.
- If shader/render flow changed, run `cargo run` for a visual smoke test.
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
- If adding tests, include single-test invocation examples in PR notes.

## Commit and Change Hygiene
- Match current commit style: concise, imperative, behavior-oriented titles.
- Explain visible rendering changes in commit body when behavior shifts.
- Avoid mixing formatting-only and functional changes unless required.

## Agent Completion Checklist
- Formatting done (`cargo fmt`).
- Build check done (`cargo check`).
- Lint status reported (pass or known warnings/failures).
- Runtime smoke test done for shader/render-loop edits.
- Any limitations and follow-up work documented clearly.
