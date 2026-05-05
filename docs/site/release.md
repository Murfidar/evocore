# Release Process

1. Update version numbers in `pyproject.toml` and `Cargo.toml`.
2. Update `CHANGELOG.md`.
3. Run local verification:
   - `cargo test`
   - `python -m maturin develop --release`
   - `python -m pytest tests/unit/ tests/integration/ -v`
   - `ruff format --check`
   - `ruff check --select ALL`
   - `cargo fmt --check`
   - `cargo clippy --all-targets -- -D warnings`
4. Commit the release preparation changes.
5. Create and push a version tag such as `v0.6.0`.
6. Wait for release artifacts to build.
7. Download and inspect the artifacts.
8. Approve or edit the draft GitHub Release.
9. Trigger the manual PyPI publish workflow with the tag name.
10. Approve the `pypi` environment.
11. Verify the PyPI page and install in a clean environment.
