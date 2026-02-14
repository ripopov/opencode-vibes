#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
source_dir="$repo_root/.githooks"
target_dir="$repo_root/.git/hooks"

install_hook() {
    local hook_name="$1"
    local source_path="$source_dir/$hook_name"
    local target_path="$target_dir/$hook_name"

    if [[ ! -f "$source_path" ]]; then
        echo "Missing hook source: $source_path" >&2
        exit 1
    fi

    cp "$source_path" "$target_path"
    chmod +x "$target_path"
    echo "Installed $hook_name"
}

install_hook pre-commit
install_hook pre-push

echo "Git hooks installed from .githooks/ into .git/hooks/."
