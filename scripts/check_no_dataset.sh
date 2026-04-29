#!/usr/bin/env bash
# Pre-commit guard: block accidental dataset/model commits.
# PDF Chapter V says committing dataset/model files = grade 0 at defense.
set -euo pipefail
forbidden_pattern='\.(zip|pt|pth)$|^images/|^augmented_directory/|^artifacts/'
violations=$(git diff --cached --name-only | grep -E "$forbidden_pattern" || true)
if [ -n "$violations" ]; then
    echo "Cannot commit dataset or model files:"
    echo "$violations" | sed 's/^/   /'
    echo "If unintentional, remove from index: git rm --cached <path>"
    exit 1
fi
