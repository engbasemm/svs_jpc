#!/usr/bin/env bash
# Usage: bash push.sh https://github.com/<owner>/<repo>.git [branch]
# Pushes this folder's history to the given repository (creates the branch if needed).
set -euo pipefail
url="${1:?usage: bash push.sh <repo url> [branch]}"; branch="${2:-thesis-eval}"
git remote remove origin 2>/dev/null || true
git remote add origin "$url"
git push -u origin "HEAD:$branch"
