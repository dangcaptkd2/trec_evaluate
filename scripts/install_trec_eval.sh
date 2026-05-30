#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="$ROOT_DIR/tools"
REPO_DIR="$TOOLS_DIR/trec_eval"

mkdir -p "$TOOLS_DIR"

if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone https://github.com/usnistgov/trec_eval.git "$REPO_DIR"
fi

make -C "$REPO_DIR"

echo "$REPO_DIR/trec_eval"
