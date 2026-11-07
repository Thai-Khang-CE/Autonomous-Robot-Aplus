#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---check}"
FORMATTER="${CLANG_FORMAT_BIN:-clang-format-18}"

if [[ "$MODE" != "--check" && "$MODE" != "--fix" ]]; then
  echo "Usage: $0 [--check|--fix]"
  exit 2
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if ! command -v "$FORMATTER" >/dev/null 2>&1; then
  echo "ERROR: $FORMATTER is not installed."
  exit 1
fi

if [[ ! -f ".clang-format" ]]; then
  echo "ERROR: .clang-format is missing."
  exit 1
fi

# Include tracked and new untracked C++ files.
# Exclude ignored files such as build artifacts.
mapfile -d '' -t candidates < <(
  git ls-files --cached --others --exclude-standard -z -- \
    '*.cpp' '*.cc' '*.cxx' \
    '*.hpp' '*.hh' '*.hxx' '*.h'
)

files=()

for file in "${candidates[@]}"; do
  # Skip files deleted from the working tree.
  if [[ -f "$file" ]]; then
    files+=("$file")
  fi
done

if (( ${#files[@]} == 0 )); then
  echo "NOTICE: No C++ files found."
  exit 0
fi

echo "Formatter: $("$FORMATTER" --version)"
echo "Found ${#files[@]} C++ files."

case "$MODE" in
  --check)
    echo "Checking formatting..."
    "$FORMATTER" --dry-run --Werror "${files[@]}"
    echo "PASS: C++ formatting is valid."
    ;;
  --fix)
    echo "Formatting C++ files..."
    "$FORMATTER" -i "${files[@]}"
    echo "DONE: C++ files formatted."
    ;;
esac
