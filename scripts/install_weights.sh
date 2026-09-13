#!/usr/bin/env bash
# Install a trained KC->MBON weight set as the active `fly-mb` policy.
#
#   scripts/install_weights.sh round3            # models/mb_weights_round3.npz -> data/compiled/mb_weights.npz (current release)
#   scripts/install_weights.sh round2            # previous release
#   scripts/install_weights.sh round1
#   scripts/install_weights.sh path/to/weights.npz
#   scripts/install_weights.sh --list
#
# `python -m flysweeper.server --condition fly-mb`, `validate.py` and `mb_eval.py` read
# data/compiled/mb_weights.npz by default (FLYSWEEPER_DATA overrides the data root, as in
# flysweeper/paths.py). Named rounds are checked against models/SHA256SUMS before copying; an
# existing installed file is kept as mb_weights.npz.bak.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="$ROOT/models"
DATA_ROOT="${FLYSWEEPER_DATA:-$ROOT/data}"
TARGET="$DATA_ROOT/compiled/mb_weights.npz"

usage() {
  sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
}

sha256_of() {
  if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}';
  elif command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}';
  else echo "need shasum or sha256sum" >&2; exit 1; fi
}

if [[ $# -ne 1 || "$1" == "-h" || "$1" == "--help" ]]; then usage; exit 1; fi

if [[ "$1" == "--list" ]]; then
  echo "available rounds (models/):"
  for f in "$MODELS"/mb_weights_round*.npz; do
    [[ -e "$f" ]] || continue
    b="$(basename "$f")"; r="${b#mb_weights_}"; r="${r%.npz}"
    printf '  %-8s %s  %s\n' "$r" "$(sha256_of "$f")" "$b"
  done
  if [[ -e "$TARGET" ]]; then echo "installed: $TARGET  $(sha256_of "$TARGET")"; else echo "installed: (none)"; fi
  exit 0
fi

case "$1" in
  round[0-9]*)
    SRC="$MODELS/mb_weights_$1.npz"
    if [[ ! -e "$SRC" ]]; then
      echo "no such weight set: $SRC (try --list; see models/README.md)" >&2; exit 1
    fi
    expected="$(awk -v f="mb_weights_$1.npz" '$2 == f {print $1}' "$MODELS/SHA256SUMS" || true)"
    actual="$(sha256_of "$SRC")"
    if [[ -n "$expected" && "$expected" != "$actual" ]]; then
      echo "sha256 mismatch for $SRC" >&2
      echo "  expected $expected" >&2
      echo "  actual   $actual" >&2
      exit 1
    fi
    [[ -z "$expected" ]] && echo "warning: $1 is not listed in models/SHA256SUMS; installing unverified" >&2
    ;;
  *)
    SRC="$1"
    [[ -e "$SRC" ]] || { echo "no such file: $SRC" >&2; exit 1; }
    ;;
esac

mkdir -p "$(dirname "$TARGET")"
if [[ -e "$TARGET" ]]; then
  if [[ "$(sha256_of "$TARGET")" == "$(sha256_of "$SRC")" ]]; then
    echo "already installed: $TARGET == $SRC"; exit 0
  fi
  cp -p "$TARGET" "$TARGET.bak"
  echo "backed up previous weights to $TARGET.bak"
fi
cp -p "$SRC" "$TARGET"
echo "installed $SRC -> $TARGET ($(sha256_of "$TARGET"))"
echo "run: python -m flysweeper.server --condition fly-mb"
