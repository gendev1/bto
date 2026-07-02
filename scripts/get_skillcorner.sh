#!/usr/bin/env bash
# Download one match of SkillCorner Open Data (github.com/SkillCorner/opendata)
# into data/skillcorner/<id>/.
#
# Repo layout (explored via GitHub API against the `master` branch, 2026-07):
#   data/matches/<id>/<id>_match.json                 -- match/roster/team metadata
#   data/matches/<id>/<id>_tracking_extrapolated.jsonl -- per-frame tracking (Git LFS)
#   data/matches/<id>/<id>_dynamic_events.csv          -- events (not used here)
#   data/matches/<id>/<id>_phases_of_play.csv          -- phases (not used here)
#
# The tracking file is stored via Git LFS, so raw.githubusercontent.com only
# serves the small LFS pointer stub. The actual bytes are served by GitHub's
# LFS media proxy at media.githubusercontent.com/media/<owner>/<repo>/<ref>/<path>.
#
# Usage: scripts/get_skillcorner.sh [match_id]
# Default match_id is 1886347 (Auckland FC vs Newcastle Jets, A-League 2024/25).
set -euo pipefail

MATCH_ID="${1:-1886347}"
REPO="SkillCorner/opendata"
REF="master"
RAW_BASE="https://raw.githubusercontent.com/${REPO}/${REF}/data/matches/${MATCH_ID}"
LFS_BASE="https://media.githubusercontent.com/media/${REPO}/${REF}/data/matches/${MATCH_ID}"

OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/skillcorner/${MATCH_ID}"
mkdir -p "${OUT_DIR}"

echo "Downloading SkillCorner match ${MATCH_ID} into ${OUT_DIR}"

echo "-> match.json"
curl -fSL "${RAW_BASE}/${MATCH_ID}_match.json" -o "${OUT_DIR}/match_data.json"

echo "-> tracking_extrapolated.jsonl (via LFS media proxy, ~85-95MB)"
curl -fSL "${LFS_BASE}/${MATCH_ID}_tracking_extrapolated.jsonl" -o "${OUT_DIR}/structured_data.jsonl"

echo "Done. Files:"
ls -lh "${OUT_DIR}"
