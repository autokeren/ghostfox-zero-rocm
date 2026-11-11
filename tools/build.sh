#!/usr/bin/env bash
# Assemble the Match 4 image WITHOUT pulling the 29 GiB ROCm base.
#
#   base (digest-pinned, never downloaded)  --crane append-->  :stage1
#   + /opt/gz/site + /app (code + deps)                       :stage1
#   + /models/Qwen3-4B (weights)             --crane append--> :stage2
#   config (env + CMD sleep infinity)        --crane mutate--> :match4
#
# Requires: crane, docker-config auth for ghcr.io.
set -euo pipefail
cd "$(dirname "$0")/.."

REPO="${GZ_REPO:-ghcr.io/autokeren/ghostfox-zero-rocm}"
BASE_REF="rocm/pytorch@sha256:3174cb7061d94c427da96c0edef4adea28046fa3f3b2ff3948dc4e995665ff8c"
CRANE="${CRANE:-$HOME/.local/bin/crane}"
WORK="${GZ_BUILD_DIR:-/tmp/gz_build}"

mkdir -p "$WORK"

echo "[1/4] pip site-packages (pure-python + abi3 wheels, py3.14-safe)"
rm -rf "$WORK/site"
pip install --quiet --target "$WORK/site" --no-cache-dir \
    "transformers==4.57.1" >/dev/null
rm -f "$WORK/site"/numpy* 2>/dev/null || true   # base image ships numpy

echo "[2/4] app layer tarball"
rm -rf "$WORK/approot"
mkdir -p "$WORK/approot/opt/gz" "$WORK/approot/app"
cp -r app/. "$WORK/approot/app/"
cp -r "$WORK/site" "$WORK/approot/opt/gz/site"
tar -C "$WORK/approot" -cf "$WORK/app_layer.tar" .
stat -c "app layer: %s bytes" "$WORK/app_layer.tar"

echo "[3/4] models layer tarball (7.6 GiB)"
test -d models/Qwen3-4B-Instruct-2507
tar -C models -cf "$WORK/models_layer.tar" Qwen3-4B-Instruct-2507

echo "[4/4] crane: append + mutate + push"
"$CRANE" append -b "$BASE_REF" -f "$WORK/app_layer.tar" \
    -t "$REPO:stage1"
"$CRANE" append -b "$REPO:stage1" -f "$WORK/models_layer.tar" \
    -t "$REPO:stage2"
"$CRANE" mutate "$REPO:stage2" \
    --env "PYTHONPATH=/opt/gz/site" \
    --env "GZ_MODEL_DIR=/models/Qwen3-4B-Instruct-2507" \
    --env "GZ_M4_SOCKET=/tmp/gz_m4.sock" \
    --env "HF_HUB_OFFLINE=1" \
    --env "TRANSFORMERS_OFFLINE=1" \
    --env "PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    --env "VIRTUAL_ENV=/opt/venv" \
    --cmd sleep infinity \
    -t "$REPO:match4"

"$CRANE" manifest "$REPO:match4" | python3 -c "import json,sys; m=json.load(sys.stdin); print('layers:', len(m['layers']))"
echo "DONE -> $REPO:match4"
