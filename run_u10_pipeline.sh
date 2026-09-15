#!/usr/bin/env bash
# ellipsoid_u10 LHS-300 pipeline driver.
# Spec: docs/superpowers/specs/2026-09-15-u10-lhs300-pipeline-design.md
#
# Usage: ./run_u10_pipeline.sh [0|1|2|3|all]   (default: all)
#   0 = prep + smoke test
#   1 = LHS 310 re-sample + OpenFOAM batch (--jobs 12)
#   2 = update pipod_config.json (iters=20000, n_field=60000)
#   3 = PiPOD-DeepONet stage 1->2->3 (chained init_from)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PY="$REPO_ROOT/.env/bin/python"
[ -x "$PY" ] || { echo "missing $PY (run Task 1 first)" >&2; exit 1; }

DATA_ROOT="data/openfoam/ellipsoids_u10"
SMOKE_ROOT="data/openfoam/_smoke_u10"
EXPERIMENT="examples/ellipsoids_of4"
CONFIG="$EXPERIMENT/pipod_config.json"
OF_BASHRC="/opt/openfoam14/etc/bashrc"
LOG="$DATA_ROOT/pipeline_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$DATA_ROOT"
exec > >(tee -a "$LOG") 2>&1
echo "[$(date '+%F %T')] log: $LOG ($(realpath "$LOG"))"

# OpenFOAM 14 env: its bashrc returns non-zero under 'set -e' (benign
# foamEtcFile/readlink warnings), so relax around it and validate by PATH.
set +e
set +u
source "$OF_BASHRC"
set -eu
command -v blockMesh >/dev/null || {
    echo "OpenFOAM env load failed (blockMesh not on PATH)" >&2
    exit 1
}

STAGE="${1:-all}"

stage0() {
    echo "[$(date '+%F %T')] === Stage 0: prep + smoke ==="
    # 移走旧 manifest（全新重采；保留备份防误删）
    if [ -f "$DATA_ROOT/lhs_latents.npz" ]; then
        mv "$DATA_ROOT/lhs_latents.npz" "$DATA_ROOT/lhs_latents.npz.pre300.bak"
        echo "old manifest -> lhs_latents.npz.pre300.bak"
    fi
    # 清理上次被 kill 的 4 个半成品 case
    rm -rf "$DATA_ROOT"/cases/lhs_shape_00{0,1,2,3}_case000
    # 冒烟：临时 root 跑 4 采样/2 并发，验证 mesh->solver->postProcess 全链路
    rm -rf "$SMOKE_ROOT"
    "$PY" generate_openfoam_snapshots.py --root "$SMOKE_ROOT" \
        --lhs 4 --seed 99 --fixed_bc 10 1 0 0 --jobs 2 --grid_stretch \
        --experiment "$EXPERIMENT" --data_source data/ellipsoids
    n=$(find "$SMOKE_ROOT/snapshots" -name '*.npz' | wc -l)
    rm -rf "$SMOKE_ROOT"
    if [ "${n// /}" -lt 1 ]; then
        echo "SMOKE TEST FAILED: no snapshot produced" >&2
        exit 1
    fi
    echo "smoke test ok (${n// /} snapshots)"
}

stage1() {
    echo "[$(date '+%F %T')] === Stage 1: LHS 310 + OpenFOAM batch (jobs=12) ==="
    "$PY" generate_openfoam_snapshots.py --root "$DATA_ROOT" \
        --lhs 310 --seed 3 --fixed_bc 10 1 0 0 --jobs 12 --grid_stretch \
        --experiment "$EXPERIMENT" --data_source data/ellipsoids
    # 有 failed case 则中止（训练要求 manifest 每个形状至少一个快照）
    "$PY" - <<'EOF'
import json, sys
s = json.load(open("data/openfoam/ellipsoids_u10/batch_summary.json"))
print("batch:", s["cases"])
failed = s["cases"]["failed"]
if failed:
    print("FAILED CASES: " + ", ".join(f["case"] for f in failed), file=sys.stderr)
    sys.exit(1)
EOF
}

stage2() {
    echo "[$(date '+%F %T')] === Stage 2: update pipod_config.json ==="
    "$PY" - <<'EOF'
import json
p = "examples/ellipsoids_of4/pipod_config.json"
cfg = json.load(open(p))
cfg["iters"] = 20000
cfg["n_field"] = 60000
with open(p, "w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
print("iters =", cfg["iters"], "| n_field =", cfg["n_field"])
EOF
}

stage3() {
    # optional arg: subset of substages, e.g. "./run_u10_pipeline.sh 3 1"
    local substages="${1:-1 2 3}"
    for st in $substages; do
        echo "[$(date '+%F %T')] === Stage 3: train stage $st ==="
        "$PY" train_pipod_deeponet.py --config "$CONFIG" --stage "$st"
    done
}

case "$STAGE" in
    0) stage0 ;;
    1) stage1 ;;
    2) stage2 ;;
    3) stage3 "${2:-}" ;;
    all) stage0; stage1; stage2; stage3 ;;
    *) echo "usage: $0 [0|1|2|3|all]" >&2; exit 2 ;;
esac
echo "[$(date '+%F %T')] stage '$STAGE' finished OK"
