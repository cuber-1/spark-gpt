#!/usr/bin/env bash
set -euo pipefail

spark_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
spark_repo_root=$(cd -- "$spark_script_dir/.." && pwd)
cd "$spark_repo_root"

spark_python=${SPARKGPT_PYTHON:-python3}
spark_config=${SPARKGPT_CONFIG:-configs/spark_300m.toml}
spark_out_dir=${SPARKGPT_OUT_DIR:-runs/spark-300m}
spark_max_restarts=${SPARKGPT_MAX_RESTARTS:-5}

if [[ ! "$spark_max_restarts" =~ ^[0-9]+$ ]]; then
  echo "SPARKGPT_MAX_RESTARTS must be a non-negative integer" >&2
  exit 2
fi

if [[ -x /usr/local/cuda/bin/ptxas ]]; then
  export TRITON_PTXAS_PATH=${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}
fi

if [[ -n "${SPARKGPT_PYTHON_DEV_ROOT:-}" ]]; then
  spark_include_root=$SPARKGPT_PYTHON_DEV_ROOT/usr/include
  export CPATH="$spark_include_root:$spark_include_root/python3.12${CPATH:+:$CPATH}"
fi
export PYTHONPATH="$spark_repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

spark_attempt=0
while true; do
  spark_resume_args=()
  if [[ -f "$spark_out_dir/last.pt" ]]; then
    spark_resume_args=(--resume "$spark_out_dir/last.pt")
  fi

  set +e
  "$spark_python" -m sparkgpt train --config "$spark_config" "${spark_resume_args[@]}"
  spark_status=$?
  set -e
  if [[ "$spark_status" -eq 0 ]]; then
    exit 0
  fi
  if [[ ! -f "$spark_out_dir/last.pt" || "$spark_attempt" -ge "$spark_max_restarts" ]]; then
    exit "$spark_status"
  fi
  spark_attempt=$((spark_attempt + 1))
  echo "training failed; resuming last.pt in 30 seconds (attempt $spark_attempt)" >&2
  sleep 30
done
