#!/usr/bin/env bash

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# If COMFYUI_MODE is worker, start worker + handler headless
if [ "${COMFYUI_MODE:-worker}" = "worker" ]; then
  echo "worker-comfyui: Starting ComfyUI Worker headless..."
  comfy worker --workspace /comfyui \
    --manager-host "${COMFYUI_MANAGER_HOST:-comfy-manager}" \
    --manager-port "${COMFYUI_MANAGER_PORT:-8188}" \
    --verbose "${COMFY_LOG_LEVEL}" --log-stdout &
  echo "worker-comfyui: Starting RunPod Handler"
  exec python -u /handler.py
fi

echo "worker-comfyui: Starting ComfyUI"

# Allow operators to tweak verbosity; default is DEBUG.
: "${COMFY_LOG_LEVEL:=DEBUG}"