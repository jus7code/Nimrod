#!/usr/bin/env bash
# Arranca FaceSwap Lab. Si se ejecuta dentro de un contenedor (VS Code Flatpak, toolbox…)
# donde el driver NVIDIA solo existe en el host (/run/host), expone sus librerías al venv.
set -euo pipefail
cd "$(dirname "$0")/.."

HOST_LIB=/run/host/usr/lib64
if ! ldconfig -p 2>/dev/null | grep -q 'libcuda.so.1' && [ -e "$HOST_LIB/libcuda.so.1" ]; then
    NVLIB="$PWD/.venv/host-nvidia"
    mkdir -p "$NVLIB"
    for f in "$HOST_LIB"/libcuda.so* "$HOST_LIB"/libnvidia-ml.so* \
             "$HOST_LIB"/libnvidia-ptxjitcompiler.so* "$HOST_LIB"/libnvidia-nvvm*.so* \
             "$HOST_LIB"/libnvidia-gpucomp.so*; do
        [ -e "$f" ] && ln -sf "$f" "$NVLIB/"
    done
    export LD_LIBRARY_PATH="$NVLIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "[run.sh] Usando el driver NVIDIA del host ($HOST_LIB)"
fi

exec .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port "${PORT:-8000}" "$@"
