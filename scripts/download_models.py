"""
Descarga inswapper_128.onnx en ./models/

InsightFace ya no distribuye este archivo desde su repositorio oficial, así que se usan
mirrors públicos de la comunidad. Si todos fallan, descárgalo a mano y colócalo en
models/inswapper_128.onnx (≈ 530 MB).

El paquete buffalo_l (detector + ArcFace) NO hace falta descargarlo: InsightFace lo baja
automáticamente la primera vez que arranca la app.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

DEST = Path(__file__).resolve().parent.parent / "models" / "inswapper_128.onnx"
MIRRORS = [
    "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
    "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx",
]
MIN_SIZE = 500 * 1024 * 1024  # sanity check: el archivo real pesa ~530 MB


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(".part")

    def hook(blocks: int, block_size: int, total: int) -> None:
        if total > 0:
            pct = min(100, blocks * block_size * 100 / total)
            sys.stdout.write(f"\r  {pct:5.1f}%  ({blocks * block_size / 1e6:.0f}/{total / 1e6:.0f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, hook)
    print()
    if tmp.stat().st_size < MIN_SIZE:
        tmp.unlink()
        raise RuntimeError("archivo demasiado pequeño, probablemente no es el modelo")
    tmp.replace(dest)


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists() and DEST.stat().st_size >= MIN_SIZE:
        print(f"Ya existe: {DEST}")
        return
    for url in MIRRORS:
        print(f"Descargando desde {url}")
        try:
            download(url, DEST)
            print(f"OK -> {DEST}")
            return
        except Exception as exc:  # noqa: BLE001
            print(f"  falló: {exc}")
    sys.exit("No se pudo descargar inswapper_128.onnx de ningún mirror. Descárgalo manualmente.")


if __name__ == "__main__":
    main()
