"""Descargas de modelos con progreso y reanudación (archivo .part + cabecera HTTP Range)."""
from __future__ import annotations

import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Optional

StatusCb = Optional[Callable[[str, float], None]]


def download_file(urls: list[str], dest: Path, label: str, status_cb: StatusCb = None) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last_error: Exception | None = None

    for url in urls:
        for attempt in range(8):
            done = part.stat().st_size if part.exists() else 0
            headers = {"User-Agent": "faceswap-lab"}
            if done:
                headers["Range"] = f"bytes={done}-"
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as resp:
                    if done and resp.status != 206:  # el servidor no admite reanudar
                        done = 0
                    length = resp.headers.get("Content-Length")
                    total = done + int(length) if length else 0
                    last_report = 0.0
                    with part.open("ab" if done else "wb") as f:
                        while chunk := resp.read(1 << 16):
                            f.write(chunk)
                            done += len(chunk)
                            now = time.time()
                            if status_cb and now - last_report > 0.5:
                                last_report = now
                                status_cb(
                                    f"Descargando {label} (solo la 1ª vez): "
                                    f"{done / 1e6:.0f}/{total / 1e6:.0f} MB" if total else
                                    f"Descargando {label}: {done / 1e6:.0f} MB",
                                    done / total if total else 0.0,
                                )
                part.replace(dest)
                return dest
            except urllib.error.HTTPError as exc:
                if exc.code == 416 and part.exists():  # ya estaba completo
                    part.replace(dest)
                    return dest
                last_error = exc
                if exc.code in (403, 404):
                    break  # este mirror no lo tiene, probar el siguiente
            except Exception as exc:  # noqa: BLE001  (cortes de red)
                last_error = exc
                if status_cb:
                    status_cb(f"Conexión cortada descargando {label}; reintento {attempt + 1}/8...", 0.0)
                time.sleep(2)
    raise RuntimeError(f"No se pudo descargar {label}: {last_error}")


def download_and_extract_zip(urls: list[str], target_dir: Path, label: str, status_cb: StatusCb = None) -> Path:
    if target_dir.is_dir() and any(target_dir.glob("*.onnx")):
        return target_dir
    zip_path = target_dir.with_suffix(".zip")
    download_file(urls, zip_path, label, status_cb)
    if status_cb:
        status_cb(f"Descomprimiendo {label}...", 1.0)
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)
    nested = target_dir / target_dir.name  # algunos zips traen una subcarpeta
    if nested.is_dir():
        for f in nested.iterdir():
            f.replace(target_dir / f.name)
        nested.rmdir()
    zip_path.unlink(missing_ok=True)
    return target_dir
