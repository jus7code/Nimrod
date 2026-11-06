"""
API web (FastAPI) del laboratorio de face swap.

    GET  /api/models            -> catálogo de modelos para la página
    POST /api/jobs              -> sube fuente + video y crea un trabajo
    GET  /api/jobs/{id}         -> estado y progreso
    GET  /api/jobs/{id}/result  -> video resultante
    GET  /api/health            -> GPU / estado
"""
from __future__ import annotations

import os
import shutil
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .engine import IMAGE_EXT, VIDEO_EXT, FaceEngine, Masker, PipelineOptions, get_providers, process_video
from .models_registry import SWAP_MODELS, public_catalog

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS = BASE_DIR / "data" / "uploads"
OUTPUTS = BASE_DIR / "data" / "outputs"
ENCODER_CKPTS = BASE_DIR / "data" / "encoder_models"
for d in (UPLOADS, OUTPUTS):
    d.mkdir(parents=True, exist_ok=True)

WATERMARK = os.getenv("FACESWAP_WATERMARK", "1") != "0"

app = FastAPI(title="FaceSwap Lab")
executor = ThreadPoolExecutor(max_workers=1)  # una GPU -> un trabajo a la vez
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()

_engine: FaceEngine | None = None
_engine_lock = threading.Lock()


def get_engine(status_cb=None) -> FaceEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = FaceEngine(status_cb=status_cb)
        return _engine


def update_job(job_id: str, **fields) -> None:
    with jobs_lock:
        jobs[job_id].update(fields)


def run_job(job_id: str, source: Path, video: Path, params: dict) -> None:
    def status(msg: str, frac: float = 0.0, phase: str | None = None):
        fields = {"message": msg, "progress": frac}
        if phase:
            fields["status"] = phase
        update_job(job_id, **fields)

    try:
        status("Cargando modelos...", 0, "loading_models")
        engine = get_engine(status)
        masker = Masker(engine, params["occlusion"], params["region"], status)

        if params["method"] == "encoder":
            from .encoder.trainer import build_encoder_swapper  # importa PyTorch solo si hace falta

            engine.release_swapper()  # deja la VRAM libre para entrenar
            status("Preparando entrenamiento...", 0, "training")
            swapper = build_encoder_swapper(engine, masker, source, video, params["train_minutes"],
                                            ENCODER_CKPTS, status)
        else:
            from .swappers import OnnxSwapper

            identity = engine.identity_from(source, status)
            swapper = OnnxSwapper(engine, params["model"], identity, status)

        status("Procesando video...", 0, "processing")
        opts = PipelineOptions(mode=params["mode"], color=params["color"], smooth=params["smooth"],
                               watermark=WATERMARK)

        def on_progress(done: int, total: int, fps: float):
            eta = (total - done) / fps if fps > 0 else 0
            status(f"Frame {done}/{total} · {fps:.1f} fps · quedan ~{eta / 60:.1f} min", done / total)

        out = OUTPUTS / f"{job_id}.mp4"
        process_video(engine, swapper, masker, opts, video, out, on_progress)
        update_job(job_id, status="done", progress=1.0, message="Listo", result=str(out))
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        update_job(job_id, status="error", message=str(exc))
    finally:
        swapper = None
        if _engine is not None:
            _engine.release_swapper()
        if params["method"] == "encoder":
            import torch

            torch.cuda.empty_cache()


def _save_upload(upload: UploadFile, allowed: set[str], stem: str) -> Path:
    ext = Path(upload.filename or "").suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Formato no soportado: {ext or '(sin extensión)'}")
    dest = UPLOADS / f"{stem}{ext}"
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest


@app.get("/api/health")
def health():
    return {"providers": get_providers(), "engine_loaded": _engine is not None, "watermark": WATERMARK}


@app.get("/api/models")
def models():
    return {"pretrained": public_catalog()}


@app.post("/api/jobs")
def create_job(
    source: UploadFile = File(...),
    video: UploadFile = File(...),
    method: str = Form("pretrained"),
    model: str = Form("hyperswap_1a_256"),
    mode: str = Form("largest"),
    occlusion: bool = Form(False),
    region: bool = Form(True),
    color: bool = Form(True),
    smooth: bool = Form(True),
    train_minutes: int = Form(15),
    consent: bool = Form(False),
):
    if not consent:
        raise HTTPException(400, "Debes confirmar que tienes el consentimiento de las personas.")
    if method not in {"pretrained", "encoder"}:
        raise HTTPException(400, "method debe ser 'pretrained' o 'encoder'")
    if method == "pretrained" and model not in SWAP_MODELS:
        raise HTTPException(400, f"Modelo desconocido: {model}")
    if mode not in {"largest", "all"}:
        raise HTTPException(400, "mode debe ser 'largest' o 'all'")

    job_id = uuid.uuid4().hex[:12]
    source_path = _save_upload(source, IMAGE_EXT | VIDEO_EXT, f"{job_id}_source")
    video_path = _save_upload(video, VIDEO_EXT, f"{job_id}_video")
    params = {
        "method": method, "model": model, "mode": mode, "occlusion": occlusion, "region": region,
        "color": color, "smooth": smooth, "train_minutes": max(1, min(train_minutes, 240)),
    }
    with jobs_lock:
        jobs[job_id] = {"id": job_id, "status": "queued", "progress": 0.0, "message": "En cola", "params": params}
    executor.submit(run_job, job_id, source_path, video_path, params)
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Trabajo no encontrado")
        return {k: v for k, v in job.items() if k != "result"}


@app.get("/api/jobs/{job_id}/result")
def get_result(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job.get("status") != "done":
        raise HTTPException(404, "Resultado no disponible")
    return FileResponse(job["result"], media_type="video/mp4", filename=f"faceswap_{job_id}.mp4")


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
