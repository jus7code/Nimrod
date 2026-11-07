"""
API web (FastAPI) del laboratorio de face swap / character swap.

    GET  /api/models            -> catálogo de modelos (face swap y character swap)
    POST /api/scan              -> sube el video destino, detecta y agrupa los rostros (para elegir a quién cambiar)
    POST /api/jobs              -> sube la fuente y crea un trabajo sobre un video escaneado
    GET  /api/jobs              -> lista de trabajos (incluye los interrumpidos)
    GET  /api/jobs/{id}         -> estado y progreso
    GET  /api/jobs/{id}/result  -> video resultante
    GET  /api/jobs/{id}/partial -> lo generado hasta ahora (face swap), con audio
    POST /api/jobs/{id}/resume  -> reanuda un trabajo interrumpido
    DELETE /api/jobs/{id}       -> descarta un trabajo que no está en marcha
    GET  /api/health            -> GPU / estado

Cada trabajo guarda su estado en data/work/<id>/job.json y el face swap escribe el video en
tramos de ~10 s (engine.SegmentWriter). Si el servidor se cae a mitad, al volver a arrancar el
trabajo aparece como "interrumpido" y se puede reanudar desde el último tramo terminado.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .engine import (IMAGE_EXT, VIDEO_EXT, FaceEngine, Masker, PipelineOptions, concat_segments, get_providers,
                     process_video)
from .models_registry import (CHARACTER_MODELS, SWAP_MODELS, character_availability, public_catalog,
                              public_character_catalog)

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS = BASE_DIR / "data" / "uploads"
OUTPUTS = BASE_DIR / "data" / "outputs"
ENCODER_CKPTS = BASE_DIR / "data" / "encoder_models"
WORK = BASE_DIR / "data" / "work"
for d in (UPLOADS, OUTPUTS, WORK):
    d.mkdir(parents=True, exist_ok=True)

WATERMARK = os.getenv("FACESWAP_WATERMARK", "1") != "0"

app = FastAPI(title="FaceSwap Lab")
executor = ThreadPoolExecutor(max_workers=1)  # una GPU -> un trabajo a la vez
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
scans: OrderedDict[str, dict] = OrderedDict()  # scan_id -> {"video": Path, "people": [... con embedding]}
MAX_SCANS = 20

_engine: FaceEngine | None = None
_engine_lock = threading.Lock()


def get_engine(status_cb=None) -> FaceEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = FaceEngine(status_cb=status_cb)
        return _engine


RUNNING = {"queued", "loading_models", "training", "processing", "preparing", "generating", "compositing",
           "finalizing"}


def job_dir(job_id: str) -> Path:
    return WORK / job_id


def public_job(job: dict) -> dict:
    """Lo que ve el navegador: sin rutas internas ni el estado de reanudación."""
    st = job.get("_", {})
    return {k: v for k, v in job.items() if k not in ("result", "_")} | {
        "done_frames": sum(n for _, n in st.get("segments", [])),
        "total_frames": st.get("total_frames", 0),
        "parts": len(st.get("segments", [])),
    }


def save_job(job_id: str) -> None:
    """Escribe data/work/<id>/job.json (a un temporal y luego renombra: nunca queda a medias).
    Si falla (p. ej. OneDrive tiene el archivo bloqueado un instante) se reintenta y, como último
    recurso, se sigue sin guardar: no debe tumbar el trabajo."""
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            return
        data = json.dumps(job, ensure_ascii=False, indent=1, default=str)
    d = job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "job.json.tmp"
    for attempt in range(5):
        try:
            tmp.write_text(data, encoding="utf-8")
            os.replace(tmp, d / "job.json")
            return
        except OSError as exc:
            if attempt == 4:
                print(f"[aviso] no se pudo guardar el estado del trabajo {job_id}: {exc}")
            time.sleep(0.2)


def update_job(job_id: str, persist: bool = False, **fields) -> None:
    with jobs_lock:
        jobs[job_id].update(fields)
    if persist or "status" in fields:  # cada cambio de fase se guarda; el progreso fino no
        save_job(job_id)


def load_jobs() -> None:
    """Al arrancar: recupera los trabajos guardados. Los que estaban en marcha cuando se cayó el
    servidor quedan como "interrupted" (reanudables)."""
    for f in sorted(WORK.glob("*/job.json")):
        try:
            job = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        st = job.get("_", {})
        if job.get("status") == "done":
            if not Path(job.get("result", "")).is_file():
                continue
        elif job.get("status") != "error":
            done = sum(n for _, n in st.get("segments", []))
            total = st.get("total_frames") or 0
            if job.get("params", {}).get("task") == "character":
                msg = "Interrumpido. El character swap se volverá a generar desde el principio."
            elif done:
                msg = f"Interrumpido con {done}/{total or '?'} frames ya generados. Puedes reanudarlo o descargar lo generado."
            elif st.get("trained_minutes"):
                msg = f"Interrumpido durante el entrenamiento ({st['trained_minutes']:.0f} min guardados). Puedes reanudarlo."
            else:
                msg = "Interrumpido antes de generar ningún frame. Puedes reanudarlo."
            job.update(status="interrupted", message=msg, progress=(done / total) if total else 0.0)
        job["resumable"] = (all(Path(st.get(k) or "").is_file() for k in ("source", "video"))
                            and (job_dir(job["id"]) / "targets.npy").is_file())
        jobs[job["id"]] = job
        if job["status"] == "interrupted":
            save_job(job["id"])


def run_job(job_id: str, source: Path, video: Path, params: dict) -> None:
    def status(msg: str, frac: float = 0.0, phase: str | None = None):
        fields = {"message": msg, "progress": frac}
        if phase:
            fields["status"] = phase
        update_job(job_id, **fields)

    with jobs_lock:
        st = jobs[job_id]["_"]  # estado de reanudación (se modifica en sitio y se guarda con save_job)
    targets: np.ndarray = params["targets"]
    swapper = None
    try:
        status("Cargando modelos...", 0, "loading_models")
        engine = get_engine(status)
        out = OUTPUTS / f"{job_id}.mp4"

        if params["task"] == "character":
            from .character.pipeline import run_character_swap  # importa PyTorch solo si hace falta

            run_character_swap(engine, params, source, video, targets[0], out, job_dir(job_id), WATERMARK, status)
            update_job(job_id, status="done", progress=1.0, message="Listo", result=str(out))
            return

        seg_dir = job_dir(job_id) / "segments"
        masker = Masker(engine, params["occlusion"], params["region"], status)
        if params["method"] == "encoder":
            from .encoder.trainer import build_encoder_swapper  # importa PyTorch solo si hace falta

            engine.release_swapper()  # deja la VRAM libre para entrenar
            # al reanudar se entrena solo lo que faltaba (nada, si ya se estaba generando el video)
            trained = float(st.get("trained_minutes", 0.0))
            minutes = 0.0 if st["segments"] or st.get("processed") else max(0.0, params["train_minutes"] - trained)
            status("Preparando entrenamiento..." if minutes > 0 else "Cargando el encoder entrenado...", 0, "training")

            def on_checkpoint(m: float):
                st["trained_minutes"] = trained + m
                save_job(job_id)

            swapper = build_encoder_swapper(engine, masker, source, video, minutes, ENCODER_CKPTS, status,
                                            targets, on_checkpoint)
            st["trained_minutes"] = float(params["train_minutes"])
            save_job(job_id)
        else:
            from .swappers import OnnxSwapper

            identity = engine.identity_from(source, status)
            swapper = OnnxSwapper(engine, params["model"], identity, status)

        if not st.get("processed"):
            # tramos válidos: los registrados cuyo archivo existe, en orden y sin huecos
            valid = []
            for name, n in st["segments"]:
                if not (seg_dir / name).is_file():
                    break
                valid.append([name, n])
            with jobs_lock:
                st["segments"] = valid
            keep = {name for name, _ in valid}
            if seg_dir.exists():
                for f in seg_dir.glob("seg_*.mp4"):  # .part.mp4 a medias y tramos sin registrar
                    if f.name not in keep:
                        f.unlink(missing_ok=True)
            start = sum(n for _, n in valid)
            status(f"Reanudando desde el frame {start} (avanzando por el video)..." if start else "Procesando video...",
                   0, "processing")
            opts = PipelineOptions(mode="selected", color=params["color"], smooth=params["smooth"],
                                   watermark=WATERMARK, targets=targets)

            def on_progress(done: int, total: int, fps: float):
                st["total_frames"] = total
                eta = (total - done) / fps if fps > 0 else 0
                status(f"Frame {done}/{total} · {fps:.1f} fps · quedan ~{eta / 60:.1f} min", done / total)

            def on_segment(name: str, n: int):
                with jobs_lock:
                    st["segments"].append([name, n])
                save_job(job_id)

            process_video(engine, swapper, masker, opts, video, seg_dir, on_progress, start, on_segment)
            st["processed"] = True
            save_job(job_id)

        status("Uniendo los tramos y añadiendo el audio...", 1.0, "finalizing")
        concat_segments([seg_dir / name for name, _ in st["segments"]], out, video)
        update_job(job_id, status="done", progress=1.0, message="Listo", result=str(out))
        shutil.rmtree(seg_dir, ignore_errors=True)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        update_job(job_id, status="error", message=str(exc))
    finally:
        swapper = None
        if _engine is not None:
            _engine.release_swapper()
        if params["method"] == "encoder" or params["task"] == "character":
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
    return {"pretrained": public_catalog(), "character": public_character_catalog()}


@app.post("/api/scan")
def scan(video: UploadFile = File(...)):
    """Escaneo rápido (~2 frames/s) del video destino: devuelve una miniatura por persona."""
    from .scan import scan_video

    scan_id = uuid.uuid4().hex[:12]
    video_path = _save_upload(video, VIDEO_EXT, f"scan_{scan_id}_video")
    try:
        result = scan_video(get_engine(), video_path)
    except Exception as exc:  # noqa: BLE001
        video_path.unlink(missing_ok=True)
        raise HTTPException(400, f"No se pudo escanear el video: {exc}") from exc
    with jobs_lock:
        scans[scan_id] = {"video": video_path, "people": result["people"]}
        while len(scans) > MAX_SCANS:
            scans.popitem(last=False)
    people = [{k: v for k, v in p.items() if k != "embedding"} for p in result["people"]]
    return {"scan_id": scan_id, "info": result["info"], "people": people}


@app.post("/api/jobs")
def create_job(
    source: UploadFile = File(...),
    scan_id: str = Form(...),
    targets: str = Form(...),               # ids de personas del escaneo, separados por comas
    task: str = Form("face"),               # "face" (face swap) | "character" (character swap)
    # face swap
    method: str = Form("pretrained"),
    model: str = Form("hyperswap_1a_256"),
    occlusion: bool = Form(False),
    region: bool = Form(True),
    color: bool = Form(True),
    smooth: bool = Form(True),
    train_minutes: int = Form(15),
    # character swap
    char_model: str = Form("wan_animate_space"),
    prompt: str = Form(""),
    start: float = Form(0.0),
    duration: float = Form(5.0),
    turbo: bool = Form(True),
    hq: bool = Form(False),
    hf_token: str = Form(""),               # opcional: token de Hugging Face para el Space gratuito
    consent: bool = Form(False),
):
    if not consent:
        raise HTTPException(400, "Debes confirmar que todas las personas involucradas están autorizadas.")
    if task not in {"face", "character"}:
        raise HTTPException(400, "task debe ser 'face' o 'character'")
    with jobs_lock:
        scan_data = scans.get(scan_id)
    if scan_data is None:
        raise HTTPException(400, "Escaneo no encontrado: vuelve a subir el video destino.")
    try:
        ids = sorted({int(x) for x in targets.split(",") if x.strip()})
    except ValueError:
        raise HTTPException(400, "targets debe ser una lista de ids separados por comas") from None
    people = {p["id"]: p for p in scan_data["people"]}
    if not ids or any(i not in people for i in ids):
        raise HTTPException(400, "Selecciona al menos una persona del escaneo.")
    if task == "character":
        if len(ids) != 1:
            raise HTTPException(400, "En character swap se cambia a una sola persona: selecciona solo una.")
        if char_model not in CHARACTER_MODELS:
            raise HTTPException(400, f"Modelo desconocido: {char_model}")
        ok, note = character_availability(char_model)
        if not ok:
            raise HTTPException(400, f"{CHARACTER_MODELS[char_model]['label']}: {note}")
    else:
        if method not in {"pretrained", "encoder"}:
            raise HTTPException(400, "method debe ser 'pretrained' o 'encoder'")
        if method == "pretrained" and model not in SWAP_MODELS:
            raise HTTPException(400, f"Modelo desconocido: {model}")

    job_id = uuid.uuid4().hex[:12]
    source_path = _save_upload(source, IMAGE_EXT | VIDEO_EXT, f"{job_id}_source")
    job_dir(job_id).mkdir(parents=True, exist_ok=True)
    params = {
        "task": task, "method": method, "model": model, "occlusion": occlusion, "region": region,
        "color": color, "smooth": smooth, "train_minutes": max(1, min(train_minutes, 240)),
        "char_model": char_model, "prompt": prompt.strip()[:500], "start": max(0.0, start),
        "duration": float(min(max(duration, 1.0), CHARACTER_MODELS.get(char_model, {}).get("max_seconds", 30))), "turbo": turbo, "hq": hq,
        "targets": np.stack([people[i]["embedding"] for i in ids]),
        "hf_token": hf_token.strip() or None,  # solo en memoria para este trabajo; nunca se devuelve
    }
    public = {k: v for k, v in params.items() if k not in ("targets", "hf_token")} | {"target_ids": ids}
    np.save(job_dir(job_id) / "targets.npy", params["targets"])
    with jobs_lock:
        jobs[job_id] = {
            "id": job_id, "status": "queued", "progress": 0.0, "message": "En cola", "params": public,
            "created": time.time(), "resumable": True,
            "_": {"source": str(source_path), "video": str(scan_data["video"]), "segments": [],
                  "processed": False, "trained_minutes": 0.0, "total_frames": 0},
        }
    save_job(job_id)
    executor.submit(run_job, job_id, source_path, scan_data["video"], params)
    return {"id": job_id}


@app.get("/api/jobs")
def list_jobs():
    with jobs_lock:
        items = [public_job(j) for j in jobs.values()]
    return sorted(items, key=lambda j: j.get("created", 0), reverse=True)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Trabajo no encontrado")
        return public_job(job)


@app.post("/api/jobs/{job_id}/resume")
def resume_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Trabajo no encontrado")
        if job["status"] in RUNNING:
            raise HTTPException(409, "El trabajo ya está en marcha.")
        if job["status"] == "done":
            raise HTTPException(409, "El trabajo ya terminó.")
        if not job.get("resumable"):
            raise HTTPException(400, "Faltan los archivos originales de este trabajo: no se puede reanudar.")
        st = job["_"]
        params = dict(job["params"]) | {
            "targets": np.load(job_dir(job_id) / "targets.npy"),
            "hf_token": None,  # el token nunca se guarda en disco
        }
        job.update(status="queued", progress=0.0, message="En cola para reanudar")
    save_job(job_id)
    executor.submit(run_job, job_id, Path(st["source"]), Path(st["video"]), params)
    return {"id": job_id}


partial_lock = threading.Lock()


@app.get("/api/jobs/{job_id}/partial")
def get_partial(job_id: str):
    """Une los tramos ya terminados (con su audio) en un video parcial. Sirve tanto para un
    trabajo interrumpido como para ver el avance de uno en marcha."""
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Trabajo no encontrado")
        st = job.get("_", {})
        segments = [job_dir(job_id) / "segments" / name for name, _ in st.get("segments", [])]
        video = Path(st.get("video") or "")
    segments = [s for s in segments if s.is_file()]
    if not segments:
        raise HTTPException(404, "Todavía no hay ningún tramo terminado.")
    out = job_dir(job_id) / "partial.mp4"
    with partial_lock:
        try:
            concat_segments(segments, out, video if video.is_file() else None)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"No se pudo unir lo generado: {exc}") from exc
    return FileResponse(out, media_type="video/mp4", filename=f"faceswap_{job_id}_parcial.mp4")


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Trabajo no encontrado")
        if job["status"] in RUNNING:
            raise HTTPException(409, "No se puede descartar un trabajo en marcha.")
        jobs.pop(job_id)
    shutil.rmtree(job_dir(job_id), ignore_errors=True)
    source = job.get("_", {}).get("source")
    if source and job["status"] != "done":  # la foto/video fuente subida solo servía para este trabajo
        Path(source).unlink(missing_ok=True)
    return {"ok": True}


@app.get("/api/jobs/{job_id}/result")
def get_result(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job.get("status") != "done":
        raise HTTPException(404, "Resultado no disponible")
    prefix = "characterswap" if job["params"].get("task") == "character" else "faceswap"
    return FileResponse(job["result"], media_type="video/mp4", filename=f"{prefix}_{job_id}.mp4")


load_jobs()
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
