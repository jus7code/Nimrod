"""
API web (FastAPI) del laboratorio de face swap / character swap.

    GET  /api/models            -> catálogo de modelos (face swap y character swap)
    POST /api/sources           -> sube una o varias fuentes (personas autorizadas) y detecta su cara
    POST /api/scan              -> sube el video destino, detecta y agrupa los rostros (para elegir a quién cambiar)
    POST /api/jobs              -> sube la fuente y crea un trabajo sobre un video escaneado
    GET  /api/jobs              -> lista de trabajos (incluye los interrumpidos)
    GET  /api/jobs/{id}         -> estado y progreso
    GET  /api/jobs/{id}/result  -> video resultante
    GET  /api/jobs/{id}/partial -> lo generado hasta ahora (face swap), con audio
    POST /api/jobs/{id}/resume  -> reanuda un trabajo interrumpido
    DELETE /api/jobs/{id}       -> descarta un trabajo que no está en marcha
    GET  /api/health            -> GPU / estado

    Buscador de videos sugeridos (app/scraper.py):
    POST /api/scrapes           -> analiza una página web: rostros de las miniaturas de sus videos
    GET  /api/scrapes           -> páginas ya analizadas (índice de rostros guardado en data/scrape)
    GET  /api/scrapes/{id}      -> progreso de un análisis
    DELETE /api/scrapes/{id}    -> borra un análisis del índice
    POST /api/similar           -> videos del índice cuyos rostros se parecen a las fuentes elegidas

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
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
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
SCRAPE = BASE_DIR / "data" / "scrape"
for d in (UPLOADS, OUTPUTS, WORK):
    d.mkdir(parents=True, exist_ok=True)

WATERMARK = os.getenv("FACESWAP_WATERMARK", "1") != "0"

app = FastAPI(title="FaceSwap Lab")
executor = ThreadPoolExecutor(max_workers=1)  # una GPU -> un trabajo a la vez
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
scans: OrderedDict[str, dict] = OrderedDict()  # scan_id -> {"video": Path, "people": [... con embedding]}
MAX_SCANS = 20
sources: OrderedDict[str, dict] = OrderedDict()  # source_id -> {"path", "embedding" (normalizado), "thumb", "name"}
MAX_SOURCES = 50

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


def job_sources(job: dict) -> list[str]:
    """Rutas de las fuentes de un trabajo (los trabajos antiguos guardaban una sola, en "source")."""
    st = job.get("_", {})
    return list(st.get("sources") or ([st["source"]] if st.get("source") else [""]))


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
        job["resumable"] = (all(Path(p).is_file() for p in job_sources(job) + [st.get("video") or ""])
                            and (job_dir(job["id"]) / "targets.npy").is_file())
        jobs[job["id"]] = job
        if job["status"] == "interrupted":
            save_job(job["id"])


def run_job(job_id: str, source_paths: list[Path], video: Path, params: dict) -> None:
    def status(msg: str, frac: float = 0.0, phase: str | None = None):
        fields = {"message": msg, "progress": frac}
        if phase:
            fields["status"] = phase
        update_job(job_id, **fields)

    with jobs_lock:
        st = jobs[job_id]["_"]  # estado de reanudación (se modifica en sitio y se guarda con save_job)
    targets: np.ndarray = params["targets"]
    # índice (en source_paths) de la fuente asignada a cada persona elegida, en el orden de targets
    target_sources: list[int] = params.get("target_sources") or [0] * len(targets)
    source = source_paths[target_sources[0]]
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

            identities = {}
            for n, s in enumerate(sorted(set(target_sources))):
                if len(set(target_sources)) > 1:
                    status(f"Calculando identidad de la fuente {s + 1} ({n + 1}/{len(set(target_sources))})...", 0)
                identities[s] = engine.identity_from(source_paths[s], status)
            swapper = OnnxSwapper(engine, params["model"], [identities[s] for s in target_sources], status)

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


def _register_source(path: Path, name: str) -> dict:
    """Detecta la cara de la fuente, calcula su identidad y la guarda (en memoria y en disco, para
    que sobreviva a un reinicio del servidor). Lanza ValueError si no hay ninguna cara."""
    from .scan import _jpeg_data_uri, face_thumb
    from .engine import imread_unicode, sample_frames

    engine = get_engine()
    identity = engine.identity_from(path)
    frame = imread_unicode(path) if path.suffix.lower() in IMAGE_EXT else next(iter(sample_frames(path, 1)), None)
    det = engine.largest(frame) if frame is not None else None
    thumb = _jpeg_data_uri(face_thumb(frame, det.bbox)) if det is not None else ""
    sid = path.stem.removeprefix("src_")
    np.save(path.with_suffix(".npy"), identity.normed.astype(np.float32))
    path.with_suffix(".json").write_text(json.dumps({"name": name, "thumb": thumb}), encoding="utf-8")
    return {"id": sid, "path": path, "embedding": identity.normed.astype(np.float32), "thumb": thumb, "name": name}


def get_source(sid: str) -> dict:
    """Fuente subida con /api/sources (si el servidor se reinició, se recupera del disco)."""
    with jobs_lock:
        src = sources.get(sid)
    if src is not None:
        return src
    if not sid.isalnum():
        raise HTTPException(400, "Id de fuente no válido.")
    files = [f for f in UPLOADS.glob(f"src_{sid}.*") if f.suffix.lower() in IMAGE_EXT | VIDEO_EXT]
    emb = UPLOADS / f"src_{sid}.npy"
    if not files or not emb.is_file():
        raise HTTPException(400, "Fuente no encontrada: vuelve a subirla.")
    meta = {}
    try:
        meta = json.loads((UPLOADS / f"src_{sid}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    src = {"id": sid, "path": files[0], "embedding": np.load(emb), "thumb": meta.get("thumb", ""),
           "name": meta.get("name", files[0].name)}
    with jobs_lock:
        sources[sid] = src
    return src


@app.post("/api/sources")
def upload_sources(files: list[UploadFile] = File(...)):
    """Sube una o varias fuentes (foto o video de cada persona autorizada). Devuelve, por archivo,
    la miniatura de la cara detectada o el error."""
    out = []
    for upload in files:
        sid = uuid.uuid4().hex[:12]
        name = Path(upload.filename or "").name
        try:
            path = _save_upload(upload, IMAGE_EXT | VIDEO_EXT, f"src_{sid}")
        except HTTPException as exc:
            out.append({"name": name, "error": exc.detail})
            continue
        try:
            src = _register_source(path, name)
        except Exception as exc:  # noqa: BLE001
            path.unlink(missing_ok=True)
            out.append({"name": name, "error": str(exc)})
            continue
        with jobs_lock:
            sources[sid] = src
            while len(sources) > MAX_SOURCES:
                sources.popitem(last=False)
        out.append({"id": sid, "name": name, "thumb": src["thumb"],
                    "kind": "video" if path.suffix.lower() in VIDEO_EXT else "image"})
    return out


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
    scan_id: str = Form(...),
    # asignación persona del escaneo -> fuente subida con /api/sources: "persona:fuente,persona:fuente"
    targets: str = Form(...),
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
    mapping: dict[int, str] = {}
    try:
        for pair in filter(None, (x.strip() for x in targets.split(","))):
            person, sid = pair.split(":")
            mapping[int(person)] = sid.strip()
    except ValueError:
        raise HTTPException(400, "targets debe tener la forma 'persona:fuente,persona:fuente'") from None
    ids = sorted(mapping)
    people = {p["id"]: p for p in scan_data["people"]}
    if not ids or any(i not in people for i in ids):
        raise HTTPException(400, "Asigna una fuente al menos a una persona del escaneo.")
    source_ids = sorted(set(mapping.values()), key=list(mapping.values()).index)
    srcs = [get_source(sid) for sid in source_ids]
    target_sources = [source_ids.index(mapping[i]) for i in ids]
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
        if method == "encoder" and len(source_ids) > 1:
            raise HTTPException(400, "El encoder propio se entrena para una sola fuente: asigna la misma "
                                     "fuente a todas las personas o usa un modelo pre-entrenado.")

    job_id = uuid.uuid4().hex[:12]
    job_dir(job_id).mkdir(parents=True, exist_ok=True)
    params = {
        "task": task, "method": method, "model": model, "occlusion": occlusion, "region": region,
        "color": color, "smooth": smooth, "train_minutes": max(1, min(train_minutes, 240)),
        "char_model": char_model, "prompt": prompt.strip()[:500], "start": max(0.0, start),
        "duration": float(min(max(duration, 1.0), CHARACTER_MODELS.get(char_model, {}).get("max_seconds", 30))), "turbo": turbo, "hq": hq,
        "targets": np.stack([people[i]["embedding"] for i in ids]),
        "target_sources": target_sources,
        "hf_token": hf_token.strip() or None,  # solo en memoria para este trabajo; nunca se devuelve
    }
    public = {k: v for k, v in params.items() if k not in ("targets", "hf_token")} | {"target_ids": ids, "source_ids": source_ids}
    np.save(job_dir(job_id) / "targets.npy", params["targets"])
    with jobs_lock:
        jobs[job_id] = {
            "id": job_id, "status": "queued", "progress": 0.0, "message": "En cola", "params": public,
            "created": time.time(), "resumable": True,
            "_": {"sources": [str(s["path"]) for s in srcs], "video": str(scan_data["video"]), "segments": [],
                  "processed": False, "trained_minutes": 0.0, "total_frames": 0},
        }
    save_job(job_id)
    executor.submit(run_job, job_id, [s["path"] for s in srcs], scan_data["video"], params)
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
    executor.submit(run_job, job_id, [Path(p) for p in job_sources(job)], Path(st["video"]), params)
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
    source = job.get("_", {}).get("source")  # trabajos antiguos: la fuente solo servía para ese trabajo
    if source and job["status"] != "done":    # (las de /api/sources se pueden reutilizar: se conservan)
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


# ------------------------------------------------------------------ buscador de videos sugeridos
scrape_executor = ThreadPoolExecutor(max_workers=1)  # un análisis de página a la vez
_scrape_index = None


def scrape_index():
    global _scrape_index
    if _scrape_index is None:
        from .scraper import ScrapeIndex

        _scrape_index = ScrapeIndex(SCRAPE)
    return _scrape_index


@app.post("/api/scrapes")
def start_scrape(url: str = Body(...), deep: bool = Body(False), max_pages: int = Body(1),
                 max_videos: int = Body(150)):
    """Analiza una página con videos: guarda en el índice los rostros de sus miniaturas."""
    from urllib.parse import urlparse

    from .scraper import analyze_page, new_scrape_id

    url = url.strip()
    if urlparse(url).scheme not in ("http", "https") or not urlparse(url).netloc:
        raise HTTPException(400, "Introduce un enlace completo (https://…).")
    sid = new_scrape_id()
    index = scrape_index()
    index.set_progress(sid, url=url, created=time.time(), status="queued", message="En cola", progress=0.0,
                       n_videos=0, n_faces=0, n_images=0, title="")
    scrape_executor.submit(analyze_page, get_engine(), index, sid, url, deep,
                           max(1, min(max_pages, 10)), max(1, min(max_videos, 500)))
    return {"id": sid}


@app.get("/api/scrapes")
def list_scrapes():
    return scrape_index().list()


@app.get("/api/scrapes/{sid}")
def get_scrape(sid: str):
    item = scrape_index().get(sid)
    if item is None:
        raise HTTPException(404, "Análisis no encontrado")
    return item


@app.delete("/api/scrapes/{sid}")
def delete_scrape(sid: str):
    if not scrape_index().delete(sid):
        raise HTTPException(404, "Análisis no encontrado (o todavía en marcha)")
    return {"ok": True}


@app.post("/api/similar")
def similar(source_ids: list[str] = Body(...), scrape_ids: list[str] | None = Body(None),
            min_sim: float = Body(0.0), limit: int = Body(200)):
    """Videos analizados cuyos rostros se parecen a las fuentes elegidas, del más parecido al menos."""
    from .scraper import search

    if not source_ids:
        raise HTTPException(400, "Elige al menos una fuente (modelo) para buscar parecidos.")
    models = [(sid, get_source(sid)["embedding"]) for sid in source_ids]
    return search(scrape_index(), models, scrape_ids or None, min_sim, max(1, min(limit, 500)))


load_jobs()
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
