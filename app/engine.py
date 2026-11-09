"""
Motor común a todos los métodos de face swap.

Para cada frame del video:
    1. SCRFD detecta caras y 5 landmarks                         (FaceEngine.detect)
    2. Los landmarks se suavizan en el tiempo (anti-temblor)      (KpsSmoother)
    3. Se alinea/recorta la cara a la plantilla del modelo        (warp_face)
    4. El swapper elegido genera la cara nueva                    (swappers.py / encoder/)
    5. Se construye una máscara suave = caja ∩ oclusión ∩ región  (Masker)
    6. Se iguala el color de la cara nueva al de la original      (color_transfer)
    7. Se devuelve al frame con la transformación inversa         (paste_back)
"""
from __future__ import annotations

import os
import queue
import subprocess
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Protocol

import cv2
import numpy as np
import onnxruntime as ort

from .downloads import StatusCb, download_and_extract_zip, download_file
from .models_registry import BUFFALO_URLS, INSIGHTFACE_ROOT, OCCLUDER, PARSER, WARP_TEMPLATES

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
WATERMARK_TEXT = "CONTENIDO SINTETICO (IA)"


# --------------------------------------------------------------------------- utilidades
def get_providers() -> list[str]:
    """GPU (CUDA) si onnxruntime-gpu funciona; si no, CPU."""
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception:  # noqa: BLE001
            pass
    if "CUDAExecutionProvider" in ort.get_available_providers():
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def imread_unicode(path: str | Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {path}")
    return img


def warp_face(frame: np.ndarray, kps: np.ndarray, template: str, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Transformación de similitud (rotación+escala+traslación) que lleva los 5 landmarks a la plantilla."""
    dst = np.asarray(WARP_TEMPLATES[template], dtype=np.float32) * size
    M = cv2.estimateAffinePartial2D(kps.astype(np.float32), dst, method=cv2.RANSAC, ransacReprojThreshold=100)[0]
    crop = cv2.warpAffine(frame, M, (size, size), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA)
    return crop, M


def paste_back(frame: np.ndarray, crop: np.ndarray, mask: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Devuelve la cara generada a su sitio con M⁻¹, trabajando solo en la región afectada."""
    h, w = frame.shape[:2]
    size = crop.shape[0]
    IM = cv2.invertAffineTransform(M)
    corners = np.array([[[0, 0], [size, 0], [0, size], [size, size]]], dtype=np.float32)
    pts = cv2.transform(corners, IM)[0]
    x1, y1 = np.maximum(np.floor(pts.min(0)).astype(int), 0)
    x2, y2 = np.minimum(np.ceil(pts.max(0)).astype(int), [w, h])
    if x2 <= x1 or y2 <= y1:
        return frame
    IM[:, 2] -= (x1, y1)
    pw, ph = x2 - x1, y2 - y1
    inv_crop = cv2.warpAffine(crop, IM, (pw, ph), borderMode=cv2.BORDER_REPLICATE)
    inv_mask = cv2.warpAffine(mask, IM, (pw, ph)).clip(0, 1)[..., None]
    roi = frame[y1:y2, x1:x2].astype(np.float32)
    frame[y1:y2, x1:x2] = (roi * (1 - inv_mask) + inv_crop * inv_mask).clip(0, 255).astype(np.uint8)
    return frame


def color_transfer(swapped: np.ndarray, original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Iguala media y desviación de cada canal LAB de la cara nueva a la original (solo dentro
    de la cara). Corrige el tono de piel y la luz para que no se note el 'parche'."""
    sel = mask > 0.5
    if sel.sum() < 100:
        return swapped
    src = cv2.cvtColor(swapped.clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
    s_mean, s_std = src[sel].mean(0), src[sel].std(0) + 1e-6
    r_mean, r_std = ref[sel].mean(0), ref[sel].std(0) + 1e-6
    # Luminosidad: ratio acotado para no exagerar contrastes. Color (A/B): puede bajar hasta 0,
    # así una fuente en color sobre un video en blanco y negro queda bien desaturada.
    ratio = np.clip(r_std / s_std, [0.6, 0.0, 0.0], [1.6, 1.6, 1.6])
    out = (src - s_mean) * ratio + r_mean
    return cv2.cvtColor(out.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)


def add_watermark(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = max(0.4, w / 1600)
    thickness = max(1, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(WATERMARK_TEXT, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    pad = int(10 * scale) + 4
    x, y = pad, h - pad
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - pad // 2, y - th - pad // 2), (x + tw + pad // 2, y + pad // 2), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    cv2.putText(frame, WATERMARK_TEXT, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


# --------------------------------------------------------------------------- detección
@dataclass
class Detection:
    bbox: np.ndarray  # x1, y1, x2, y2
    kps: np.ndarray   # 5×2
    score: float
    target: int = 0   # índice de la persona elegida en el escaneo a la que corresponde (TargetMatcher)

    @property
    def size(self) -> float:
        return float(max(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]))

    @property
    def area(self) -> float:
        return float((self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1]))


@dataclass
class Identity:
    embedding: np.ndarray  # embedding ArcFace "crudo" (promedio si hay varios frames)

    @property
    def normed(self) -> np.ndarray:
        return self.embedding / np.linalg.norm(self.embedding)


class FaceEngine:
    def __init__(self, status_cb: StatusCb = None, det_size: tuple[int, int] = (640, 640)):
        from insightface.app import FaceAnalysis

        download_and_extract_zip(BUFFALO_URLS, INSIGHTFACE_ROOT / "models" / "buffalo_l", "buffalo_l", status_cb)
        if status_cb:
            status_cb("Cargando detector y ArcFace en la GPU...", 0.0)
        self.providers = get_providers()
        ctx_id = 0 if "CUDAExecutionProvider" in self.providers else -1
        self.analyzer = FaceAnalysis(
            name="buffalo_l", root=str(INSIGHTFACE_ROOT),
            allowed_modules=["detection", "recognition"], providers=self.providers,
        )
        self.analyzer.prepare(ctx_id=ctx_id, det_size=det_size)
        self.detector = self.analyzer.models["detection"]
        self.recognizer = self.analyzer.models["recognition"]
        self._sessions: dict[str, ort.InferenceSession] = {}
        self._swap_session: tuple[str, ort.InferenceSession] | None = None

    def _new_session(self, path: str) -> ort.InferenceSession:
        providers: list = list(self.providers)
        if providers[0] == "CUDAExecutionProvider":
            # búsqueda exhaustiva del mejor algoritmo de convolución + layout NHWC (Tensor Cores);
            # la memoria crece solo lo necesario (con 6 GB compartidos con PyTorch, importa)
            providers[0] = ("CUDAExecutionProvider", {
                "cudnn_conv_algo_search": "EXHAUSTIVE", "prefer_nhwc": "1",
                "arena_extend_strategy": "kSameAsRequested",
            })
        return ort.InferenceSession(path, providers=providers)

    def release_swapper(self) -> None:
        """Libera de la GPU el modelo de swap (cientos de MB) entre trabajos."""
        self._swap_session = None
        import gc

        gc.collect()

    # Sesiones ONNX: las auxiliares se quedan en memoria; del swapper solo la última usada
    def session(self, path: Path, keep: bool = True) -> ort.InferenceSession:
        key = str(path)
        if keep:
            if key not in self._sessions:
                self._sessions[key] = self._new_session(key)
            return self._sessions[key]
        if self._swap_session is None or self._swap_session[0] != key:
            self._swap_session = None
            self._swap_session = (key, self._new_session(key))
        return self._swap_session[1]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        bboxes, kpss = self.detector.detect(frame, max_num=0, metric="default")
        if kpss is None:
            return []
        return [Detection(bboxes[i, :4], kpss[i], float(bboxes[i, 4])) for i in range(bboxes.shape[0])]

    def embed(self, frame: np.ndarray, faces: list[Detection]) -> np.ndarray:
        """Embeddings ArcFace normalizados (N×512) de varias caras de un mismo frame, en un solo lote."""
        if not faces:
            return np.zeros((0, 512), np.float32)
        from insightface.utils import face_align

        size = self.recognizer.input_size[0]
        crops = [face_align.norm_crop(frame, landmark=d.kps, image_size=size) for d in faces]
        emb = np.asarray(self.recognizer.get_feat(crops), np.float32).reshape(len(faces), -1)
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)

    def largest(self, frame: np.ndarray) -> Optional[Detection]:
        faces = self.detect(frame)
        return max(faces, key=lambda d: d.area) if faces else None

    def identity_from(self, path: Path, status_cb: StatusCb = None, max_frames: int = 24) -> Identity:
        """Embedding de identidad. Si la fuente es un video, promedia varios frames:
        da una identidad más estable que una sola foto."""
        if path.suffix.lower() in IMAGE_EXT:
            frames = [imread_unicode(path)]
        else:
            frames = list(sample_frames(path, max_frames))
        embeddings = []
        for i, frame in enumerate(frames):
            faces = self.analyzer.get(frame)
            if faces:
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                embeddings.append(face.embedding)
            if status_cb and len(frames) > 1:
                status_cb(f"Calculando identidad de la fuente ({i + 1}/{len(frames)})...", (i + 1) / len(frames))
        if not embeddings:
            raise ValueError("No se detectó ninguna cara en la imagen/video de origen.")
        # promedio de vectores normalizados, re-escalado a la norma típica del embedding crudo
        normed = np.stack([e / np.linalg.norm(e) for e in embeddings]).mean(0)
        norm = float(np.mean([np.linalg.norm(e) for e in embeddings]))
        return Identity(embedding=(normed / np.linalg.norm(normed) * norm).astype(np.float32))


def sample_frames(video: Path, count: int):
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or count
    for idx in np.linspace(0, max(total - 1, 0), num=min(count, total), dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            yield frame
    cap.release()


# --------------------------------------------------------------------------- máscaras
class Masker:
    """Máscara final = caja difuminada ∩ oclusión (XSeg) ∩ región facial (BiSeNet)."""

    def __init__(self, engine: FaceEngine, occlusion: bool, region: bool, status_cb: StatusCb = None):
        self.engine = engine
        self.occ = None
        self.parser = None
        if occlusion:
            download_file(OCCLUDER["urls"], OCCLUDER["file"], "XSeg (oclusión)", status_cb)
            self.occ = engine.session(OCCLUDER["file"])
        if region:
            download_file(PARSER["urls"], PARSER["file"], "BiSeNet (segmentación)", status_cb)
            self.parser = engine.session(PARSER["file"])
        self._box_cache: dict[int, np.ndarray] = {}

    def box(self, size: int, blur: float = 0.3) -> np.ndarray:
        if size not in self._box_cache:
            blur_amount = int(size * 0.5 * blur)
            area = max(blur_amount // 2, 1)
            m = np.ones((size, size), np.float32)
            m[:area, :] = 0
            m[-area:, :] = 0
            m[:, :area] = 0
            m[:, -area:] = 0
            if blur_amount > 0:
                m = cv2.GaussianBlur(m, (0, 0), blur_amount * 0.25)
            self._box_cache[size] = m
        return self._box_cache[size]

    @staticmethod
    def _soften(mask: np.ndarray) -> np.ndarray:
        return (cv2.GaussianBlur(mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2

    def occlusion(self, crop: np.ndarray) -> np.ndarray:
        size = OCCLUDER["size"]
        x = cv2.resize(crop, (size, size))[None].astype(np.float32) / 255.0  # NHWC, BGR
        out = self.occ.run(None, {self.occ.get_inputs()[0].name: x})[0][0]
        out = np.squeeze(out).clip(0, 1).astype(np.float32)
        return self._soften(cv2.resize(out, crop.shape[:2][::-1]))

    def region(self, crop: np.ndarray) -> np.ndarray:
        size = PARSER["size"]
        x = cv2.resize(crop.clip(0, 255).astype(np.uint8), (size, size))[:, :, ::-1].astype(np.float32) / 255.0
        x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        x = x.transpose(2, 0, 1)[None]
        out = self.parser.run([self.parser.get_outputs()[0].name], {self.parser.get_inputs()[0].name: x})[0][0]
        out = out[:, ::2, ::2]  # 19×256×256: argmax 4× más barato y la máscara final es ≤256 px
        mask = np.isin(out.argmax(0), PARSER["regions"]).astype(np.float32)
        return self._soften(cv2.resize(mask, crop.shape[:2][::-1]))

    def build(self, original_crop: np.ndarray, swapped_crop: np.ndarray, region_on_original: bool = False) -> np.ndarray:
        masks = [self.box(original_crop.shape[0])]
        if self.occ is not None:
            masks.append(self.occlusion(original_crop))
        if self.parser is not None:
            # Por defecto sobre la cara generada (como FaceFusion). El encoder propio puede generar
            # caras borrosas al principio del entrenamiento que BiSeNet no reconoce: ahí se usa la original.
            masks.append(self.region(original_crop if region_on_original else swapped_crop))
        return np.minimum.reduce(masks).clip(0, 1)


# --------------------------------------------------------------------------- suavizado temporal
class KpsSmoother:
    """Filtro adaptativo sobre los 5 landmarks de cada cara: si la cara casi no se mueve,
    promedia mucho (elimina el temblor del detector); si se mueve rápido, sigue al detector
    (sin retraso). Asocia caras entre frames por cercanía del centro."""

    def __init__(self, min_alpha: float = 0.35):
        self.min_alpha = min_alpha
        self.tracks: list[np.ndarray] = []

    def __call__(self, detections: list[Detection]) -> list[Detection]:
        new_tracks = []
        for det in detections:
            center = det.kps.mean(0)
            best, best_d = None, det.size * 0.5
            for prev in self.tracks:
                d = float(np.linalg.norm(prev.mean(0) - center))
                if d < best_d:
                    best, best_d = prev, d
            if best is not None:
                disp = float(np.abs(det.kps - best).mean())
                alpha = float(np.clip(self.min_alpha + disp / (0.03 * det.size), self.min_alpha, 1.0))
                det.kps = alpha * det.kps + (1 - alpha) * best
            new_tracks.append(det.kps.copy())
        self.tracks = new_tracks
        return detections


# --------------------------------------------------------------------------- pipeline de video
class Swapper(Protocol):
    size: int
    template: str

    def swap(self, crop: np.ndarray) -> np.ndarray: ...  # uint8 BGR -> float32 BGR 0..255


@dataclass
class PipelineOptions:
    mode: str = "largest"      # "largest" | "all" | "selected"
    color: bool = True
    smooth: bool = True
    watermark: bool = True
    targets: Optional[np.ndarray] = None  # K×512: identidades elegidas en el escaneo (modo "selected")


def swap_frame(frame, faces, swapper: Swapper, masker: Masker, opts: PipelineOptions) -> np.ndarray:
    for det in faces:
        crop, M = warp_face(frame, det.kps, swapper.template, swapper.size)
        # con varias fuentes, cada persona elegida se cambia por la fuente que se le asignó
        fake = swapper.swap(crop, det.target) if getattr(swapper, "multi", False) else swapper.swap(crop)
        mask = masker.build(crop, fake, getattr(swapper, "region_on_original", False))
        if opts.color:
            fake = color_transfer(fake, crop, mask)
        frame = paste_back(frame, fake, mask, M)
    return frame


SEGMENT_SECONDS = 10.0  # el video se escribe en tramos independientes de ~10 s (ver SegmentWriter)
RESUME_WARMUP = 8       # frames previos que se vuelven a analizar al reanudar (suavizado y seguimiento)


def segment_frames(fps: float) -> int:
    return max(1, int(round(fps * SEGMENT_SECONDS)))


def process_video(
    engine: FaceEngine, swapper: Swapper, masker: Masker, opts: PipelineOptions,
    video_in: Path, seg_dir: Path, progress_cb: Optional[Callable[[int, int, float], None]] = None,
    start_frame: int = 0, on_segment: Optional[Callable[[str, int], None]] = None,
) -> int:
    """Procesa el video desde `start_frame` y lo escribe en tramos dentro de `seg_dir`
    (ver SegmentWriter). `start_frame` debe ser un límite de tramo (el total de frames de los
    tramos ya terminados). Devuelve el número total de frames del video procesado."""
    cap = cv2.VideoCapture(str(video_in))
    if not cap.isOpened():
        raise ValueError("No se pudo abrir el video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    seg_len = segment_frames(fps)
    if start_frame % seg_len:
        raise ValueError(f"start_frame={start_frame} no es un límite de tramo ({seg_len} frames)")

    # Hilo lector: decodifica frames mientras la GPU procesa los anteriores. Al reanudar, los
    # frames ya hechos se saltan con grab() (exacto, a diferencia de buscar por posición), salvo
    # los últimos RESUME_WARMUP, que se analizan sin escribirse para que el suavizado y el
    # seguimiento de las personas lleguen al punto de reanudación en el mismo estado.
    frames: queue.Queue = queue.Queue(maxsize=16)
    stop = threading.Event()

    def reader():
        i, skip = 0, max(0, start_frame - RESUME_WARMUP)
        while i < skip and not stop.is_set():
            if not cap.grab():
                frames.put(None)
                return
            i += 1
        while not stop.is_set():
            ok, frame = cap.read()
            if not ok:
                frames.put(None)
                return
            frames.put((i, frame))
            i += 1

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()

    smoother = KpsSmoother() if opts.smooth else None
    matcher = None
    if opts.mode == "selected":
        from .scan import TargetMatcher

        matcher = TargetMatcher(engine, opts.targets)
    writer: Optional[SegmentWriter] = None
    started, idx = time.time(), start_frame

    def work(frame, faces):
        frame = swap_frame(frame, faces, swapper, masker, opts)
        return add_watermark(frame) if opts.watermark else frame

    # La detección y el suavizado van en orden (dependen del frame anterior); el swap + máscaras
    # de varios frames se solapan en 2 hilos: mientras la GPU procesa uno, la CPU prepara otro.
    pending: deque[Future] = deque()
    pool = ThreadPoolExecutor(max_workers=2)

    def flush(limit: int):
        nonlocal idx
        while len(pending) > limit:
            writer.write(pending.popleft().result())
            idx += 1
            if progress_cb:
                progress_cb(idx, max(total, idx), (idx - start_frame) / max(time.time() - started, 1e-6))

    ok = False
    try:
        while (item := frames.get()) is not None:
            i, frame = item
            faces = engine.detect(frame)
            if matcher is not None:
                faces = matcher(frame, faces)
            elif opts.mode != "all" and faces:
                faces = [max(faces, key=lambda d: d.area)]
            if smoother is not None:
                faces = smoother(faces)
            if i < start_frame:
                continue  # calentamiento: solo actualiza el estado del suavizado y del seguimiento
            if writer is None:
                h, w = frame.shape[:2]
                writer = SegmentWriter(seg_dir, w, h, fps, seg_len, start_frame // seg_len, on_segment)
                started = time.time()
            pending.append(pool.submit(work, frame, faces))
            flush(limit=4)
        if writer is not None:
            flush(limit=0)
        ok = True
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        stop.set()
        while True:  # vacía la cola para que el lector no se quede bloqueado y suelte el video
            try:
                frames.get_nowait()
            except queue.Empty:
                break
        reader_thread.join(timeout=5)
        if writer is not None:
            if ok:
                writer.close()  # cierra (y da por terminado) el último tramo, aunque sea más corto
            else:
                writer.abort()
        cap.release()
    if writer is None and start_frame == 0:
        raise ValueError("El video no tiene frames legibles.")
    return idx


class SegmentWriter:
    """Escribe el video en tramos de `seg_len` frames: seg_00000.mp4, seg_00001.mp4...

    Cada tramo es un MP4 completo e independiente (solo video). Mientras se escribe se llama
    `seg_XXXXX.part.mp4`; al cerrarse bien se renombra y se avisa con on_segment(nombre, frames).
    Si el proceso muere a mitad, solo se pierde el tramo en curso (su .part.mp4 queda corrupto
    y se descarta al reanudar). Al final, concat_segments() une los tramos y añade el audio."""

    def __init__(self, seg_dir: Path, w: int, h: int, fps: float, seg_len: int, first_index: int,
                 on_segment: Optional[Callable[[str, int], None]] = None):
        self.dir, self.w, self.h, self.fps, self.seg_len = seg_dir, w, h, fps, seg_len
        self.index, self.on_segment = first_index, on_segment
        self.cur: Optional[FfmpegWriter] = None
        self.n = 0
        seg_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _part(self) -> Path:
        return self.dir / f"seg_{self.index:05d}.part.mp4"

    def write(self, frame: np.ndarray) -> None:
        if self.cur is None:
            self.cur = FfmpegWriter(self._part, None, self.w, self.h, self.fps)
        self.cur.write(frame)
        self.n += 1
        if self.n >= self.seg_len:
            self._finish()

    def _finish(self) -> None:
        cur, self.cur = self.cur, None
        cur.close()
        name = f"seg_{self.index:05d}.mp4"
        os.replace(self._part, self.dir / name)
        if self.on_segment:
            self.on_segment(name, self.n)
        self.index, self.n = self.index + 1, 0

    def close(self) -> None:
        if self.cur is not None:
            self._finish()

    def abort(self) -> None:
        cur, self.cur = self.cur, None
        if cur is not None:
            try:
                cur.close()
            except Exception:  # noqa: BLE001
                pass
            self._part.unlink(missing_ok=True)


def concat_segments(segments: list[Path], out: Path, audio_src: Optional[Path]) -> Path:
    """Une los tramos sin recodificar (copia el H.264) y añade el audio original, recortado a la
    duración del video (sirve también para un resultado parcial). Escribe a un archivo temporal y
    lo renombra al final: si se corta a mitad, `out` no queda corrupto."""
    import imageio_ffmpeg

    if not segments:
        raise ValueError("No hay tramos que unir.")
    seg_dir = segments[0].parent
    listing = out.with_name(out.stem + ".concat.txt")
    lines = []
    for s in segments:
        p = s.resolve().as_posix().replace("'", r"'\''")
        lines.append(f"file '{p}'")
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp = out.with_name(out.stem + ".tmp.mp4")
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "concat", "-safe", "0", "-i", str(listing)]
    if audio_src is not None:
        cmd += ["-i", str(audio_src), "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac", "-shortest"]
    else:
        cmd += ["-c", "copy"]
    cmd += ["-movflags", "+faststart", str(tmp)]
    try:
        proc = subprocess.run(cmd, capture_output=True, cwd=seg_dir)
        if proc.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg falló al unir los tramos: {proc.stderr.decode(errors='replace').strip()}")
        os.replace(tmp, out)
    finally:
        listing.unlink(missing_ok=True)
    return out


class FfmpegWriter:
    """Envía los frames crudos a ffmpeg, que codifica H.264 y añade el audio original
    en un solo paso (sin archivo intermedio)."""

    def __init__(self, out: Path, audio_src: Optional[Path], w: int, h: int, fps: float,
                 audio_start: float = 0.0, crf: int = 17):
        import imageio_ffmpeg

        self.log = tempfile.TemporaryFile()
        cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-",
        ]
        if audio_src is not None:  # audio original (desde audio_start: el tramo procesado)
            cmd += ["-ss", f"{audio_start:.3f}", "-i", str(audio_src), "-map", "0:v:0", "-map", "1:a:0?"]
        cmd += [
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
        ]
        if audio_src is not None:
            cmd += ["-c:a", "aac", "-shortest"]
        cmd += ["-movflags", "+faststart", str(out)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self.log)

    def write(self, frame: np.ndarray) -> None:
        try:
            self.proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError):
            self.close()
            raise

    def close(self) -> None:
        if self.proc.stdin and not self.proc.stdin.closed:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
        code = self.proc.wait()
        if code != 0:
            self.log.seek(0)
            raise RuntimeError(f"ffmpeg falló: {self.log.read().decode(errors='replace').strip()}")
