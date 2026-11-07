"""
Character swap de principio a fin:

    1. Referencia: cuerpo de la persona autorizada sobre fondo blanco + primer plano de su cara.
    2. Seguimiento: en el tramo elegido (a 16 fps) se localiza a la persona seleccionada en el
       escaneo y se obtiene su máscara en cada frame.
    3. Recorte: una caja FIJA que contiene a la persona en todo el tramo, con la proporción del
       modelo (480×832 vertical u 832×480 horizontal). Así el generador dedica todos sus píxeles
       a la persona, y el resto del frame (otras personas, fondo) no se toca.
    4. Generación con el modelo elegido (backends.py).
    5. Composición: la persona generada vuelve al frame original con la transformación inversa,
       mezclada con una máscara suave = (silueta original ∪ silueta generada) dilatada y difuminada.
       Fuera de esa máscara el video queda idéntico al original.
    6. ffmpeg: H.264 + el audio original del tramo + marca de contenido sintético.
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..downloads import StatusCb
from ..engine import FaceEngine, FfmpegWriter, add_watermark
from ..scan import video_info
from .backends import ClipInputs, make_backend
from .persons import PersonSegmenter, TrackedFrame, iter_frames, prepare_reference, track_target


def crop_transform(frames: list[TrackedFrame], frame_wh: tuple[int, int], out_wh: tuple[int, int],
                   margin: float = 0.10) -> np.ndarray:
    """Matriz afín (2×3) que lleva la caja que envuelve a la persona en todo el tramo al tamaño del
    modelo, conservando la proporción (sin deformar)."""
    boxes = np.stack([f.bbox for f in frames if f.bbox is not None])
    x1, y1 = boxes[:, :2].min(0)
    x2, y2 = boxes[:, 2:].max(0)
    bw, bh = (x2 - x1) * (1 + 2 * margin), (y2 - y1) * (1 + 2 * margin)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    W, H = out_wh
    if bw / bh < W / H:
        bw = bh * W / H
    else:
        bh = bw * H / W
    # si cabe, se desplaza la caja para no salirse del frame (menos relleno replicado en los bordes)
    fw, fh = frame_wh
    if bw <= fw:
        cx = float(np.clip(cx, bw / 2, fw - bw / 2))
    if bh <= fh:
        cy = float(np.clip(cy, bh / 2, fh - bh / 2))
    s = W / bw
    return np.float32([[s, 0, W / 2 - cx * s], [0, s, H / 2 - cy * s]])


def paste(frame: np.ndarray, crop: np.ndarray, alpha: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Devuelve `crop` (W×H del modelo) al frame con M⁻¹, trabajando solo en la región afectada."""
    h, w = frame.shape[:2]
    H, W = crop.shape[:2]
    IM = cv2.invertAffineTransform(M)
    pts = cv2.transform(np.float32([[[0, 0], [W, 0], [0, H], [W, H]]]), IM)[0]
    x1, y1 = np.maximum(np.floor(pts.min(0)).astype(int), 0)
    x2, y2 = np.minimum(np.ceil(pts.max(0)).astype(int), [w, h])
    if x2 <= x1 or y2 <= y1:
        return frame
    IM[:, 2] -= (x1, y1)
    inv = cv2.warpAffine(crop, IM, (x2 - x1, y2 - y1), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    a = cv2.warpAffine(alpha, IM, (x2 - x1, y2 - y1), flags=cv2.INTER_LINEAR)[..., None].clip(0, 1)
    roi = frame[y1:y2, x1:x2].astype(np.float32)
    frame[y1:y2, x1:x2] = (roi * (1 - a) + inv.astype(np.float32) * a).clip(0, 255).astype(np.uint8)
    return frame


def _dilate(mask: np.ndarray, frac: float) -> np.ndarray:
    k = max(3, int(round(max(mask.shape) * frac)) | 1)
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def run_character_swap(engine: FaceEngine, params: dict, source: Path, video: Path, target: np.ndarray,
                       out_path: Path, work_dir: Path, watermark: bool, status: StatusCb) -> Path:
    backend = make_backend(params["char_model"])
    info = video_info(video)
    start = float(np.clip(params.get("start", 0.0), 0.0, max(info["duration"] - 0.5, 0.0)))
    duration = float(min(params.get("duration", 5.0), info["duration"] - start))
    if duration < 0.5:
        raise ValueError("El tramo elegido es demasiado corto.")
    fps_gen = backend.gen_fps

    status("Cargando el segmentador de personas...", 0.0, "loading_models")
    seg = PersonSegmenter(status)
    try:
        status("Preparando la referencia de la persona autorizada...", 0.0, "preparing")
        reference, face_ref = prepare_reference(engine, seg, source)
        tracked = track_target(engine, seg, video, start, duration, fps_gen, target,
                               lambda m, f: status(m, f, "preparing"))
        visible = sum(f.bbox is not None for f in tracked)
        if visible == 0:
            raise ValueError("No se encontró a la persona elegida en el tramo indicado. Prueba otro inicio.")

        boxes = np.stack([f.bbox for f in tracked if f.bbox is not None])
        portrait = float(np.median(boxes[:, 3] - boxes[:, 1])) > 1.1 * float(np.median(boxes[:, 2] - boxes[:, 0]))
        W, H = backend.resolution(portrait)
        M = crop_transform(tracked, (info["width"], info["height"]), (W, H))

        # recortes del tramo a 16 fps + máscaras (dilatadas y con máximo temporal ±1 frame: la ropa o
        # el pelo nuevos pueden ocupar algo más que la silueta original)
        frames, masks = [], []
        for (t, frame), tf in zip(iter_frames(video, start, duration, fps_gen), tracked):
            frames.append(cv2.warpAffine(frame, M, (W, H), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE))
            full = tf.full_mask(frame.shape[0], frame.shape[1])
            masks.append(cv2.warpAffine(full, M, (W, H), flags=cv2.INTER_NEAREST))
        n = min(len(frames), len(masks))
        frames, masks = frames[:n], masks[:n]
        masks = [_dilate(m, 0.025) for m in masks]
        masks = [np.maximum.reduce(masks[max(0, i - 1):i + 2]) for i in range(n)]

        work_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(work_dir / "reference_body.png"), reference)
        cv2.imwrite(str(work_dir / "reference_face.png"), face_ref)
        clip = ClipInputs(
            frames=frames, masks=masks, reference=reference, face_reference=face_ref,
            prompt=params.get("prompt") or "a person, natural lighting, realistic, high quality",
            fps=fps_gen, work_dir=work_dir, turbo=params.get("turbo", True),
            extra={"resolution": "720p" if params.get("hq") else "480p", "hf_token": params.get("hf_token")},
        )

        status(f"Generando con {backend.spec['label']}...", 0.0, "generating")
        gen = backend.generate(clip, lambda m, f: status(m, f, "generating"))
        backend.close()
        gc.collect()

        # silueta de la persona generada (puede ser más ancha: chaqueta, pelo...) para la máscara final
        status("Integrando a la persona generada en el video...", 0.0, "compositing")
        alphas = []
        for g, m in zip(gen, masks):
            region = m.copy()
            persons = seg.persons(g)
            if persons and m.any():
                # persona generada = la que cae mayoritariamente dentro de la zona regenerada
                overlap = [((p.mask > 0) & (m > 0)).sum() / max((p.mask > 0).sum(), 1) for p in persons]
                best = int(np.argmax(overlap))
                if overlap[best] > 0.5:
                    region = np.maximum(region, _dilate(persons[best].mask, 0.01))
            if not m.any():
                region[:] = 0  # la persona no estaba en este frame: no se toca nada
            soft = cv2.GaussianBlur(region.astype(np.float32) / 255.0, (0, 0), max(W, H) * 0.008)
            alphas.append(soft)
    finally:
        del seg
        gc.collect()

    # composición a los fps ORIGINALES: el fondo conserva su fluidez; la persona usa el frame
    # generado más cercano en el tiempo (16 fps)
    writer: Optional[FfmpegWriter] = None
    total = max(1, int(duration * info["fps"]))
    try:
        for k, (t, frame) in enumerate(iter_frames(video, start, duration, info["fps"])):
            i = min(int(round((t - start) * fps_gen)), n - 1)
            frame = paste(frame, gen[i], alphas[i], M)
            if watermark:
                frame = add_watermark(frame)
            if writer is None:
                writer = FfmpegWriter(out_path, video, frame.shape[1], frame.shape[0], info["fps"], audio_start=start)
            writer.write(frame)
            if k % 10 == 0:
                status(f"Componiendo frame {k + 1}/{total}", (k + 1) / total, "compositing")
    finally:
        if writer is not None:
            writer.close()
    return out_path
