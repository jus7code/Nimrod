"""
Personas en el video: segmentación, seguimiento de la persona elegida y foto de referencia.

    PersonSegmenter     YOLO11s-seg (Ultralytics): cajas + máscara de cada persona y, con ByteTrack,
                        un id estable por persona entre frames.
    track_target()      une la identidad elegida en el escaneo (cara, ArcFace) con su cuerpo: la
                        persona cuya máscara contiene la cara reconocida. Si la cara no se ve
                        (de espaldas, tapada), sigue el id de ByteTrack.
    prepare_reference() de la foto/video de la persona autorizada saca un recorte de cuerpo entero
                        sobre fondo blanco y un primer plano de la cara.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from ..downloads import StatusCb, download_file
from ..engine import IMAGE_EXT, FaceEngine, imread_unicode, sample_frames
from ..models_registry import PERSON_SEG
from ..scan import TargetMatcher, face_thumb


@dataclass
class Person:
    bbox: np.ndarray       # x1, y1, x2, y2 (frame completo)
    mask: np.ndarray       # uint8 0/255, tamaño del frame
    track_id: int = -1
    score: float = 0.0


@dataclass
class TrackedFrame:
    """Máscara de la persona elegida en un frame muestreado, guardada recortada a su caja."""
    t: float
    bbox: Optional[np.ndarray] = None   # None = la persona no aparece en este frame
    patch: Optional[np.ndarray] = None  # máscara uint8 dentro de bbox (enteros)

    def full_mask(self, h: int, w: int) -> np.ndarray:
        m = np.zeros((h, w), np.uint8)
        if self.bbox is not None:
            x1, y1, x2, y2 = self.bbox.astype(int)
            m[y1:y2, x1:x2] = self.patch
        return m


class PersonSegmenter:
    def __init__(self, status_cb: StatusCb = None):
        import torch
        from ultralytics import YOLO

        download_file(PERSON_SEG["urls"], PERSON_SEG["file"], "YOLO11s-seg (personas)", status_cb)
        self.model = YOLO(str(PERSON_SEG["file"]))
        self.cuda = torch.cuda.is_available()

    def persons(self, frame: np.ndarray, track: bool = False) -> list[Person]:
        kw = dict(classes=[0], verbose=False, conf=0.3, imgsz=640, device=0 if self.cuda else "cpu")
        r = (self.model.track(frame, persist=True, tracker="bytetrack.yaml", **kw) if track
             else self.model.predict(frame, **kw))[0]
        if r.masks is None or r.boxes is None:
            return []
        h, w = frame.shape[:2]
        ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [-1] * len(r.boxes)
        out = []
        for poly, box, conf, tid in zip(r.masks.xy, r.boxes.xyxy.cpu().numpy(), r.boxes.conf.tolist(), ids):
            m = np.zeros((h, w), np.uint8)
            if len(poly) >= 3:
                cv2.fillPoly(m, [poly.astype(np.int32)], 255)
            out.append(Person(box.astype(np.float32), m, int(tid), float(conf)))
        return out


def _face_in_mask(face_bbox: np.ndarray, mask: np.ndarray) -> float:
    """Fracción de la caja de la cara que cae dentro de la máscara de la persona."""
    x1, y1, x2, y2 = np.clip(face_bbox, 0, [mask.shape[1], mask.shape[0]] * 2).astype(int)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float((mask[y1:y2, x1:x2] > 0).mean())


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def iter_frames(video: Path, start: float, duration: float, fps_out: float) -> Iterator[tuple[float, np.ndarray]]:
    """Frames del tramo [start, start+duration) remuestreados a fps_out (por tiempo, sin interpolar)."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_out = max(1, int(math.floor(duration * fps_out + 1e-6)))
    first = int(round(start * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    idx, frame = first - 1, None
    for k in range(n_out):
        want = int(round((start + k / fps_out) * fps))
        while idx < want:
            ok, f = cap.read()
            if not ok:
                cap.release()
                return
            idx, frame = idx + 1, f
        yield start + k / fps_out, frame
    cap.release()


def track_target(engine: FaceEngine, seg: PersonSegmenter, video: Path, start: float, duration: float,
                 fps_out: float, target: np.ndarray, status_cb: StatusCb = None) -> list[TrackedFrame]:
    matcher = TargetMatcher(engine, target[None])
    out: list[TrackedFrame] = []
    current_id, last_box = None, None
    n_total = max(1, int(duration * fps_out))
    for k, (t, frame) in enumerate(iter_frames(video, start, duration, fps_out)):
        persons = seg.persons(frame, track=True)
        faces = matcher(frame, engine.detect(frame))
        chosen: Optional[Person] = None
        if faces and persons:  # la cara reconocida manda: su cuerpo es la persona que la contiene
            face = max(faces, key=lambda d: d.area)
            score, best = max(((_face_in_mask(face.bbox, p.mask), p) for p in persons), key=lambda x: x[0])
            if score > 0.25:
                chosen, current_id = best, best.track_id
        if chosen is None and current_id is not None:  # cara no visible: seguir el id de ByteTrack
            chosen = next((p for p in persons if p.track_id == current_id and p.track_id >= 0), None)
        if chosen is None and last_box is not None and persons:  # ByteTrack cambió el id
            iou, best = max(((_iou(p.bbox, last_box), p) for p in persons), key=lambda x: x[0])
            if iou > 0.5:
                chosen, current_id = best, best.track_id
        if chosen is None or not chosen.mask.any():
            out.append(TrackedFrame(t))
        else:
            ys, xs = np.nonzero(chosen.mask)
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            out.append(TrackedFrame(t, np.array([x1, y1, x2, y2], np.float32), chosen.mask[y1:y2, x1:x2].copy()))
            last_box = chosen.bbox
        if status_cb:
            seen = sum(f.bbox is not None for f in out)
            status_cb(f"Siguiendo a la persona elegida: {k + 1}/{n_total} frames (visible en {seen})",
                      (k + 1) / n_total)
    return out


def prepare_reference(engine: FaceEngine, seg: PersonSegmenter, source: Path) -> tuple[np.ndarray, np.ndarray]:
    """(cuerpo recortado sobre fondo blanco, primer plano de la cara 512×512) de la persona autorizada."""
    frames = [imread_unicode(source)] if source.suffix.lower() in IMAGE_EXT else list(sample_frames(source, 24))
    best = None
    for frame in frames:
        faces = engine.detect(frame)
        if not faces:
            continue
        face = max(faces, key=lambda d: d.area)
        k = face.kps
        eye_w = max(float(np.linalg.norm(k[1] - k[0])), 1e-3)
        frontal = 1.0 - min(abs((k[2, 0] - (k[0, 0] + k[1, 0]) / 2) / eye_w), 1.0)
        q = face.score * math.sqrt(face.area) * (0.5 + frontal)
        if best is None or q > best[0]:
            best = (q, frame, face)
    if best is None:
        raise ValueError("No se detectó ninguna cara en la foto/video de la persona autorizada.")
    _, frame, face = best
    face_crop = face_thumb(frame, face.bbox, size=512, scale=1.8)

    persons = seg.persons(frame)
    body = None
    if persons:
        score, p = max(((_face_in_mask(face.bbox, p.mask), p) for p in persons), key=lambda x: x[0])
        if score > 0.25:
            body = p
    if body is None:  # selfie muy cerrada: se usa la imagen completa
        return frame.copy(), face_crop
    mask = cv2.dilate(body.mask, np.ones((5, 5), np.uint8))
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 1.5)[..., None]
    cut = (frame * alpha + 255 * (1 - alpha)).astype(np.uint8)
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = body.bbox
    pad = 0.04 * max(x2 - x1, y2 - y1)
    x1, y1 = int(max(0, x1 - pad)), int(max(0, y1 - pad))
    x2, y2 = int(min(w, x2 + pad)), int(min(h, y2 + pad))
    return cut[y1:y2, x1:x2].copy(), face_crop
