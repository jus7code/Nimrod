"""
Escaneo rápido del video destino y selección de a quién cambiar.

    scan_video()     muestrea ~2 frames/s, detecta caras (SCRFD), calcula su embedding ArcFace y
                     agrupa las caras por identidad (clustering voraz por similitud coseno).
                     Devuelve una miniatura por persona para que el usuario elija.
    TargetMatcher    durante el procesado, decide en cada frame qué caras pertenecen a las
                     personas elegidas: similitud ArcFace + continuidad espacial con el frame anterior
                     (para no perder a la persona cuando gira la cabeza y el embedding empeora).
"""
from __future__ import annotations

import base64
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .downloads import StatusCb
from .engine import Detection, FaceEngine

SAME_PERSON = 0.40   # coseno ArcFace a partir del cual dos caras del escaneo se consideran la misma persona
MERGE = 0.40         # fusión final de clusters cuyos centroides se parecen mucho
MIN_FACE_PX = 28     # caras más pequeñas no dan un embedding fiable


def _jpeg_data_uri(img: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def face_thumb(frame: np.ndarray, bbox: np.ndarray, size: int = 128, scale: float = 1.6) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    cx, cy, half = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * scale / 2
    M = np.float32([[size / (2 * half), 0, size / 2 - cx * size / (2 * half)],
                    [0, size / (2 * half), size / 2 - cy * size / (2 * half)]])
    return cv2.warpAffine(frame, M, (size, size), borderMode=cv2.BORDER_REPLICATE)


def context_thumb(frame: np.ndarray, bbox: np.ndarray, width: int = 360) -> np.ndarray:
    h, w = frame.shape[:2]
    s = width / w
    small = cv2.resize(frame, (width, int(h * s)), interpolation=cv2.INTER_AREA)
    x1, y1, x2, y2 = (np.asarray(bbox) * s).astype(int)
    cv2.rectangle(small, (x1, y1), (x2, y2), (60, 220, 90), 2)
    return small


@dataclass
class _Cluster:
    emb_sum: np.ndarray
    count: int = 0
    first_t: float = 0.0
    best_q: float = -1.0
    best_frame: np.ndarray | None = None
    best_bbox: np.ndarray | None = None
    times: list[float] = field(default_factory=list)

    @property
    def centroid(self) -> np.ndarray:
        return self.emb_sum / np.linalg.norm(self.emb_sum)

    def add(self, emb: np.ndarray, t: float, det: Detection, frame: np.ndarray) -> None:
        self.emb_sum = self.emb_sum + emb
        self.count += 1
        self.times.append(t)
        # mejor miniatura: cara grande, nítida para el detector y frontal (ojos nivelados, nariz centrada)
        k = det.kps
        eye_w = max(np.linalg.norm(k[1] - k[0]), 1e-3)
        frontal = 1.0 - min(abs((k[2, 0] - (k[0, 0] + k[1, 0]) / 2) / eye_w), 1.0)
        q = det.score * math.sqrt(det.area) * (0.5 + frontal)
        if q > self.best_q:
            self.best_q, self.best_frame, self.best_bbox = q, frame, det.bbox.copy()


def video_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("No se pudo abrir el video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": fps, "frames": total, "width": w, "height": h, "duration": total / fps if fps else 0.0}


def scan_video(engine: FaceEngine, path: Path, status_cb: StatusCb = None,
               samples_per_s: float = 2.0, max_samples: int = 120) -> dict:
    """Devuelve {"info", "people": [{id, count, share, first_t, thumb, context, embedding}]}."""
    info = video_info(path)
    fps, total = info["fps"], max(info["frames"], 1)
    step = max(1, round(fps / samples_per_s))
    if total / step > max_samples:
        step = math.ceil(total / max_samples)
    n_samples = math.ceil(total / step)

    clusters: list[_Cluster] = []
    cap = cv2.VideoCapture(str(path))
    started, i, sampled = time.time(), 0, 0
    while cap.grab():
        if i % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                sampled += 1
                t = i / fps
                faces = [d for d in engine.detect(frame) if d.size >= MIN_FACE_PX and d.score >= 0.5]
                embs = engine.embed(frame, faces)
                for det, emb in zip(faces, embs):
                    best, best_sim = None, SAME_PERSON
                    for c in clusters:
                        sim = float(emb @ c.centroid)
                        if sim > best_sim:
                            best, best_sim = c, sim
                    if best is None:
                        best = _Cluster(emb_sum=np.zeros_like(emb), first_t=t)
                        clusters.append(best)
                    best.add(emb, t, det, frame)
                if status_cb:
                    status_cb(f"Escaneando rostros: {sampled}/{n_samples} frames · "
                              f"{len(clusters)} personas", sampled / n_samples)
        i += 1
    cap.release()

    # fusiona clusters que resultaron ser la misma persona (p. ej. de perfil al principio y luego de frente)
    clusters.sort(key=lambda c: -c.count)
    merged: list[_Cluster] = []
    for c in clusters:
        target = next((m for m in merged if float(c.centroid @ m.centroid) >= MERGE), None)
        if target is None:
            merged.append(c)
            continue
        target.emb_sum = target.emb_sum + c.emb_sum
        target.count += c.count
        target.times += c.times
        target.first_t = min(target.first_t, c.first_t)
        if c.best_q > target.best_q:
            target.best_q, target.best_frame, target.best_bbox = c.best_q, c.best_frame, c.best_bbox

    people = []
    for idx, c in enumerate(sorted(merged, key=lambda c: -c.count)):
        people.append({
            "id": idx,
            "count": c.count,
            "share": round(len(set(c.times)) / max(sampled, 1), 3),  # fracción de frames donde aparece
            "first_t": round(c.first_t, 2),
            "thumb": _jpeg_data_uri(face_thumb(c.best_frame, c.best_bbox)),
            "context": _jpeg_data_uri(context_thumb(c.best_frame, c.best_bbox), 75),
            "embedding": c.centroid.astype(np.float32),
        })
    info["scan_seconds"] = round(time.time() - started, 2)
    info["sampled_frames"] = sampled
    return {"info": info, "people": people}


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


class TargetMatcher:
    """Filtra las detecciones de un frame dejando solo a las personas elegidas.

    Una cara se acepta si:
      · su similitud ArcFace con alguna identidad elegida ≥ `strong`, o
      · su similitud ≥ `weak` y solapa (IoU > 0.3) con una cara aceptada en los últimos frames
        (perfiles, desenfoque de movimiento: el embedding baja pero sigue siendo la misma cara).
    """

    def __init__(self, engine: FaceEngine, targets: np.ndarray, strong: float = 0.36, weak: float = 0.12,
                 memory: int = 8):
        self.engine = engine
        t = np.asarray(targets, np.float32).reshape(-1, 512)
        self.targets = t / np.linalg.norm(t, axis=1, keepdims=True)
        self.strong, self.weak, self.memory = strong, weak, memory
        # (bbox, frames desde la última vez vista, índice de la persona elegida)
        self.tracks: list[tuple[np.ndarray, int, int]] = []

    def similarities(self, frame: np.ndarray, faces: list[Detection]) -> np.ndarray:
        """N×K: similitud de cada cara con cada persona elegida."""
        if not faces:
            return np.zeros((0, len(self.targets)), np.float32)
        return self.engine.embed(frame, faces) @ self.targets.T

    def similarity(self, frame: np.ndarray, faces: list[Detection]) -> np.ndarray:
        sims = self.similarities(frame, faces)
        return sims.max(1) if len(sims) else np.zeros(0, np.float32)

    def __call__(self, frame: np.ndarray, faces: list[Detection]) -> list[Detection]:
        """Devuelve las caras aceptadas con `det.target` = índice de la persona elegida que son."""
        sims = self.similarities(frame, faces)
        chosen = []
        for det, row in zip(faces, sims):
            best = int(row.argmax())
            sim = float(row[best])
            near, near_target = 0.0, best
            for b, _, tgt in self.tracks:
                iou = _iou(det.bbox, b)
                if iou > near:
                    near, near_target = iou, tgt
            if sim >= self.strong:
                det.target = best
            elif sim >= self.weak and near > 0.3:
                det.target = near_target  # de perfil o borrosa: sigue siendo quien era en el frame anterior
            else:
                continue
            chosen.append(det)
        kept = [(b, age + 1, tgt) for b, age, tgt in self.tracks
                if age + 1 < self.memory and all(_iou(b, d.bbox) < 0.3 for d in chosen)]
        self.tracks = [(d.bbox.copy(), 0, d.target) for d in chosen] + kept
        return chosen
