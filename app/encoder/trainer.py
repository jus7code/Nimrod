"""
Entrenamiento rápido del autoencoder para un par de personas (A = fuente, B = video).

1. Extracción: se detecta y alinea la cara (128×128, plantilla arcface_128) en frames de A y B,
   y BiSeNet da una máscara de la cara para que la pérdida se concentre en ella.
2. Entrenamiento con presupuesto de tiempo: aumentos en GPU (afín aleatoria + "random warp")
   y pérdida = L1 + DSSIM pesadas por la máscara.
3. Se guarda un checkpoint por par: si se repite el mismo par, continúa donde se quedó.
"""
from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..downloads import StatusCb
from ..engine import IMAGE_EXT, FaceEngine, Masker, imread_unicode, warp_face
from ..models_registry import ENCODER_MODEL
from .model import RES, SwapAutoencoder, dssim

TEMPLATE = ENCODER_MODEL["template"]


def file_key(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()[:12]


def extract_faces(engine: FaceEngine, masker: Masker, path: Path, max_faces: int, label: str,
                  status_cb: StatusCb = None) -> tuple[np.ndarray, np.ndarray]:
    crops, masks = [], []

    def add(frame):
        det = engine.largest(frame)
        if det is None:
            return
        crop, _ = warp_face(frame, det.kps, TEMPLATE, RES)
        crops.append(crop)
        masks.append(masker.region(crop) if masker.parser is not None else masker.box(RES))

    if path.suffix.lower() in IMAGE_EXT:
        add(imread_unicode(path))
    else:
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        step = max(1, math.ceil(total / max_faces))
        i = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if i % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    add(frame)
                if status_cb:
                    status_cb(f"Extrayendo caras · {label}: {len(crops)} caras ({i + 1}/{total} frames)", (i + 1) / total)
            i += 1
        cap.release()
    if not crops:
        raise ValueError(f"No se detectaron caras en {label}.")
    return np.stack(crops), np.stack(masks)


class GpuAugment:
    """Afín aleatoria (rotación ±10°, escala ±5 %, desplazamiento ±5 %, espejo) para entrada y
    objetivo, y además un "random warp" (deformación suave) SOLO en la entrada: la red tiene que
    reconstruir la cara sin deformar, así que aprende la estructura facial en vez de copiar píxeles."""

    def __init__(self, device):
        self.device = device

    def __call__(self, imgs: torch.Tensor, masks: torch.Tensor):
        b, _, h, w = imgs.shape
        r = lambda: torch.rand(b, device=self.device) * 2 - 1  # noqa: E731
        ang, sc, tx, ty = r() * math.radians(10), 1 + r() * 0.05, r() * 0.05, r() * 0.05
        flip = torch.where(torch.rand(b, device=self.device) < 0.5, -1.0, 1.0)
        cos, sin = torch.cos(ang) * sc, torch.sin(ang) * sc
        theta = torch.stack([
            torch.stack([cos * flip, -sin, tx], 1),
            torch.stack([sin * flip, cos, ty], 1),
        ], 1)
        grid = F.affine_grid(theta, (b, 3, h, w), align_corners=False)
        target = F.grid_sample(imgs, grid, padding_mode="border", align_corners=False)
        mask = F.grid_sample(masks, grid, padding_mode="zeros", align_corners=False)
        noise = (torch.rand(b, 2, 5, 5, device=self.device) * 2 - 1) * 0.05
        noise = F.interpolate(noise, size=(h, w), mode="bicubic", align_corners=True).permute(0, 2, 3, 1)
        warped = F.grid_sample(imgs, grid + noise, padding_mode="border", align_corners=False)
        return warped, target, mask


def to_tensor(crops: np.ndarray, masks: np.ndarray, device) -> tuple[torch.Tensor, torch.Tensor]:
    imgs = torch.from_numpy(crops).to(device).permute(0, 3, 1, 2).contiguous()  # uint8 BGR
    m = torch.from_numpy(masks.astype(np.float16)).to(device)[:, None]
    return imgs, m


def train(model: SwapAutoencoder, opt, A, B, minutes: float, device, status_cb: StatusCb = None,
          batch: int = 8, start_iter: int = 0) -> int:
    aug = GpuAugment(device)
    a_img, a_mask = A
    b_img, b_mask = B
    deadline = time.time() + minutes * 60
    started, last_report, it, loss_ema = time.time(), 0.0, start_iter, None
    model.train()

    def batch_from(imgs, masks):
        idx = torch.randint(0, imgs.shape[0], (batch,), device=device)
        w, t, m = aug(imgs[idx].float() / 255.0, masks[idx].float())
        return w.contiguous(memory_format=torch.channels_last), t, m

    while time.time() < deadline:
        wa, ta, ma = batch_from(a_img, a_mask)
        wb, tb, mb = batch_from(b_img, b_mask)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            ra = model.decoder_a(model.encoder(wa))
            rb = model.decoder_b(model.encoder(wb))
        loss = 0.0
        for rec, tgt, m in ((ra.float(), ta, ma), (rb.float(), tb, mb)):
            w = 0.1 + 0.9 * m                          # la cara pesa 10× más que el fondo
            loss = loss + 10 * (torch.abs(rec - tgt) * w).mean() + 10 * dssim(rec * w, tgt * w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        it += 1
        loss_ema = loss.item() if loss_ema is None else 0.98 * loss_ema + 0.02 * loss.item()
        now = time.time()
        if status_cb and now - last_report > 1.0:
            last_report = now
            elapsed = now - started
            status_cb(
                f"Entrenando encoder: {elapsed / 60:.1f}/{minutes:.0f} min · iteración {it} · "
                f"pérdida {loss_ema:.3f} · {(it - start_iter) / elapsed:.1f} it/s",
                min(elapsed / (minutes * 60), 1.0),
            )
    return it


class EncoderSwapper:
    size = RES
    template = TEMPLATE
    region_on_original = True  # la máscara facial se calcula sobre la cara del video, no sobre la generada

    def __init__(self, model: SwapAutoencoder, device):
        self.model = model.eval()
        self.device = device

    @torch.no_grad()
    def swap(self, crop: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(crop).to(self.device).permute(2, 0, 1)[None].float() / 255.0
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            y = self.model.swap_to_a(x)
        return (y.float()[0].permute(1, 2, 0).clamp(0, 1) * 255).cpu().numpy()


def build_encoder_swapper(engine: FaceEngine, masker: Masker, source: Path, video: Path,
                          minutes: float, ckpt_dir: Path, status_cb: StatusCb = None) -> EncoderSwapper:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / f"{file_key(source)}_{file_key(video)}.pt"

    A = extract_faces(engine, masker, source, 1500, "fuente (A)", status_cb)
    B = extract_faces(engine, masker, video, 1500, "video (B)", status_cb)
    A, B = to_tensor(*A, device), to_tensor(*B, device)

    model = SwapAutoencoder().to(device).to(memory_format=torch.channels_last)  # Tensor Cores
    opt = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))
    start_iter = 0
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_iter = state["iter"]
        if status_cb:
            status_cb(f"Continuando entrenamiento previo de este par (iteración {start_iter})...", 0.0)

    it = train(model, opt, A, B, minutes, device, status_cb, start_iter=start_iter)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "iter": it}, ckpt)
    del A, B
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return EncoderSwapper(model, device)
