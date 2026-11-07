"""
Generadores de "character swap". Todos reciben lo mismo (ClipInputs: el recorte del video alrededor
de la persona elegida a 16 fps, su máscara por frame y las fotos de referencia) y devuelven los
frames generados del recorte, uno por cada frame de entrada.

    VaceBackend         Wan2.1-VACE 1.3B en local con diffusers (vídeo enmascarado + referencia).
    WanAnimateBackend   Wan2.2-Animate-14B, modo "replace", en la nube (fal.ai).
    MochaBackend        MoCha (Wan2.1-14B) con el repositorio oficial, en una máquina con GPU grande.
"""
from __future__ import annotations

import gc
import hashlib
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..downloads import StatusCb
from ..engine import FfmpegWriter
from ..models_registry import CHAR_DIR, CHARACTER_MODELS

RESOLUTIONS = {"portrait": (480, 832), "landscape": (832, 480)}  # (ancho, alto): 480p nativo de Wan

DEFAULT_NEGATIVE = (
    "bright tones, overexposed, static, blurred details, subtitles, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, "
    "disfigured, misshapen limbs, fused fingers, messy background, three legs, many people in the background"
)


@dataclass
class ClipInputs:
    frames: list[np.ndarray]      # BGR uint8, W×H del modelo, a gen_fps
    masks: list[np.ndarray]       # uint8 0/255: zona a regenerar (persona dilatada)
    reference: np.ndarray         # BGR: cuerpo de la persona autorizada sobre fondo blanco
    face_reference: np.ndarray    # BGR 512×512: primer plano de su cara
    prompt: str
    fps: float
    work_dir: Path
    turbo: bool = True
    seed: int = 42
    extra: dict = field(default_factory=dict)


def resample(frames: list[np.ndarray], n: int) -> list[np.ndarray]:
    """Ajusta una secuencia a n frames por tiempo normalizado (por si el generador cambió los fps)."""
    if not frames:
        raise RuntimeError("El generador no devolvió frames.")
    if len(frames) == n:
        return frames
    idx = np.round(np.linspace(0, len(frames) - 1, n)).astype(int)
    return [frames[i] for i in idx]


def read_video(path: Path, size: Optional[tuple[int, int]] = None) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f, size, interpolation=cv2.INTER_AREA) if size and f.shape[1::-1] != size else f)
    cap.release()
    return frames


def write_video(frames: list[np.ndarray], path: Path, fps: float) -> Path:
    h, w = frames[0].shape[:2]
    writer = FfmpegWriter(path, None, w, h, fps, crf=12)
    for f in frames:
        writer.write(f)
    writer.close()
    return path


def pad_on_white(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    W, H = size
    h, w = img.shape[:2]
    s = min(W / w, H / h)
    nw, nh = int(w * s), int(h * s)
    canvas = np.full((H, W, 3), 255, np.uint8)
    x, y = (W - nw) // 2, (H - nh) // 2
    canvas[y:y + nh, x:x + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return canvas


class CharacterBackend:
    key: str

    def __init__(self, key: str):
        self.key = key
        self.spec = CHARACTER_MODELS[key]
        self.gen_fps = self.spec["gen_fps"]

    def resolution(self, portrait: bool) -> tuple[int, int]:
        orient = "portrait" if portrait and "portrait" in self.spec["orientations"] else self.spec["orientations"][-1]
        return RESOLUTIONS[orient]

    def generate(self, clip: ClipInputs, status_cb: StatusCb = None) -> list[np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- 1. VACE 1.3B (local)
class VaceBackend(CharacterBackend):
    """VACE ("Video All-in-one Creation and Editing") sobre Wan2.1-1.3B.

    Entrada por trozo de 81 frames:  vídeo + máscara (blanco = generar) + imagen de referencia.
    Internamente el vídeo se parte en  inactivo = vídeo·(1−máscara)  y  reactivo = vídeo·máscara,
    ambos se codifican con el VAE y entran como "contexto" a bloques VACE paralelos al DiT,
    cuyas salidas se suman a las capas del DiT (como un ControlNet). La referencia se codifica
    como frames extra al principio de la secuencia latente.
    """

    def __init__(self, key: str = "vace_1_3b"):
        super().__init__(key)
        self.pipe = None

    # -- descarga con progreso (huggingface_hub no expone progreso: se mide el tamaño de la carpeta)
    def _download(self, patterns: list[str], label: str, expected_gb: float, status_cb: StatusCb) -> Path:
        from huggingface_hub import snapshot_download

        target: Path = self.spec["dir"]
        done = threading.Event()

        def report():
            while not done.wait(1.0):
                size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 1e9 if target.exists() else 0
                if status_cb:
                    status_cb(f"Descargando {label} (solo la 1ª vez): {size:.1f}/{expected_gb:.1f} GB",
                              min(size / expected_gb, 0.99))

        threading.Thread(target=report, daemon=True).start()
        try:
            snapshot_download(self.spec["repo"], local_dir=target, allow_patterns=patterns)
        finally:
            done.set()
        return target

    def _prompt_embeds(self, prompt: str, status_cb: StatusCb):
        """Codifica el texto con umT5-XXL (5.7B parámetros, 11 GB) en la CPU UNA vez y lo cachea:
        así el codificador de texto nunca ocupa la VRAM de 6 GB."""
        import torch

        key = hashlib.sha1(f"{prompt}\n{DEFAULT_NEGATIVE}".encode()).hexdigest()[:16]
        cache = CHAR_DIR / "prompt_cache" / f"vace_{key}.pt"
        if cache.exists():
            data = torch.load(cache)
            return data["pos"], data["neg"]
        root = self._download(["text_encoder/*", "tokenizer/*", "model_index.json"], "codificador de texto umT5",
                              11.4, status_cb)
        if status_cb:
            status_cb("Codificando la descripción con umT5 en la CPU (solo la 1ª vez por descripción)...", 0.0)
        from diffusers import WanVACEPipeline
        from transformers import AutoTokenizer, UMT5EncoderModel

        tok = AutoTokenizer.from_pretrained(root, subfolder="tokenizer")
        enc = UMT5EncoderModel.from_pretrained(root, subfolder="text_encoder", torch_dtype=torch.bfloat16)
        helper = WanVACEPipeline(tokenizer=tok, text_encoder=enc, vae=None, scheduler=None)
        with torch.no_grad():
            pos, neg = helper.encode_prompt(prompt, DEFAULT_NEGATIVE, do_classifier_free_guidance=True,
                                            max_sequence_length=512, device=torch.device("cpu"),
                                            dtype=torch.bfloat16)
        del helper, enc
        gc.collect()
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"pos": pos, "neg": neg}, cache)
        return pos, neg

    def _load(self, turbo: bool, status_cb: StatusCb):
        import torch
        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanVACEPipeline, WanVACETransformer3DModel

        root = self._download(["transformer/*", "vae/*", "scheduler/*", "model_index.json"],
                              "Wan2.1-VACE 1.3B", 7.7, status_cb)
        if status_cb:
            status_cb("Cargando VACE 1.3B...", 0.0)
        transformer = WanVACETransformer3DModel.from_pretrained(root, subfolder="transformer", torch_dtype=torch.bfloat16)
        vae = AutoencoderKLWan.from_pretrained(root, subfolder="vae", torch_dtype=torch.float32)
        vae.enable_tiling()  # decodificar 81 frames de golpe no cabe en 6 GB
        scheduler = UniPCMultistepScheduler.from_pretrained(root, subfolder="scheduler", flow_shift=3.0)  # 3.0 = 480p
        pipe = WanVACEPipeline(tokenizer=None, text_encoder=None, vae=vae, scheduler=scheduler, transformer=transformer)
        self.lora_ok = False
        if turbo:
            try:
                from huggingface_hub import hf_hub_download

                lora = self.spec["turbo_lora"]
                path = hf_hub_download(lora["repo"], lora["file"], local_dir=CHAR_DIR / "loras")
                pipe.load_lora_weights(path, adapter_name="causvid")
                pipe.set_adapters(["causvid"], [0.7])
                self.lora_ok = True
            except Exception as exc:  # noqa: BLE001
                print(f"[VACE] no se pudo cargar la LoRA turbo ({exc}); se usa el modo normal")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
        if vram >= 16:
            pipe.to("cuda")
        elif vram > 0:
            pipe.enable_model_cpu_offload()  # transformer y VAE se turnan en la GPU
        self.pipe = pipe

    def generate(self, clip: ClipInputs, status_cb: StatusCb = None) -> list[np.ndarray]:
        import torch
        from PIL import Image

        pos, neg = self._prompt_embeds(clip.prompt, status_cb)
        if self.pipe is None:
            self._load(clip.turbo, status_cb)
        turbo = clip.turbo and self.lora_ok
        steps, cfg = (8, 1.0) if turbo else (int(clip.extra.get("steps", 25)), 5.0)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        pos, neg = pos.to(dev), neg.to(dev)

        H, W = clip.frames[0].shape[:2]
        ref = Image.fromarray(pad_on_white(clip.reference, (W, H))[:, :, ::-1])
        n, L = len(clip.frames), self.spec["chunk"]
        overlap = min(8, L // 4)
        out: list[Optional[np.ndarray]] = [None] * n
        starts = list(range(0, max(n - overlap, 1), L - overlap))
        for ci, s in enumerate(starts):
            idx = list(range(s, min(s + L, n)))
            # Wan necesita 4k+1 frames: se completa repitiendo el último
            padded = idx + [idx[-1]] * ((-(len(idx) - 1)) % 4)
            if len(padded) < 5:
                padded += [idx[-1]] * (5 - len(padded))
            video, mask = [], []
            for j, i in enumerate(padded):
                if ci > 0 and j < overlap and out[i] is not None:
                    # continuidad entre trozos: los primeros frames son los ya generados (máscara negra = fijo)
                    video.append(Image.fromarray(out[i][:, :, ::-1]))
                    mask.append(Image.new("L", (W, H), 0))
                else:
                    video.append(Image.fromarray(clip.frames[i][:, :, ::-1]))
                    mask.append(Image.fromarray(clip.masks[i]))

            def on_step(pipe, step, t, kw, ci=ci):
                if status_cb:
                    status_cb(f"Generando con VACE: trozo {ci + 1}/{len(starts)} · paso {step + 1}/{steps}"
                              f"{' (turbo)' if turbo else ''}", (ci + (step + 1) / steps) / len(starts))
                return kw

            with torch.inference_mode():
                result = self.pipe(
                    video=video, mask=mask, reference_images=[ref], height=H, width=W, num_frames=len(padded),
                    prompt_embeds=pos, negative_prompt_embeds=neg if cfg > 1 else None,
                    num_inference_steps=steps, guidance_scale=cfg,
                    generator=torch.Generator("cpu").manual_seed(clip.seed),
                    output_type="np", callback_on_step_end=on_step,
                ).frames[0]
            for j, i in enumerate(idx):
                out[i] = (np.clip(result[j], 0, 1) * 255).astype(np.uint8)[:, :, ::-1].copy()
        return [o if o is not None else f for o, f in zip(out, clip.frames)]

    def close(self) -> None:
        self.pipe = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- 2. Wan-Animate 14B (fal.ai)
class WanAnimateBackend(CharacterBackend):
    """Wan2.2-Animate-14B en modo "replace" a través de la API de fal.ai.

    El servicio extrae del vídeo el esqueleto (pose) y los rasgos faciales implícitos de la persona,
    la segmenta, y genera a la persona de la imagen con ese movimiento sobre el fondo original.
    Una LoRA de re-iluminación ajusta su luz y color a los de la escena.
    """

    def generate(self, clip: ClipInputs, status_cb: StatusCb = None) -> list[np.ndarray]:
        if not os.getenv("FAL_KEY"):
            raise RuntimeError("Wan-Animate se ejecuta en fal.ai: define la variable de entorno FAL_KEY (ver README).")
        import fal_client

        clip.work_dir.mkdir(parents=True, exist_ok=True)
        vid = write_video(clip.frames, clip.work_dir / "target_crop.mp4", clip.fps)
        ref = clip.work_dir / "reference.png"
        cv2.imwrite(str(ref), clip.reference)
        if status_cb:
            status_cb("Subiendo el recorte y la referencia a fal.ai...", 0.0)
        args = {
            "video_url": fal_client.upload_file(str(vid)),
            "image_url": fal_client.upload_file(str(ref)),
            "resolution": clip.extra.get("resolution", "480p"),
            "use_turbo": bool(clip.turbo),
            "video_quality": "high",
        }
        started = time.time()

        def on_update(update):
            if status_cb:
                name = type(update).__name__
                pos = getattr(update, "position", None)
                msg = f"en cola (posición {pos})" if pos is not None else "generando"
                status_cb(f"Wan-Animate en fal.ai: {msg} · {time.time() - started:.0f} s ({name})", 0.0)

        result = fal_client.subscribe(self.spec["endpoint"], arguments=args, with_logs=False, on_queue_update=on_update)
        url = result["video"]["url"]
        out = clip.work_dir / "wan_animate.mp4"
        urllib.request.urlretrieve(url, out)
        W, H = clip.frames[0].shape[1], clip.frames[0].shape[0]
        return resample(read_video(out, (W, H)), len(clip.frames))


# --------------------------------------------------------------------------- 2b. Wan-Animate en un Space gratuito
class WanAnimateSpaceBackend(CharacterBackend):
    """El mismo Wan2.2-Animate-14B ("replace"), ejecutado en un Space de Hugging Face con ZeroGPU.

    No descarga ningún modelo: solo se suben el recorte (unos MB) y la foto, y se baja el resultado.
    Es gratis dentro de la cuota diaria de GPU de Hugging Face (mayor con HF_TOKEN de una cuenta).
    """

    def generate(self, clip: ClipInputs, status_cb: StatusCb = None) -> list[np.ndarray]:
        from gradio_client import Client, handle_file

        clip.work_dir.mkdir(parents=True, exist_ok=True)
        vid = write_video(clip.frames, clip.work_dir / "target_crop.mp4", clip.fps)
        ref = clip.work_dir / "reference.png"
        cv2.imwrite(str(ref), clip.reference)
        seconds = len(clip.frames) / clip.fps
        token = clip.extra.get("hf_token") or os.getenv("HF_TOKEN") or None
        if status_cb:
            status_cb(f"Conectando con el Space {self.spec['space']}"
                      f"{' (con tu token de HF)' if token else ' (anónimo: cuota muy pequeña)'}...", 0.0)
        # download_files=False: el resultado se baja a mano con las cabeceras de la sesión (la descarga
        # automática de gradio_client recibía 403 de este Space)
        try:
            client = Client(self.spec["space"], token=token, verbose=False, download_files=False)
        except TypeError:  # versiones antiguas de gradio_client usan hf_token
            client = Client(self.spec["space"], hf_token=token, verbose=False, download_files=False)
        job = client.submit(
            input_video=handle_file(str(vid)),
            max_duration_s=float(min(20, max(2, math.ceil(seconds)))),
            edited_frame=handle_file(str(ref)),
            rc_str="Character Swap",
            resolution_choice="Medium Res" if clip.extra.get("resolution") == "720p" else "Low Res",
            api_name="/animate_scene",
        )
        started = time.time()
        while not job.done():
            st = job.status()
            if status_cb:
                code = getattr(st.code, "name", str(st.code))
                rank = f" · posición {st.rank} de {st.queue_size}" if st.rank is not None else ""
                msg = {"IN_QUEUE": "en cola", "SENDING_DATA": "subiendo", "PROCESSING": "generando en la GPU de HF",
                       "STARTING": "arrancando", "JOINING_QUEUE": "entrando en cola"}.get(code, code.lower())
                status_cb(f"Wan-Animate (Space gratuito): {msg}{rank} · {time.time() - started:.0f} s", 0.0)
            time.sleep(2)
        try:
            result = job.result()
        except BaseException as exc:  # noqa: BLE001  (CancelledError no hereda de Exception)
            raise RuntimeError(self._explain(exc, token)) from exc
        first = result[0] if isinstance(result, (list, tuple)) else result
        if isinstance(first, dict) and "video" in first:
            first = first["video"]
        out = clip.work_dir / "wan_animate_space.mp4"
        self._fetch(client, first, out, token)
        W, H = clip.frames[0].shape[1], clip.frames[0].shape[0]
        return resample(read_video(out, (W, H)), len(clip.frames))

    @staticmethod
    def _explain(exc: BaseException, token: Optional[str]) -> str:
        text = str(exc).strip()
        low = text.lower()
        if "quota" in low:
            hint = ("Se agotó la cuota diaria de GPU gratuita de Hugging Face. " +
                    ("Espera a que se renueve (el mensaje indica cuánto falta) o usa otra cuenta/plan PRO."
                     if token else
                     "Sin token la cuota anónima es casi nula: pega el token de tu cuenta gratuita de Hugging Face "
                     "(huggingface.co/settings/tokens) en la casilla de la página."))
            return f"{hint} Detalle del Space: {text}"
        if not text or type(exc).__name__ == "CancelledError":
            return ("El Space cerró la conexión sin dar un motivo (suele pasar cuando se queda sin cuota de GPU, "
                    "se reinicia o la conexión se corta mientras genera). " +
                    ("" if token else "Prueba con tu token de Hugging Face (más cuota y prioridad). ") +
                    "Vuelve a intentarlo en unos minutos.")
        return f"El Space devolvió un error: {type(exc).__name__}: {text}"

    @staticmethod
    def _fetch(client, file, out: Path, token: Optional[str]) -> None:
        """Baja el video resultante (ruta local, URL o FileData), con las cabeceras de la sesión."""
        import httpx

        if isinstance(file, dict):
            file = file.get("url") or file.get("path")
        if not file:
            raise RuntimeError("El Space no devolvió ningún video.")
        if Path(str(file)).exists():
            shutil.copy(file, out)
            return
        url = str(file)
        if not url.startswith("http"):
            url = f"{client.src.rstrip('/')}/gradio_api/file={url}"
        headers = dict(getattr(client, "headers", {}) or {})
        if token:
            headers["Authorization"] = f"Bearer {token}"
        cookies = getattr(client, "cookies", None)
        last = None
        for attempt in range(3):
            try:
                with httpx.stream("GET", url, headers=headers, cookies=cookies, follow_redirects=True, timeout=300) as r:
                    if r.status_code == 200:
                        with out.open("wb") as f:
                            for chunk in r.iter_bytes():
                                f.write(chunk)
                        return
                    last = f"HTTP {r.status_code}"
            except httpx.HTTPError as exc:
                last = str(exc)
            time.sleep(2)
        raise RuntimeError(f"El Space generó el video pero no dejó descargarlo ({last}). "
                           + ("" if token else "Con tu token de Hugging Face suele funcionar."))


# --------------------------------------------------------------------------- 3. MoCha (externo)
class MochaBackend(CharacterBackend):
    """MoCha: sustitución de personaje "end-to-end" sobre Wan2.1-T2V-14B.

    Concatena en una sola secuencia de tokens el vídeo original, la máscara del PRIMER frame y las
    referencias (cuerpo + cara), y deja que la atención completa del DiT encuentre a la persona y la
    siga en el resto del vídeo, sin esqueletos ni máscaras por frame. Se ejecuta el script oficial
    `inference_mocha.py` (repositorio Orange-3DV-Team/MoCha) como subproceso.
    """

    def generate(self, clip: ClipInputs, status_cb: StatusCb = None) -> list[np.ndarray]:
        repo = Path(os.getenv("MOCHA_DIR", ""))
        if not (repo / "inference_mocha.py").exists():
            raise RuntimeError("MoCha no está instalado: define MOCHA_DIR con la ruta del repositorio oficial "
                               "(con inference_mocha.py configurado) — ver README.")
        python = os.getenv("MOCHA_PYTHON", sys.executable)
        work = clip.work_dir / "mocha"
        if work.exists():
            shutil.rmtree(work)
        (work / "out").mkdir(parents=True)
        cv2.imwrite(str(work / "ref_body.png"), clip.reference)
        cv2.imwrite(str(work / "ref_face.png"), clip.face_reference)

        n, L = len(clip.frames), self.spec["chunk"]
        rows, chunks = [], []
        for ci, s in enumerate(range(0, n, L)):
            idx = list(range(s, min(s + L, n)))
            first = next((i for i in idx if clip.masks[i].any()), None)
            if first is None:
                continue  # la persona no aparece en este trozo
            idx = [i for i in idx if i >= first]
            name = f"chunk{ci:02d}"
            write_video([clip.frames[i] for i in idx], work / f"{name}.mp4", clip.fps)
            cv2.imwrite(str(work / f"{name}_mask.png"), clip.masks[first])
            rows.append(f"{work / f'{name}.mp4'},{work / f'{name}_mask.png'},{work / 'ref_body.png'},{work / 'ref_face.png'}")
            chunks.append((name, idx))
        if not rows:
            raise RuntimeError("La persona elegida no aparece en el tramo.")
        csv = work / "data.csv"
        csv.write_text("source_video,source_mask,reference_1,reference_2\n" + "\n".join(rows) + "\n", encoding="utf-8")

        cmd = [python, "inference_mocha.py", "--data_path", str(csv), "--output_dir", str(work / "out")]
        if os.getenv("MOCHA_CKPT"):
            cmd += ["--ckpt_path", os.environ["MOCHA_CKPT"]]
        if status_cb:
            status_cb(f"Ejecutando MoCha ({len(rows)} trozo(s) de hasta 81 frames)...", 0.0)
        proc = subprocess.Popen(cmd, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors="replace", bufsize=1)
        tail: list[str] = []
        for line in proc.stdout:  # tqdm escribe con \r: se muestra la última línea útil
            for part in line.replace("\r", "\n").splitlines():
                if part.strip():
                    tail = (tail + [part.strip()])[-20:]
                    if status_cb:
                        status_cb(f"MoCha: {part.strip()[:160]}", 0.0)
        if proc.wait() != 0:
            raise RuntimeError("MoCha falló:\n" + "\n".join(tail))

        out = list(clip.frames)
        W, H = clip.frames[0].shape[1], clip.frames[0].shape[0]
        for name, idx in chunks:
            res = work / "out" / f"{name}_replaced.mp4"
            if not res.exists():
                raise RuntimeError(f"MoCha no generó {res.name}")
            gen = resample(read_video(res, (W, H)), len(idx))
            for i, f in zip(idx, gen):
                out[i] = f
        return out


BACKENDS = {"wan_animate_space": WanAnimateSpaceBackend, "vace_1_3b": VaceBackend, "wan_animate_14b": WanAnimateBackend, "mocha_14b": MochaBackend}


def make_backend(key: str) -> CharacterBackend:
    return BACKENDS[key](key)
