"""
Swappers "one-shot" pre-entrenados (ONNX): inswapper_128, hyperswap_1a_256, ghost_2_256.

Los tres siguen el mismo esquema:  cara destino alineada + vector de identidad  ->  cara nueva.
Lo que cambia entre ellos es CÓMO se prepara ese vector de identidad y la normalización:

    inswapper : embedding ArcFace normalizado × matriz "emap" (dentro del ONNX), re-normalizado
    hyperswap : embedding ArcFace normalizado tal cual
    ghost     : embedding ArcFace crudo -> red "crossface_ghost" -> espacio de identidad propio de GHOST
"""
from __future__ import annotations

import numpy as np

from .downloads import StatusCb, download_file
from .engine import FaceEngine, Identity
from .models_registry import SWAP_MODELS


class OnnxSwapper:
    def __init__(self, engine: FaceEngine, key: str, identity: Identity | list[Identity], status_cb: StatusCb = None):
        """`identity` puede ser una lista: una identidad por persona elegida en el escaneo (en el mismo
        orden que los targets). Entonces `swap(crop, i)` usa la identidad de la persona i."""
        spec = SWAP_MODELS[key]
        self.key, self.spec = key, spec
        self.type = spec["type"]
        self.size = spec["size"]
        self.template = spec["template"]
        self.mean = np.array(spec["mean"], np.float32)
        self.std = np.array(spec["std"], np.float32)

        download_file(spec["urls"], spec["file"], spec["label"], status_cb)
        if "converter_file" in spec:
            download_file(spec["converter_urls"], spec["converter_file"], "conversor de identidad", status_cb)
        if status_cb:
            status_cb(f"Cargando {spec['label']} en la GPU...", 0.0)

        self.session = engine.session(spec["file"], keep=False)
        self.inputs = {i.name: i for i in self.session.get_inputs()}
        self.fp16 = any("float16" in i.type for i in self.inputs.values())
        self.output_name = self.session.get_outputs()[0].name  # GHOST devuelve además 8 mapas internos
        identities = identity if isinstance(identity, list) else [identity]
        cache: dict[int, np.ndarray] = {}  # la misma fuente asignada a varias personas se prepara una vez
        self.latents = [cache.setdefault(id(i), self._prepare_identity(engine, i)) for i in identities]
        self.multi = len(self.latents) > 1

    # -------------------------------------------------------------- identidad
    def _prepare_identity(self, engine: FaceEngine, identity: Identity) -> np.ndarray:
        if self.type == "inswapper":
            emap = load_inswapper_emap(self.spec["file"])
            latent = identity.normed.reshape(1, -1) @ emap
            latent /= np.linalg.norm(latent)
        elif self.type == "hyperswap":
            latent = identity.normed.reshape(1, -1)
        elif self.type == "ghost":
            conv = engine.session(self.spec["converter_file"])
            latent = conv.run(None, {conv.get_inputs()[0].name: identity.embedding.reshape(1, -1).astype(np.float32)})[0]
            latent = latent.reshape(1, -1)
        else:
            raise ValueError(self.type)
        return latent.astype(np.float16 if self.fp16 else np.float32)

    # -------------------------------------------------------------- inferencia
    def swap(self, crop: np.ndarray, which: int = 0) -> np.ndarray:
        x = crop[:, :, ::-1].astype(np.float32) / 255.0          # BGR -> RGB, 0..1
        x = ((x - self.mean) / self.std).transpose(2, 0, 1)[None]
        if self.fp16:
            x = x.astype(np.float16)
        out = self.session.run([self.output_name], {"target": x, "source": self.latents[which]})[0][0].astype(np.float32)
        out = out.transpose(1, 2, 0)
        if self.type in ("ghost", "hyperswap"):                  # salida en [-1, 1]
            out = out * self.std + self.mean
        return (out.clip(0, 1)[:, :, ::-1] * 255.0).astype(np.float32)


def load_inswapper_emap(model_path) -> np.ndarray:
    """La matriz emap (512×512) es el último initializer del grafo ONNX. Se cachea en .npy
    para no volver a leer los 550 MB del modelo cada vez."""
    cache = model_path.with_suffix(".emap.npy")
    if cache.exists():
        return np.load(cache)
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(model_path))
    emap = numpy_helper.to_array(model.graph.initializer[-1]).astype(np.float32)
    np.save(cache, emap)
    return emap
