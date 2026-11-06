"""Catálogo de modelos: de dónde se descargan y cómo hay que preparar sus entradas.

Las especificaciones (plantilla de alineación, tamaño, media/desviación, conversor de embedding)
coinciden con las que usa FaceFusion (https://github.com/facefusion/facefusion), que publica
estos modelos convertidos a ONNX.
"""
from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
FF_DIR = MODELS_DIR / "facefusion"
INSIGHTFACE_ROOT = MODELS_DIR / "insightface"

BUFFALO_URLS = ["https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip"]


def ff_urls(tag: str, file: str) -> list[str]:
    return [
        f"https://github.com/facefusion/facefusion-assets/releases/download/{tag}/{file}",
        f"https://huggingface.co/facefusion/{tag}/resolve/main/{file}",
    ]


# Puntos de referencia (normalizados 0..1) a los que se llevan los 5 landmarks al alinear.
WARP_TEMPLATES = {
    "arcface_112_v1": [
        [0.35473214, 0.45658929], [0.64526786, 0.45658929], [0.50000000, 0.61154464],
        [0.37913393, 0.77687500], [0.62086607, 0.77687500],
    ],
    "arcface_128": [
        [0.36167656, 0.40387734], [0.63696719, 0.40235469], [0.50019687, 0.56044219],
        [0.38710391, 0.72160547], [0.61507734, 0.72034453],
    ],
}

SWAP_MODELS: dict[str, dict] = {
    "hyperswap_1a_256": {
        "label": "HyperSwap 1a · 256 px",
        "summary": "Recomendado: el más preciso y además el más rápido. Cara a 256 px (4× más "
                   "píxeles que inswapper), conserva muy bien expresión e iluminación.",
        "speed": "alta (~22 fps)", "quality": "alta", "license": "ResearchRAIL (uso de investigación, gratis)",
        "vendor": "FaceFusion, 2025",
        "type": "hyperswap", "template": "arcface_128", "size": 256,
        "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
        "file": FF_DIR / "hyperswap_1a_256.onnx", "urls": ff_urls("models-3.3.0", "hyperswap_1a_256.onnx"),
    },
    "ghost_2_256": {
        "label": "GHOST 2 · 256 px",
        "summary": "Licencia Apache-2.0 (totalmente libre). 256 px con más textura de piel; a veces "
                   "deja rastros de la cara original (vello, bigote).",
        "speed": "media (~16 fps)", "quality": "media-alta", "license": "Apache-2.0",
        "vendor": "ai-forever (Sber AI), 2022",
        "type": "ghost", "template": "arcface_112_v1", "size": 256,
        "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
        "file": FF_DIR / "ghost_2_256.onnx", "urls": ff_urls("models-3.0.0", "ghost_2_256.onnx"),
        "converter_file": FF_DIR / "crossface_ghost.onnx",
        "converter_urls": ff_urls("models-3.4.0", "crossface_ghost.onnx"),
    },
    "inswapper_128": {
        "label": "inswapper · 128 px (InsightFace)",
        "summary": "El clásico de InsightFace: identidad muy marcada, pero la cara sale a 128 px "
                   "y en primeros planos se ve suave/borrosa.",
        "speed": "media (~13 fps)", "quality": "media", "license": "No comercial (InsightFace)",
        "vendor": "InsightFace, 2023",
        "type": "inswapper", "template": "arcface_128", "size": 128,
        "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0],
        "file": MODELS_DIR / "inswapper_128.onnx",
        "urls": [
            "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
            *ff_urls("models-3.0.0", "inswapper_128.onnx"),
        ],
    },
}

# Modelos auxiliares para que la cara no parezca una "máscara pegada".
OCCLUDER = {  # XSeg (DeepFaceLab): qué píxeles de la cara están tapados (manos, micrófono, pelo)
    "file": FF_DIR / "xseg_1.onnx", "urls": ff_urls("models-3.1.0", "xseg_1.onnx"), "size": 256,
}
PARSER = {  # BiSeNet: segmenta piel, cejas, ojos, nariz, boca... (excluye pelo, cuello y fondo)
    "file": FF_DIR / "bisenet_resnet_18.onnx", "urls": ff_urls("models-3.1.0", "bisenet_resnet_18.onnx"),
    "size": 512,
    # 1 piel, 2-3 cejas, 4-5 ojos, 6 gafas, 10 nariz, 11 boca, 12-13 labios
    "regions": [1, 2, 3, 4, 5, 6, 10, 11, 12, 13],
}

ENCODER_MODEL = {"template": "arcface_128", "size": 128}


def public_catalog() -> list[dict]:
    out = []
    for key, m in SWAP_MODELS.items():
        out.append({
            "key": key, "label": m["label"], "summary": m["summary"], "speed": m["speed"],
            "quality": m["quality"], "license": m["license"], "vendor": m["vendor"],
            "downloaded": m["file"].exists() and (
                "converter_file" not in m or m["converter_file"].exists()),
        })
    return out
