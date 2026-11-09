"""
Buscador de videos sugeridos: encuentra, en una página web con muchos videos, aquellos en los que
aparece alguien con rasgos faciales parecidos a los de una fuente (para elegir un video destino
donde el face swap quede más natural: forma de la cara, edad y tono similares).

Dos pasos:

    analyze_page()   1. Descarga la página (y, opcionalmente, las siguientes de la paginación).
                     2. Extrae cada enlace a un video con su miniatura (<a><img></a>, lazy-loading
                        data-src/srcset, <video poster>, og:image...).
                     3. Opcional ("a fondo"): visita cada página de video para sacar más fotos
                        (og:image, poster, thumbnailUrl de JSON-LD).
                     4. Detecta las caras de esas imágenes (SCRFD) y guarda su embedding ArcFace
                        en data/scrape/<id>/ (meta.json + emb.npy): el índice de "actores".
    search()         Compara los embeddings de las fuentes elegidas con todo el índice y devuelve
                     los videos ordenados por parecido (foto de la cara encontrada + enlace).

Límites deliberados: respeta robots.txt, solo sigue enlaces del mismo sitio, pausa entre
peticiones, no inicia sesión ni salta muros de pago, y no descarga videos (solo miniaturas);
el usuario abre el enlace y decide. Las páginas que generan su lista de videos con JavaScript
no exponen las miniaturas en el HTML y no se pueden analizar así.
"""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Optional
from urllib import robotparser
from urllib.parse import urljoin, urlparse, urldefrag

import cv2
import numpy as np
import requests

from .engine import FaceEngine
from .scan import _jpeg_data_uri, face_thumb

USER_AGENT = "FaceSwapLab-research/1.0 (+uso academico; respeta robots.txt)"
TIMEOUT = 15
MAX_HTML_BYTES = 8 * 1024 * 1024
MAX_IMAGE_BYTES = 6 * 1024 * 1024
REQUEST_DELAY = 0.5        # s entre páginas HTML del mismo sitio
MIN_FACE_PX = 20           # en miniaturas pequeñas las caras son chicas; SCRFD ya reescala la imagen
MAX_THUMBS_PER_VIDEO = 4

ProgressCb = Optional[Callable[[str, float], None]]

IMG_ATTRS = ("data-src", "data-original", "data-lazy-src", "data-thumb", "data-thumb-url", "data-poster",
             "data-mediumthumb", "data-image", "data-lazy", "src")
NEXT_TEXT = {"next", "siguiente", "›", "»", ">", "next page", "página siguiente", "pagina siguiente", "→"}
BG_URL = re.compile(r"url\(\s*['\"]?([^'\")]+)['\"]?\s*\)")
SKIP_IMG = re.compile(r"(logo|icon|sprite|avatar-default|blank|placeholder|spacer|pixel)\.|\.svg(\?|$)", re.I)


# --------------------------------------------------------------------------- HTML
class _PageParser(HTMLParser):
    """Recorre el HTML una vez y junta: enlaces con sus imágenes dentro, posters de <video>,
    meta og:image/twitter:image, enlaces de paginación y thumbnailUrl de JSON-LD."""

    def __init__(self, base: str):
        super().__init__(convert_charrefs=True)
        self.base = base
        self.anchors: list[dict] = []           # {"href", "imgs": [...], "title", "text"}
        self.stack: list[dict] = []             # <a> abiertos
        self.posters: list[str] = []
        self.meta_imgs: list[str] = []
        self.title = ""
        self.next_links: list[str] = []
        self._in_title = False
        self._in_ldjson = False
        self._ld = ""

    def _abs(self, url: str | None) -> str | None:
        if not url:
            return None
        url = url.strip()
        if url.startswith("data:") or url.startswith("javascript:"):
            return None
        u = urldefrag(urljoin(self.base, url))[0]
        return u if urlparse(u).scheme in ("http", "https") else None

    def _img_url(self, attrs: dict) -> str | None:
        for key in ("data-srcset", "srcset"):
            if attrs.get(key):  # el último candidato del srcset suele ser el más grande
                parts = [p.strip().split(" ")[0] for p in attrs[key].split(",") if p.strip()]
                if parts:
                    return self._abs(parts[-1])
        for key in IMG_ATTRS:
            if attrs.get(key) and not attrs[key].startswith("data:"):
                return self._abs(attrs[key])
        return None

    def handle_starttag(self, tag, attrs_list):
        attrs = {k: (v or "") for k, v in attrs_list}
        if tag == "a":
            href = self._abs(attrs.get("href"))
            a = {"href": href, "imgs": [], "title": attrs.get("title", ""), "text": ""}
            self.stack.append(a)
            if href:
                self.anchors.append(a)
            if href and (attrs.get("rel", "").lower() == "next"):
                self.next_links.append(href)
        elif tag == "link" and attrs.get("rel", "").lower() == "next":
            if (u := self._abs(attrs.get("href"))):
                self.next_links.append(u)
        elif tag in ("img", "source"):
            url = self._img_url(attrs)
            if url and self.stack:
                self.stack[-1]["imgs"].append(url)
                if not self.stack[-1]["title"]:
                    self.stack[-1]["title"] = attrs.get("alt", "")
        elif tag == "video":
            if (u := self._abs(attrs.get("poster") or attrs.get("data-poster"))):
                self.posters.append(u)
        elif tag == "meta":
            prop = (attrs.get("property") or attrs.get("name") or "").lower()
            if prop in ("og:image", "og:image:url", "twitter:image", "twitter:image:src"):
                if (u := self._abs(attrs.get("content"))):
                    self.meta_imgs.append(u)
        elif tag == "title":
            self._in_title = True
        elif tag == "script" and "ld+json" in attrs.get("type", ""):
            self._in_ldjson, self._ld = True, ""
        # miniaturas como background-image en cualquier elemento dentro de un enlace
        if self.stack and "style" in attrs and (m := BG_URL.search(attrs["style"])):
            if (u := self._abs(m.group(1))):
                self.stack[-1]["imgs"].append(u)

    def handle_endtag(self, tag):
        if tag == "a" and self.stack:
            self.stack.pop()
        elif tag == "title":
            self._in_title = False
        elif tag == "script" and self._in_ldjson:
            self._in_ldjson = False
            for m in re.finditer(r'"thumbnailUrl"\s*:\s*(\[[^\]]*\]|"[^"]+")', self._ld):
                try:
                    v = json.loads(m.group(1))
                except ValueError:
                    continue
                for u in (v if isinstance(v, list) else [v]):
                    if isinstance(u, str) and (au := self._abs(u)):
                        self.meta_imgs.append(au)

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if self._in_ldjson:
            self._ld += data
        if self.stack:
            a = self.stack[-1]
            a["text"] = (a["text"] + " " + data.strip()).strip()[:200]
            if data.strip().lower() in NEXT_TEXT and a["href"]:
                self.next_links.append(a["href"])


def _host(url: str) -> str:
    h = urlparse(url).hostname or ""
    return h[4:] if h.startswith("www.") else h


def _same_site(a: str, b: str) -> bool:
    ha, hb = _host(a), _host(b)
    return ha == hb or ha.endswith("." + hb) or hb.endswith("." + ha)


# --------------------------------------------------------------------------- red
class Fetcher:
    """requests con User-Agent propio, robots.txt (cacheado por sitio), límite de tamaño y una
    pausa mínima entre páginas HTML."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "es,en;q=0.8"})
        self._robots: dict[str, robotparser.RobotFileParser | None] = {}
        self._last_html = 0.0
        self._lock = threading.Lock()

    def allowed(self, url: str) -> bool:
        p = urlparse(url)
        root = f"{p.scheme}://{p.netloc}"
        with self._lock:
            if root not in self._robots:
                rp = robotparser.RobotFileParser()
                try:
                    r = self.session.get(root + "/robots.txt", timeout=TIMEOUT)
                    if r.status_code >= 400:
                        rp = None  # sin robots.txt: todo permitido
                    else:
                        rp.parse(r.text.splitlines())
                except requests.RequestException:
                    rp = None
                self._robots[root] = rp
            rp = self._robots[root]
        return rp is None or rp.can_fetch(USER_AGENT, url)

    def _get(self, url: str, limit: int, accept: str) -> tuple[bytes, str]:
        with self.session.get(url, timeout=TIMEOUT, stream=True, headers={"Accept": accept}) as r:
            r.raise_for_status()
            ctype = r.headers.get("Content-Type", "")
            data = bytearray()
            for chunk in r.iter_content(64 * 1024):
                data += chunk
                if len(data) > limit:
                    raise ValueError("archivo demasiado grande")
            return bytes(data), ctype

    def html(self, url: str) -> tuple[str, str]:
        """Devuelve (html, url final). Lanza PermissionError si robots.txt lo prohíbe."""
        if not self.allowed(url):
            raise PermissionError(f"robots.txt del sitio no permite analizar {url}")
        wait = REQUEST_DELAY - (time.time() - self._last_html)
        if wait > 0:
            time.sleep(wait)
        self._last_html = time.time()
        data, ctype = self._get(url, MAX_HTML_BYTES, "text/html,application/xhtml+xml")
        if "html" not in ctype and "xml" not in ctype and ctype:
            raise ValueError(f"no es una página HTML ({ctype})")
        m = re.search(r"charset=([\w-]+)", ctype)
        return data.decode(m.group(1) if m else "utf-8", errors="replace"), url

    def image(self, url: str) -> np.ndarray | None:
        if not self.allowed(url):
            return None
        try:
            data, ctype = self._get(url, MAX_IMAGE_BYTES, "image/avif,image/webp,image/*")
        except (requests.RequestException, ValueError):
            return None
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None or min(img.shape[:2]) < 48:
            return None
        return img


# --------------------------------------------------------------------------- extracción
def _candidates(parser: _PageParser, page_url: str) -> list[dict]:
    """Enlaces que parecen videos: tienen una imagen dentro y no son la propia página."""
    out: dict[str, dict] = {}
    for a in parser.anchors:
        imgs = [u for u in dict.fromkeys(a["imgs"]) if not SKIP_IMG.search(u)]
        if not imgs or not a["href"] or a["href"] == page_url:
            continue
        c = out.setdefault(a["href"], {"link": a["href"], "title": "", "imgs": []})
        c["title"] = c["title"] or (a["title"] or a["text"]).strip()[:160]
        c["imgs"] += [u for u in imgs if u not in c["imgs"]]
    # la propia página es un video (poster / og:image)
    own = [u for u in dict.fromkeys(parser.posters + parser.meta_imgs) if not SKIP_IMG.search(u)]
    if own:
        out.setdefault(page_url, {"link": page_url, "title": parser.title.strip()[:160], "imgs": own})
    return list(out.values())


def _enrich(fetcher: Fetcher, cand: dict) -> None:
    """Modo "a fondo": añade las imágenes de la página del propio video."""
    try:
        html, url = fetcher.html(cand["link"])
    except Exception:  # noqa: BLE001
        return
    p = _PageParser(url)
    p.feed(html)
    extra = [u for u in dict.fromkeys(p.meta_imgs + p.posters) if not SKIP_IMG.search(u)]
    cand["imgs"] += [u for u in extra if u not in cand["imgs"]]
    if not cand["title"]:
        cand["title"] = p.title.strip()[:160]


def _small(img: np.ndarray, width: int = 320) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= width:
        return img
    return cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- índice en disco
class ScrapeIndex:
    """data/scrape/<id>/meta.json  (página, estado, videos y caras sin embedding)
       data/scrape/<id>/emb.npy    (N×512 float16, una fila por cara, en el orden de meta["faces"])"""

    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.progress: dict[str, dict] = {}   # análisis en marcha

    def list(self) -> list[dict]:
        items = []
        for f in self.root.glob("*/meta.json"):
            try:
                m = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            items.append({k: m.get(k) for k in ("id", "url", "title", "created", "status", "message",
                                                 "n_videos", "n_faces", "n_images")})
        with self.lock:
            live = {k: dict(v) for k, v in self.progress.items()}
        known = {i["id"] for i in items}
        items += [v for k, v in live.items() if k not in known]
        for i in items:
            if i["id"] in live:
                i.update(live[i["id"]])
        return sorted(items, key=lambda i: i.get("created") or 0, reverse=True)

    def get(self, sid: str) -> dict | None:
        return next((i for i in self.list() if i["id"] == sid), None)

    def delete(self, sid: str) -> bool:
        d = self.root / sid
        if not sid.isalnum() or not d.is_dir():
            return False
        shutil.rmtree(d, ignore_errors=True)
        return True

    def load(self, sid: str) -> tuple[dict, np.ndarray] | None:
        d = self.root / sid
        try:
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            emb = np.load(d / "emb.npy").astype(np.float32) if (d / "emb.npy").is_file() else np.zeros((0, 512))
        except (OSError, ValueError):
            return None
        return meta, emb

    def save(self, sid: str, meta: dict, emb: np.ndarray) -> None:
        d = self.root / sid
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "emb.npy", emb.astype(np.float16))
        tmp = d / "meta.json.tmp"
        tmp.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        tmp.replace(d / "meta.json")

    def set_progress(self, sid: str, **fields) -> None:
        with self.lock:
            self.progress.setdefault(sid, {"id": sid}).update(fields)

    def clear_progress(self, sid: str) -> None:
        with self.lock:
            self.progress.pop(sid, None)


def new_scrape_id() -> str:
    return uuid.uuid4().hex[:12]


def analyze_page(engine: FaceEngine, index: ScrapeIndex, sid: str, url: str, deep: bool = False,
                 max_pages: int = 1, max_videos: int = 150) -> None:
    """Paso 1: analiza `url` (y hasta `max_pages` páginas de su paginación) y guarda en el índice
    los rostros encontrados en las miniaturas de sus videos. Pensado para correr en un hilo."""
    created = time.time()
    index.set_progress(sid, url=url, created=created, status="running", message="Descargando la página…",
                       progress=0.0, n_videos=0, n_faces=0, n_images=0, title="")
    fetcher = Fetcher()
    try:
        # --- 1. páginas y candidatos
        cands: dict[str, dict] = {}
        title, page, visited = "", url, set()
        for n in range(max(1, max_pages)):
            if page in visited:
                break
            visited.add(page)
            index.set_progress(sid, message=f"Leyendo la página {n + 1}/{max_pages}…")
            html, final = fetcher.html(page)
            p = _PageParser(final)
            p.feed(html)
            title = title or p.title.strip()[:160]
            for c in _candidates(p, final):
                if c["link"] in cands:
                    cands[c["link"]]["imgs"] += [u for u in c["imgs"] if u not in cands[c["link"]]["imgs"]]
                else:
                    cands[c["link"]] = c
            nxt = next((u for u in p.next_links if _same_site(u, url) and u not in visited), None)
            if not nxt or len(cands) >= max_videos:
                break
            page = nxt
        videos = list(cands.values())[:max_videos]
        if not videos:
            raise ValueError("No se encontraron enlaces con miniaturas en el HTML de la página. "
                             "Puede que el sitio cargue los videos con JavaScript.")
        index.set_progress(sid, title=title, n_videos=len(videos))

        # --- 2. (opcional) más fotos desde cada página de video, solo del mismo sitio
        if deep:
            same = [c for c in videos if _same_site(c["link"], url) and c["link"] != url]
            for i, c in enumerate(same):
                index.set_progress(sid, message=f"Visitando páginas de video {i + 1}/{len(same)}…",
                                   progress=0.3 * (i + 1) / max(len(same), 1))
                _enrich(fetcher, c)

        # --- 3. miniaturas -> caras -> embeddings
        jobs = [(vi, u) for vi, c in enumerate(videos) for u in c["imgs"][:MAX_THUMBS_PER_VIDEO]]
        faces_meta, embs, n_img = [], [], 0
        thumbs: dict[int, str] = {}
        base = 0.3 if deep else 0.0
        with ThreadPoolExecutor(max_workers=8) as pool:   # descarga en paralelo, detección en este hilo
            for k, ((vi, u), img) in enumerate(zip(jobs, pool.map(lambda j: fetcher.image(j[1]), jobs))):
                index.set_progress(sid, message=f"Buscando rostros en las miniaturas {k + 1}/{len(jobs)}…",
                                   progress=base + (1 - base) * (k + 1) / len(jobs))
                if img is None:
                    continue
                n_img += 1
                dets = [d for d in engine.detect(img) if d.size >= MIN_FACE_PX and d.score >= 0.55]
                if not dets:
                    continue
                if vi not in thumbs:
                    thumbs[vi] = _jpeg_data_uri(_small(img), 70)
                for det, emb in zip(dets, engine.embed(img, dets)):
                    faces_meta.append({"video": vi, "img": u,
                                       "face": _jpeg_data_uri(face_thumb(img, det.bbox, size=112), 80)})
                    embs.append(emb)
                index.set_progress(sid, n_faces=len(faces_meta), n_images=n_img)

        out_videos = [{"link": c["link"], "title": c["title"], "thumb": thumbs.get(i, ""),
                       "img": c["imgs"][0] if c["imgs"] else ""} for i, c in enumerate(videos)]
        meta = {"id": sid, "url": url, "title": title, "created": created, "status": "done",
                "message": f"{len(faces_meta)} rostros en {n_img} imágenes de {len(videos)} videos",
                "n_videos": len(videos), "n_faces": len(faces_meta), "n_images": n_img,
                "videos": out_videos, "faces": faces_meta}
        index.save(sid, meta, np.stack(embs) if embs else np.zeros((0, 512), np.float32))
    except Exception as exc:  # noqa: BLE001
        meta = {"id": sid, "url": url, "title": "", "created": created, "status": "error",
                "message": str(exc), "n_videos": 0, "n_faces": 0, "n_images": 0, "videos": [], "faces": []}
        index.save(sid, meta, np.zeros((0, 512), np.float32))
    finally:
        index.clear_progress(sid)


def search(index: ScrapeIndex, models: list[tuple[str, np.ndarray]], scrape_ids: list[str] | None = None,
           min_sim: float = 0.0, limit: int = 200) -> list[dict]:
    """Paso 2: videos del índice ordenados por parecido con alguna de las fuentes.

    `models` = [(id de la fuente, embedding ArcFace normalizado)]. Por video se queda la cara más
    parecida. Referencia de la similitud coseno ArcFace: ≥ 0.40 suele ser la misma persona;
    0.20–0.40, rasgos parecidos (forma de cara, edad, etnia); < 0.15, nada que ver."""
    if not models:
        return []
    M = np.stack([m for _, m in models]).astype(np.float32)
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    ids = scrape_ids or [i["id"] for i in index.list() if i.get("status") == "done"]
    best: dict[str, dict] = {}
    for sid in ids:
        loaded = index.load(sid)
        if loaded is None:
            continue
        meta, emb = loaded
        if not len(emb):
            continue
        emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-6)
        sims = emb @ M.T                                   # caras × fuentes
        for fi, row in enumerate(sims):
            mi = int(row.argmax())
            s = float(row[mi])
            if s < min_sim:
                continue
            f = meta["faces"][fi]
            v = meta["videos"][f["video"]]
            prev = best.get(v["link"])
            if prev is None or s > prev["similarity"]:
                best[v["link"]] = {"link": v["link"], "title": v["title"], "thumb": v["thumb"], "face": f["face"],
                                   "similarity": round(s, 3), "source_id": models[mi][0],
                                   "page": meta["url"], "scrape_id": sid}
    return sorted(best.values(), key=lambda r: -r["similarity"])[:limit]
