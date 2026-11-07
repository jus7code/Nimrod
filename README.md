# FaceSwap Lab — face swap y character swap

Aplicación web para estudiar la sustitución de personas en video. Subes una **fuente** (foto o video
corto de la persona autorizada A) y un **video destino**, eliges en el video **a quién cambiar** y la
app genera el resultado conservando pose, expresión, movimiento, iluminación y audio.

Hay **dos modos**, que se eligen con una casilla al principio de la página:

| Modo | Qué cambia | Cómo | Velocidad |
|---|---|---|---|
| **Face swap** | Solo la cara (pelo, cuerpo, ropa y fondo se conservan) | Modelos de swap facial de 128–256 px + máscaras | ~20 fps, video completo |
| **Character swap** | La persona entera: cara, pelo, cuerpo y ropa (p. ej. tú con traje y corbata en una conferencia) | Modelos de **difusión de video** que regeneran la silueta de la persona | minutos por cada 5 s, tramos de hasta 30 s |

Antes de cualquiera de los dos, el video destino se **escanea** en unos segundos: se detectan y
agrupan los rostros por identidad y eliges la persona (o personas) a alterar. Al final, una casilla
obligatoria confirma que todas las personas involucradas están autorizadas.

### Face swap: cuatro métodos

| | Método | Tipo | Resolución de la cara | Necesita entrenar | Velocidad medida* |
|---|---|---|---|---|---|
| 1 | **HyperSwap 1a** (FaceFusion) | pre-entrenado, *one-shot* | 256 px | No | **~22 fps** |
| 2 | **GHOST 2** (ai-forever) | pre-entrenado, *one-shot* | 256 px | No | ~16 fps |
| 3 | **inswapper** (InsightFace) | pre-entrenado, *one-shot* | 128 px | No | ~13 fps |
| 4 | **Encoder propio** (autoencoder) | entrenado por nosotros para un par A↔B | 128 px | Sí (5–120 min) | ~20 fps al convertir |

\* RTX 4050 Laptop (6 GB), video vertical de 720×1280, una cara, segmentación + color + suavizado
activados y XSeg desactivado. Un video de 1:20 a 30 fps (2 400 frames) tarda unos 2 minutos con HyperSwap.

### Character swap: tres modelos (cuatro formas de ejecutarlos)

| | Modelo | Dónde corre | Precisión | Velocidad (5 s de video) | Licencia |
|---|---|---|---|---|---|
| 0 | **Wan2.2-Animate 14B · Space gratuito** — *sin descargar nada* (por defecto) | Space público de Hugging Face (GPU A10G, ZeroGPU) | alta, con re-iluminación | ~2–6 min + cola* | gratis dentro de la cuota de HF |
| 1 | **Wan2.1-VACE 1.3B** — *el ligero* | **en esta GPU** (6 GB, con offload) | media | ~3–8 min en modo turbo* | Apache-2.0 |
| 2 | **Wan2.2-Animate 14B** (replace) — *el equilibrado* | nube (fal.ai, de pago) | alta, con re-iluminación | ~1–3 min | Apache-2.0 (pesos) |
| 3 | **MoCha 14B** — *el más preciso* | GPU externa de ≥ 48 GB | muy alta (mejor en benchmarks) | ≥ 10 min en A100 | ver repositorio |

\* Estimaciones: el cableado de VACE se verificó con un modelo diminuto de pesos aleatorios, pero el
modelo real de 19 GB no se ha ejecutado todavía en esta máquina, y el Space aún no se ha llamado con un
video real (ver sección 7.7).

**¿No quieres descargar 19 GB?** Usa la opción 0: el mismo Wan2.2-Animate que la opción 2, pero
ejecutado gratis en un Space de Hugging Face. La app solo sube el recorte de la persona (unos MB) y la
foto, y baja el resultado; en tu disco no se guarda ningún modelo de difusión (sección 7.4b).

No existe hoy un modelo de character swap que sea a la vez ligero, rápido **y** muy preciso: los
precisos son de 14 000 millones de parámetros y no caben en 6 GB. Por eso la app ofrece los tres
extremos del compromiso (sección 7).

> **Uso académico y con consentimiento.** Proyecto personal con personas que han autorizado el uso
> de su imagen. La página exige confirmar el consentimiento y cada frame lleva la marca
> **"CONTENIDO SINTETICO (IA)"** (se quita para pruebas internas con `FACESWAP_WATERMARK=0`).

---

## Índice

1. [Instalación y arranque](#1-instalación-y-arranque)
2. [Cómo se usa la página](#2-cómo-se-usa-la-página)
3. [El pipeline común, paso a paso](#3-el-pipeline-común-paso-a-paso)
4. [Por qué antes parecía una máscara y cómo se evita](#4-por-qué-antes-parecía-una-máscara-y-cómo-se-evita)
5. [Los modelos pre-entrenados y en qué se diferencian](#5-los-modelos-pre-entrenados-y-en-qué-se-diferencian)
6. [El encoder propio: cómo funciona y cómo se entrena](#6-el-encoder-propio-cómo-funciona-y-cómo-se-entrena)
7. [Character swap: la persona completa](#7-character-swap-la-persona-completa)
8. [Escaneo de rostros y selección de la persona](#8-escaneo-de-rostros-y-selección-de-la-persona)
9. [Comparativa final: qué método usar](#9-comparativa-final-qué-método-usar)
10. [Modelos auxiliares](#10-modelos-auxiliares)
11. [Rendimiento: qué cuesta cada paso](#11-rendimiento-qué-cuesta-cada-paso)
12. [Estructura del código](#12-estructura-del-código)
13. [Limitaciones y siguientes experimentos](#13-limitaciones-y-siguientes-experimentos)
14. [Ética](#14-ética)
15. [Referencias](#15-referencias)

---

## 1. Instalación y arranque

Requisitos: Python 3.10+ (probado con 3.13), driver NVIDIA reciente. CUDA y cuDNN llegan por pip,
no hace falta instalar el CUDA Toolkit.

```powershell
cd C:\Users\rodri\OneDrive\Documentos\nimrod
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# insightface instala "onnxruntime" (CPU), que no puede convivir con la versión GPU:
.\.venv\Scripts\python.exe -m pip uninstall -y onnxruntime
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps onnxruntime-gpu
# PyTorch con CUDA (lo usan el "Encoder propio" y el character swap):
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu130
# ultralytics instala torchvision SOLO-CPU desde PyPI; se reemplaza por la de CUDA:
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps torchvision --index-url https://download.pytorch.org/whl/cu130
```

### Configuración opcional del character swap

| Variable | Para qué |
|---|---|
| `HF_TOKEN` | Token (gratuito) de una cuenta de [Hugging Face](https://huggingface.co/settings/tokens) para el **Space gratuito**: sin él se usa la cuota anónima de GPU, que es muy pequeña |
| `WAN_ANIMATE_SPACE` | Space a usar (por defecto `alexnasa/Wan2.2-Animate-ZEROGPU`). Puedes duplicarlo en tu cuenta y poner aquí el tuyo |
| `FAL_KEY` | Clave de API de [fal.ai](https://fal.ai) para **Wan2.2-Animate** (de pago, por uso). Sin ella el modelo aparece desactivado |
| `WAN_ANIMATE_ENDPOINT` | Endpoint de fal.ai (por defecto `fal-ai/wan/v2.2-14b/animate/replace`; fal.ai anuncia que lo retira el 4-ene-2027) |
| `MOCHA_DIR` | Ruta al repositorio oficial de **MoCha** ya configurado (ver 7.5). Sin ella el modelo aparece desactivado |
| `MOCHA_PYTHON` | Python del entorno de MoCha (por defecto el de esta app) |
| `MOCHA_CKPT` | Ruta al checkpoint de MoCha si no está en `checkpoints/step18500.ckpt` |

```powershell
$env:FAL_KEY = "tu-clave"     # en la misma terminal, antes de arrancar uvicorn
```

Arrancar:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

y abrir <http://127.0.0.1:8000>. Para pararlo, `Ctrl + C`.

Los modelos se descargan solos la primera vez que se usan (con progreso en la página y reanudación si
se corta la conexión): `buffalo_l` (~280 MB), HyperSwap (~400 MB), GHOST (~740 MB + 22 MB),
inswapper (~550 MB), BiSeNet (~53 MB) y XSeg (~70 MB). Quedan en `models/`.

Además, la primera vez que se usa el character swap se descargan YOLO11s-seg (~21 MB) y, si eliges
VACE, el modelo en `models/character/` (**~19 GB**: 11,4 GB del codificador de texto umT5 + 7,2 GB del
transformer + 0,5 GB del VAE). Comprueba que tienes espacio en disco.

## 2. Cómo se usa la página

0. **Modo**: *Face swap* (solo la cara) o *Character swap* (persona completa).
1. **Persona autorizada (fuente)**.
   - Face swap: una selfie frontal o, mejor, un video de 10–60 s girando un poco la cabeza. Con video,
     los modelos pre-entrenados promedian la identidad de 24 frames, y el encoder tiene muchas caras.
   - Character swap: una **foto de cuerpo entero**, de frente, bien iluminada y **con la ropa que debe
     aparecer** (traje, corbata…). Si subes un video se elige automáticamente el frame más frontal.
2. **Video destino**. Al subirlo se **escanea** solo (unos segundos; sección 8).
3. **¿A quién se cambia?**: aparece una miniatura por persona detectada, con el porcentaje del video
   en el que sale (al pasar el ratón se ve el frame completo con su caja). En face swap puedes marcar una
   o varias; en character swap, una. Las personas que salen en un solo frame se ocultan bajo
   "Mostrar N más".
4. **Modelo**.
   - Face swap: uno de los tres pre-entrenados o el *Encoder propio*, y las opciones de integración.
   - Character swap: VACE / Wan-Animate / MoCha, el **tramo** (inicio y duración, 1–30 s), una
     descripción opcional en inglés de la persona nueva (la usa VACE, p. ej. *"a man wearing a dark navy
     suit, white shirt and a red tie, giving a talk on a conference stage"*), modo turbo y 720p.
5. **Verificación de autorización**: casilla obligatoria confirmando que todas las personas
   involucradas —la de la fuente y las seleccionadas en el video— han dado su consentimiento. Sin ella
   el botón está desactivado y el servidor rechaza el trabajo. Pulsar **Generar**. La barra muestra la
   fase, el progreso y el tiempo restante.

### Si se corta a mitad (se cierra el servidor, se apaga el portátil…)

El face swap no escribe un único MP4 de golpe: lo escribe en **tramos de ~10 s**
(`data/work/<id>/segments/seg_00000.mp4`, `seg_00001.mp4`…), y cada trabajo guarda su estado en
`data/work/<id>/job.json`. Al terminar, los tramos se unen sin recodificar y se añade el audio.

- Si el servidor muere, como mucho se pierden los ~10 s del tramo que se estaba escribiendo.
- Al volver a arrancarlo, la página muestra la tarjeta **Trabajos interrumpidos** con
  **Reanudar** (sigue desde el último tramo terminado), **Descargar lo generado** (une lo que hay,
  con audio) y **Descartar**. El video reanudado es idéntico al que habría salido sin interrupción.
- Mientras un trabajo está en marcha también aparece *Descargar lo generado hasta ahora*.
- Si solo se pierde la conexión con la página, esta sigue reintentando y retoma el seguimiento sola.
- Encoder propio: el entrenamiento guarda un checkpoint cada 5 min; al reanudar solo entrena lo que
  faltaba (o nada, si ya estaba generando el video).
- Character swap: se guardan los parámetros y se puede relanzar, pero se genera **desde el principio**
  (el token de Hugging Face no se guarda: sin él se usa la cuota anónima).

Los tramos se borran al terminar bien. `data/` está dentro de OneDrive: conviene excluir `data/work`
de la sincronización para que no suba esos archivos temporales.

---

## 3. El pipeline común, paso a paso

Los cuatro métodos comparten todo excepto el paso 4 (el generador de la cara):

```
 frame del video
    │
    ▼
 (1) SCRFD ── cajas + 5 landmarks por cara
    │
    ▼
 (1b) Filtro de persona ── ArcFace de cada cara vs. las personas elegidas en el escaneo (sección 8)
    │
    ▼
 (2) Suavizado temporal de los landmarks ── quita el temblor entre frames
    │
    ▼
 (3) Alineación ── transformación de similitud M que lleva los 5 puntos a una plantilla fija
    │               → recorte de 128×128 o 256×256 con la cara siempre en la misma posición
    ▼
 (4) GENERADOR ── HyperSwap │ GHOST │ inswapper │ Encoder propio      ◄── identidad de la fuente
    │
    ▼
 (5) Máscara suave = caja difuminada ∩ segmentación BiSeNet ∩ oclusión XSeg (opcional)
    │
    ▼
 (6) Transferencia de color LAB (dentro de la máscara)
    │
    ▼
 (7) Paste-back con M⁻¹ y mezcla alfa con la máscara
    │
    ▼
 ffmpeg (H.264 + audio original, en un solo paso)
```

### (1) Detección — SCRFD

El detector `det_10g.onnx` (paquete `buffalo_l` de InsightFace) devuelve cada cara con su caja, una
puntuación y **5 landmarks**: ojo izquierdo, ojo derecho, punta de la nariz y las dos comisuras de la
boca. Trabaja con la imagen reescalada a 640×640.

### (2) Suavizado temporal

El detector analiza cada frame por separado, así que los landmarks "tiemblan" 1–3 píxeles aunque la
cara esté quieta, y la cara generada vibra. `KpsSmoother` ([app/engine.py](app/engine.py)) asocia cada
cara con la del frame anterior y mezcla posiciones con un peso **adaptativo**:

```
desplazamiento pequeño  → α ≈ 0.35  (promedia mucho: elimina el temblor)
desplazamiento grande   → α = 1.0   (sigue al detector: sin retraso en movimientos rápidos)
kps_suavizados = α · kps_detectados + (1 − α) · kps_anteriores
```

### (3) Alineación

Se estima la transformación de similitud (rotación + escala + traslación) que lleva los 5 landmarks a
la **plantilla** del modelo (`arcface_128` o `arcface_112_v1`, ver
[app/models_registry.py](app/models_registry.py)) y se recorta la cara. Así el generador siempre ve
los ojos, la nariz y la boca en el mismo sitio. La matriz `M` se guarda para deshacer el recorte en (7).

### (4) Generador

Ver secciones 5 y 6.

### (5)–(7) Integración

Ver sección 4.

### Identidad de la fuente (ArcFace)

En los modelos pre-entrenados la persona A se representa con un **vector de 512 números** calculado por
ArcFace (`w600k_r50.onnx`, ResNet-50 entrenada con ~600 000 identidades). ArcFace se entrena con la
pérdida *Additive Angular Margin*:

```
L = −log( e^{s·cos(θ_y + m)} / ( e^{s·cos(θ_y + m)} + Σ_{j≠y} e^{s·cos θ_j} ) )      s = 64, m = 0.5
```

Eso agrupa las fotos de una misma persona en una dirección de una hiperesfera y separa a personas
distintas por al menos `m` radianes. Resultado: el vector describe **quién** es la persona y no su pose,
expresión o luz. Si la fuente es un video, se promedian los vectores de 24 frames.

---

## 4. Por qué antes parecía una máscara y cómo se evita

La primera versión pegaba la salida de inswapper con un **rectángulo erosionado y difuminado**. Eso
causaba cuatro problemas visibles:

| Síntoma | Causa | Solución en esta versión |
|---|---|---|
| Borde visible en frente, mejillas o mandíbula | El rectángulo incluye pelo, fondo y cuello | **Segmentación BiSeNet**: solo se sustituyen piel, cejas, ojos, nariz, boca y labios |
| La cara tiene otro tono o luz | El generador no reproduce exactamente la iluminación | **Transferencia de color en LAB**: se igualan media y desviación de cada canal dentro de la cara |
| La cara "flota" o vibra | Landmarks con ruido en cada frame | **Suavizado temporal adaptativo** de los landmarks |
| Las manos u objetos desaparecen detrás de la cara | La máscara no sabe qué tapa la cara | **XSeg** (opcional): máscara de oclusión |
| Cara borrosa | inswapper genera a 128 px | **Modelos de 256 px** (HyperSwap, GHOST) |

**Cómo se construye la máscara final** (en el espacio del recorte alineado):

```
máscara = mínimo( caja_difuminada,          # bordes del recorte a 0, difuminado gaussiano (30 %)
                  región_BiSeNet,           # 1 en piel/ojos/nariz/boca, 0 en pelo/fondo/cuello
                  oclusión_XSeg )           # 0 donde algo tapa la cara (si está activada)
cada máscara binaria se suaviza:  (blur_gauss(m, σ=5).clip(0.5, 1) − 0.5) · 2
```

El último paso es importante: convierte el borde duro en una **rampa suave de unos 10 px hacia dentro**,
así la transición entre cara nueva y piel original es gradual y no se ve una costura.

**Transferencia de color** (`color_transfer`): se pasa la cara generada y la original a espacio LAB
(L = luminosidad, A/B = color), y dentro de la máscara:

```
nueva = (generada − media_gen) · ratio + media_orig
ratio = clip(std_orig / std_gen, 0.6, 1.6)   para L   (no exagerar contrastes)
ratio = clip(std_orig / std_gen, 0.0, 1.6)   para A/B (una selfie en color sobre un video en
                                                       blanco y negro se desatura del todo)
```

**Dónde se calcula la segmentación**: en los modelos pre-entrenados, BiSeNet se aplica a la cara
**generada** (como hace FaceFusion). En el encoder propio se aplica a la cara **original** del video,
porque al principio del entrenamiento el encoder genera caras borrosas en las que BiSeNet no encuentra
piel y la máscara saldría vacía.

**Paste-back**: se aplica `M⁻¹` solo a la región de la cara (no a todo el frame) y se mezcla:
`frame = máscara · cara_nueva + (1 − máscara) · frame`.

---

## 5. Los modelos pre-entrenados y en qué se diferencian

Los tres son **one-shot**: se entrenaron una vez con millones de caras y funcionan con **cualquier
persona** a partir de una sola foto, sin entrenar nada. Todos tienen la misma interfaz:

```
generador( cara_destino_alineada [1×3×N×N],  vector_identidad [1×512] )  →  cara_nueva [1×3×N×N]
```

y la misma idea interna: un **encoder** extrae los atributos de la cara destino (pose, expresión,
luz), la **identidad** se inyecta mediante capas de normalización condicionadas, y un **decoder**
dibuja la cara nueva. Lo que cambia es la arquitectura, la resolución, el entrenamiento y **cómo se
codifica la identidad**:

### 5.1 inswapper_128 (InsightFace, 2023)

- **Entrada**: cara 128×128 en RGB `[0, 1]`, plantilla `arcface_128`.
- **Identidad**: `latente = normalizar(embedding_ArcFace_normalizado · emap)`, donde `emap` es una
  matriz 512×512 guardada dentro del ONNX que traduce el espacio de ArcFace al espacio que usa el generador.
- **Arquitectura** (deducida del grafo; no hay paper ni código de entrenamiento): encoder convolucional →
  ~6 bloques residuales con **AdaIN** → decoder.
  `AdaIN(h, id) = γ(id) · (h − μ(h)) / σ(h) + β(id)`: se borran las estadísticas de cada canal y se
  imponen unas calculadas a partir de la identidad.
- **Pros**: identidad muy marcada; es el más usado y documentado.
- **Contras**: 128 px se nota en primeros planos (piel suave, dientes poco definidos), y en esta GPU
  es el más lento de los tres porque su generador tiene muchos canales.
- **Licencia**: modelo solo para investigación no comercial.

### 5.2 HyperSwap 1a 256 (FaceFusion, 2025) — recomendado

- **Entrada**: cara 256×256 normalizada a `[−1, 1]`, plantilla `arcface_128`.
- **Identidad**: el embedding ArcFace **normalizado, directamente**, sin matriz de proyección.
- **Salidas**: la cara generada y además una máscara propia (esta app usa BiSeNet en su lugar).
- **Arquitectura**: no publicada en detalle; es un generador condicionado por identidad entrenado por el
  equipo de FaceFusion a 256 px. Existen variantes 1a/1b/1c entrenadas con distintas configuraciones.
- **Pros**: 4× más píxeles que inswapper, conserva muy bien expresión, mirada e iluminación, y además
  es el **más rápido** de los tres en esta GPU (~22 ms por cara).
- **Contras**: identidad algo menos "agresiva" que inswapper en perfiles extremos.
- **Licencia**: ResearchRAIL (uso de investigación, gratuito).

### 5.3 GHOST 2 256 (ai-forever / Sber AI, 2022)

- **Paper**: *GHOST — A New Face Swap Approach for Image and Video Domains* (IEEE Access 2022).
- **Arquitectura**: basada en **AEI-Net de FaceShifter**:
  - un **encoder de atributos multinivel** (tipo U-Net) extrae mapas de características de la cara
    destino a 8 escalas (de 2×2 a 256×256; el ONNX los devuelve como salidas extra),
  - un generador con bloques **AAD** (*Adaptive Attentional Denormalization*): en cada escala, una
    máscara de atención **aprendida** decide píxel a píxel cuánto tomar de la identidad (AdaIN) y cuánto
    de los atributos del destino. "2" indica 2 bloques AAD por nivel (hay variantes 1, 2 y 3).
  - Pérdidas extra respecto a FaceShifter: **pérdida de ojos** (conservar la dirección de la mirada) y
    técnicas de estabilización para video.
- **Identidad**: GHOST se entrenó con **otro** modelo ArcFace. Para usar nuestro embedding se pasa por
  `crossface_ghost.onnx`, una red pequeña (creada por FaceFusion) que **traduce** el embedding de
  `w600k_r50` al espacio de identidad que GHOST espera.
- **Pros**: licencia **Apache-2.0** (la única totalmente libre, incluso comercial), buena textura de piel.
- **Contras**: a veces deja rastros de la cara original (vello facial, bigote) porque la atención AAD
  conserva más "atributos" del destino.

### 5.4 Resumen de diferencias técnicas

| | inswapper_128 | HyperSwap 1a 256 | GHOST 2 256 |
|---|---|---|---|
| Plantilla de alineación | arcface_128 | arcface_128 | arcface_112_v1 |
| Normalización entrada | `[0, 1]` | `[−1, 1]` | `[−1, 1]` |
| Identidad | ArcFace × emap | ArcFace normalizado | ArcFace → crossface_ghost |
| Inyección de identidad | AdaIN | condicionamiento (no publicado) | AAD (AdaIN + atención) |
| ms por cara (RTX 4050) | ~49 | ~18–22 | ~31 |
| Tamaño | 554 MB | 403 MB | 739 MB + 22 MB |
| Licencia | no comercial | ResearchRAIL | Apache-2.0 |

---

## 6. El encoder propio: cómo funciona y cómo se entrena

A diferencia de los anteriores, este modelo **no sabe nada de caras al empezar**: se entrena desde cero
en la página para **un par concreto de personas** (A = fuente, B = video). Es el enfoque original de los
*deepfakes* (DeepFaceLab, faceswap).

### 6.1 Arquitectura ([app/encoder/model.py](app/encoder/model.py))

```
                          ┌──► Decoder_A ──► reconstrucción de A
 cara 128×128 ──► Encoder ┤
                          └──► Decoder_B ──► reconstrucción de B

 Encoder:  128 → 64 → 32 → 16 → 8   (4 convoluciones 5×5 con stride 2: 64,128,256,512 canales)
           aplanar → Linear(32768 → 512) → Linear(512 → 32768)      ← cuello de botella
           → 8×8×512 → upsample → 16×16×512
 Decoder:  16 → 32 → 64 → 128  (Conv + PixelShuffle; bloques residuales a 32 y 64 px) → Conv → sigmoide (RGB)
```

El bloque residual a 128 px se quitó a propósito: costaba ~15 % del cómputo y casi no mejoraba la
calidad en entrenamientos cortos.

- **Encoder compartido**: el mismo encoder codifica caras de A y de B. Para que los dos decoders puedan
  reconstruir desde ese código, el encoder aprende una representación común de **pose, expresión,
  mirada e iluminación**.
- **Cuello de botella de 512**: es tan estrecho que no caben los detalles de identidad. Esos detalles
  quedan "dibujados" en los pesos de cada decoder.
- **Swap**: `cara de B → Encoder → Decoder_A`. El decoder de A solo sabe dibujar a A, así que dibuja
  a A con la pose y expresión que el encoder extrajo de B.
- **PixelShuffle** para subir resolución (reordena canales en píxeles) en vez de convoluciones
  transpuestas, que producen artefactos de "tablero de ajedrez".

### 6.2 Entrenamiento ([app/encoder/trainer.py](app/encoder/trainer.py))

1. **Extracción**: se detectan y alinean hasta 1 500 caras de A y de B (128×128, plantilla
   `arcface_128`) y BiSeNet calcula la máscara de cada cara.
2. **Aumento de datos en la GPU**, a cada iteración:
   - transformación afín aleatoria (rotación ±10°, escala ±5 %, desplazamiento ±5 %, espejo) aplicada
     igual a la entrada y al objetivo;
   - **random warp** (deformación suave aleatoria) **solo en la entrada**: la red tiene que reconstruir
     la cara sin deformar, así que aprende la *estructura* facial en vez de copiar píxeles.
3. **Pérdida**, pesada por la máscara (la cara cuenta 10× más que el fondo):
   `L = 10 · L1 + 10 · DSSIM`. L1 mide error por píxel; **DSSIM** = (1 − SSIM)/2 mide diferencias de
   estructura y contraste local, y da caras más nítidas que L1 sola.
4. **Optimización**: Adam (lr = 2·10⁻⁴, β₁ = 0.5), *batch* 8 de A + 8 de B, precisión mixta
   **bfloat16** y formato `channels_last` (usan los Tensor Cores de la RTX). Velocidad medida en la
   RTX 4050 limitada a 45 W: **~9 iteraciones/s**, es decir, ~5 400 iteraciones en 10 minutos. Se eligió
   *batch* 8 porque da casi el doble de actualizaciones por minuto que 16, y en entrenamientos cortos eso
   converge antes.
5. **Presupuesto de tiempo**: entrena hasta agotar los minutos elegidos y muestra iteraciones, pérdida e it/s.
6. **Checkpoint por par**: se guarda en `data/encoder_models/<hash_fuente>_<hash_video>.pt`. Si se
   vuelven a subir **los mismos archivos**, continúa donde se quedó; así se puede entrenar en sesiones de
   20 minutos y acumular horas.
7. **Conversión**: igual que los demás métodos (mismas máscaras, color y suavizado), pero el generador
   es `Decoder_A(Encoder(cara_B))`.

### 6.3 Qué esperar

| Tiempo acumulado | Resultado típico |
|---|---|
| 5–10 min | Se reconoce la forma de la cara de A, pero borrosa |
| 30–60 min | Rasgos claros, algo suave; expresiones básicas correctas |
| 3–6 h (varias sesiones) | Comparable a un modelo pre-entrenado en ese par concreto |

- **Necesita un video de la fuente.** Con una sola selfie, el decoder A solo ve una expresión y siempre
  la dibuja (boca cerrada, misma mirada).
- **Ventaja**: puede capturar detalles personales (lunares, forma exacta de los dientes, arrugas) que un
  modelo one-shot no ve en una sola foto, y la identidad no depende de ArcFace.
- **Desventaja**: hay que entrenar para cada par y tarda mucho más.

---

## 7. Character swap: la persona completa

### 7.1 Por qué hace falta otro tipo de modelo

Un face swap solo redibuja un recorte de cara alineado de 128–256 px: el resto del cuerpo queda
intacto. Para "ponerte a ti con corbata dando la conferencia" hay que **inventar píxeles que no
existen** (tu ropa, tu pelo, tus manos) en cada frame, **con el movimiento de la persona original** y
de forma **coherente en el tiempo**. Eso lo hacen hoy los **modelos de difusión de video**:

```
ruido 3D (frames × alto × ancho, en el espacio latente de un VAE de video)
   │   repetir N pasos:  DiT (transformer de difusión) predice cómo quitar ruido,
   │                     condicionado por: el video original (lo que se conserva),
   │                     la máscara (dónde generar), la referencia (a quién dibujar),
   │                     y opcionalmente texto, pose o rasgos faciales
   ▼
latentes limpios ──► decoder del VAE ──► frames
```

Los tres modelos usan como base **Wan** (Alibaba, 2025): un VAE causal 3D que comprime ×4 en el
tiempo y ×8 en espacio, y un DiT con atención completa espacio-temporal entrenado con *flow matching*.
Trabajan a 16 fps y 480p (832×480), en trozos de 81 frames (≈ 5 s).

### 7.2 El pipeline común de la app ([app/character/pipeline.py](app/character/pipeline.py))

```
 foto/video de la persona autorizada ──► (1) Referencia: frame más frontal → YOLO11-seg →
                                             cuerpo recortado sobre fondo blanco + primer plano de la cara
 video destino + persona elegida en el escaneo
    │
    ▼
 (2) Seguimiento a 16 fps en el tramo elegido:
       YOLO11s-seg + ByteTrack (máscara e id de cada persona)  +  SCRFD/ArcFace (¿cuál es la elegida?)
       la persona elegida = la máscara que contiene la cara reconocida; si se gira y no se ve la cara,
       se sigue su id de ByteTrack
    ▼
 (3) Recorte FIJO alrededor de la persona en todo el tramo, con la proporción del modelo
       (480×832 vertical u 832×480 horizontal): el generador dedica todos sus píxeles a la persona
       y no se toca a nadie más. Máscaras dilatadas un 2,5 % y con máximo temporal ±1 frame
    ▼
 (4) GENERADOR ── VACE 1.3B │ Wan-Animate 14B │ MoCha 14B
    ▼
 (5) Composición a los fps originales: el recorte generado vuelve al frame con M⁻¹, mezclado con
       máscara suave = (silueta original ∪ silueta de la persona generada) difuminada.
       Fuera de esa máscara el video es idéntico al original
    ▼
 ffmpeg: H.264 + audio original del tramo + marca "CONTENIDO SINTETICO (IA)"
```

Detalles:
- La silueta de la persona **generada** se vuelve a segmentar con YOLO y se une a la original, porque
  la ropa nueva (una chaqueta) puede ser más ancha que la original (una camiseta).
- La persona se genera a 16 fps; el fondo conserva los fps del video (30/60), y en cada frame se pega
  el frame generado más cercano en el tiempo.
- El resultado es **solo el tramo procesado** (1–30 s) con su audio.

### 7.3 Modelo 1 — Wan2.1-VACE 1.3B (el ligero, local)

- **Qué es**: *VACE: All-in-One Video Creation and Editing* (Alibaba, ICCV 2025) sobre Wan2.1-T2V-1.3B.
  Un solo modelo para texto→video, referencia→video y edición enmascarada; su tarea *Swap Anything*
  es exactamente esto: máscara + imagen de referencia.
- **Cómo condiciona** (*Video Condition Unit*): el video se parte en
  `inactivo = video · (1 − máscara)` (lo que se conserva) y `reactivo = video · máscara` (lo que se
  regenera); ambos se codifican con el VAE y, junto con la máscara reducida a la resolución latente,
  entran a unos **bloques "VACE" paralelos** al DiT (como un ControlNet) cuyas salidas se suman a 15 de
  sus 30 capas. La **referencia** se codifica como frames extra al principio de la secuencia, así la
  atención espacio-temporal puede "copiar" su apariencia.
- **Texto**: el DiT necesita la descripción codificada por **umT5-XXL** (5 700 M parámetros, 11 GB).
  Para que no ocupe la GPU, la app la ejecuta **una vez en la CPU** y guarda el resultado en
  `models/character/prompt_cache/` (reusar la misma descripción no vuelve a cargarlo).
- **Memoria en 6 GB**: transformer en bf16 (3,6 GB) con `enable_model_cpu_offload` (transformer y VAE
  se turnan en la GPU) y el VAE decodificando por baldosas (`enable_tiling`).
- **Videos largos**: trozos de 81 frames con **8 frames de solapamiento**: los primeros frames de cada
  trozo son los últimos generados del anterior con máscara negra (= fijos), para que la persona no
  "salte" entre trozos.
- **Modo turbo**: LoRA **CausVid** (destilación de pasos, Kijai/WanVideo_comfy) → 8 pasos sin CFG en
  vez de 25 con CFG (×2 pasadas): unas 6 veces menos evaluaciones del transformer. Si la LoRA no carga,
  se usa el modo normal automáticamente.
- **Pros**: corre en local, licencia Apache-2.0, no sube nada a internet.
- **Contras**: 1.3B es pequeño: cara y manos menos fieles, y la identidad depende solo de la foto de
  referencia (no hay rasgos faciales explícitos). No re-ilumina como Wan-Animate.

### 7.4 Modelo 2 — Wan2.2-Animate 14B, modo *replace* (el equilibrado, nube)

- **Qué es**: *Wan-Animate: Unified Character Animation and Replacement with Holistic Replication*
  (Alibaba, 2025). Un modelo de 14B entrenado específicamente para animar o **sustituir** personajes.
- **Cómo funciona**: del video se extraen el **esqueleto 2D** (pose del cuerpo, alineada espacialmente
  con los latentes) y **rasgos faciales implícitos** (un codificador de la cara del actor original que
  captura la expresión, inyectados por atención cruzada). La persona de la imagen se genera con ese
  movimiento y expresión. En modo *replace* se le da además el fondo y la máscara de la persona
  original, y una **LoRA de re-iluminación** ajusta su luz y color a los de la escena (lo que más se nota
  en un escenario de conferencia).
- **En la app**: se envía a fal.ai el **recorte** de la persona (a 16 fps) y la referencia; fal.ai hace
  la pose, la segmentación y la generación. El resultado vuelve y se compone igual que los demás.
  Opciones: turbo (fal.ai optimiza los parámetros) y 720p.
- **Pros**: mucho más fiel que VACE en cuerpo, manos y expresión; integra la luz de la escena; rápido.
- **Contras**: de pago y **los fotogramas de las personas se suben a un servidor externo** (tenlo en
  cuenta en el consentimiento). Elige al "sujeto principal" del recorte: si dentro del recorte hay otra
  persona de tamaño parecido, puede confundirse.
- **Por qué no en local**: 14B en bf16 son ~28 GB de pesos + 11 GB de umT5 + preprocesado de pose;
  no cabe en 6 GB de VRAM ni en los 16 GB de RAM de este portátil.

### 7.4b Wan2.2-Animate en un Space gratuito (sin descargar nada)

- **Qué es**: el mismo modelo que 7.4, ejecutado en el Space público
  [`alexnasa/Wan2.2-Animate-ZEROGPU`](https://huggingface.co/spaces/alexnasa/Wan2.2-Animate-ZEROGPU)
  (Gradio + **ZeroGPU**: Hugging Face presta una GPU A10G solo mientras dura cada petición).
- **En la app** ([backends.py](app/character/backends.py), `WanAnimateSpaceBackend`): con `gradio_client`
  se llama a su API `/animate_scene` con el recorte de la persona a 16 fps, la foto de referencia, el modo
  `"Character Swap"`, la duración (2–20 s) y `"Low Res"` / `"Medium Res"` (casilla "Más resolución").
  El Space calcula la pose, la cara, el fondo y la máscara, y devuelve el video. La app lo compone en
  el original igual que con los demás modelos. La página muestra la posición en la cola.
- **Qué se descarga**: nada del modelo. En tu equipo solo se usan SCRFD/ArcFace (ya instalados) y
  YOLO11s-seg (21 MB) para encontrar y recortar a la persona.
- **Cuota (medido el 6-nov-2026)**: cada petición **reserva ~110 s de GPU**, incluso para 2 s de video
  (en la práctica generó 2 s en ~65 s). La cuota **anónima** (por IP) se agota con **una sola** petición
  al día, así que en la práctica **hace falta un token** de una cuenta de Hugging Face: pégalo en el
  campo "Token de Hugging Face" de la página (solo se usa para ese trabajo; no se guarda) o defínelo en
  `HF_TOKEN`. Las cuentas gratuitas también tienen un límite diario de pocos minutos; una cuenta PRO
  tiene bastante más. Si se agota, la página muestra cuánto falta para que se renueve.
- **Descarga del resultado**: la descarga automática de `gradio_client` recibía *403 Forbidden* de este
  Space; la app pide solo la URL del resultado y la descarga con las cabeceras de la sesión y el token.
- **Limitaciones**: máximo 20 s por trabajo. El Space lo mantiene un usuario de la comunidad y puede
  cambiar su API o quedar pausado; en ese caso duplícalo en tu cuenta y apunta `WAN_ANIMATE_SPACE` a tu
  copia. El Space elige automáticamente a la persona principal del recorte.
- **Privacidad**: el recorte y la foto se envían a Hugging Face y al código de ese Space.

### 7.5 Modelo 3 — MoCha 14B (el más preciso, GPU externa)

- **Qué es**: *MoCha: End-to-End Video Character Replacement without Structural Guidance* (Orange 3DV
  Team, CVPR 2026), un ajuste de Wan2.1-T2V-14B.
- **Cómo funciona**: **no usa esqueleto, ni profundidad, ni máscaras por frame**. Concatena en una sola
  secuencia de tokens el video original, la **máscara del primer frame** y las referencias (cuerpo +
  primer plano de la cara), y deja que la atención completa del DiT **encuentre a la persona y la siga**
  el resto del video. Se entrenó en dos fases: aprendizaje en contexto de todas las capas de
  auto-atención, y una LoRA con **recompensa facial** para mejorar el parecido de la cara.
- **Por qué es el más preciso**: evitar el esqueleto elimina sus errores típicos (manos cruzadas,
  oclusiones, poses de perfil). En su benchmark sintético obtiene SSIM 0,746 / LPIPS 0,152 / PSNR 23,1
  frente a Wan-Animate 0,692 / 0,213 / 19,2 y VACE 0,572 / 0,253 / 17,1; también gana en consistencia
  de sujeto y fondo y en suavidad de movimiento (VBench).
- **En la app**: se generan los trozos de 81 frames (832×480), la máscara del primer frame de cada uno,
  `ref_body.png` y `ref_face.png` (el primer plano que el script llama `reference_2`), un CSV, y se
  ejecuta `inference_mocha.py` del repositorio oficial como subproceso, mostrando su progreso.
- **Requisitos**: el checkpoint de MoCha (28,6 GB) + Wan2.1-T2V-14B (~70 GB) y una GPU de ≥ 48 GB
  (A6000/A100/H100). En la práctica: instalar esta app en una máquina en la nube con esa GPU.
  Pasos:
  1. `git clone https://github.com/Orange-3DV-Team/MoCha` e instalar su `requirements.txt`.
  2. Descargar `Orange-3DV-Team/MoCha` (HF) a `checkpoints/` y `Wan-AI/Wan2.1-T2V-14B`.
  3. Editar en `inference_mocha.py` las rutas `/path/to/...` del DiT, el umT5 y el VAE de Wan2.1.
  4. Definir `MOCHA_DIR` (y `MOCHA_PYTHON` si usa otro entorno) antes de arrancar la app.
- **Contras**: muy pesado y lento (50 pasos con CFG), sin paso de texto, y cada trozo de 81 frames se
  genera por separado (puede haber pequeños saltos entre trozos).

### 7.6 Comparativa de los tres

| | VACE 1.3B | Wan-Animate 14B | MoCha 14B |
|---|---|---|---|
| Parámetros | 1,3 B (+ bloques VACE) | 14 B | 14 B |
| Guía estructural | máscara por frame | esqueleto + rasgos faciales + máscara | solo máscara del 1er frame |
| Identidad de la persona nueva | 1 imagen de referencia | 1 imagen | cuerpo + primer plano de cara |
| Re-iluminación | no | sí (LoRA) | implícita |
| Texto | sí (umT5) | no | no |
| Dónde corre | local, 6 GB | nube (fal.ai) | GPU ≥ 48 GB |
| SSIM / LPIPS (benchmark de MoCha) | 0,572 / 0,253 | 0,692 / 0,213 | **0,746 / 0,152** |
| Privacidad | todo en local | sube recorte y foto | depende de dónde la ejecutes |

### 7.7 Qué está verificado y qué no

- Verificado en esta máquina: escaneo y selección, seguimiento de la persona con YOLO11-seg + ByteTrack,
  recorte, trozos con solapamiento, la llamada a `WanVACEPipeline` (con un VACE diminuto de pesos
  aleatorios: misma arquitectura, sin descargar 19 GB), la composición a los fps originales y el audio.
- **No** verificado aún: una llamada real al Space gratuito (sube imágenes de personas a un tercero, así
  que necesita una prueba con tu autorización); la calidad y velocidad reales de VACE 1.3B en la RTX 4050 (requiere la descarga
  de 19 GB), la carga de la LoRA CausVid en VACE, y las llamadas reales a fal.ai y a MoCha (requieren
  clave de pago / GPU grande). Las velocidades de la tabla son estimaciones.

---

## 8. Escaneo de rostros y selección de la persona

Al subir el video destino, `POST /api/scan` ([app/scan.py](app/scan.py)) hace en pocos segundos:

1. **Muestreo** a ~2 frames por segundo (máximo 120 frames).
2. **Detección** con SCRFD (caras < 28 px o con confianza < 0,5 se descartan: no dan un embedding fiable).
3. **Embedding** ArcFace de cada cara, en un solo lote por frame.
4. **Clustering voraz**: cada cara se asigna al grupo cuyo centroide tenga similitud coseno ≥ 0,40; si
   no hay ninguno, abre un grupo nuevo. Al final se fusionan grupos cuyos centroides se parecen ≥ 0,40
   (la misma persona de perfil al principio y de frente después).
5. **Miniatura** de cada persona: la cara con mejor puntuación = confianza × √área × frontalidad
   (nariz centrada entre los ojos), más el frame completo con su caja.

Medido: 18 s de video 640×640 → 36 frames analizados en ~2 s.

**Durante el procesado** (`TargetMatcher`), en cada frame se calcula el ArcFace de las caras detectadas
y se aceptan las que:
- tienen similitud ≥ 0,36 con alguna persona elegida, **o**
- tienen similitud ≥ 0,12 **y** se solapan (IoU > 0,3) con una cara aceptada en los últimos 8 frames.

La segunda regla evita perder a la persona cuando gira la cabeza o hay desenfoque de movimiento (el
embedding empeora pero sigue estando en el mismo sitio). Coste: ~3 ms por cara (HyperSwap pasó de ~22 a
~19 fps). En el *Encoder propio*, el mismo filtro hace que solo se extraigan para entrenar las caras de
la persona elegida.

En character swap la identidad se une al **cuerpo**: la persona cuya máscara de YOLO contiene la cara
reconocida (sección 7.2).

---

## 9. Comparativa final: qué método usar

| Situación | Recomendación |
|---|---|
| Resultado rápido y de buena calidad | **HyperSwap** |
| Necesitas licencia totalmente libre (Apache-2.0) | **GHOST** |
| Comparar con el *baseline* clásico | **inswapper** |
| Estudiar cómo se aprende un face swap desde cero / tienes video de ambas personas y tiempo | **Encoder propio** |
| Cambiar la persona entera **sin descargar modelos** y gratis | **Character swap · Wan-Animate (Space gratuito)** |
| Cambiar la persona entera (ropa, pelo, cuerpo) sin salir de este portátil | **Character swap · VACE 1.3B** |
| Cambiar la persona entera con buena calidad y luz de la escena, rápido | **Character swap · Wan-Animate** (nube) |
| La máxima fidelidad posible y tienes una GPU de ≥ 48 GB | **Character swap · MoCha** |
| Solo necesitas otra cara y el resto de la persona sirve | **Face swap**: mucho más rápido y nítido que cualquier character swap |

Métricas útiles para comparar los cuatro en el experimento:

| Métrica | Cómo medirla |
|---|---|
| Similitud de identidad | Coseno entre ArcFace(cara generada) y ArcFace(fuente). Mejor con un reconocedor distinto al usado por el modelo |
| Conservación de pose | Diferencia de yaw/pitch/roll entre destino y resultado (`1k3d68.onnx` de buffalo_l) |
| Conservación de expresión | Distancia entre landmarks normalizados (`2d106det.onnx`) |
| Realismo | FID entre caras generadas y reales |
| Estabilidad temporal | Diferencia media entre frames consecutivos dentro de la máscara |

---

## 10. Modelos auxiliares

| Modelo | Archivo | Qué hace | Licencia |
|---|---|---|---|
| **SCRFD-10GF** | `det_10g.onnx` | Detector de caras + 5 landmarks. Una sola etapa, pirámide de características con *strides* 8/16/32; su aportación es cómo repartir el cómputo entre *backbone*, *neck* y *head* y cómo muestrear caras pequeñas | No comercial (InsightFace) |
| **ArcFace R50** | `w600k_r50.onnx` | Embedding de identidad de 512 dimensiones | No comercial (InsightFace) |
| **BiSeNet ResNet-18** | `bisenet_resnet_18.onnx` | *Face parsing*: clasifica cada píxel en 19 clases (piel, cejas, ojos, gafas, nariz, boca, labios, pelo, cuello…). Dos ramas: una espacial (detalle) y una de contexto (semántica) | MIT |
| **XSeg** | `xseg_1.onnx` | Segmentación de oclusiones de DeepFaceLab: marca qué parte de la cara es visible | GPL-3.0 |
| **crossface_ghost** | `crossface_ghost.onnx` | Traduce embeddings ArcFace al espacio de GHOST | FaceFusion |
| **YOLO11s-seg** | `yolo11s-seg.pt` | Detección + segmentación de instancias de personas (character swap) | AGPL-3.0 (Ultralytics) |
| **ByteTrack** | (en `ultralytics`) | Asocia las cajas de personas entre frames con filtro de Kalman + emparejamiento en dos etapas (detecciones de alta y baja confianza) | MIT |
| **umT5-XXL** | `text_encoder/` de VACE | Codificador de texto de Wan (5,7 B parámetros); se ejecuta en CPU y se cachea | Apache-2.0 |
| **CausVid LoRA** | `Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32` | Destilación a pocos pasos para Wan 1.3B (modo turbo de VACE) | ver repositorio |

---

## 11. Rendimiento: qué cuesta cada paso

Medido por cara en la RTX 4050 (video 720×1280):

| Paso | Tiempo | Notas |
|---|---|---|
| Detección SCRFD | ~16 ms | Siempre activo |
| Generador | 18–49 ms | Según el modelo (sección 5.4) |
| BiSeNet (segmentación) | ~15 ms | La salida se procesa a 256 px para ahorrar CPU |
| XSeg (oclusión) | ~28 ms | Por eso viene desactivado |
| Color + paste-back | ~10 ms | CPU |

Optimizaciones aplicadas:

- **cuDNN en modo exhaustivo + NHWC**: ONNX Runtime prueba todos los algoritmos de convolución la
  primera vez y elige el más rápido para los Tensor Cores (HyperSwap pasó de 22 a 18 ms).
- **Solapamiento CPU/GPU**: la detección va en orden (el suavizado depende del frame anterior), pero
  el swap + máscaras de dos frames se procesan en paralelo. Mientras la GPU genera una cara, la CPU
  mezcla la anterior: ×1.6 en total.
- **Lectura en un hilo aparte** y **escritura directa a ffmpeg** por tubería (sin archivo intermedio).

> **Potencia de la GPU del portátil.** Todas las medidas se hicieron con la RTX 4050 limitada a
> **45 W** (`nvidia-smi` mostraba 45/45 W y 1 935 de 3 105 MHz). Según el fabricante del portátil, esta
> GPU puede funcionar entre 35 y 115 W. Con el cargador conectado y el modo "Rendimiento" del software del
> portátil, tanto el entrenamiento como el procesado de video pueden ir bastante más rápido. Puedes
> comprobarlo con `nvidia-smi --query-gpu=power.draw,enforced.power.limit --format=csv` mientras procesa.

---

## 12. Estructura del código

```
app/
├── main.py              API FastAPI: escaneo, trabajos, progreso, catálogo de modelos
├── engine.py            Pipeline de face swap: detección, embeddings, suavizado, alineación, máscaras, color, video
├── scan.py              Escaneo rápido + clustering de rostros; TargetMatcher (filtrar a las personas elegidas)
├── swappers.py          inswapper / HyperSwap / GHOST (ONNX)
├── models_registry.py   Catálogo: URLs, plantillas y normalización (face swap) y modelos de character swap
├── downloads.py         Descargas con progreso y reanudación
├── encoder/
│   ├── model.py         Autoencoder (encoder compartido + 2 decoders) y DSSIM
│   └── trainer.py       Extracción de caras, aumentos en GPU, entrenamiento, conversión
├── character/
│   ├── persons.py       YOLO11-seg + ByteTrack, seguimiento de la persona elegida, foto de referencia
│   ├── backends.py      Wan-Animate (Space HF gratuito · fal.ai) · VACE 1.3B (diffusers, local) · MoCha (subproceso)
│   └── pipeline.py      Recorte → generación → composición → ffmpeg
└── static/index.html    La página
scripts/download_models.py   (opcional) descarga inswapper por adelantado
models/                      modelos descargados (no se suben a git)
data/                        uploads/, outputs/, encoder_models/, work/ (no se suben a git)
```

## 13. Limitaciones y siguientes experimentos

- **Perfiles extremos (> 70°)**: con 5 landmarks la alineación pierde precisión.
- **Mejorar nitidez**: añadir un restaurador facial (GFPGAN / CodeFormer / GPEN) tras el swap, a costa
  de ~20–40 ms por frame.
- **Encoder más grande**: subir a 256 px y añadir pérdida perceptual (VGG) y adversarial (GAN), o
  entrenar el decoder para que prediga también su máscara (como DeepFaceLab).
- **Escaneo**: si una persona aparece siempre de perfil o muy pequeña puede quedar repartida en
  varios grupos; se puede marcar más de uno (face swap). Una persona que nunca muestra la cara no se
  puede elegir.
- **Character swap**:
  - el recorte es fijo: si la persona cruza todo el escenario, el recorte es grande y ella sale pequeña
    (menos detalle). Un recorte que la siga suavemente daría más resolución;
  - la persona se genera a 16 fps y como máximo 30 s por trabajo;
  - con VACE 1.3B las manos y la cara pueden deformarse; un face swap (HyperSwap) **encima** del
    resultado del character swap mejora mucho el parecido facial (siguiente experimento natural);
  - Wan-Animate en fal.ai elige al sujeto principal del recorte, no recibe nuestra máscara.
- **Detector de deepfakes**: entrenar un clasificador con los videos que genera esta app.

## 14. Ética

- Consentimiento informado y por escrito de **todas** las personas que aparecen en la fuente y en el video.
- No publicar resultados sin la marca de contenido sintético ni hacerlos pasar por reales.
- Borrar `data/` al acabar el experimento.
- Con **Wan-Animate** (fal.ai o el Space gratuito de Hugging Face) los fotogramas del recorte y la foto
  de la persona se suben a un servidor de terceros: el consentimiento debe cubrir también ese tratamiento.
- Si el video destino es una grabación de terceros (p. ej. una conferencia), además de la persona que
  se sustituye pueden aparecer otras: el character swap no las modifica, pero el video resultante las
  sigue mostrando junto a contenido sintético.
- La Ley de IA de la UE y muchas legislaciones nacionales exigen transparencia en los *deepfakes*;
  suplantar a alguien sin su consentimiento puede ser delito.
- Las licencias de los modelos (secciones 10, 5.4 y 7.6) permiten uso académico; solo GHOST y BiSeNet permiten
  uso comercial.

## 15. Referencias

- InsightFace — <https://github.com/deepinsight/insightface>
- FaceFusion (especificaciones y ONNX de HyperSwap, GHOST, XSeg, BiSeNet) — <https://github.com/facefusion/facefusion>
- Deng et al. *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*. CVPR 2019.
- Guo et al. *Sample and Computation Redistribution for Efficient Face Detection (SCRFD)*. ICLR 2022.
- Groshev et al. *GHOST — A New Face Swap Approach for Image and Video Domains*. IEEE Access 2022.
- Li et al. *FaceShifter: Towards High Fidelity and Occlusion Aware Face Swapping*. 2019 (AEI-Net, AAD).
- Chen et al. *SimSwap: An Efficient Framework for High Fidelity Face Swapping*. ACM MM 2020.
- Huang & Belongie. *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization*. ICCV 2017.
- Perov et al. *DeepFaceLab: Integrated, flexible and extensible face-swapping framework*. 2020.
- Yu et al. *BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation*. ECCV 2018.
- Wang et al. *Image Quality Assessment: From Error Visibility to Structural Similarity (SSIM)*. IEEE TIP 2004.
- Shi et al. *Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel CNN (PixelShuffle)*. CVPR 2016.
- Wan Team. *Wan: Open and Advanced Large-Scale Video Generative Models*. 2025 — <https://github.com/Wan-Video/Wan2.1>
- Jiang et al. *VACE: All-in-One Video Creation and Editing*. ICCV 2025 — <https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers>
- Cheng et al. *Wan-Animate: Unified Character Animation and Replacement with Holistic Replication*. 2025 — <https://arxiv.org/abs/2509.14055>
- Xu et al. *MoCha: End-to-End Video Character Replacement without Structural Guidance*. CVPR 2026 — <https://arxiv.org/abs/2601.08587>, <https://github.com/Orange-3DV-Team/MoCha>
- Yin et al. *From Slow Bidirectional to Fast Autoregressive Video Diffusion Models (CausVid)*. CVPR 2025.
- Zhang et al. *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*. ECCV 2022.
- Ultralytics YOLO11 — <https://docs.ultralytics.com>
- Space gratuito de Wan2.2-Animate (ZeroGPU) — <https://huggingface.co/spaces/alexnasa/Wan2.2-Animate-ZEROGPU>
- fal.ai — Wan 2.2 Animate Replace — <https://fal.ai/models/fal-ai/wan/v2.2-14b/animate/replace>
