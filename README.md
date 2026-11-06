# FaceSwap Lab — modelos pre-entrenados vs. encoder propio

Aplicación web para estudiar el *face swapping*. Subes una **fuente** (selfie o video corto de la
persona A) y un **video destino** (persona B), y la app genera un video donde B tiene la cara de A,
conservando su pose, expresión, mirada, iluminación y audio.

Se pueden comparar **cuatro métodos** desde la misma página:

| | Método | Tipo | Resolución de la cara | Necesita entrenar | Velocidad medida* |
|---|---|---|---|---|---|
| 1 | **HyperSwap 1a** (FaceFusion) | pre-entrenado, *one-shot* | 256 px | No | **~22 fps** |
| 2 | **GHOST 2** (ai-forever) | pre-entrenado, *one-shot* | 256 px | No | ~16 fps |
| 3 | **inswapper** (InsightFace) | pre-entrenado, *one-shot* | 128 px | No | ~13 fps |
| 4 | **Encoder propio** (autoencoder) | entrenado por nosotros para un par A↔B | 128 px | Sí (5–120 min) | ~20 fps al convertir |

\* RTX 4050 Laptop (6 GB), video vertical de 720×1280, una cara, segmentación + color + suavizado
activados y XSeg desactivado. Un video de 1:20 a 30 fps (2 400 frames) tarda unos 2 minutos con HyperSwap.

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
7. [Comparativa final: qué método usar](#7-comparativa-final-qué-método-usar)
8. [Modelos auxiliares](#8-modelos-auxiliares)
9. [Rendimiento: qué cuesta cada paso](#9-rendimiento-qué-cuesta-cada-paso)
10. [Estructura del código](#10-estructura-del-código)
11. [Limitaciones y siguientes experimentos](#11-limitaciones-y-siguientes-experimentos)
12. [Ética](#12-ética)
13. [Referencias](#13-referencias)

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
# PyTorch con CUDA (solo lo usa el modo "Encoder propio"):
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu130
```

Arrancar:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

y abrir <http://127.0.0.1:8000>. Para pararlo, `Ctrl + C`.

Los modelos se descargan solos la primera vez que se usan (con progreso en la página y reanudación si
se corta la conexión): `buffalo_l` (~280 MB), HyperSwap (~400 MB), GHOST (~740 MB + 22 MB),
inswapper (~550 MB), BiSeNet (~53 MB) y XSeg (~70 MB). Quedan en `models/`.

## 2. Cómo se usa la página

1. **Fuente**: una selfie frontal o, mejor, un video de 10–60 s de la persona girando un poco la cabeza.
   Con video, los modelos pre-entrenados promedian la identidad de 24 frames (más estable que una foto),
   y el encoder tiene muchas caras para aprender.
2. **Video destino**.
3. **Método**: uno de los tres modelos pre-entrenados, o *Encoder propio* con un tiempo de entrenamiento.
4. **Integración**: segmentación facial, igualar color, suavizado temporal (activados por defecto) y
   máscara de oclusión (desactivada; actívala si algo tapa la cara).
5. Confirmar el consentimiento y pulsar **Generar video**. La barra muestra la fase
   (preparar modelos → entrenar → procesar), los fps y el tiempo restante.

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

## 7. Comparativa final: qué método usar

| Situación | Recomendación |
|---|---|
| Resultado rápido y de buena calidad | **HyperSwap** |
| Necesitas licencia totalmente libre (Apache-2.0) | **GHOST** |
| Comparar con el *baseline* clásico | **inswapper** |
| Estudiar cómo se aprende un face swap desde cero / tienes video de ambas personas y tiempo | **Encoder propio** |

Métricas útiles para comparar los cuatro en el experimento:

| Métrica | Cómo medirla |
|---|---|
| Similitud de identidad | Coseno entre ArcFace(cara generada) y ArcFace(fuente). Mejor con un reconocedor distinto al usado por el modelo |
| Conservación de pose | Diferencia de yaw/pitch/roll entre destino y resultado (`1k3d68.onnx` de buffalo_l) |
| Conservación de expresión | Distancia entre landmarks normalizados (`2d106det.onnx`) |
| Realismo | FID entre caras generadas y reales |
| Estabilidad temporal | Diferencia media entre frames consecutivos dentro de la máscara |

---

## 8. Modelos auxiliares

| Modelo | Archivo | Qué hace | Licencia |
|---|---|---|---|
| **SCRFD-10GF** | `det_10g.onnx` | Detector de caras + 5 landmarks. Una sola etapa, pirámide de características con *strides* 8/16/32; su aportación es cómo repartir el cómputo entre *backbone*, *neck* y *head* y cómo muestrear caras pequeñas | No comercial (InsightFace) |
| **ArcFace R50** | `w600k_r50.onnx` | Embedding de identidad de 512 dimensiones | No comercial (InsightFace) |
| **BiSeNet ResNet-18** | `bisenet_resnet_18.onnx` | *Face parsing*: clasifica cada píxel en 19 clases (piel, cejas, ojos, gafas, nariz, boca, labios, pelo, cuello…). Dos ramas: una espacial (detalle) y una de contexto (semántica) | MIT |
| **XSeg** | `xseg_1.onnx` | Segmentación de oclusiones de DeepFaceLab: marca qué parte de la cara es visible | GPL-3.0 |
| **crossface_ghost** | `crossface_ghost.onnx` | Traduce embeddings ArcFace al espacio de GHOST | FaceFusion |

---

## 9. Rendimiento: qué cuesta cada paso

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

## 10. Estructura del código

```
app/
├── main.py              API FastAPI: trabajos, progreso, catálogo de modelos
├── engine.py            Pipeline común: detección, suavizado, alineación, máscaras, color, paste-back, video
├── swappers.py          inswapper / HyperSwap / GHOST (ONNX)
├── models_registry.py   Catálogo: URLs, plantillas, normalización de cada modelo
├── downloads.py         Descargas con progreso y reanudación
├── encoder/
│   ├── model.py         Autoencoder (encoder compartido + 2 decoders) y DSSIM
│   └── trainer.py       Extracción de caras, aumentos en GPU, entrenamiento, conversión
└── static/index.html    La página
scripts/download_models.py   (opcional) descarga inswapper por adelantado
models/                      modelos descargados (no se suben a git)
data/                        uploads/, outputs/, encoder_models/ (no se suben a git)
```

## 11. Limitaciones y siguientes experimentos

- **Perfiles extremos (> 70°)**: con 5 landmarks la alineación pierde precisión.
- **Mejorar nitidez**: añadir un restaurador facial (GFPGAN / CodeFormer / GPEN) tras el swap, a costa
  de ~20–40 ms por frame.
- **Encoder más grande**: subir a 256 px y añadir pérdida perceptual (VGG) y adversarial (GAN), o
  entrenar el decoder para que prediga también su máscara (como DeepFaceLab).
- **Seleccionar a quién cambiar** cuando hay varias personas: elegir la cara destino por similitud
  ArcFace con una foto de referencia.
- **Detector de deepfakes**: entrenar un clasificador con los videos que genera esta app.

## 12. Ética

- Consentimiento informado y por escrito de **todas** las personas que aparecen en la fuente y en el video.
- No publicar resultados sin la marca de contenido sintético ni hacerlos pasar por reales.
- Borrar `data/` al acabar el experimento.
- La Ley de IA de la UE y muchas legislaciones nacionales exigen transparencia en los *deepfakes*;
  suplantar a alguien sin su consentimiento puede ser delito.
- Las licencias de los modelos (sección 8 y 5.4) permiten uso académico; solo GHOST y BiSeNet permiten
  uso comercial.

## 13. Referencias

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
