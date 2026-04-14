# ADR-0006: Facial Detection — Model Selection

**Status:** Proposed  
**Date:** 2026-04

---

## Context

The "Facial Detection" feature will add a new top-level page that:

1. **Groups photos by person** — detects faces, generates embeddings, and clusters
   them into named identity groups.
2. **Finds "similar photos"** where a person's face may be hidden (side-profile,
   back-turned, masked, crowd) and surfaces those photos for inclusion in the
   same person group.

The app is **local-only** and privacy-preserving by design; no images or
embeddings may ever leave the user's machine.  The feature must work on a
typical consumer laptop (CPU-only fallback required; GPU optional acceleration
desirable).

---

## Problem decomposition

The feature decomposes into two distinct technical challenges:

| Challenge | Description |
|-----------|-------------|
| **Face detection** | Locate faces in each photo (bounding box + optional landmarks). |
| **Face recognition** | Produce a compact identity embedding so that photos of the *same* person cluster together regardless of age, pose, lighting, or expression. |
| **Body/context matching** | Group photos where the face is hidden by using full-frame or person-crop embeddings (clothes, posture, context). |
| **Clustering** | Unsupervised grouping of embeddings into identity clusters; cluster labelling by the user. |

A good implementation will combine a dedicated face-recognition model (for cases
where a face is visible) with a complementary full-frame similarity method (for
hidden-face cases).

---

## Candidates evaluated

### 1. `face_recognition` / dlib

| Attribute | Detail |
|-----------|--------|
| Detection | HOG-based or CNN detector bundled in dlib |
| Recognition | 128-dimensional ResNet embeddings (dlib pretrained, ~2018) |
| Dependencies | `face_recognition`, `dlib` (requires C++ build tools or prebuilt wheel) |
| Model size | ~100 MB (dlib shape predictor + recognition model) |
| Accuracy | ~95 % on LFW; noticeably behind 2020+ margin-based models |
| CPU speed | Medium (single-core C++ inference) |
| GPU support | Limited (CUDA via dlib, complex to set up) |

**Pros:**
- Simple, well-documented API; lowest barrier to entry.
- Pure Python install on most platforms when prebuilt wheels are available.
- Widely used in tutorials; many clustering examples exist.

**Cons:**
- Model is static (~2017–2019); no active improvement.
- 128-d embeddings less discriminative than modern 512-d ArcFace embeddings,
  especially on challenging poses, large age gaps, or look-alike individuals.
- `dlib` dependency requires C++ toolchain or platform-specific wheels; fragile
  cross-platform packaging.
- Does **not** natively handle hidden-face photos.

---

### 2. DeepFace

| Attribute | Detail |
|-----------|--------|
| Detection | Supports OpenCV, SSD, MTCNN, RetinaFace, MediaPipe as backends |
| Recognition | Wraps VGG-Face, Facenet, Facenet512, OpenFace, DeepFace (native), DeepID, ArcFace, Dlib |
| Dependencies | `deepface`; heavyweight backends (TensorFlow or PyTorch) pulled on demand |
| Model size | Varies; ArcFace via DeepFace ~250 MB; VGG-Face ~550 MB |
| Accuracy | Backend-dependent; ArcFace backend is competitive |
| CPU speed | Slow (TensorFlow eager mode for some models) |
| GPU support | Yes (TensorFlow / PyTorch auto-detected) |

**Pros:**
- "Swiss-army knife" — lets you swap detection/recognition backends via config
  with no code change.
- Built-in `DeepFace.find()` and cluster utilities.
- Includes demographic analysis (age, gender, emotion) as a bonus.

**Cons:**
- High abstraction hides performance-critical details; harder to optimise
  batched inference.
- TensorFlow is a heavy transitive dependency (>400 MB install); conflicts with
  existing PyTorch-based scorers are possible.
- Model zoo relies on Dropbox / Drive CDN mirrors; download reliability varies.
- Not the fastest option at inference time on large photo collections.

---

### 3. InsightFace (recommended)

| Attribute | Detail |
|-----------|--------|
| Detection | SCRFD (Sample and Computation Redistribution Face Detector) — ONNX, very fast |
| Recognition | ArcFace on ResNet50/ResNet100 — 512-d L2-normalised embeddings, ONNX |
| Bundle | `buffalo_l` pack: SCRFD 10G + ArcFace R100 (≈ 350 MB); lighter `buffalo_sc` also available |
| Dependencies | `insightface`, `onnxruntime` (CPU) or `onnxruntime-gpu` (GPU) |
| Accuracy | SOTA: ~99.8 % on LFW, top rankings on MegaFace / IJB-C |
| CPU speed | Fast (ONNX Runtime; optimised for CPU AVX2) |
| GPU speed | Excellent (CUDA/TensorRT via `onnxruntime-gpu`) |

**Pros:**
- Best-in-class accuracy thanks to ArcFace Additive Angular Margin Loss; embeddings
  are highly discriminative across poses, illumination, and age.
- Runs purely via **ONNX Runtime** — no PyTorch/TF dependency conflict; plays
  nicely with the existing PyTorch scorers because they share no runtime.
- Local-only, privacy-preserving: `insightface` downloads models from its own
  CDN to `~/.insightface`; no images or embeddings are transmitted.
- Landmark output enables precise face alignment before embedding, improving
  accuracy on oblique angles.
- Multiple model packs let operators trade accuracy for speed/size:
  `buffalo_l` (best accuracy) → `buffalo_m` → `buffalo_sc` (smallest).
- Actively maintained research toolkit; model zoo is regularly updated.

**Cons:**
- `insightface` package requires `numpy`, `scikit-image` (small footprint, already
  in the dependency graph via other scorers).
- `buffalo_l` weights are ~350 MB; a one-time download on first use.
- The InsightFace `buffalo_l` license is **non-commercial** (CC BY-NC 4.0);
  suitable for the app's personal/non-commercial positioning, but must be noted
  in the UI and docs.  The lighter `buffalo_sc` uses the MIT-licensed SCRFD
  detector but the same ArcFace recognizer license applies.
- Does **not** directly handle hidden-face photos (see "body/context matching"
  below).

---

### 4. FaceNet (facenet-pytorch)

| Attribute | Detail |
|-----------|--------|
| Detection | MTCNN (Multi-task Cascaded CNN) bundled in `facenet-pytorch` |
| Recognition | InceptionResnet V1 pretrained on VGGFace2 or CASIA-WebFace (512-d) |
| Dependencies | `facenet-pytorch` (PyTorch) |
| Accuracy | ~99.6 % on LFW — slightly below ArcFace |
| CPU speed | Moderate (PyTorch eager) |
| GPU support | Yes (PyTorch CUDA) |

**Pros:**
- Pure PyTorch; compatible with existing scorer stack.
- MTCNN detector is reliable and well-understood.
- Triplet-loss embeddings still competitive on clean datasets.

**Cons:**
- Accuracy trail-off on hard cases (large age gaps, extreme poses) vs ArcFace.
- MTCNN detector slower than SCRFD; more resource-intensive for large albums.
- No ONNX option natively; less flexible deployment path.

---

### 5. MediaPipe Face Detection

| Attribute | Detail |
|-----------|--------|
| Type | Detection-only (no recognition / embedding) |
| Dependencies | `mediapipe` (Google) |
| Model size | Very small (<5 MB) |
| CPU speed | Excellent (TFLite) |
| GPU support | Limited GPU acceleration on desktop |

**Pros:**
- Extremely fast and lightweight; ideal for quick face presence detection.
- Minimal install.

**Cons:**
- **Detection only** — produces bounding boxes, not embeddings.
- Cannot do recognition or clustering on its own; needs a separate recognition
  model, adding complexity.
- Accuracy drops on small faces and extreme poses.

*Verdict:* Only suitable as a pre-filter (is a face present?), not a complete
solution.

---

### 6. RetinaFace (standalone)

| Attribute | Detail |
|-----------|--------|
| Type | Detection + 5-point landmarks only |
| Dependencies | `retina-face` (PyTorch) or via InsightFace |
| Accuracy | Very high detection accuracy on WIDER FACE |

**Pros:**
- Best standalone detection quality; returns 5 landmarks for alignment.

**Cons:**
- Detection-only; a second model is still needed for recognition.
- InsightFace already bundles the SCRFD detector (comparable accuracy) together
  with ArcFace recognition in one package, making standalone RetinaFace
  redundant when using InsightFace.

*Verdict:* Superseded by the InsightFace bundle.

---

## Handling hidden-face photos (body/context matching)

The issue explicitly requires surfacing photos where the person's face is hidden.
Two complementary approaches exist:

### Option A — CLIP global-frame similarity (recommended: already in codebase)

The app already computes and stores **CLIP ViT-L/14 embeddings** for every image
(schema `clip_embeddings`, schema v10, Iteration 12).  These 768-d vectors encode
global scene context — clothing, body posture, background, overall composition —
and can be used to find photos that *look similar* to a reference person group,
even when the face is absent.

Workflow:
1. Build a set of "reference embeddings" from photos where the person **is**
   recognised by the face model.
2. Compute the mean reference vector (or use all positives with cosine threshold).
3. Search the `clip_embeddings` table for photos within a cosine distance
   threshold — these become candidates for the person group.

**Pros:**
- Zero additional dependencies; CLIP is already embedded.
- Finds similar clothing, settings, body shape, and environment.

**Cons:**
- Clothing changes between photos reduce recall; different people in similar
  outfits increase false positives.
- Works best when a person has a consistent look across multiple photos
  (e.g., the same event / outfit).

### Option B — Person Re-ID model (Torchreid / OSNet)

Person re-identification (ReID) models (e.g. OSNet from Torchreid, trained on
Market-1501 / MSMT17) produce body-crop embeddings specifically designed to
match the same person across camera views, independent of pose or viewpoint.

**Pros:**
- Trained specifically for "same body, different angle" matching.
- More robust than CLIP when clothing is consistent across multiple cameras.

**Cons:**
- Adds `torchreid` as a new heavyweight dependency (PyTorch + torchvision).
- Models are trained on surveillance-camera crops; accuracy on typical phone-
  camera photos (varied framing, partial crops) is uncertain.
- Full-body crop required; photos where the person is partially in frame are
  harder to handle.
- Not well-tested on consumer photo archives.

*Verdict:* CLIP (Option A) is preferred as it leverages existing infrastructure
at zero cost.  OSNet (Option B) can be explored in a later iteration if CLIP
similarity proves insufficient.

---

## Clustering algorithm

Both face recognition and CLIP embeddings are best clustered with **DBSCAN**
(`scikit-learn`, already in the dependency graph via other features):

- Works well on embeddings of unknown density (no need to pre-specify the number
  of people / clusters).
- Returns noise points (label `-1`) for photos where the face detector found no
  face, or where the embedding is too dissimilar from any cluster.
- `eps` (neighbourhood radius) and `min_samples` are tunable per-run and can
  be stored as `params_json` in a new `face_clusters` table analogous to the
  existing `clusters` table.

For face embeddings: cosine distance metric, `eps` ≈ 0.4–0.6 (512-d ArcFace
space).  For CLIP embeddings: cosine distance, `eps` ≈ 0.15–0.25 (768-d ViT
space).

---

## Decision

**Primary face recognition pipeline: InsightFace (`insightface` + `onnxruntime`)**

- Model pack: `buffalo_l` (default, best accuracy); expose `buffalo_sc` as an
  alternative variant for users with slow CPUs or limited disk space.
- SCRFD detector + ArcFace R100 recognizer; runs entirely via ONNX Runtime (no
  TF/PyTorch conflict).
- Produces 512-d L2-normalised face embeddings per detected face.

**Hidden-face "similar photos" search: existing CLIP embeddings**

- Leverage `clip_embeddings` table already populated by the embed job.
- No new dependencies.
- Search UI surfaces CLIP-similar candidates for manual review/assignment.

**Clustering: DBSCAN from `scikit-learn`** (already available).

**No cloud calls; all inference is local.**

---

## Consequences

**Positive:**
- InsightFace delivers SOTA accuracy with a lightweight ONNX Runtime dependency
  that does not conflict with the existing PyTorch scorer stack.
- Reusing CLIP embeddings for hidden-face matching adds zero new dependencies.
- DBSCAN is already in use conceptually (pHash clustering); the pattern is
  familiar to the codebase.
- The modular design (detection → embedding → clustering) allows future model
  upgrades without breaking stored clusters (new `face_scorer_run` rows, old
  ones preserved).

**Negative / risks:**
- `buffalo_l` is CC BY-NC 4.0; must be disclosed in the UI and docs.  Consider
  making the model pack selection an explicit user choice at job start.
- First-run download of ~350 MB may be slow on poor connections; a progress
  indicator is needed.
- Face clustering accuracy depends heavily on photo quality; user education
  (e.g., tooltip explaining that the tool groups *likely* matches, not definite
  ones) is important.
- CLIP-based hidden-face matching is best-effort and will generate false
  positives; the UI must present these as *suggestions*, not confirmed matches.

---

## Rejected alternatives summary

| Option | Rejected because |
|--------|-----------------|
| `face_recognition` / dlib | Outdated embeddings; C++ build dependency; lower accuracy |
| DeepFace | TensorFlow conflict; opaque batching; fragile CDN mirrors |
| FaceNet (facenet-pytorch) | Slightly lower accuracy; no ONNX path; MTCNN slower than SCRFD |
| MediaPipe | Detection-only; not sufficient on its own |
| Standalone RetinaFace | Superseded by InsightFace bundle |
| Torchreid/OSNet (body ReID) | Extra heavyweight dep; uncertain accuracy on consumer photos; deferred to later iteration |
