# Speech Emotion Recognition

End-to-end system that detects emotion from a speech clip. Train an ML classifier on RAVDESS, serve it behind a FastAPI endpoint, and interact with it through a Next.js web app that supports both audio file upload and live mic recording.

---

## Project structure

```
voice-emotion-detection/
├── Zenodo_processed/         # RAVDESS dataset (24 actors, 1440 .npy clips)
├── training.ipynb            # full training + evaluation pipeline (single notebook)
├── emotion_model.pkl         # trained model bundle (SVM + scaler + label encoder)
├── requirements.txt          # Python dependencies
├── api/
│   └── app.py                # FastAPI server: GET /, POST /predict
└── frontend/                 # Next.js 16 + React 19 + Tailwind v4
    └── app/
        ├── page.tsx          # upload + mic UI, client-side WAV conversion
        └── layout.tsx
```

---

## Quick start

### 1. Backend setup

```bash
cd voice-emotion-detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the model (optional — `emotion_model.pkl` is already shipped)

```bash
jupyter notebook training.ipynb
# Run all cells. Takes ~2 min. Saves emotion_model.pkl on completion.
```

### 3. Start the API

```bash
source .venv/bin/activate
uvicorn api.app:app --reload --port 8000
```

Verify: `curl http://localhost:8000/` → returns model metadata.

### 4. Start the frontend (separate terminal)

```bash
cd frontend
npm install        # only first time
npm run dev
```

Open **http://localhost:3000**.

---

## Model details

### Dataset

- **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
- Preprocessed version from Zenodo: raw audio waveforms reshaped into 2D `.npy` matrices, normalized to `[-1, 1]`, sample rate **22050 Hz**
- **24 actors** (12 male, 12 female), **1440 clips total**
- **8 emotion classes**: `neutral` (96 samples), `calm` `happy` `sad` `angry` `fearful` `disgust` `surprised` (192 each)

### Feature extraction (librosa)

Each clip is flattened back to a 1D waveform, silence-trimmed (`top_db=25`), then we extract five feature families and mean+std-pool each over the time axis:

| Feature | Dim | Per clip after pooling |
|---|---|---|
| MFCC | 40 | 80 |
| ΔMFCC (1st-order delta) | 40 | 80 |
| Chroma STFT | 12 | 24 |
| Log-mel spectrogram | 64 | 128 |
| Spectral contrast | 7 | 14 |
| **Total feature vector** | | **326** |

### Train/eval split

- **Speaker-independent** — actors 1–18 train, 19–20 val, 21–24 test
- **Gender-balanced** at every split (RAVDESS odd IDs = male, even = female)
- This is harder than a random split (~10% lower accuracy) but is the honest measure: the model has never heard the test speakers' voices.

### Models compared

Two baselines trained on the same 326-dim feature vector:

| Model | Validation accuracy |
|---|---|
| MLPClassifier `(512, 256)`, ReLU, early stopping | 59.17% |
| **SVC RBF kernel, C=10, γ=scale** ← chosen | **63.33%** |

### Final test set accuracy (actors 21–24, never seen during training)

**52.08%** (chance = 12.5% for 8 classes).

Per-class (F1 score):

| Emotion | F1 | Notes |
|---|---|---|
| surprised | 0.75 | strong — distinct prosody |
| calm | 0.65 | strong |
| angry | 0.61 | strong |
| disgust | 0.50 | OK |
| happy | 0.48 | OK |
| fearful | 0.43 | confused with surprised/sad |
| neutral | 0.41 | only 16 test samples — high variance |
| sad | 0.31 | weakest — confused with calm/neutral |

### Cross-validation (more honest single number)

**Leave-one-actor-group-out** across 6 gender-balanced folds (each fold tests 2 male + 2 female actors).

| Fold | Test actors | Accuracy |
|---|---|---|
| 1 | 1, 2, 3, 4 | 65.42% |
| 2 | 5, 6, 7, 8 | 64.17% |
| 3 | 9, 10, 11, 12 | 50.83% |
| 4 | 13, 14, 15, 16 | 52.92% |
| 5 | 17, 18, 19, 20 | 52.50% |
| 6 | 21, 22, 23, 24 | 54.58% |
| **Mean** | | **56.74% ± 5.81%** |

This range (~57% with ~6% std) is consistent with published speaker-independent SER baselines on RAVDESS using shallow models — deeper CNN/LSTM approaches reach 70–75%.

---

## API reference

`POST http://localhost:8000/predict`

**Request:** `multipart/form-data` with a single `file` field. Accepted formats: `.wav .mp3 .ogg .flac .m4a .webm` (uploaded files must be in a format `libsndfile` understands; the frontend always converts to WAV before sending).

**Response:**
```json
{
  "emotion": "angry",
  "confidence": 0.864,
  "all_scores": {
    "angry": 0.864, "calm": 0.001, "disgust": 0.007,
    "fearful": 0.037, "happy": 0.067, "neutral": 0.001,
    "sad": 0.016, "surprised": 0.007
  },
  "duration_seconds": 6.132
}
```

`GET http://localhost:8000/` returns model metadata (model name, classes, feature dim, sample rate).

CORS is open (`*`) — fine for local dev, lock down before any deploy.

---

## Frontend details

- **Next.js 16** with App Router, **React 19**, **Tailwind CSS v4**, TypeScript
- Two input paths:
  - **File upload** — accepts any audio format the browser's `AudioContext.decodeAudioData()` understands
  - **Mic recording** — uses `MediaRecorder` API, returns WebM/Opus or MP4/AAC depending on browser
- **Client-side WAV conversion** — both inputs go through `blobToWav()`, which decodes via `AudioContext` and re-encodes as 16-bit PCM WAV. This avoids needing ffmpeg on the server.
- Result UI shows: emoji + emotion label, confidence %, sorted probability bars per emotion, inline audio player.
- Configurable via `NEXT_PUBLIC_API_URL` env var (defaults to `http://localhost:8000`).

---

## Tech stack summary

| Layer | Tools |
|---|---|
| Audio features | librosa 0.11, soundfile, numpy |
| ML | scikit-learn (`MLPClassifier`, `SVC`, `StandardScaler`, `LabelEncoder`) |
| Model serialization | joblib |
| Notebook | Jupyter, matplotlib, seaborn |
| API | FastAPI, uvicorn, python-multipart |
| Frontend | Next.js 16, React 19, Tailwind CSS v4, TypeScript 5 |
| Audio decoding (browser) | Web Audio API (`AudioContext`, `MediaRecorder`) |

---

## Known limitations

- ~57% accuracy means roughly **2 in 5 predictions are wrong** on unseen speakers — a baseline, not a production-ready model.
- Trained on **acted, scripted English** ("Kids are talking by the door" / "Dogs are sitting by the door"). Real-world spontaneous speech, accents, and short utterances will degrade further.
- `sad` is consistently the hardest class — overlaps acoustically with `calm` and `neutral`.
- No noise-robustness training: background noise will hurt accuracy.
- Only 96 training examples for `neutral` (half of the others) — class imbalance.

---

## Improvement roadmap

Things we can add next (in rough order of impact-per-effort):

1. **Data augmentation** — pitch shift, time stretch, additive noise. Should add 5–10%.
2. **Better hyperparameter search** — grid/random search over SVM `C, γ` and MLP architecture. Probably +2–4%.
3. **Replace mean+std pooling with a small CNN** on log-mel spectrograms (PyTorch). Reach 70–75% range.
4. **Add more datasets** — TESS, CREMA-D, EmoDB. Lots more actors → far better generalization.
5. **Class-balanced loss / oversampling** for `neutral`.
6. **Real-time streaming** — chunk mic input every 1–2s and update prediction live.
7. **Lock down CORS** + add API key before any public deploy.
