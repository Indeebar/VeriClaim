# VeriClaim — Phase 6 Part B
# Prepare Codebase for Deployment

---

## Context

Models are uploaded to Hugging Face at:
    Indeebar/VeriClaim_models

Both files confirmed uploaded:
    best_model.pt       (16.3 MB)
    xgb_fraud_model.pkl (1.02 MB)

Your job is to write the files below exactly as specified,
then commit and push to GitHub. Do not modify any other files.

---

## Step 1 — Install huggingface_hub

```bash
pip install huggingface_hub==0.23.4
pip freeze > requirements.txt
```

---

## Step 2 — Create scripts/__init__.py

Create an empty file at:
    scripts/__init__.py

---

## Step 3 — Create scripts/download_models.py

Write exactly this content:

```python
"""
Downloads model files from Hugging Face at startup.
Only downloads if files do not already exist locally.
"""
import os
import shutil
from huggingface_hub import hf_hub_download

HF_REPO = os.environ.get('HF_REPO', 'Indeebar/VeriClaim_models')

MODELS = [
    {
        'filename':   'best_model.pt',
        'local_path': 'models/damage_classifier/best_model.pt'
    },
    {
        'filename':   'xgb_fraud_model.pkl',
        'local_path': 'models/fraud_classifier/xgb_fraud_model.pkl'
    }
]


def download_models():
    os.makedirs('models/damage_classifier', exist_ok=True)
    os.makedirs('models/fraud_classifier',  exist_ok=True)

    for m in MODELS:
        if os.path.exists(m['local_path']):
            size = os.path.getsize(m['local_path'])
            print(f"[HF] Already exists: {m['local_path']} ({size} bytes)")
            continue

        print(f"[HF] Downloading {m['filename']} from {HF_REPO}...")
        path = hf_hub_download(
            repo_id   = HF_REPO,
            filename  = m['filename'],
            local_dir = '.'
        )
        shutil.move(path, m['local_path'])
        size = os.path.getsize(m['local_path'])
        print(f"[HF] Saved: {m['local_path']} ({size} bytes)")


if __name__ == '__main__':
    download_models()
    print('[HF] All models ready.')
```

---

## Step 4 — Update api/main.py

Replace the entire lifespan function with this.
Do not change anything else in main.py.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Download models from Hugging Face if not present
    print('Checking model files...')
    from scripts.download_models import download_models
    download_models()

    # Load all models into memory
    print('Loading models...')
    load_model('models/damage_classifier/best_model.pt')
    load_nlp_model('models/claim_nlp/fraud_patterns.json')
    load_fraud_model('models/fraud_classifier/xgb_fraud_model.pkl')
    load_explainer('models/fraud_classifier/xgb_fraud_model.pkl')
    print('All models loaded. API ready.')
    yield
```

---

## Step 5 — Update app.py

Find this line near the top of app.py:

    API_URL = 'http://localhost:8000'

Replace it with exactly this:

```python
import os
API_URL = os.environ.get('API_URL', 'https://vericlaim-api.onrender.com')
```

Do not change anything else in app.py.

---

## Step 6 — Create render.yaml

Create render.yaml in the VeriClaim root with exactly this content:

```yaml
services:
  - type: web
    name: vericlaim-api
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn api.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: HF_REPO
        value: Indeebar/VeriClaim_models
      - key: PYTHON_VERSION
        value: 3.11.0
```

---

## Step 7 — Test the download script locally

Run this to confirm it works:

```bash
python scripts/download_models.py
```

Expected output:
    [HF] Already exists: models/damage_classifier/best_model.pt (16341753 bytes)
    [HF] Already exists: models/fraud_classifier/xgb_fraud_model.pkl (1024376 bytes)
    [HF] All models ready.

If it prints Already exists for both files the script is working correctly.
If it tries to download, let it finish — it means the path resolution worked.

---

## Step 8 — Test uvicorn still works locally

```bash
uvicorn api.main:app --port 8000
```

Must still print all 4 model load messages and API ready.
If it does, stop uvicorn with Ctrl+C.

---

## Step 9 — Commit and push

```bash
git add .
git status
```

Confirm git status does NOT show:
    - any .pt files
    - any .pkl files
    - kaggle.json

Then commit:

```bash
git commit -m "Phase 6: Hugging Face model download, render.yaml, deployment config"
git push origin main
```

---

## Done — Report back with:

1. Output of python scripts/download_models.py
2. Confirmation that uvicorn still starts correctly
3. Confirmation that git push succeeded
4. The exact contents of render.yaml
