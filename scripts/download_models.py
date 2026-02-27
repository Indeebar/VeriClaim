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