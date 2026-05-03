# Pulmonary Embolism Detection & Segmentation — Web System

Flask + SQLite + Attention U-Net inference pipeline with a presentation-ready medical AI UI.

## Quick start

```bash
cd pulmonary_embolism_system
python -m pip install -r requirements.txt
python run.py
```

Open **http://127.0.0.1:5000**

### Demo login

- **Username:** `demo`  
- **Password:** `demo12345`

## Model weights

Place a compatible PyTorch `.pth` file at `models/weights/attention_unet_pe.pth` or set `MODEL_WEIGHTS` in the environment.  
If weights are missing or incompatible, the system uses a **deterministic demo segmentation** so the UI remains fully functional.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `SECRET_KEY` | Flask session secret |
| `MODEL_WEIGHTS` | Path to `.pth` checkpoint |
| `HOSPITAL_NAME` | Shown in PDF reports and UI |
| `PORT` | Server port (default 5000) |

## Project layout

- `app/` — Flask application (routes, templates, static)
- `ml/` — Attention U-Net definition + inference service
- `database/` — SQLite file (`pe_system.db`)
- `uploads/scans`, `uploads/results`, `uploads/reports` — user data output
