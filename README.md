# Handwritten Character Recognition

AI-powered handwritten character recognition system built with FastAPI, PyTorch, and EMNIST ByClass. Supports digits, uppercase, and lowercase letters with real-time character prediction, confidence scoring, and an interactive drawing canvas.

## Run

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python run.py
```

Open `http://127.0.0.1:8000`.

## Optional ML Dependencies

Install these only when you are training or loading a real PyTorch checkpoint:

```powershell
.\.venv\Scripts\python -m pip install -r requirements-ml.txt
```

To rebuild the included baseline checkpoint:

```powershell
.\.venv\Scripts\python scripts\train_baseline_model.py
```

## Model

Place the trained PyTorch checkpoint at:

```text
trained_models/best_character_model.pth
```

The web app runs on the current Python version without PyTorch installed. If
PyTorch is available and the checkpoint exists, predictions use the CNN model;
otherwise the endpoint returns a placeholder result so the UI remains usable.
