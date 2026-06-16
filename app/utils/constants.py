from pathlib import Path


APP_NAME = "Handwritten Character Recognition"

BASE_DIR = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = str(BASE_DIR / "templates")
STATIC_DIR = str(BASE_DIR / "static")
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
MODEL_PATH = BASE_DIR / "trained_models" / "best_character_model.pth"

