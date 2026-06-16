from functools import lru_cache
from pathlib import Path

from app.models.character_mapping import CHARACTER_MAPPING
from app.utils.constants import MODEL_PATH


@lru_cache(maxsize=1)
def load_model():
    path = Path(MODEL_PATH)

    if not path.exists() or path.stat().st_size == 0:
        return None

    try:
        import torch

        from app.models.cnn_model import CharacterCNN
    except ImportError:
        return None

    model = CharacterCNN(num_classes=len(CHARACTER_MAPPING))
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model
