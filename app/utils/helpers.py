from pathlib import Path
from uuid import uuid4


def unique_filename(original_name: str) -> str:
    suffix = Path(original_name).suffix.lower()
    return f"{uuid4().hex}{suffix}"

