from app.models.character_mapping import CHARACTER_MAPPING
from app.services.image_processor import preprocess_image
from app.services.model_loader import load_model


CHARACTER_SETS = {
    "all": set(CHARACTER_MAPPING.keys()),
    "digits": {index for index, value in CHARACTER_MAPPING.items() if value.isdigit()},
    "uppercase": {index for index, value in CHARACTER_MAPPING.items() if value.isupper()},
    "lowercase": {index for index, value in CHARACTER_MAPPING.items() if value.islower()},
}


def _allowed_indices(recognition_mode: str) -> list[int]:
    return sorted(CHARACTER_SETS.get(recognition_mode, CHARACTER_SETS["all"]))


def predict_character(image_bytes: bytes, recognition_mode: str = "all") -> dict:
    features = preprocess_image(image_bytes)
    model = load_model()
    allowed_indices = _allowed_indices(recognition_mode)

    if model is None:
        ink_score = sum(max(value, 0.0) for value in features)
        predicted_index = allowed_indices[int(ink_score * 10) % len(allowed_indices)]
        return {
            "prediction": CHARACTER_MAPPING[predicted_index],
            "confidence": 0.0,
            "mode": "placeholder",
        }

    import torch

    tensor = torch.tensor(features, dtype=torch.float32).view(1, 1, 28, 28)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
        filtered_probabilities = probabilities[:, allowed_indices]
        filtered_probabilities = filtered_probabilities / filtered_probabilities.sum(
            dim=1,
            keepdim=True,
        )
        confidence, filtered_index = torch.max(filtered_probabilities, dim=1)
        predicted_index = allowed_indices[int(filtered_index.item())]
        top_confidences, top_indices = torch.topk(
            filtered_probabilities,
            k=min(5, len(allowed_indices)),
            dim=1,
        )

    label = CHARACTER_MAPPING.get(predicted_index, "?")
    confidence_value = round(float(confidence.item()), 4)
    return {
        "prediction": label,
        "confidence": confidence_value,
        "mode": "model",
        "recognition_mode": recognition_mode,
        "uncertain": confidence_value < 0.65,
        "top_predictions": [
            {
                "prediction": CHARACTER_MAPPING.get(allowed_indices[int(index)], "?"),
                "confidence": round(float(score), 4),
            }
            for index, score in zip(top_indices[0].tolist(), top_confidences[0].tolist())
        ],
    }
