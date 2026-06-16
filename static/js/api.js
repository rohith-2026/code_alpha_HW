async function predictImageBlob(blob, recognitionMode = "all") {
  const formData = new FormData();
  formData.append("file", blob, "character.png");
  formData.append("recognition_mode", recognitionMode);

  const response = await fetch("/api/predict", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Prediction failed." }));
    throw new Error(error.detail || "Prediction failed.");
  }

  return response.json();
}

function goToResult(result) {
  const params = new URLSearchParams({
    prediction: result.prediction,
    confidence: result.confidence,
  });
  if (Array.isArray(result.top_predictions)) {
    params.set(
      "alternatives",
      result.top_predictions
        .map((item) => `${item.prediction} ${(item.confidence * 100).toFixed(1)}%`)
        .join(", ")
    );
  }
  window.location.href = `/result?${params.toString()}`;
}
