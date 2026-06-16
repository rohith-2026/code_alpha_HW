const input = document.getElementById("imageInput");
const button = document.getElementById("uploadButton");
const statusText = document.getElementById("uploadStatus");
const recognitionMode = document.getElementById("recognitionMode");

button.addEventListener("click", async () => {
  const file = input.files[0];
  if (!file) {
    statusText.textContent = "Choose an image first.";
    return;
  }

  statusText.textContent = "Predicting...";
  try {
    const result = await predictImageBlob(file, recognitionMode.value);
    goToResult(result);
  } catch (error) {
    statusText.textContent = error.message;
  }
});
