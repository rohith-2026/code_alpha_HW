const canvas = document.getElementById("drawingCanvas");
const context = canvas.getContext("2d");
const clearButton = document.getElementById("clearCanvas");
const predictButton = document.getElementById("predictCanvas");
const statusText = document.getElementById("canvasStatus");
const recognitionMode = document.getElementById("recognitionMode");

let drawing = false;
let lastPoint = null;

function resetCanvas() {
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.lineWidth = 24;
  context.lineCap = "round";
  context.lineJoin = "round";
  context.strokeStyle = "#000000";
}

function pointerPosition(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * canvas.width,
    y: ((event.clientY - rect.top) / rect.height) * canvas.height,
  };
}

canvas.addEventListener("pointerdown", (event) => {
  event.preventDefault();
  drawing = true;
  const point = pointerPosition(event);
  lastPoint = point;
  context.beginPath();
  context.moveTo(point.x, point.y);
  context.arc(point.x, point.y, context.lineWidth / 2, 0, Math.PI * 2);
  context.fillStyle = context.strokeStyle;
  context.fill();
  context.beginPath();
  context.moveTo(point.x, point.y);
});

canvas.addEventListener("pointermove", (event) => {
  if (!drawing) return;
  event.preventDefault();
  const point = pointerPosition(event);
  if (!lastPoint) {
    lastPoint = point;
  }
  const midPoint = {
    x: (lastPoint.x + point.x) / 2,
    y: (lastPoint.y + point.y) / 2,
  };
  context.quadraticCurveTo(lastPoint.x, lastPoint.y, midPoint.x, midPoint.y);
  context.lineTo(point.x, point.y);
  context.stroke();
  lastPoint = point;
});

window.addEventListener("pointerup", () => {
  drawing = false;
  lastPoint = null;
});

clearButton.addEventListener("click", () => {
  resetCanvas();
  statusText.textContent = "";
});

predictButton.addEventListener("click", async () => {
  statusText.textContent = "Predicting...";
  canvas.toBlob(async (blob) => {
    try {
      const result = await predictImageBlob(blob, recognitionMode.value);
      goToResult(result);
    } catch (error) {
      statusText.textContent = error.message;
    }
  }, "image/png");
});

resetCanvas();
