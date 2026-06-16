from pathlib import Path
import random
import sys

from PIL import Image, ImageDraw, ImageFilter, ImageFont
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.models.character_mapping import CHARACTER_MAPPING
from app.models.cnn_model import CharacterCNN
from app.services.image_processor import image_to_features
from app.utils.constants import MODEL_PATH


CHARACTERS = [CHARACTER_MAPPING[index] for index in range(len(CHARACTER_MAPPING))]
IMAGE_SIZE = 28
SAMPLES_PER_CLASS = 140
BATCH_SIZE = 128
EPOCHS = 22


def font_paths() -> list[Path]:
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf"),
        Path("C:/Windows/Fonts/consola.ttf"),
        Path("C:/Windows/Fonts/times.ttf"),
        Path("C:/Windows/Fonts/timesbd.ttf"),
        Path("C:/Windows/Fonts/verdana.ttf"),
        Path("C:/Windows/Fonts/verdanab.ttf"),
        Path("C:/Windows/Fonts/tahoma.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf"),
    ]
    return [path for path in candidates if path.exists()]


def load_font(paths: list[Path], size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if not paths:
        return ImageFont.load_default()
    return ImageFont.truetype(str(random.choice(paths)), size=size)


def render_character(character: str, fonts: list[Path]) -> list[float]:
    canvas = Image.new("L", (96, 96), 255)
    draw = ImageDraw.Draw(canvas)
    font = load_font(fonts, random.randint(34, 64))
    bbox = draw.textbbox((0, 0), character, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = (96 - width) // 2 - bbox[0] + random.randint(-12, 12)
    y = (96 - height) // 2 - bbox[1] + random.randint(-12, 12)
    draw.text((x, y), character, fill=random.randint(0, 40), font=font)

    angle = random.uniform(-28, 28)
    canvas = canvas.rotate(angle, fillcolor=255)
    if random.random() < 0.45:
        shear = random.uniform(-0.18, 0.18)
        canvas = canvas.transform(
            canvas.size,
            Image.Transform.AFFINE,
            (1, shear, 0, random.uniform(-0.08, 0.08), 1, 0),
            fillcolor=255,
        )
    if random.random() < 0.35:
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.7)))

    return image_to_features(canvas)


def build_dataset() -> TensorDataset:
    fonts = font_paths()
    samples = []
    labels = []
    for label, character in enumerate(CHARACTERS):
        for _ in range(SAMPLES_PER_CLASS):
            samples.append(render_character(character, fonts))
            labels.append(label)

    x = torch.tensor(samples, dtype=torch.float32).view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
    y = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(x, y)


def train() -> None:
    random.seed(7)
    torch.manual_seed(7)

    dataset = build_dataset()
    train_size = int(len(dataset) * 0.9)
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(7),
    )
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    model = CharacterCNN(num_classes=len(CHARACTERS))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        correct = 0
        total = 0
        total_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * labels.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += labels.size(0)

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                logits = model(images)
                val_correct += int((logits.argmax(dim=1) == labels).sum().item())
                val_total += labels.size(0)
        print(
            "epoch="
            f"{epoch + 1} loss={total_loss / total:.4f} "
            f"train_accuracy={correct / total:.4f} "
            f"val_accuracy={val_correct / val_total:.4f}"
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"saved={MODEL_PATH}")


if __name__ == "__main__":
    train()
