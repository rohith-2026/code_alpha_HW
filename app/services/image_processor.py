from io import BytesIO

from PIL import Image, ImageOps


IMAGE_SIZE = 28
INK_THRESHOLD = 220
MIN_INK_PIXELS = 8


class BlankImageError(ValueError):
    pass


def load_grayscale_image(image_bytes: bytes) -> Image.Image:
    image = Image.open(BytesIO(image_bytes)).convert("RGBA")
    background = Image.new("RGBA", image.size, "WHITE")
    return Image.alpha_composite(background, image).convert("L")


def _border_average(image: Image.Image) -> float:
    width, height = image.size
    pixels = []
    pixels.extend(image.crop((0, 0, width, 1)).getdata())
    pixels.extend(image.crop((0, height - 1, width, height)).getdata())
    pixels.extend(image.crop((0, 0, 1, height)).getdata())
    pixels.extend(image.crop((width - 1, 0, width, height)).getdata())
    return sum(pixels) / len(pixels)


def normalize_character_image(image: Image.Image) -> Image.Image:
    image = image.convert("L")
    if _border_average(image) < 128:
        image = ImageOps.invert(image)

    ink_mask = image.point(lambda value: 255 if value < INK_THRESHOLD else 0)
    bbox = ink_mask.getbbox()
    if bbox is None:
        raise BlankImageError("No character detected. Draw or upload a darker character.")

    ink_pixels = sum(1 for value in ink_mask.getdata() if value)
    if ink_pixels < MIN_INK_PIXELS:
        raise BlankImageError("Character is too light or too small to detect.")

    cropped = image.crop(bbox)
    width, height = cropped.size
    side = max(width, height)
    padding = max(8, int(side * 0.25))
    square_side = side + padding * 2
    square = Image.new("L", (square_side, square_side), 255)
    square.paste(cropped, ((square_side - width) // 2, (square_side - height) // 2))
    return square.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)


def image_to_features(image: Image.Image) -> list[float]:
    normalized_image = normalize_character_image(image)
    get_pixels = getattr(normalized_image, "get_flattened_data", normalized_image.getdata)
    pixels = list(get_pixels())
    normalized = [((255 - value) / 255.0 - 0.5) / 0.5 for value in pixels]
    return normalized


def preprocess_image(image_bytes: bytes):
    return image_to_features(load_grayscale_image(image_bytes))
