from PIL import Image
from PIL.Image import Image as pil_image
from PIL.Image import Transpose, TRANSVERSE
from PIL.ImageEnhance import Brightness, Contrast
import numpy as np


def image_enhancement_change(
    image: pil_image, brightness: float, contrast: float
) -> pil_image:
    image = Brightness(image).enhance(brightness)
    image = Contrast(image).enhance(contrast)

    return image


def gradient_flip_horizontal(image: pil_image):
    return image.transpose(Transpose.FLIP_LEFT_RIGHT)


def gradient_rotate_to_vertical(image: pil_image, is_change: bool):
    return (
        image.transpose(Transpose.ROTATE_90)
        if is_change
        else image.transpose(TRANSVERSE)
    )


def gradient_flip_vertical(image: pil_image):
    return image.transpose(Transpose.FLIP_TOP_BOTTOM)


def image_transform_change(
    image: pil_image,
    is_flip_horizontal: bool,
    is_to_vertical: bool,
    is_flip_vertical: bool,
) -> pil_image:
    is_vertical_operation = not is_flip_horizontal

    if is_vertical_operation:
        if is_to_vertical:
            image = image.transpose(Transpose.ROTATE_90)

        if is_flip_vertical:
            image = image.transpose(Transpose.FLIP_TOP_BOTTOM)
    else:
        image = image.transpose(Transpose.FLIP_LEFT_RIGHT)

    return image


def gradient_calculate(
    image_width: float,
    image_height: float,
    strength: float,
    brightness: float,
    contrast: float,
    is_flip_horizontal: bool,
    is_to_vertical: bool,
    is_flip_vertical: bool,
) -> pil_image:
    image = create_gradient(image_width, image_height, strength)
    image = image_enhancement_change(image, brightness, contrast)
    image = image_transform_change(
        image, is_flip_horizontal, is_to_vertical, is_flip_vertical
    )

    return image


# linear interpolation
def l_interp(start: list[int], end: list[int], alpha: float) -> list[int]:
    return [
        min(255, max(0, int(start[index] + alpha * (end[index] - start[index]))))
        for index in range(3)
    ]


def create_gradient(
    image_width: int,
    image_height: int,
    strength: float,
) -> pil_image:
    start_pixel = (255, 255, 255)  # white
    end_pixel = (0, 0, 0)  # black

    row_pixels = []

    for width in range(image_width):
        current_pixel = l_interp(
            start_pixel, end_pixel, ((float(width) / float(image_width)) * strength)
        )

        row_pixels.append(current_pixel)

    image_pixels = []
    image_pixels.extend(row_pixels * image_height)
    image_array = np.array(image_pixels, dtype=np.uint8)
    image_array = np.reshape(image_array, [image_height, image_width, 3])

    image = Image.fromarray(image_array)

    image.convert("L")

    return image
