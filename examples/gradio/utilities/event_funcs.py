from PIL.Image import Image as pil_image
from PIL.Image import Transpose, TRANSVERSE
from PIL.ImageEnhance import Brightness, Contrast
from gradio import Column

from .gradient import create_gradient


def image_enhancement_change(
    image: pil_image, brightness: float, contrast: float
) -> pil_image:
    image = Brightness(image).enhance(brightness)
    image = Contrast(image).enhance(contrast)

    return image


def image_transform_change(
    image: pil_image,
    is_flip_horizontal: bool,
    is_to_vertical: bool,
    is_flip_vertical: bool,
) -> pil_image:
    if is_to_vertical:
        image = image.transpose(Transpose.ROTATE_90)

    if is_flip_vertical:
        image = image.transpose(Transpose.FLIP_TOP_BOTTOM)

    if is_flip_horizontal:
        image = image.transpose(Transpose.FLIP_LEFT_RIGHT)

    return image


def gradient_calculate(
    image_width: int,
    image_height: int,
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


def image_upload_change():
    return Column(visible=False)


def image_clear_change():
    return Column(visible=True)
