from PIL import Image
from PIL.Image import Image as pil_image
import numpy as np

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
