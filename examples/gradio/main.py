from gradio import Blocks, Tab, Row, Column, Image, Slider, Checkbox, on
from utilities.gradient import create_gradient
from PIL.Image import Image as pil_image
from PIL.Image import Transpose, TRANSVERSE
from PIL.ImageEnhance import Brightness, Contrast
from utilities.event_funcs import image_enhancement_change


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


from tabs.gradient_tab import GradientTab

gradient_tab = GradientTab()

with Blocks() as example:
    with Row() as main_row:
        with Tab("Gradient Mask"):
            gradient_tab.render()

        with Tab("Image Mask"):
            with Column() as uploaded_image_main_column:
                uploaded_mask_image = Image(
                    image_mode="L",
                    sources=["upload", "clipboard"],
                    type="pil",
                    label="Image Mask",
                    interactive=True,
                )

            with Column() as uploaded_image_enhancement_column:
                uploaded_image_brightness_slider = Slider(
                    minimum=0.0,
                    maximum=10.0,
                    value=1.0,
                    step=0.5,
                    label="Image Brightness",
                    interactive=True,
                )

                uploaded_image_contrast_slider = Slider(
                    minimum=0.0,
                    maximum=10.0,
                    value=1.0,
                    step=0.5,
                    label="Image Contrast",
                    interactive=True,
                )

            with Row() as uploaded_image_transformation_row:
                uploaded_image_flip_horizontal_checkbox = Checkbox(
                    value=False, label="Flip Horizontal", interactive=True
                )

                uploaded_image_to_vertical_checkbox = Checkbox(
                    value=False, label="To Vertical", interactive=True
                )

                uploaded_image_flip_vertical_checkbox = Checkbox(
                    value=False, label="Flip Vertical", interactive=True
                )

        with Column():
            mask_image = Image(sources=None, label="Mask Image")
            output_image = Image(sources=None, label="Output Image")

    """
    Gradient Event Functions
    """

    gradient_tab.attach_event(mask_image)

    """
    Uploaded Image Event Functions
    """

if __name__ == "__main__":
    example.launch()
