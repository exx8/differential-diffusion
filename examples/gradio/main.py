from gradio import Blocks, Tab, Row, Column, Image, Slider, Checkbox
from utilities.gradient import (
    gradient_calculate,
    image_enhancement_change,
    image_transform_change,
)

with Blocks() as example:
    with Row() as main_row:
        with Tab("Gradient"):
            with Column() as gradient_main_column:
                gradient_image = Image(
                    value=gradient_calculate(
                        512, 512, 1.0, 1.0, 1.0, False, False, False
                    ),
                    image_mode="L",
                    type="pil",
                    label="Gradient Mask",
                    sources=["upload", "clipboard"],
                    interactive=True,
                )

                with Column() as gradient_creation_column:
                    gradient_width_slider = Slider(
                        minimum=512,
                        maximum=4096,
                        value=512,
                        step=1,
                        label="Gradient Image Width",
                        interactive=True,
                    )

                    gradient_height_slider = Slider(
                        minimum=512,
                        maximum=4096,
                        value=512,
                        step=1,
                        label="Gradient Image Height",
                        interactive=True,
                    )

                    gradient_strength_slider = Slider(
                        minimum=0,
                        maximum=20,
                        value=1,
                        step=0.5,
                        label="Gradient Strength",
                        interactive=True,
                    )

                with Column() as image_enhancement_column:
                    image_brightness_slider = Slider(
                        minimum=0.0,
                        maximum=10.0,
                        value=1.0,
                        step=0.5,
                        label="Gradient Brightness",
                        interactive=True,
                    )

                    image_contrast_slider = Slider(
                        minimum=0.0,
                        maximum=10.0,
                        value=1.0,
                        step=0.5,
                        label="Gradient Contrast",
                        interactive=True,
                    )

                with Row() as image_transformation_row:
                    image_flip_horizontal_checkbox = Checkbox(
                        value=False, label="Flip Horizontal", interactive=True
                    )

                    image_to_vertical_checkbox = Checkbox(
                        value=False, label="To Vertical", interactive=True
                    )

                    image_flip_vertical_checkbox = Checkbox(
                        value=False, label="Flip Vertical", interactive=True
                    )

        with Column():
            mask_image = Image(sources=None, label="Mask Image")
            output_image = Image(sources=None, label="Output Image")

    """
    Gradient Values Event Functions
    """

    gradient_width_slider.release(
        gradient_calculate,
        inputs=[
            gradient_width_slider,
            gradient_height_slider,
            gradient_strength_slider,
            image_brightness_slider,
            image_contrast_slider,
            image_flip_horizontal_checkbox,
            image_to_vertical_checkbox,
            image_flip_vertical_checkbox,
        ],
        outputs=mask_image,
        show_progress="hidden",
    )

    gradient_height_slider.release(
        gradient_calculate,
        inputs=[
            gradient_width_slider,
            gradient_height_slider,
            gradient_strength_slider,
            image_brightness_slider,
            image_contrast_slider,
            image_flip_horizontal_checkbox,
            image_to_vertical_checkbox,
            image_flip_vertical_checkbox,
        ],
        outputs=mask_image,
        show_progress="hidden",
    )

    gradient_strength_slider.release(
        gradient_calculate,
        inputs=[
            gradient_width_slider,
            gradient_height_slider,
            gradient_strength_slider,
            image_brightness_slider,
            image_contrast_slider,
            image_flip_horizontal_checkbox,
            image_to_vertical_checkbox,
            image_flip_vertical_checkbox,
        ],
        outputs=mask_image,
        show_progress="hidden",
    )

    image_brightness_slider.release(
        image_enhancement_change,
        inputs=[
            gradient_image,
            image_brightness_slider,
            image_contrast_slider,
        ],
        outputs=mask_image,
        show_progress="hidden",
    )

    image_contrast_slider.release(
        image_enhancement_change,
        inputs=[
            gradient_image,
            image_brightness_slider,
            image_contrast_slider,
        ],
        outputs=mask_image,
        show_progress="hidden",
    )

    image_flip_horizontal_checkbox.select(
        image_transform_change,
        inputs=[
            gradient_image,
            image_flip_horizontal_checkbox,
            image_to_vertical_checkbox,
            image_flip_vertical_checkbox,
        ],
        outputs=mask_image,
        show_progress=False,
    )

    image_to_vertical_checkbox.select(
        image_transform_change,
        inputs=[
            gradient_image,
            image_flip_horizontal_checkbox,
            image_to_vertical_checkbox,
            image_flip_vertical_checkbox,
        ],
        outputs=mask_image,
        show_progress="hidden",
    )

    image_flip_vertical_checkbox.select(
        image_transform_change,
        inputs=[
            gradient_image,
            image_flip_horizontal_checkbox,
            image_to_vertical_checkbox,
            image_flip_vertical_checkbox,
        ],
        outputs=mask_image,
        show_progress="hidden",
    )

if __name__ == "__main__":
    example.launch()
