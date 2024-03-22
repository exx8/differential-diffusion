from gradio import Blocks, Tab, Row, Column, Image, Slider, Checkbox
from utilities.gradient import (
    create_gradient,
    gradient_flip_horizontal,
    gradient_flip_vertical,
    gradient_rotate_to_vertical,
)

with Blocks() as example:
    with Row() as example_main_row:
        with Tab("Gradient"):
            with Column() as example_gradient_row:
                gradient_image = Image(
                    value=create_gradient(512, 512, 1.0),
                    image_mode="L",
                    type="pil",
                    label="Gradient Mask",
                    sources=["upload", "clipboard"],
                    interactive=True,
                )

                with Column() as example_gradient_column:
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

                    gradient_brightness_slider = Slider(
                        minimum=0.0,
                        maximum=10.0,
                        value=1.0,
                        step=0.5,
                        label="Gradient Brightness",
                        interactive=True,
                    )

                    gradient_contrast_slider = Slider(
                        minimum=0.0,
                        maximum=10.0,
                        value=1.0,
                        step=0.5,
                        label="Gradient Contrast",
                        interactive=True,
                    )

                    with Row():
                        gradient_flip_horizontal_checkbox = Checkbox(
                            value=False, label="Flip Horizontal", interactive=True
                        )

                        gradient_to_vertical_checkbox = Checkbox(
                            value=False, label="To Vertical", interactive=True
                        )

                        gradient_flip_vertical_checkbox = Checkbox(
                            value=False, label="Flip Vertical", interactive=True
                        )

        output_image = Image(sources=None)

    """
    Gradient Values Event Functions
    """

    gradient_width_slider.release(
        create_gradient,
        inputs=[
            gradient_width_slider,
            gradient_height_slider,
            gradient_strength_slider,
            gradient_brightness_slider,
            gradient_contrast_slider,
            gradient_flip_horizontal_checkbox,
            gradient_to_vertical_checkbox,
            gradient_flip_vertical_checkbox,
        ],
        outputs=gradient_image,
        show_progress="hidden",
    )

    gradient_height_slider.release(
        create_gradient,
        inputs=[
            gradient_width_slider,
            gradient_height_slider,
            gradient_strength_slider,
            gradient_brightness_slider,
            gradient_contrast_slider,
            gradient_flip_horizontal_checkbox,
            gradient_to_vertical_checkbox,
            gradient_flip_vertical_checkbox,
        ],
        outputs=gradient_image,
        show_progress="hidden",
    )

    gradient_strength_slider.release(
        create_gradient,
        inputs=[
            gradient_width_slider,
            gradient_height_slider,
            gradient_strength_slider,
            gradient_brightness_slider,
            gradient_contrast_slider,
            gradient_flip_horizontal_checkbox,
            gradient_to_vertical_checkbox,
            gradient_flip_vertical_checkbox,
        ],
        outputs=gradient_image,
        show_progress="hidden",
    )

    gradient_brightness_slider.release(
        create_gradient,
        inputs=[
            gradient_width_slider,
            gradient_height_slider,
            gradient_strength_slider,
            gradient_brightness_slider,
            gradient_contrast_slider,
            gradient_flip_horizontal_checkbox,
            gradient_to_vertical_checkbox,
            gradient_flip_vertical_checkbox,
        ],
        outputs=gradient_image,
        show_progress="hidden",
    )

    gradient_contrast_slider.release(
        create_gradient,
        inputs=[
            gradient_width_slider,
            gradient_height_slider,
            gradient_strength_slider,
            gradient_brightness_slider,
            gradient_contrast_slider,
            gradient_flip_horizontal_checkbox,
            gradient_to_vertical_checkbox,
            gradient_flip_vertical_checkbox,
        ],
        outputs=gradient_image,
        show_progress="hidden",
    )

    gradient_flip_horizontal_checkbox.select(
        gradient_flip_horizontal,
        inputs=gradient_image,
        outputs=gradient_image,
        show_progress=False,
    )

    gradient_to_vertical_checkbox.select(
        gradient_rotate_to_vertical,
        inputs=[gradient_image, gradient_to_vertical_checkbox],
        outputs=gradient_image,
        show_progress="hidden",
    )

    gradient_flip_vertical_checkbox.select(
        gradient_flip_vertical,
        inputs=gradient_image,
        outputs=gradient_image,
        show_progress="hidden",
    )

if __name__ == "__main__":
    example.launch()
