from gradio import Column, Slider, Image, on
from PIL.Image import Image as pil_image

from utilities.gradient import create_gradient
from utilities.event_funcs import image_enhancement_change, image_transform_change

from .image_edit_block import ImageEditBlock


class GradientTab:
    def __init__(self) -> None:
        from utilities.event_funcs import gradient_calculate

        self.default_image = Image(
            value=gradient_calculate(512, 512, 1.0, 1.0, 1.0, False, False, False),
            image_mode="L",
            sources=None,
            type="pil",
            label="Gradient Mask",
            interactive=False,
            width=256,
            height=256,
        )

        self.width_slider = Slider(
            minimum=512,
            maximum=4096,
            value=512,
            step=1,
            label="Gradient Image Width",
            interactive=True,
        )

        self.height_slider = Slider(
            minimum=512,
            maximum=4096,
            value=512,
            step=1,
            label="Gradient Image Height",
            interactive=True,
        )

        self.strength_slider = Slider(
            minimum=0,
            maximum=20,
            value=1,
            step=0.5,
            label="Gradient Strength",
            interactive=True,
        )

        self.image_edit = ImageEditBlock()

    def render(self) -> None:
        self.default_image.render()
        self.width_slider.render()
        self.height_slider.render()
        self.strength_slider.render()
        self.image_edit.render()

    def attach_event(
        self,
        output_image,
    ) -> None:
        @on(
            triggers=[
                self.width_slider.release,
                self.height_slider.release,
                self.strength_slider.release,
            ],
            inputs=[
                self.width_slider,
                self.height_slider,
                self.strength_slider,
                self.image_edit.brightness_slider,
                self.image_edit.contrast_slider,
                self.image_edit.flip_horizontal_checkbox,
                self.image_edit.to_vertical_checkbox,
                self.image_edit.flip_vertical_checkbox,
            ],
            outputs=output_image,
            show_progress="hidden",
        )
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

        self.image_edit.attach_event(self.default_image, output_image)
