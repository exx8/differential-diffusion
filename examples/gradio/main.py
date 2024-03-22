from gradio import Blocks, Tab, Row, Column, Image

from utilities.event_funcs import gradient_calculate, image_enhancement_change

from tabs.gradient_tab import GradientTab
from tabs.image_mask_tab import ImageMaskTab
from tabs.generate_tab import GenerateTab

gradient_tab = GradientTab()
image_mask_tab = ImageMaskTab()
generate_tab = GenerateTab()

with Blocks() as example:
    with Row() as main_row:
        with Tab("Gradient Mask") as tab_gradient:
            gradient_tab.render()

        with Tab("Image Mask") as tab_image_mask:
            image_mask_tab.render()

        with Tab("Generate") as tab_generate:
            generate_tab.render()

        with Column():
            mask_image = Image(sources=None, label="Mask Image", width=512, height=512)
            output_image = Image(sources=None, label="Output Image")

    tab_gradient.select(
        gradient_calculate,
        inputs=[
            gradient_tab.width_slider,
            gradient_tab.height_slider,
            gradient_tab.strength_slider,
            gradient_tab.image_edit.brightness_slider,
            gradient_tab.image_edit.contrast_slider,
            gradient_tab.image_edit.flip_horizontal_checkbox,
            gradient_tab.image_edit.to_vertical_checkbox,
            gradient_tab.image_edit.flip_vertical_checkbox,
        ],
        outputs=mask_image,
        show_progress="hidden",
    )

    tab_image_mask.select(
        image_enhancement_change,
        inputs=[
            image_mask_tab.mask_image,
            image_mask_tab.image_edit.brightness_slider,
            image_mask_tab.image_edit.contrast_slider,
        ],
        outputs=mask_image,
        show_progress="hidden",
    )

    gradient_tab.attach_event(mask_image)
    image_mask_tab.attach_event(mask_image)
    generate_tab.attach_event(mask_image, output_image)

if __name__ == "__main__":
    example.launch()
