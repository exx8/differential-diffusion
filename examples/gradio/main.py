from gradio import Blocks, Tab, Row, Column, Image

from utilities.event_funcs import gradient_calculate, image_edit_change

from tabs.gradient_tab import GradientTab
from tabs.image_mask_tab import ImageMaskTab
from tabs.generate_tab import GenerateTab
from tabs.extract_depth_tab import ExtractDepthTab

from importlib.util import find_spec


def check_package(package_name: str) -> None:
    if find_spec(package_name):
        print(f"/_\ {package_name} is found")
    else:
        print(
            f"/_\ {package_name} is not found. Please install the {package_name} with pip"
        )
        exit(1)


print(" Differential Diffusion Gradio Example ".center(100, "-"))
print("/_\ Checking Packages")

check_package("diffusers")
check_package("transformers")
check_package("accelerate")
check_package("torch")
check_package("gradio")

print("/_\ Launching example")

gradient_tab = GradientTab()
image_mask_tab = ImageMaskTab()
extract_depth_tab = ExtractDepthTab()
generate_tab = GenerateTab()

with Blocks() as example:
    with Row() as main_row:
        with Tab("Gradient Mask") as tab_gradient:
            gradient_tab.render()

        with Tab("Image Mask") as tab_image_mask:
            image_mask_tab.render()

        with Tab("Extract Depth") as tab_extracted_depth:
            extract_depth_tab.render()

        with Tab("Generate") as tab_generate:
            generate_tab.render()

        with Column():
            mask_image = Image(
                value=gradient_calculate(512, 512, 1.0, 1.0, 1.0, False, False, False),
                sources=None,
                label="Mask Image",
                width=512,
                height=512,
            )
            output_image = Image(
                sources=None, label="Output Image", width=512, height=512
            )

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
        image_edit_change,
        inputs=[
            image_mask_tab.mask_image,
            image_mask_tab.image_edit.brightness_slider,
            image_mask_tab.image_edit.contrast_slider,
            image_mask_tab.image_edit.flip_horizontal_checkbox,
            image_mask_tab.image_edit.to_vertical_checkbox,
            image_mask_tab.image_edit.flip_vertical_checkbox,
        ],
        outputs=mask_image,
        show_progress="hidden",
    )

    tab_extracted_depth.select(
        image_edit_change,
        inputs=[
            extract_depth_tab.extracted_depth_image,
            extract_depth_tab.image_edit.brightness_slider,
            extract_depth_tab.image_edit.contrast_slider,
            extract_depth_tab.image_edit.flip_horizontal_checkbox,
            extract_depth_tab.image_edit.to_vertical_checkbox,
            extract_depth_tab.image_edit.flip_vertical_checkbox,
        ],
        outputs=mask_image,
        show_progress="hidden",
    )

    gradient_tab.attach_event(mask_image)
    image_mask_tab.attach_event(mask_image)
    extract_depth_tab.attach_event(mask_image)
    generate_tab.attach_event(mask_image, output_image)


if __name__ == "__main__":
    example.launch()
