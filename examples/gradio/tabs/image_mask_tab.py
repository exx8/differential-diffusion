from gradio import Column, Image
from PIL import Image as pil_image
from .image_edit_block import ImageEditBlock


class ImageMaskTab:
    def __init__(self) -> None:
        self.main_column = Column()

        from os.path import join

        self.mask_image = Image(
            value=pil_image.open(join("assets", "map2.jpg")),
            image_mode="L",
            sources=["upload", "clipboard"],
            type="pil",
            label="Image Mask",
            interactive=True,
        )

        self.image_edit = ImageEditBlock()

    def render(self) -> None:
        self.mask_image.render()
        self.image_edit.render()

    def attach_event(self, output_image) -> None:
        def upload_image(image):
            return image

        self.mask_image.upload(
            upload_image,
            inputs=self.mask_image,
            outputs=output_image,
            show_progress="hidden",
        )

        self.image_edit.attach_event(self.mask_image, output_image)
