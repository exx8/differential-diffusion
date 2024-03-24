from transformers import pipeline
from gradio import Column, Image, Row
import numpy as np
from PIL import Image as pil_image
from torch.nn.functional import interpolate

from .image_edit_block import ImageEditBlock


class ExtractDepthTab:
    def __init__(self) -> None:
        self.main_column = Column()

        self.images_row = Row()

        self.wanted_image = Image(
            sources=["upload", "clipboard"],
            type="pil",
            width=512,
            height=512,
            label="Input Image",
        )

        from os.path import join

        self.extracted_depth_image = Image(
            value=pil_image.open(join("assets", "map2.jpg")),
            width=512,
            height=512,
            type="pil",
            label="Extracted Depth Image",
            sources=None,
        )

        self.pipe = pipeline(
            "depth-estimation",
            model="Intel/dpt-large",
            framework="pt",
            torch_dtype="auto",
        )

        self.image_edit = ImageEditBlock()

    def render(self) -> None:
        with self.images_row:
            self.wanted_image.render()
            self.extracted_depth_image.render()

        self.images_row.render()
        self.image_edit.render()

    def attach_event(self, output_image) -> None:
        def extract_depth(given_image):
            outputs = self.pipe(given_image)
            predicted_depth = outputs["predicted_depth"]

            prediction = interpolate(
                input=predicted_depth.unsqueeze(1),
                size=given_image.size[::-1],
                mode="bicubic",
                align_corners=False,
            )

            output = prediction.squeeze().cpu().numpy()
            formatted = (output * 255 / np.max(output)).astype("uint8")
            pil_formatted_image = pil_image.fromarray(formatted)

            return [pil_formatted_image, pil_formatted_image]

        self.wanted_image.upload(
            extract_depth,
            inputs=self.wanted_image,
            outputs=[self.extracted_depth_image, output_image],
            show_progress="full",
        )

        self.image_edit.attach_event(self.extracted_depth_image, output_image)
