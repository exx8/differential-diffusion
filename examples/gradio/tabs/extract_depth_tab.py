from transformers import pipeline
from gradio import Column, Image

from .image_edit_block import ImageEditBlock

from torch.nn.functional import interpolate

import numpy as np

from PIL import Image as pil_image


class ExtractDepthTab:
    def __init__(self) -> None:
        self.main_column = Column()

        self.wanted_image = Image(
            sources=["upload", "clipboard"],
            width=256,
            height=256,
            label="Input Image",
        )
        self.main_column.add(self.wanted_image)

        self.extracted_depth_image = Image(
            width=256,
            height=256,
            label="Extracted Depth Image",
            sources=None,
        )
        self.main_column.add(self.extracted_depth_image)

        self.pipe = pipeline(
            "depth-estimation",
            model="Intel/dpt-large",
            framework="pt",
            torch_dtype="auto",
        )

        self.image_edit = ImageEditBlock()

    def render(self) -> None:
        self.wanted_image.render()
        self.extracted_depth_image.render()
        self.image_edit.render()

    def attach_event(self, output_image) -> None:
        def extract_depth(given_image):
            pil_given_image = pil_image.fromarray(given_image)
            outputs = self.pipe(pil_given_image)
            predicted_depth = outputs["predicted_depth"]

            prediction = interpolate(
                input=predicted_depth.unsqueeze(1),
                size=pil_given_image.size[::-1],
                mode="bicubic",
                align_corners=False,
            )

            output = prediction.squeeze().cpu().numpy()
            formatted = (output * 255 / np.max(output)).astype("uint8")
            return pil_image.fromarray(formatted)

        self.wanted_image.upload(
            extract_depth,
            inputs=self.wanted_image,
            outputs=self.extracted_depth_image,
            show_progress="full",
        )
