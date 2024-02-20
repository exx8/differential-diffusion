from diffusers import DiffusionPipeline
from diff_pipe_I import DiffIFImg2ImgPipeline
from diff_pipe_II import IFDiffImg2ImgSuperResolutionPipeline
from diffusers.utils import pt_to_pil

import torch

from PIL import Image
import torchvision.transforms as transforms

device = "cuda"


def preprocess_image(image):
    image = image.convert("RGB")
    image = transforms.CenterCrop((image.size[1] // 64 * 64, image.size[0] // 64 * 64))(image)
    return image


def preprocess_map(map):
    map = map.convert("L")
    map = transforms.CenterCrop((map.size[1] // 64 * 64, map.size[0] // 64 * 64))(map)
    # convert to tensor
    map = transforms.ToTensor()(map)
    map = map.to(device)
    return map


with Image.open("assets/input.jpg") as imageFile:
    original_image = preprocess_image(imageFile)
with Image.open("assets/map2.jpg") as mapFile:
    map = preprocess_map(mapFile)

# We support all DeepFloyd models, you can change them here
# stage 1
stage_1 = DiffIFImg2ImgPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
stage_1.enable_model_cpu_offload()

# stage 2
stage_2 = IFDiffImg2ImgSuperResolutionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
)
stage_2.enable_model_cpu_offload()

# One day we will have DeepFloyd/IF-III till then, we use this.
# stage 3
safety_modules = {
    "feature_extractor": stage_1.feature_extractor,
    "safety_checker": stage_1.safety_checker,
    "watermarker": stage_1.watermarker,
}
stage_3 = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16
)
stage_3.enable_model_cpu_offload()

prompt = "painting of a mountain landscape with a meadow and a forest, meadow background"

# text embeds
prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)

stage_1.watermarker = None  # delete me if you want to watermark the image
# stage 1
edited_image = stage_1(
    image=original_image,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    output_type="pt",
    strength=1,
    map=map
).images
pt_to_pil(edited_image)[0].save(f"output_I.png")

stage_2.watermarker = None  # delete me if you want to watermark the image
# stage 2
edited_image = stage_2(
    image=edited_image,
    original_image=original_image,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    output_type="pt",
    strength=1,
    map=map

).images
pt_to_pil(edited_image)[0].save(f"output_II.png")

stage_3.safety_checker = None  # In this version the safety checker is broken, I cross my fingers that you will be safe.
stage_3.watermarker = None  # delete me if you want to watermark the image

edited_image = stage_3(prompt=prompt, image=edited_image, noise_level=0).images
edited_image[0].save(f"./output_III.png")
