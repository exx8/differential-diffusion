import torch
from PIL import Image
from torchvision import transforms
from diff_pipe import StableDiffusionDiffImg2ImgPipeline

device = "cuda"

#This is the default model, you can use other fine tuned models as well
pipe = StableDiffusionDiffImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base",
                                                          torch_dtype=torch.float16).to(device)


def preprocess_image(image):
    image = image.convert("RGB")
    image = transforms.CenterCrop((image.size[1] // 64 * 64, image.size[0] // 64 * 64))(image)
    image = transforms.ToTensor()(image)
    image = image * 2 - 1
    image = image.unsqueeze(0).to(device)
    return image


def preprocess_map(map):
    map = map.convert("L")
    map = transforms.CenterCrop((map.size[1] // 64 * 64, map.size[0] // 64 * 64))(map)
    # convert to tensor
    map = transforms.ToTensor()(map)
    map = map.to(device)
    return map


with Image.open("assets/input.jpg") as imageFile:
    image = preprocess_image(imageFile)

with Image.open("assets/map.jpg") as mapFile:
    map = preprocess_map(mapFile)

edited_image = \
pipe(prompt=["painting of a mountain landscape with a meadow and a forest, meadow background"], image=image,
     guidance_scale=7,
     num_images_per_prompt=1,
     negative_prompt=["blurry, shadow polaroid photo, scary angry pose"], map=map, num_inference_steps=100).images[0]
edited_image.save("output.png")

print("Done!")
