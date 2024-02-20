import torch
from PIL import Image
from torchvision import transforms
from diff_pipe import StableDiffusionXLDiffImg2ImgPipeline

device = "cuda"

base = StableDiffusionXLDiffImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to(device)

refiner = StableDiffusionXLDiffImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to(device)


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


with Image.open("assets/input2.jpg") as imageFile:
    image = preprocess_image(imageFile)

with Image.open("assets/map2.jpg") as mapFile:
    map = preprocess_map(mapFile)

prompt = ["painting of a mountain landscape with a meadow and a forest, meadow background"]
negative_prompt = ["blurry, shadow polaroid photo, scary angry pose"]

edited_images = base(prompt=prompt, original_image=image, image=image, strength=1, guidance_scale=17.5,
                     num_images_per_prompt=1,
                     negative_prompt=negative_prompt,
                     map=map,
                     num_inference_steps=100, denoising_end=0.8, output_type="latent").images

edited_images = refiner(prompt=prompt, original_image=image, image=edited_images, strength=1, guidance_scale=17.5,
                        num_images_per_prompt=1,
                        negative_prompt=negative_prompt,
                        map=map,
                        num_inference_steps=100, denoising_start=0.8).images[0]

# Despite we use here both of the refiner and the base models,
# one can use only the base model, or only the refiner (for low strengths).

edited_images.save("output.png")

print("Done!")
