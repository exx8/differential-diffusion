from diffusers import  KandinskyV22PriorPipeline
from diff_pipe import KandinskyV22DiffImg2ImgPipeline
from diffusers.utils import load_image
import torch
from torchvision.transforms import ToTensor
from PIL import Image
from torchvision import transforms

device = "cuda"

pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16
)
pipe_prior.to(device)

# Define the prompt
prompt = "painting of a mountain landscape with a meadow and a forest, meadow background"
negative_prompt = "blurry, shadow polaroid photo, scary angry pose"

# Generate image embeddings using the prior pipeline
image_emb, negative_image_emb = pipe_prior(prompt, negative_prompt, return_dict=False)

# Initialize the image-to-image pipeline
pipe = KandinskyV22DiffImg2ImgPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16
)
pipe.to(device)



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
    image = preprocess_image(imageFile)

with Image.open("assets/map3.jpg") as mapFile:
    map = preprocess_map(mapFile)

edited_image = pipe(
        image=image,
        image_embeds=image_emb,
        negative_image_embeds=negative_image_emb,
        height=512,
        width=512,
        num_inference_steps=150,
        strength=1,
        map=map,
    ).images[0]
edited_image.save("output.png")

print("Done!")