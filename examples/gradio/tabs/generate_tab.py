from gradio import Column, Row, Textbox, Image, Button, Slider
from utilities.SD2.diff_pipe import StableDiffusionDiffImg2ImgPipeline
from torchvision import transforms
from torch.cuda import is_available
from PIL import Image as pil_image

device = "cuda" if is_available() else "cpu"


def preprocess_image(image):
    image = image.convert("RGB")
    image = transforms.CenterCrop((image.size[1] // 64 * 64, image.size[0] // 64 * 64))(
        image
    )
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


class GenerateTab:
    def __init__(self) -> None:
        self.config_row = Row()
        self.parameter_column = Column()

        from os.path import join

        self.input_image = Image(
            value=pil_image.open(join("assets", "input.jpg")),
            width=512,
            height=512,
            show_download_button=False,
            sources=["upload", "clipboard"],
            interactive=True,
        )

        self.guidance_scale_slider = Slider(
            minimum=0,
            maximum=20,
            value=7,
            label="Guidance Scale",
            step=0.5,
            interactive=True,
        )

        self.number_of_steps_slider = Slider(
            minimum=0,
            maximum=200,
            step=1,
            value=100,
            label="Inference Step",
            interactive=True,
        )

        self.strength_slider = Slider(
            minimum=0, maximum=1, value=1, step=0.1, label="Strength", interactive=True
        )

        self.positive_prompt_textbox = Textbox(
            value="painting of a mountain landscape with a meadow and a forest, meadow background, anime countryside landscape, anime nature wallpap, anime landscape wallpaper, studio ghibli landscape, anime landscape, mountain behind meadow, anime background art, studio ghibli environment, background of flowery hill, anime beautiful peace scene, forrest background, anime scenery, landscape background, background art, anime scenery concept art",
            max_lines=300,
            label="Positive Prompt",
        )

        self.negative_prompt_textbox = Textbox(
            value="blurry, shadow polaroid photo, scary angry pose, worn decay texture, portrait fashion model, piercing stare, bruised face, demoness",
            max_lines=300,
            label="Negative Prompt",
        )

        self.generate_button = Button(value="Generate")

        self.pipe = StableDiffusionDiffImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base",
            device_map="auto",
        )

    def render(self) -> None:
        with self.parameter_column:
            self.guidance_scale_slider.render()
            self.number_of_steps_slider.render()
            self.strength_slider.render()

        with self.config_row:
            self.input_image.render()
            self.parameter_column.render()

        self.config_row.render()
        self.positive_prompt_textbox.render()
        self.negative_prompt_textbox.render()
        self.generate_button.render()

    def attach_event(self, mask_image, result_image) -> None:

        def generate_image(
            map_image,
            input_image,
            guidance_scale,
            inference_steps,
            strength,
            positive_prompt,
            negative_prompt,
        ):
            processed_mask_image = preprocess_map(pil_image.fromarray(map_image))
            processed_input_image = preprocess_image(pil_image.fromarray(input_image))

            return self.pipe(
                prompt=[positive_prompt],
                image=processed_input_image,
                num_images_per_prompt=1,
                negative_prompt=[negative_prompt],
                map=processed_mask_image,
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
            ).images[0]

        self.generate_button.click(
            generate_image,
            inputs=[
                mask_image,
                self.input_image,
                self.guidance_scale_slider,
                self.number_of_steps_slider,
                self.strength_slider,
                self.positive_prompt_textbox,
                self.negative_prompt_textbox,
            ],
            outputs=result_image,
            show_progress="full",
        )
