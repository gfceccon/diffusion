import os
import torch
from diffusers import StableDiffusionPipeline
import time
from PIL.Image import Image


def generate_image(prompt: str,
                   model_id: str = "stabilityai/stable-diffusion-2-1",
                   output_path: str = "./images",
                   guidance_scale: float = 7.5,
                   num_inference_steps: int = 50):

    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype="auto",
        use_safetensors=True,
        output_loading_info=True,
    ).to("cuda")

    with torch.autocast("cuda"):
        start = time.perf_counter()
        images = pipe(
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps

        )
        end = time.perf_counter()
        print(f"Time taken to generate image: {(end - start):.3f} seconds")

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i, image in enumerate(images.images):
        image.save(f"{output_path}/image_{i}.png")
        image.show()
        print(f"Image {i} saved to {output_path}/image_{i}.png")
