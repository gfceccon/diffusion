import torch
from diffusers import StableDiffusionPipeline
from stable_diffusion_2_1 import generate_image

def main():
    text_prompt = "A city made of diamonds"
    generate_image(text_prompt)


if __name__ == "__main__":
    main()
