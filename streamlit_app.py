import streamlit as st
import torch
from transformers import DiffusionPipeline

# Load the Diffusion model
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
pipe.to("cuda")

# Streamlit app
st.title("Image Generation Tool with Diffusion Model")

# User input
user_input = st.text_input("Enter a prompt for image generation:")

# Generate image on button click
if st.button("Generate Image"):
    if user_input:
        try:
            # Generate image
            with torch.no_grad():
                image = pipe(user_input, num_return_sequences=1)[0]

            # Display generated image
            st.image(image.permute(1, 2, 0).cpu().numpy(), caption='Generated Image', use_column_width=True)
        except Exception as e:
            st.error(f"An error occurred: {e}")
