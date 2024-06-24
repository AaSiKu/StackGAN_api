from HelperFunctions import *
import streamlit as st
from PIL import Image
import io

def Generate_image(text):
    # Dummy implementation for illustration
    # Replace this with your actual image generation model
    embeddings = Tokenize(text)
    tensor = generate_images(embeddings)
    tensor = tensor.squeeze().permute(1, 2, 0)
    # img = Image.new('RGB', (200, 100), color = (73, 109, 137))
    tensor = (tensor * 255).clamp(0, 255).byte()  # Scale to [0, 255] and convert to byte
    array = tensor.cpu().numpy()  # Convert to NumPy array
    return Image.fromarray(array)

    return img

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    
    .custom-text-input > div > input {
        font-size: 25px !important;
        height: 50px !important;
        font-family: 'Roboto', sans-serif;
    }
    .custom-label {
        font-size: 25px !important;
        font-weight: bold;
        font-family: 'Roboto', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Pokémon Image Generator")

st.markdown('<div class="custom-label">Enter a sentence:</div>', unsafe_allow_html=True)
input_text = st.text_input("", key='input', help='Type your sentence here', label_visibility='collapsed')


st.markdown(
    """
    <style>
    .custom-text-input {
        display: flex;
        flex-direction: column;
    }
    .custom-text-input label {
        font-size: 20px;
    }
    .custom-text-input > div > input {
        font-size: 20px !important;
        height: 50px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if st.button("Generate Image"):
    if input_text:
        generated_image = Generate_image(input_text)
        
        img_bytes = io.BytesIO()
        generated_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        st.image(img_bytes, caption="Generated Image", use_column_width=True)
    
    else:
        st.error("Please enter a sentence.")
        
